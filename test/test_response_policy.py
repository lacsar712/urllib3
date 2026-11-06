"""
Tests for the response-phase runtime policy stack
(``urllib3.response_policy``).

Covers:

* empty-stack pin equivalence and push/pop nesting with exception cleanup;
* ``body_admit``: deny_new / one_more on read/read1/stream/json, in-flight
  protection and per-chunk dynamic re-resolution;
* ``decode_gate``: skip_decode / force_raw raw-byte delivery without faking
  truncation/decode failures and without mutating permanent configuration;
* ``conn_return``: keep_out / force_drop at response.release_conn/close and
  the pool urlopen cleanup chokepoint, blocking-pool semantics, cross-thread
  isolation, mid-stream teardown observability.
"""

from __future__ import annotations

import contextvars
import gzip
import http.server
import threading
from io import BufferedReader, BytesIO
from unittest import mock

import pytest

from urllib3 import HTTPConnectionPool
from urllib3.exceptions import (
    BodyAdmissionDenied,
    DecodeError,
    EmptyPoolError,
    ProtocolError,
)
from urllib3.response import HTTPResponse
from urllib3.response_policy import (
    BODY_DENY_NEW,
    BODY_HONOR_CFG,
    BODY_ONE_MORE,
    CONN_FORCE_DROP,
    CONN_KEEP_OUT,
    DECODE_FORCE_RAW,
    DECODE_SKIP_DECODE,
    ConnReturnEvent,
    ResponsePolicyFrame,
    active_response_policy,
    current_frame,
    pop_response_policy,
    push_response_policy,
    response_policy,
)

# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class FakeSocket:
    def __init__(self, fileno: int = 42) -> None:
        self._fileno = fileno

    def fileno(self) -> int:
        return self._fileno

    def shutdown(self, *args: object) -> None:
        pass


class FakeConn:
    def __init__(self) -> None:
        self.sock: FakeSocket | None = FakeSocket()
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.sock = None


class FakePool:
    def __init__(self) -> None:
        self.returned: list[FakeConn] = []

    def _put_conn(self, conn: FakeConn | None) -> None:
        self.returned.append(conn)  # type: ignore[arg-type]


def make_response(
    body: bytes = b"hello world",
    *,
    headers: dict[str, str] | None = None,
    decode_content: bool = False,
    preload_content: bool = False,
    pool: FakePool | None = None,
    conn: FakeConn | None = None,
) -> HTTPResponse:
    fp = BufferedReader(BytesIO(body))
    return HTTPResponse(
        fp,
        headers=headers,
        decode_content=decode_content,
        preload_content=preload_content,
        pool=pool,  # type: ignore[arg-type]
        connection=conn,  # type: ignore[arg-type]
    )


def gzip_body(data: bytes) -> bytes:
    buf = BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(data)
    return buf.getvalue()


class PushMidRead:
    """File-like object that pushes *frame* while a read is in flight."""

    def __init__(self, data: bytes, frame: ResponsePolicyFrame) -> None:
        self._bio = BufferedReader(BytesIO(data))
        self._frame = frame
        self._pushed = False

    def _maybe_push(self) -> None:
        if not self._pushed:
            self._pushed = True
            push_response_policy(self._frame)

    def read(self, amt: int | None = None) -> bytes:
        self._maybe_push()
        return self._bio.read(amt) if amt is not None else self._bio.read()

    def read1(self, amt: int | None = None) -> bytes:
        self._maybe_push()
        return self._bio.read1(amt)

    def close(self) -> None:
        self._bio.close()

    @property
    def closed(self) -> bool:
        return self._bio.closed


# ---------------------------------------------------------------------------
# Stack mechanics
# ---------------------------------------------------------------------------


class TestStackMechanics:
    def test_empty_stack_is_honor(self) -> None:
        assert active_response_policy() is None
        assert current_frame() is None

    def test_context_manager_push_pop(self) -> None:
        with response_policy(body_admit=BODY_DENY_NEW) as frame:
            assert current_frame() is frame
            assert frame.body_admit == BODY_DENY_NEW
        assert current_frame() is None

    def test_exception_pops_frame(self) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            with response_policy(body_admit=BODY_DENY_NEW):
                raise RuntimeError("boom")
        assert current_frame() is None

    def test_nested_frames_restore_outer(self) -> None:
        with response_policy(body_admit=BODY_DENY_NEW) as outer:
            assert current_frame() is outer
            with response_policy(body_admit=BODY_HONOR_CFG) as inner:
                assert current_frame() is inner
            assert current_frame() is outer
        assert current_frame() is None

    def test_explicit_push_pop(self) -> None:
        frame = ResponsePolicyFrame(body_admit=BODY_DENY_NEW)
        push_response_policy(frame)
        assert current_frame() is frame
        pop_response_policy(frame)
        assert current_frame() is None
        # Double pop is a safe no-op.
        pop_response_policy(frame)

    def test_frame_is_its_own_context_manager(self) -> None:
        frame = ResponsePolicyFrame(conn_return=CONN_KEEP_OUT)
        with frame:
            assert current_frame() is frame
        assert current_frame() is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"body_admit": "nope"},
            {"decode_gate": "nope"},
            {"conn_return": "nope"},
        ],
    )
    def test_invalid_values_rejected(self, kwargs: dict[str, str]) -> None:
        with pytest.raises(ValueError):
            ResponsePolicyFrame(**kwargs)

    def test_child_callback_reads_live_stack_not_entry_snapshot(self) -> None:
        # An inner callback must see the frame pushed *after* the outer
        # segment entered, not a frozen snapshot.
        seen: list[object] = []

        def child_callback() -> None:
            seen.append(current_frame())

        with response_policy() as outer:
            child_callback()
            with response_policy(body_admit=BODY_DENY_NEW) as inner:
                child_callback()
            child_callback()

        assert seen == [outer, inner, outer]

    def test_context_isolation_copy_context_sees_own_stack(self) -> None:
        # A context copied before a push must not observe the later push.
        seen: list[object] = []

        def reader() -> None:
            seen.append(current_frame())

        ctx = contextvars.copy_context()
        with response_policy(body_admit=BODY_DENY_NEW) as frame:
            ctx.run(reader)
            reader()

        assert seen == [None, frame]

    def test_concurrent_scopes_do_not_cross_stack(self) -> None:
        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def worker_a() -> None:
            with response_policy(body_admit=BODY_DENY_NEW) as frame:
                barrier.wait()
                results["a"] = current_frame()
                # Frame-scoped state is private to this frame.
                assert frame.fetches_denied == 0

        def worker_b() -> None:
            with response_policy(body_admit=BODY_ONE_MORE):
                barrier.wait()
                results["b"] = current_frame()

        ta = threading.Thread(target=worker_a)
        tb = threading.Thread(target=worker_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert results["a"] is not results["b"]
        assert current_frame() is None


# ---------------------------------------------------------------------------
# body_admit
# ---------------------------------------------------------------------------


class TestBodyAdmit:
    def test_deny_new_refuses_all_body_entry_points(self) -> None:
        r = make_response(b'{"a": 1}')
        with response_policy(body_admit=BODY_DENY_NEW) as frame:
            with pytest.raises(BodyAdmissionDenied) as ei:
                r.read()
            assert ei.value.mode == BODY_DENY_NEW
            assert ei.value.response is r

            with pytest.raises(BodyAdmissionDenied):
                r.read1(64)

            with pytest.raises(BodyAdmissionDenied):
                next(iter(r.stream(amt=4)))

            with pytest.raises(BodyAdmissionDenied):
                r.json()

        assert frame.fetches_denied == 4
        # Permanent configuration is untouched.
        assert r.decode_content is False

    def test_deny_new_does_not_cancel_in_flight_read(self) -> None:
        # Push deny_new from *inside* the underlying wire read: the fetch
        # already in flight must complete with its real bytes, not be turned
        # into an empty success; only the next fetch is refused.
        deny = ResponsePolicyFrame(body_admit=BODY_DENY_NEW)
        fp = PushMidRead(b"abcdef", deny)
        r = HTTPResponse(fp, preload_content=False)

        assert r.read(3) == b"abc"  # pushes deny mid-read, still returns data
        assert current_frame() is deny
        with pytest.raises(BodyAdmissionDenied):
            r.read(3)
        pop_response_policy(deny)
        assert r.read(3) == b"def"

    def test_one_more_allows_exactly_one_fetch(self) -> None:
        r = make_response(b"abcdef")
        with response_policy(body_admit=BODY_ONE_MORE) as frame:
            assert r.read(3) == b"abc"
            assert frame.fetches_accepted == 1
            # The second attempt re-reads the then-current top and is denied.
            with pytest.raises(BodyAdmissionDenied) as ei:
                r.read(3)
            assert ei.value.mode == BODY_ONE_MORE
        assert frame.fetches_denied == 1

    def test_one_more_second_fetch_sees_newly_pushed_inner_frame(self) -> None:
        # A frame pushed after the one_more fetch must be re-read live: an
        # inner honor frame admits the next fetch.
        r = make_response(b"abcdef")
        with response_policy(body_admit=BODY_ONE_MORE):
            assert r.read(3) == b"abc"
            with response_policy(body_admit=BODY_HONOR_CFG):
                assert r.read(3) == b"def"

    def test_one_more_stream_second_chunk_denied(self) -> None:
        r = make_response(b"0123456789" * 8)
        with response_policy(body_admit=BODY_ONE_MORE):
            chunks: list[bytes] = []
            it = iter(r.stream(amt=16))
            chunks.append(next(it))
            with pytest.raises(BodyAdmissionDenied):
                next(it)
        assert chunks == [b"0123456789012345"]

    def test_json_counts_as_single_fetch_even_though_it_drives_read(
        self,
    ) -> None:
        r = make_response(b'{"x": 1}', preload_content=False)
        with response_policy(body_admit=BODY_ONE_MORE) as frame:
            assert r.json() == {"x": 1}
            assert frame.fetches_accepted == 1

    def test_json_denied_even_with_cached_body(self) -> None:
        r = make_response(b'{"x": 1}', preload_content=True)
        assert r.data == b'{"x": 1}'
        with response_policy(body_admit=BODY_DENY_NEW):
            with pytest.raises(BodyAdmissionDenied):
                r.json()

    def test_preload_under_deny_is_refused_before_touching_body(self) -> None:
        class Probe:
            def __init__(self, data: bytes) -> None:
                self._fp = BufferedReader(BytesIO(data))
                self.read_calls = 0

            def read(self, amt: int | None = None) -> bytes:
                self.read_calls += 1
                return self._fp.read(amt) if amt is not None else self._fp.read()

            def read1(self, amt: int | None = None) -> bytes:
                self.read_calls += 1
                return self._fp.read1(amt)

            def close(self) -> None:
                self._fp.close()

            @property
            def closed(self) -> bool:
                return self._fp.closed

        fp = Probe(b"abc")
        with response_policy(body_admit=BODY_DENY_NEW):
            with pytest.raises(BodyAdmissionDenied):
                HTTPResponse(fp, preload_content=True)
        # The refusal happens at fetch entry: no wire read was ever made.
        assert fp.read_calls == 0

    def test_unpushed_scope_unaffected_while_other_scope_pushed(
        self,
    ) -> None:
        r_denied = make_response(b"aaa")
        r_free = make_response(b"bbb")
        barrier = threading.Barrier(2)
        outcome: dict[str, object] = {}

        def pushed() -> None:
            with response_policy(body_admit=BODY_DENY_NEW):
                barrier.wait()
                try:
                    r_denied.read()
                except BodyAdmissionDenied:
                    outcome["pushed"] = "denied"

        def unpushed() -> None:
            barrier.wait()
            outcome["unpushed"] = r_free.read()

        t1 = threading.Thread(target=pushed)
        t2 = threading.Thread(target=unpushed)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert outcome == {"pushed": "denied", "unpushed": b"bbb"}

    def test_pop_restores_reads(self) -> None:
        r = make_response(b"abc")
        with response_policy(body_admit=BODY_DENY_NEW):
            with pytest.raises(BodyAdmissionDenied):
                r.read(1)
        assert r.read() == b"abc"


# ---------------------------------------------------------------------------
# decode_gate
# ---------------------------------------------------------------------------


PLAIN = b"the quick brown fox " * 20
GZIPPED = gzip_body(PLAIN)


class TestDecodeGate:
    def test_empty_stack_decodes_as_pin(self) -> None:
        r = make_response(
            GZIPPED, headers={"content-encoding": "gzip"}, decode_content=True
        )
        assert r.read() == PLAIN

    def test_skip_decode_returns_raw_bytes(self) -> None:
        r = make_response(
            GZIPPED, headers={"content-encoding": "gzip"}, decode_content=True
        )
        with response_policy(decode_gate=DECODE_SKIP_DECODE) as frame:
            raw = r.read()
        assert raw == GZIPPED
        assert raw != PLAIN and raw
        assert r.decode_content is True  # permanent flag never mutated
        assert frame.decode_gate_events and frame.decode_gate_events[0].requested

    def test_force_raw_skips_decoder_path(self) -> None:
        r = make_response(
            GZIPPED, headers={"content-encoding": "gzip"}, decode_content=True
        )
        with response_policy(decode_gate=DECODE_FORCE_RAW):
            raw = r.read()
        assert raw == GZIPPED
        # The decoder machinery was never instantiated for this response.
        assert r._decoder_initialized is False
        assert r._decoder is None

    def test_force_raw_explicit_true_call_arg_still_raw(self) -> None:
        r = make_response(
            GZIPPED, headers={"content-encoding": "gzip"}, decode_content=False
        )
        with response_policy(decode_gate=DECODE_FORCE_RAW):
            assert r.read(decode_content=True) == GZIPPED

    def test_after_pop_decoding_works_again(self) -> None:
        r1 = make_response(
            GZIPPED, headers={"content-encoding": "gzip"}, decode_content=True
        )
        with response_policy(decode_gate=DECODE_FORCE_RAW):
            assert r1.read() == GZIPPED

        r2 = make_response(
            GZIPPED, headers={"content-encoding": "gzip"}, decode_content=True
        )
        assert r2.read() == PLAIN

    def test_force_raw_does_not_swallow_decode_failure_as_empty(self) -> None:
        # Garbage gzip wire data: honor fails with DecodeError (pin); under a
        # force_raw frame the same non-empty bytes come back raw instead of a
        # faked empty/decode-success body.
        garbage = b"foo"
        r_honor = make_response(
            garbage, headers={"content-encoding": "gzip"}, decode_content=True
        )
        with pytest.raises(DecodeError):
            r_honor.read()

        r_raw = make_response(
            garbage, headers={"content-encoding": "gzip"}, decode_content=True
        )
        with response_policy(decode_gate=DECODE_FORCE_RAW):
            data = r_raw.read()
        assert data == garbage and data

    def test_mix_then_force_raw_is_honest_runtimeerror(self) -> None:
        # Decoded reads happened already; raw delivery cannot be synthesized.
        r = make_response(
            GZIPPED, headers={"content-encoding": "gzip"}, decode_content=True
        )
        assert r.read(5) == PLAIN[:5]
        with response_policy(decode_gate=DECODE_FORCE_RAW):
            with pytest.raises(RuntimeError):
                r.read(5)


# ---------------------------------------------------------------------------
# conn_return -- response-side unit tests
# ---------------------------------------------------------------------------


class TestConnReturnResponseSide:
    def test_pin_release_pools_and_clears_holder(self) -> None:
        pool = FakePool()
        conn = FakeConn()
        r = make_response(pool=pool, conn=conn)
        r.release_conn()
        assert pool.returned == [conn]
        assert r.connection is None

    def test_keep_out_keeps_connection_occupied(self) -> None:
        pool = FakePool()
        conn = FakeConn()
        r = make_response(pool=pool, conn=conn)
        with response_policy(conn_return=CONN_KEEP_OUT) as frame:
            r.release_conn()
            assert pool.returned == []
            assert r.connection is conn
            assert conn.sock is not None
            # Repeated release keeps the same disposition.
            r.release_conn()
            assert pool.returned == []
        assert len(frame.connections_withheld) == 1
        assert frame.connections_withheld[0].source == "release_conn"

        # After the frame is gone the connection can still be returned.
        r.release_conn()
        assert pool.returned == [conn]

    def test_force_drop_tears_socket_down_observably(self) -> None:
        pool = FakePool()
        conn = FakeConn()
        r = make_response(b"abcdef", pool=pool, conn=conn)
        in_hand = r.read(3)
        assert in_hand == b"abc"
        with response_policy(conn_return=CONN_FORCE_DROP) as frame:
            r.release_conn()
            assert pool.returned == []
            assert conn.closed is True and conn.sock is None
            assert r.connection is conn  # evidence retained, not hidden
        ev = frame.connections_dropped[0]
        assert isinstance(ev, ConnReturnEvent)
        assert ev.socket_fileno == 42 and ev.socket_open is False
        # The drop never touched the body buffer: bytes already delivered to
        # the caller survive, and the remaining fp bytes are still there.
        assert in_hand == b"abc"
        assert r.read() == b"def"

    def test_close_keep_out_does_not_close_or_pool(self) -> None:
        pool = FakePool()
        conn = FakeConn()
        r = make_response(pool=pool, conn=conn)
        with response_policy(conn_return=CONN_KEEP_OUT):
            r.close()
            assert conn.closed is False
            assert pool.returned == []
            assert r.connection is conn
        # Still recoverable once the frame is gone.
        r.release_conn()
        assert pool.returned == [conn]

    def test_close_force_drop_records_event(self) -> None:
        pool = FakePool()
        conn = FakeConn()
        r = make_response(pool=pool, conn=conn)
        with response_policy(conn_return=CONN_FORCE_DROP) as frame:
            r.close()
        assert conn.closed is True
        assert pool.returned == []
        assert frame.connections_dropped[0].source == "close"

    def test_frame_change_during_return_is_refused(self) -> None:
        from urllib3 import response_policy as rp

        pool = FakePool()
        conn = FakeConn()
        outer = ResponsePolicyFrame(conn_return=CONN_KEEP_OUT)
        other = ResponsePolicyFrame(conn_return=CONN_KEEP_OUT)
        push_response_policy(outer)
        try:
            # Simulate a nested push landing between return-entry and
            # handoff: the guard must refuse instead of silently following
            # the stale frame.
            with mock.patch.object(rp, "current_frame", side_effect=[other]):
                with pytest.raises(rp.ConnReturnFrameChanged):
                    rp.handle_conn_return(
                        outer, CONN_KEEP_OUT, pool, conn, source="release_conn"
                    )
            assert pool.returned == []
            assert conn.sock is not None
        finally:
            pop_response_policy(outer)


# ---------------------------------------------------------------------------
# End-to-end against a real loopback HTTP/1.1 server
# ---------------------------------------------------------------------------

CHUNKS = [b"0123456789" * 10, b"abcdefghij" * 10, b"Z" * 77]
CHUNKED_BODY = b"".join(CHUNKS)
PLAIN_BODY = b"plain-response-body" * 30


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/chunked":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for piece in CHUNKS:
                self.wfile.write(b"%x\r\n%b\r\n" % (len(piece), piece))
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(PLAIN_BODY)))
            self.end_headers()
            self.wfile.write(PLAIN_BODY)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def server() -> object:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def _pool(server: object, **kw: object) -> HTTPConnectionPool:
    host, port = server.server_address  # type: ignore[attr-defined]
    kwargs = {"retries": False, "maxsize": 1}
    kwargs.update(kw)
    return HTTPConnectionPool(host, port=port, **kwargs)  # type: ignore[arg-type]


class TestPoolEndToEnd:
    def test_pin_return_reuses_connection(self, server: object) -> None:
        pool = _pool(server)
        try:
            r = pool.urlopen("GET", "/")
            assert r.status == 200 and r.data == PLAIN_BODY
            assert pool.pool.qsize() == 1
            assert pool.num_connections == 1
            pool.urlopen("GET", "/")
            assert pool.num_connections == 1
        finally:
            pool.close()

    def test_keep_out_withholds_at_urlopen_cleanup(self, server: object) -> None:
        pool = _pool(server)
        try:
            with response_policy(conn_return=CONN_KEEP_OUT) as frame:
                r = pool.urlopen("GET", "/")
                assert r.data == PLAIN_BODY
                held = r.connection
                assert held is not None and held.sock is not None
                assert pool.pool.qsize() == 0
                assert pool.num_connections == 1
                assert frame.connections_withheld[0].source == "urlopen"

                # The pool is genuinely empty: a new, second connection is
                # opened instead of a fake success from an empty queue.
                r2 = pool.urlopen("GET", "/")
                assert pool.num_connections == 2
                assert r2.connection is not held

            # Frames popped: held conns can now return to the pool.
            r.release_conn()
            r2.release_conn()
            assert pool.pool.qsize() >= 1
        finally:
            pool.close()

    def test_keep_out_blocking_pool_does_not_fake_success(self, server: object) -> None:
        pool = _pool(server, block=True, maxsize=1)
        try:
            with response_policy(conn_return=CONN_KEEP_OUT):
                held = pool.urlopen("GET", "/")
                assert pool.pool.qsize() == 0
                with pytest.raises(EmptyPoolError):
                    pool.urlopen("GET", "/", pool_timeout=0.2)
            # Return the occupied connection so pool teardown closes the
            # socket deterministically.
            held.release_conn()
        finally:
            pool.close()

    def test_force_drop_at_urlopen_cleanup(self, server: object) -> None:
        pool = _pool(server)
        try:
            with response_policy(conn_return=CONN_FORCE_DROP) as frame:
                r = pool.urlopen("GET", "/")
                assert r.data == PLAIN_BODY  # body intact, not faked
                dropped = frame.connections_dropped[0].conn
                assert dropped.sock is None
                assert pool.pool.qsize() == 0
            # Next request really opens a fresh connection.
            assert pool.num_connections == 1
            pool.urlopen("GET", "/")
            assert pool.num_connections == 2
        finally:
            pool.close()

    def test_two_threads_different_conn_return_do_not_cross(
        self, server: object
    ) -> None:
        pool = _pool(server, maxsize=2)
        barrier = threading.Barrier(2)
        outcome: dict[str, object] = {}

        def keep_out_worker() -> None:
            with response_policy(conn_return=CONN_KEEP_OUT) as frame:
                r = pool.urlopen("GET", "/")
                barrier.wait()
                # Keep a hard reference to the withheld connection past the
                # scope exit; the response remains its observable holder.
                outcome["held_conn"] = r.connection
                outcome["response"] = r
                outcome["withheld_events"] = len(frame.connections_withheld)

        def honor_worker() -> None:
            pool.urlopen("GET", "/")
            barrier.wait()
            # Preloaded response was already released to the pool by the
            # pinned urlopen cleanup; its queue slot is occupied.
            outcome["qsize_after_honor"] = pool.pool.qsize()

        try:
            t1 = threading.Thread(target=keep_out_worker)
            t2 = threading.Thread(target=honor_worker)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)
            assert not t1.is_alive() and not t2.is_alive()

            assert outcome["withheld_events"] == 1
            assert outcome["held_conn"] is not None
            assert outcome["qsize_after_honor"] == 1
            # The honor worker's connection, now pooled, is a different
            # physical connection from the withheld one.
            honor_conn = pool.pool.get_nowait()
            pool.pool.put_nowait(honor_conn)
            assert outcome["held_conn"] is not honor_conn
            # Deterministic teardown: hand the withheld connection back now
            # that its frame is gone.
            outcome["response"].release_conn()
        finally:
            pool.close()

    def test_mid_stream_deny_fails_observably_and_consumes_after_pop(
        self, server: object
    ) -> None:
        pool = _pool(server)
        try:
            r = pool.urlopen(
                "GET", "/chunked", preload_content=False, decode_content=False
            )
            it = iter(r.stream(amt=64, decode_content=False))
            first = next(it)
            assert first == CHUNKED_BODY[:64]

            with response_policy(body_admit=BODY_DENY_NEW):
                with pytest.raises(BodyAdmissionDenied):
                    next(it)
                # Not a completed body: the response still owns a live
                # connection that was never returned.
                assert r.connection is not None

            # Frame gone: the half-read body can be drained to completion.
            rest = b"".join(r.stream(amt=64, decode_content=False))
            assert first + rest == CHUNKED_BODY
            r.release_conn()
            assert pool.pool.qsize() == 1
        finally:
            pool.close()

    def test_mid_stream_one_more_allows_one_then_denies(self, server: object) -> None:
        pool = _pool(server)
        try:
            r = pool.urlopen(
                "GET", "/chunked", preload_content=False, decode_content=False
            )
            it = iter(r.stream(amt=32, decode_content=False))
            first = next(it)
            assert len(first) == 32
            with response_policy(body_admit=BODY_ONE_MORE):
                # Exactly one more fetch is admitted after the push...
                second = next(it)
                assert len(second) == 32
                # ...the one after that is refused.
                with pytest.raises(BodyAdmissionDenied):
                    next(it)
            rest = b"".join(r.stream(amt=32, decode_content=False))
            assert first + second + rest == CHUNKED_BODY
            r.release_conn()
        finally:
            pool.close()

    def test_force_drop_mid_stream_fails_later_read(self, server: object) -> None:
        pool = _pool(server)
        try:
            r = pool.urlopen(
                "GET", "/chunked", preload_content=False, decode_content=False
            )
            it = iter(r.stream(amt=64, decode_content=False))
            first = next(it)
            assert first  # bytes already handed to the caller survive

            with response_policy(conn_return=CONN_FORCE_DROP) as frame:
                r.release_conn()
                assert frame.connections_dropped[0].conn.sock is None
                # No half-body success: further reads fail observably --
                # either as an urllib3 connection error or as the stdlib
                # teardown error of the closed response file -- never as a
                # clean empty/completed stream.
                with pytest.raises((ProtocolError, OSError, AttributeError)):
                    next(it)

            assert first == CHUNKED_BODY[:64]
        finally:
            pool.close()

    def test_deny_during_preload_propagates_and_does_not_pool(
        self, server: object
    ) -> None:
        pool = _pool(server)
        qsize_before = pool.pool.qsize()
        try:
            with (
                response_policy(body_admit=BODY_DENY_NEW),
                pytest.raises(BodyAdmissionDenied),
            ):
                pool.urlopen("GET", "/", preload_content=True)
            # The refused request left no fake-pooled connection; the queue
            # still holds its original None slot(s).
            assert pool.pool.qsize() == qsize_before
        finally:
            pool.close()
