"""
Response-phase runtime policy stack.

This module implements a *runtime*, push/pop policy layer that sits on top of
the permanent response/connection-pool configuration. It governs, for the
duration of a cooperative scope only:

* ``body_admit``  -- whether new response-body fetches may be admitted
  (``read()``/``read1()``/``stream()``/``json()``);
* ``decode_gate`` -- whether content-decoding is performed;
* ``conn_return`` -- what happens when a connection is requested to be
  returned to the pool.

The stack is deliberately decoupled from:

* the permanent ``HTTPResponse.preload_content`` / ``decode_content`` /
  ``auto_close`` flags,
* the permanent pool ``retries`` / ``timeout`` configuration,
* TLS context loading, proxy-tunnel handshakes and header normalization.

None of those are ever temporarily mutated to emulate a policy frame, and no
state is attached to ``HTTPResponse``/connection/pool/thread objects. The
stack lives exclusively in a :class:`contextvars.ContextVar`: it is bound to
the current cooperative scope, child/nested scopes read the then-current top
frame (dynamic lookup, never an entry snapshot from the parent segment), and
concurrent scopes on separate contexts never share frames.

When the stack is empty every operation resolves to ``honor_cfg`` and the
behaviour is byte-for-byte the existing API behaviour.
"""

from __future__ import annotations

import contextlib
import contextvars
import typing
from dataclasses import dataclass

from .exceptions import BodyAdmissionDenied, ConnReturnFrameChanged

if typing.TYPE_CHECKING:
    from .response import BaseHTTPResponse

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

BODY_HONOR_CFG = "honor_cfg"
BODY_DENY_NEW = "deny_new"
BODY_ONE_MORE = "one_more"

DECODE_HONOR_CFG = "honor_cfg"
DECODE_SKIP_DECODE = "skip_decode"
DECODE_FORCE_RAW = "force_raw"

CONN_HONOR_CFG = "honor_cfg"
CONN_KEEP_OUT = "keep_out"
CONN_FORCE_DROP = "force_drop"

_BODY_ADMIT_VALUES = frozenset({BODY_HONOR_CFG, BODY_DENY_NEW, BODY_ONE_MORE})
_DECODE_GATE_VALUES = frozenset(
    {DECODE_HONOR_CFG, DECODE_SKIP_DECODE, DECODE_FORCE_RAW}
)
_CONN_RETURN_VALUES = frozenset({CONN_HONOR_CFG, CONN_KEEP_OUT, CONN_FORCE_DROP})


# ---------------------------------------------------------------------------
# Observable event records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyFetchEvent:
    """A decision made by the ``body_admit`` gate for one fetch entry."""

    #: The response the fetch was attempted on.
    response: typing.Any
    #: True when the fetch was admitted, False when it was refused.
    admitted: bool
    #: The ``body_admit`` value in effect at the decision.
    mode: str


@dataclass(frozen=True)
class DecodeGateEvent:
    """A case where a frame overrode the configured decoding decision."""

    #: The response being read.
    response: typing.Any
    #: The ``decode_gate`` value in effect (``skip_decode``/``force_raw``).
    gate: str
    #: What the call argument merged with ``response.decode_content`` asked
    #: for (``None`` means "use the permanent config").
    requested: bool
    #: What was actually executed (always ``False`` for an active gate).
    effective: bool


@dataclass(frozen=True)
class ConnReturnEvent:
    """A connection-return disposition performed under a pushed frame."""

    #: ``keep_out`` or ``force_drop``.
    mode: str
    #: The pool the connection was *not* handed back to.
    pool: typing.Any
    #: The connection that was kept occupied or whose socket was dropped.
    conn: typing.Any
    #: Where the return originated (``release_conn`` / ``urlopen`` /
    #: ``close``).
    source: str
    #: Whether the connection's socket was open after the disposition.
    socket_open: bool
    #: Fileno of the socket captured at disposition time (``None`` if there
    #: was no socket), so the teardown stays observable afterwards.
    socket_fileno: int | None


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


class ResponsePolicyFrame:
    """
    One entry of the response-phase policy stack.

    Instances are normally obtained through :func:`response_policy` (context
    manager) or :func:`push_response_policy` / :func:`pop_response_policy`
    (explicit push/pop). The same object doubles as a context manager.

    The frame object is the *only* place where per-scope state lives:
    admission counters, re-entrancy markers and observable event records.
    It is never attached to a response, a connection, a pool or a thread.
    """

    def __init__(
        self,
        *,
        body_admit: str = BODY_HONOR_CFG,
        decode_gate: str = DECODE_HONOR_CFG,
        conn_return: str = CONN_HONOR_CFG,
    ) -> None:
        if body_admit not in _BODY_ADMIT_VALUES:
            raise ValueError(
                f"Invalid body_admit={body_admit!r}, expected one of "
                f"{sorted(_BODY_ADMIT_VALUES)}"
            )
        if decode_gate not in _DECODE_GATE_VALUES:
            raise ValueError(
                f"Invalid decode_gate={decode_gate!r}, expected one of "
                f"{sorted(_DECODE_GATE_VALUES)}"
            )
        if conn_return not in _CONN_RETURN_VALUES:
            raise ValueError(
                f"Invalid conn_return={conn_return!r}, expected one of "
                f"{sorted(_CONN_RETURN_VALUES)}"
            )

        self.body_admit = body_admit
        self.decode_gate = decode_gate
        self.conn_return = conn_return

        self.body_fetch_events: list[BodyFetchEvent] = []
        self.decode_gate_events: list[DecodeGateEvent] = []
        self.conn_return_events: list[ConnReturnEvent] = []

        #: Number of body fetches accepted while this frame was on top.
        self._accepted_fetches = 0
        #: id(response) -> nesting depth of fetches currently in flight, so
        #: nested public entries that are part of the same in-flight fetch
        #: (e.g. ``json()`` -> ``read()``) are not charged twice.
        self._active_fetches: dict[int, int] = {}
        #: id(conn) -> disposition, so repeated return requests for the same
        #: connection inside one frame stay idempotent.
        self._handled_conns: dict[int, str] = {}
        #: (id(response), gate) pairs already reported, keeping the event log
        #: to one bypass record per response even with many chunk reads.
        self._decode_logged: set[tuple[int, str]] = set()

        self._token: contextvars.Token[typing.Any] | None = None

    # -- context-manager convenience ---------------------------------------

    def __enter__(self) -> ResponsePolicyFrame:
        return push_response_policy(self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: typing.Any,
    ) -> typing.Literal[False]:
        pop_response_policy(self)
        return False

    def push(self) -> ResponsePolicyFrame:
        return push_response_policy(self)

    def pop(self) -> None:
        pop_response_policy(self)

    # -- observable counters -----------------------------------------------

    @property
    def fetches_accepted(self) -> int:
        """Number of body fetches this frame admitted."""
        return self._accepted_fetches

    @property
    def fetches_denied(self) -> int:
        """Number of body fetches this frame observably refused."""
        return sum(1 for e in self.body_fetch_events if not e.admitted)

    @property
    def connections_withheld(self) -> list[ConnReturnEvent]:
        """Events for connections kept occupied (``keep_out``)."""
        return [e for e in self.conn_return_events if e.mode == CONN_KEEP_OUT]

    @property
    def connections_dropped(self) -> list[ConnReturnEvent]:
        """Events for connections whose socket was torn down
        (``force_drop``)."""
        return [e for e in self.conn_return_events if e.mode == CONN_FORCE_DROP]

    def __repr__(self) -> str:
        return (
            "ResponsePolicyFrame("
            f"body_admit={self.body_admit!r}, "
            f"decode_gate={self.decode_gate!r}, "
            f"conn_return={self.conn_return!r})"
        )


# ---------------------------------------------------------------------------
# The stack itself: a ContextVar, nothing per-thread/per-object
# ---------------------------------------------------------------------------

_policy_stack: contextvars.ContextVar[tuple[ResponsePolicyFrame, ...] | None] = (
    contextvars.ContextVar("urllib3_response_policy_stack", default=None)
)


def current_frame() -> ResponsePolicyFrame | None:
    """Return the frame on top of the current scope's stack, or ``None``."""
    stack = _policy_stack.get()
    if not stack:
        return None
    return stack[0]


def active_response_policy() -> ResponsePolicyFrame | None:
    """Public alias of :func:`current_frame`."""
    return current_frame()


def push_response_policy(frame: ResponsePolicyFrame) -> ResponsePolicyFrame:
    """Push an existing frame onto the current scope's stack."""
    if frame._token is not None:
        raise RuntimeError(f"Response policy frame is already pushed: {frame!r}")
    stack = _policy_stack.get()
    frame._token = _policy_stack.set((frame,) + (stack or ()))
    return frame


def pop_response_policy(frame: ResponsePolicyFrame) -> None:
    """Pop ``frame``, restoring the outer view. Safe to call from
    ``finally`` blocks; popping never raises."""
    token = frame._token
    if token is None:
        return
    frame._token = None
    try:
        _policy_stack.reset(token)
    except (ValueError, LookupError):
        # The context that pushed the frame is gone (e.g. a worker thread
        # finished with a non-popped frame). There is nothing to restore.
        pass


@contextlib.contextmanager
def response_policy(
    *,
    body_admit: str = BODY_HONOR_CFG,
    decode_gate: str = DECODE_HONOR_CFG,
    conn_return: str = CONN_HONOR_CFG,
) -> typing.Generator[ResponsePolicyFrame, None, None]:
    """
    Push a response-phase policy frame for the duration of the ``with`` block.

    The frame is always popped on exit, including when the block raises. Nested
    blocks nest: popping the inner frame restores the outer one.
    """
    frame = ResponsePolicyFrame(
        body_admit=body_admit, decode_gate=decode_gate, conn_return=conn_return
    )
    push_response_policy(frame)
    try:
        yield frame
    finally:
        pop_response_policy(frame)


# ---------------------------------------------------------------------------
# body_admit gate
# ---------------------------------------------------------------------------

# Message templates kept stable so callers can match on the mode attribute
# instead of message text.
_DENY_MESSAGES = {
    BODY_DENY_NEW: (
        "A new response-body fetch was attempted inside a response-policy "
        "frame with body_admit='deny_new'"
    ),
    BODY_ONE_MORE: (
        "The single body fetch allowed by the response-policy frame with "
        "body_admit='one_more' was already admitted"
    ),
}


def begin_body_fetch(response: BaseHTTPResponse) -> ResponsePolicyFrame | None:
    """
    Resolve the ``body_admit`` gate at the entry of *one new* body fetch.

    Raises :class:`~urllib3.exceptions.BodyAdmissionDenied` when the current
    top frame refuses the fetch. A fetch that is already in flight is never
    affected: nested entries for the same response while the outer fetch is
    running re-enter without being charged (this covers ``json()`` driving
    ``read()``). Returns the admitting frame, or ``None`` when the stack is
    empty.
    """
    frame = current_frame()
    if frame is None:
        return None

    key = id(response)
    depth = frame._active_fetches.get(key, 0)
    if depth > 0:
        # Same fetch, already admitted and possibly sitting in a blocking
        # recv(): an inner push must not cancel or rewrite it.
        frame._active_fetches[key] = depth + 1
        return frame

    mode = frame.body_admit
    if mode == BODY_DENY_NEW or (
        mode == BODY_ONE_MORE and frame._accepted_fetches >= 1
    ):
        frame.body_fetch_events.append(
            BodyFetchEvent(response=response, admitted=False, mode=mode)
        )
        raise BodyAdmissionDenied(
            _DENY_MESSAGES[mode], frame=frame, mode=mode, response=response
        )

    frame._accepted_fetches += 1
    frame._active_fetches[key] = 1
    frame.body_fetch_events.append(
        BodyFetchEvent(response=response, admitted=True, mode=mode)
    )
    return frame


def end_body_fetch(
    response: BaseHTTPResponse, frame: ResponsePolicyFrame | None
) -> None:
    """Close the admission scope opened by :func:`begin_body_fetch`."""
    if frame is None:
        return
    key = id(response)
    depth = frame._active_fetches.get(key, 0)
    if depth <= 1:
        frame._active_fetches.pop(key, None)
    else:
        frame._active_fetches[key] = depth - 1


@contextlib.contextmanager
def body_fetch(
    response: BaseHTTPResponse,
) -> typing.Generator[ResponsePolicyFrame | None, None, None]:
    """Context manager wrapping one public body-fetch operation."""
    frame = begin_body_fetch(response)
    try:
        yield frame
    finally:
        end_body_fetch(response, frame)


# ---------------------------------------------------------------------------
# decode_gate
# ---------------------------------------------------------------------------


def resolve_decode_content(
    response: BaseHTTPResponse,
    requested_arg: bool | None,
    *,
    none_default: bool | None = None,
) -> tuple[bool, str]:
    """
    Merge the read call's ``decode_content`` argument, the response's
    permanent ``decode_content`` configuration and the current top frame's
    ``decode_gate`` into the effective decoding decision.

    ``none_default`` is what ``decode_content=None`` resolves to at the
    calling site *before* consulting the stack: ``response.decode_content``
    for ``read()``/``read1()`` (the default), or ``False`` for the pinned
    chunked-streaming path where ``None`` is passed straight through.

    Returns ``(effective_decode, gate)``. An active ``skip_decode`` /
    ``force_raw`` frame always wins (raw bytes), and the bypass is recorded
    once per response on the frame.
    """
    if none_default is None:
        none_default = bool(getattr(response, "decode_content", False))
    requested = none_default if requested_arg is None else bool(requested_arg)

    frame = current_frame()
    if frame is None:
        return requested, DECODE_HONOR_CFG

    gate = frame.decode_gate
    if gate == DECODE_HONOR_CFG:
        return requested, gate

    effective = False
    log_key = (id(response), gate)
    if log_key not in frame._decode_logged:
        frame._decode_logged.add(log_key)
        frame.decode_gate_events.append(
            DecodeGateEvent(
                response=response,
                gate=gate,
                requested=requested,
                effective=effective,
            )
        )
    return effective, gate


def current_decode_gate() -> str:
    """Return the ``decode_gate`` of the current top frame, or
    ``honor_cfg``."""
    frame = current_frame()
    if frame is None:
        return DECODE_HONOR_CFG
    return frame.decode_gate


# ---------------------------------------------------------------------------
# conn_return gate
# ---------------------------------------------------------------------------


def current_conn_return() -> tuple[str, ResponsePolicyFrame | None]:
    """Return ``(mode, frame)`` for connection-return disposition."""
    frame = current_frame()
    if frame is None:
        return CONN_HONOR_CFG, None
    return frame.conn_return, frame


def _conn_fileno(conn: typing.Any) -> int | None:
    sock = getattr(conn, "sock", None)
    if sock is None:
        return None
    try:
        fileno = sock.fileno()
    except OSError:
        return None
    return fileno if fileno >= 0 else None


def _conn_socket_open(conn: typing.Any) -> bool:
    return _conn_fileno(conn) is not None


def handle_conn_return(
    frame: ResponsePolicyFrame,
    mode: str,
    pool: typing.Any,
    conn: typing.Any,
    *,
    source: str,
) -> bool:
    """
    Apply ``keep_out`` / ``force_drop`` at a connection-return chokepoint.

    The frame is *re-resolved at disposition time*. If a different frame (or a
    frame whose ``conn_return`` changed) is on top by the time the return is
    actually performed, :class:`~urllib3.exceptions.ConnReturnFrameChanged`
    is raised and no pool state is touched -- the return is explicitly
    refused instead of silently following the entry frame.

    Returns ``True`` when a disposition was performed, ``False`` when the
    same connection was already dispositioned by this frame (idempotent).
    """
    top = current_frame()
    if top is not frame or top.conn_return != mode:
        raise ConnReturnFrameChanged(
            "The response-policy frame on top at connection-return time "
            "differs from the frame resolved when the return entered; "
            "refusing to return the connection.",
            expected_frame=frame,
        )

    key = id(conn)
    if key in frame._handled_conns:
        # Repeated release/close for a connection this frame has already
        # kept out or dropped: keep the first disposition observable and
        # stable.
        return False
    frame._handled_conns[key] = mode

    if mode == CONN_FORCE_DROP:
        before_fileno = _conn_fileno(conn)
        # Tear down the underlying socket. The connection is deliberately
        # not handed back to the pool.
        conn.close()
        frame.conn_return_events.append(
            ConnReturnEvent(
                mode=mode,
                pool=pool,
                conn=conn,
                source=source,
                socket_open=False,
                socket_fileno=before_fileno,
            )
        )
    else:  # CONN_KEEP_OUT
        frame.conn_return_events.append(
            ConnReturnEvent(
                mode=mode,
                pool=pool,
                conn=conn,
                source=source,
                socket_open=_conn_socket_open(conn),
                socket_fileno=_conn_fileno(conn),
            )
        )
    return True
