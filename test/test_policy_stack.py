"""
Tests for the thread-local, nestable policy profile stack
(``urllib3._policy``): retry_budget / redirect_surface / status_force /
proxy_route.

Every assertion is written against observable egress behavior (attempt
counts, request targets, chosen pools, methods/bodies, raised exception
types) rather than private state.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

import urllib3
from urllib3 import (
    DEFAULT_PROFILE,
    HTTPConnectionPool,
    PolicyProfile,
    PoolManager,
    ProxyManager,
    ProxyRoute,
    RedirectSurface,
    Retry,
    RetryBudget,
    StatusForce,
    current_policy,
    policy_scope,
    pop_policy,
    push_policy,
)
from urllib3._policy import change_method_on_redirect
from urllib3.connectionpool import HTTPConnectionPool as HTTPConnectionPoolDirect
from urllib3.exceptions import MaxRetryError, ProtocolError, ProxyError
from urllib3.response import HTTPResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RequestRecorder:
    """Stand-in for HTTPConnectionPool._make_request.

    When installed at class level it receives the pool as ``self`` and can
    therefore record which pool (proxy vs direct) each attempt used.
    """

    def __init__(self, side_effects: list | None = None) -> None:
        self.side_effects = list(side_effects or [])
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self._per_thread: dict[int, int] = {}

    def __call__(
        self,
        pool: HTTPConnectionPool,
        conn: object,
        method: str,
        url: str,
        *args: object,
        **kwargs: object,
    ) -> HTTPResponse:
        with self._lock:
            ident = threading.get_ident()
            self._per_thread[ident] = self._per_thread.get(ident, 0) + 1
            self.calls.append(
                {
                    "thread": ident,
                    "pool_host": pool.host,
                    "pool_proxy": pool.proxy,
                    "method": method,
                    "url": url,
                    "body": kwargs.get("body"),
                    "chunked": kwargs.get("chunked", False),
                    "headers": kwargs.get("headers"),
                }
            )
            if not self.side_effects:
                effect = HTTPResponse(status=200)
            else:
                effect = self.side_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        # Mirror _make_request: hand the current Retry snapshot to the
        # response so callers can observe the (possibly frozen) counters.
        effect.retries = kwargs.get("retries")
        return effect  # type: ignore[no-any-return]

    def attempts_for(self, ident: int) -> int:
        with self._lock:
            return self._per_thread.get(ident, 0)


def _record_class(recorder):
    # A callable *instance* assigned as a class attribute is not bound as a
    # method, so wrap the recorder in a plain function: the pool is then
    # passed explicitly as the first argument.
    def _patched(pool, conn, method, url, *args, **kwargs):
        return recorder(pool, conn, method, url, *args, **kwargs)

    return patch.object(HTTPConnectionPoolDirect, "_make_request", _patched)


def _record_instance(pool: HTTPConnectionPool, recorder: RequestRecorder):
    # Instance-level attribute patches are invoked without ``self``; bind the
    # pool explicitly so the recorder always receives the same argument shape
    # as a class-level patch.
    return patch.object(
        pool, "_make_request", lambda *a, **k: recorder(pool, *a, **k)
    )


# ---------------------------------------------------------------------------
# Stack mechanics
# ---------------------------------------------------------------------------


class TestPolicyStack:
    def test_empty_stack_is_honor(self) -> None:
        assert current_policy() == DEFAULT_PROFILE

    def test_push_pop_nesting_and_restore(self) -> None:
        deny = PolicyProfile(retry_budget=RetryBudget.DENY_RETRY)
        freeze = PolicyProfile(retry_budget=RetryBudget.FREEZE_COUNTS)

        token = push_policy(deny)
        try:
            assert current_policy().retry_budget is RetryBudget.DENY_RETRY
            push_policy(freeze)
            assert current_policy().retry_budget is RetryBudget.FREEZE_COUNTS
            pop_policy()
            # Inner frame popped reveals the outer frame again.
            assert current_policy().retry_budget is RetryBudget.DENY_RETRY
        finally:
            pop_policy(token)

        assert current_policy() == DEFAULT_PROFILE

    def test_dimensions_resolve_independently(self) -> None:
        push_policy(
            PolicyProfile(
                retry_budget=RetryBudget.DENY_RETRY,
                status_force=StatusForce.IGNORE_FORCELIST,
            )
        )
        try:
            effective = current_policy()
            assert effective.retry_budget is RetryBudget.DENY_RETRY
            assert effective.status_force is StatusForce.IGNORE_FORCELIST
            # Dimensions the frame does not touch stay at honor.
            assert effective.redirect_surface is RedirectSurface.HONOR_REDIRECT
            assert effective.proxy_route is ProxyRoute.HONOR_PROXY
        finally:
            pop_policy()

    def test_inner_override_per_dimension(self) -> None:
        outer = PolicyProfile(
            redirect_surface=RedirectSurface.DENY_REDIRECT,
            status_force=StatusForce.FORBID_STATUS_RETRY,
        )
        inner = PolicyProfile(status_force=StatusForce.IGNORE_FORCELIST)
        push_policy(outer)
        try:
            push_policy(inner)
            effective = current_policy()
            # Inner override wins for status_force ...
            assert effective.status_force is StatusForce.IGNORE_FORCELIST
            # ... while the outer frame still governs redirect_surface.
            assert effective.redirect_surface is RedirectSurface.DENY_REDIRECT
            pop_policy()
            assert (
                current_policy().status_force
                is StatusForce.FORBID_STATUS_RETRY
            )
        finally:
            pop_policy()

    def test_scope_pops_even_on_exception(self) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            with policy_scope(
                PolicyProfile(retry_budget=RetryBudget.DENY_RETRY)
            ):
                assert (
                    current_policy().retry_budget is RetryBudget.DENY_RETRY
                )
                raise RuntimeError("boom")
        assert current_policy() == DEFAULT_PROFILE

    def test_pop_empty_stack_raises(self) -> None:
        with pytest.raises(IndexError):
            pop_policy()

    def test_stack_is_thread_local(self) -> None:
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                assert current_policy() == DEFAULT_PROFILE
                push_policy(
                    PolicyProfile(retry_budget=RetryBudget.ONE_MORE)
                )
                assert (
                    current_policy().retry_budget is RetryBudget.ONE_MORE
                )
                pop_policy()
                assert current_policy() == DEFAULT_PROFILE
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        push_policy(PolicyProfile(retry_budget=RetryBudget.DENY_RETRY))
        try:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            assert (
                current_policy().retry_budget is RetryBudget.DENY_RETRY
            )
        finally:
            pop_policy()
        assert errors == []

    def test_dimensions_are_distinct_enums(self) -> None:
        # The four dimensions may never be collapsed into one boolean.
        for status in (301, 302, 303, 307, 308):
            assert change_method_on_redirect(status) is (status == 303)
        values = {
            RetryBudget.DENY_RETRY,
            RedirectSurface.DENY_REDIRECT,
            StatusForce.FORBID_STATUS_RETRY,
            ProxyRoute.DENY_TUNNEL,
        }
        assert len(values) == 4


# ---------------------------------------------------------------------------
# retry_budget on the HTTPConnectionPool failure path
# ---------------------------------------------------------------------------


class TestRetryBudget:
    @pytest.mark.parametrize(
        "retries", [Retry(total=5, connect=5), Retry(5)]
    )
    def test_deny_fails_immediately_without_max_retry(
        self, retries: Retry
    ) -> None:
        recorder = RequestRecorder(side_effects=[OSError("reset")])
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(retry_budget=RetryBudget.DENY_RETRY)
                ):
                    with pytest.raises(ProtocolError):
                        pool.urlopen("GET", "/", retries=retries)

        # Exactly one attempt; the raw error propagates instead of
        # MaxRetryError and retries was never flipped to False.
        assert len(recorder.calls) == 1

    def test_honor_retries_until_budget_exhausted(self) -> None:
        recorder = RequestRecorder(
            side_effects=[OSError("reset") for _ in range(10)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with pytest.raises(MaxRetryError):
                    pool.urlopen(
                        "GET",
                        "/",
                        retries=Retry(total=2, connect=2),
                    )
        assert len(recorder.calls) == 3

    def test_freeze_counts_keeps_observable_counters_at_snapshot(self) -> None:
        retry = Retry(total=3, connect=3)

        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            # honor: the success response carries the decremented snapshot
            honor_recorder = RequestRecorder(
                side_effects=[OSError("reset"), HTTPResponse(status=200)]
            )
            with _record_instance(pool, honor_recorder):
                response = pool.urlopen("GET", "/", retries=retry)
            assert response.retries.total == 2

            # freeze: decision to retry is taken, but the observable
            # counters never decrease relative to the frame-entry snapshot.
            freeze_recorder = RequestRecorder(
                side_effects=[OSError("reset"), HTTPResponse(status=200)]
            )
            with _record_instance(pool, freeze_recorder):
                with policy_scope(
                    PolicyProfile(retry_budget=RetryBudget.FREEZE_COUNTS)
                ):
                    frozen_response = pool.urlopen(
                        "GET", "/", retries=retry
                    )
            assert frozen_response.retries.total == 3
            assert frozen_response.retries.connect == 3
            # History still records the failed attempt.
            assert len(frozen_response.retries.history) == 1

    def test_freeze_counts_still_terminates_on_exhaustion(self) -> None:
        recorder = RequestRecorder(
            side_effects=[OSError("reset") for _ in range(10)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(retry_budget=RetryBudget.FREEZE_COUNTS)
                ):
                    with pytest.raises(MaxRetryError):
                        pool.urlopen(
                            "GET",
                            "/",
                            retries=Retry(total=0, connect=0),
                        )
        # Same attempt count as the pinned budget would have allowed.
        assert len(recorder.calls) == 1

    def test_one_more_grants_exactly_one_extra_attempt(self) -> None:
        # Exhausted budget: pin fails on the first failure.
        exhausted_recorder = RequestRecorder(
            side_effects=[OSError("reset") for _ in range(5)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, exhausted_recorder):
                with pytest.raises(MaxRetryError):
                    pool.urlopen("GET", "/", retries=Retry(total=0))
        assert len(exhausted_recorder.calls) == 1

        # one_more: exactly one additional recursion, then failure.
        onemore_recorder = RequestRecorder(
            side_effects=[OSError("reset") for _ in range(5)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, onemore_recorder):
                with policy_scope(
                    PolicyProfile(retry_budget=RetryBudget.ONE_MORE)
                ):
                    with pytest.raises(MaxRetryError):
                        pool.urlopen("GET", "/", retries=Retry(total=0))
        assert len(onemore_recorder.calls) == 2

    def test_one_more_resumes_outer_budget_after_grant(self) -> None:
        recorder = RequestRecorder(
            side_effects=[OSError("reset") for _ in range(10)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(retry_budget=RetryBudget.ONE_MORE)
                ):
                    with pytest.raises(MaxRetryError):
                        pool.urlopen("GET", "/", retries=Retry(total=2))
        # grant (1) + the outer budget's three attempts = 4.
        assert len(recorder.calls) == 4

    def test_one_more_grant_is_per_frame_not_per_request(self) -> None:
        recorder = RequestRecorder(
            side_effects=[OSError("reset"), HTTPResponse(status=200)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(retry_budget=RetryBudget.ONE_MORE)
                ):
                    # First request consumes the grant and succeeds on the
                    # granted recursion.
                    response = pool.urlopen(
                        "GET", "/", retries=Retry(total=0)
                    )
                    assert response.status == 200


# ---------------------------------------------------------------------------
# redirect_surface inside the pool
# ---------------------------------------------------------------------------


class TestRedirectSurfacePool:
    def test_deny_returns_3xx_to_caller(self) -> None:
        redirect = HTTPResponse(
            status=303, headers={"location": "http://localhost/next"}
        )
        recorder = RequestRecorder(side_effects=[redirect])
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(
                        redirect_surface=RedirectSurface.DENY_REDIRECT
                    )
                ):
                    # redirect=True and a positive Retry.redirect: still no
                    # recursive follow.
                    response = pool.urlopen(
                        "GET",
                        "/",
                        retries=Retry(total=5, redirect=5),
                        redirect=True,
                    )
        assert response.status == 303
        assert len(recorder.calls) == 1

    @pytest.mark.parametrize("status", [301, 302, 307, 308])
    def test_force_get_changes_method_for_all_followed_3xx(
        self, status: int
    ) -> None:
        redirect = HTTPResponse(
            status=status, headers={"location": "http://localhost/next"}
        )
        recorder = RequestRecorder(
            side_effects=[redirect, HTTPResponse(status=200)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(
                        redirect_surface=RedirectSurface.FORCE_GET_ON_3XX
                    )
                ):
                    pool.urlopen(
                        "POST",
                        "/",
                        body=b"data",
                        retries=Retry(total=5, redirect=5),
                    )
        assert len(recorder.calls) == 2
        first, second = recorder.calls
        assert first["method"] == "POST"
        assert second["method"] == "GET"
        assert second["body"] is None
        assert second["chunked"] is False
        # Content framing headers were stripped together with the body.
        header_names = {h.lower() for h in second["headers"]}
        assert "content-length" not in header_names
        assert "content-type" not in header_names
        assert "transfer-encoding" not in header_names

    def test_preserve_method_keeps_post_body_on_303(self) -> None:
        redirect = HTTPResponse(
            status=303, headers={"location": "http://localhost/next"}
        )
        recorder = RequestRecorder(
            side_effects=[redirect, HTTPResponse(status=200)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(
                        redirect_surface=RedirectSurface.PRESERVE_METHOD
                    )
                ):
                    pool.urlopen(
                        "POST",
                        "/",
                        body=b"data",
                        retries=Retry(total=5, redirect=5),
                    )
        first, second = recorder.calls
        assert first["method"] == "POST"
        assert second["method"] == "POST"
        assert second["body"] == b"data"

    def test_honor_baseline_303_vs_307(self) -> None:
        # 303 changes to GET under the pin ...
        recorder_303 = RequestRecorder(
            side_effects=[
                HTTPResponse(
                    status=303,
                    headers={"location": "http://localhost/next"},
                ),
                HTTPResponse(status=200),
            ]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder_303):
                pool.urlopen(
                    "POST",
                    "/",
                    body=b"data",
                    retries=Retry(total=5, redirect=5),
                )
        assert recorder_303.calls[1]["method"] == "GET"
        assert recorder_303.calls[1]["body"] is None

        # ... while 307 keeps method and body -- the fork force_get closes.
        recorder_307 = RequestRecorder(
            side_effects=[
                HTTPResponse(
                    status=307,
                    headers={"location": "http://localhost/next"},
                ),
                HTTPResponse(status=200),
            ]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder_307):
                pool.urlopen(
                    "POST",
                    "/",
                    body=b"data",
                    retries=Retry(total=5, redirect=5),
                )
        assert recorder_307.calls[1]["method"] == "POST"
        assert recorder_307.calls[1]["body"] == b"data"

    def test_deny_pushed_after_drain_invalidates_pending_follow(self) -> None:
        # The 3xx was already drained and the hop prepared; a frame pushed
        # during the retry sleep must invalidate the pending recursion.
        redirect = HTTPResponse(
            status=302,
            headers={
                "location": "http://localhost/next",
                "retry-after": "0",
            },
        )
        recorder = RequestRecorder(side_effects=[redirect])

        def push_deny(*args: object) -> None:
            push_policy(
                PolicyProfile(
                    redirect_surface=RedirectSurface.DENY_REDIRECT
                )
            )

        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with patch.object(Retry, "sleep_for_retry", push_deny):
                    response = pool.urlopen(
                        "GET",
                        "/",
                        retries=Retry(total=3, redirect=3),
                    )
        pop_policy()
        assert response.status == 302
        assert len(recorder.calls) == 1

    def test_deny_retry_does_not_collapse_into_deny_redirect(self) -> None:
        # retry_budget=deny_retry must not suppress redirect following.
        redirect = HTTPResponse(
            status=303, headers={"location": "http://localhost/next"}
        )
        recorder = RequestRecorder(
            side_effects=[redirect, HTTPResponse(status=200)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(retry_budget=RetryBudget.DENY_RETRY)
                ):
                    response = pool.urlopen(
                        "POST",
                        "/",
                        body=b"data",
                        retries=Retry(total=5, redirect=5),
                    )
        assert response.status == 200
        assert len(recorder.calls) == 2
        assert recorder.calls[1]["method"] == "GET"


# ---------------------------------------------------------------------------
# redirect_surface across hosts in PoolManager
# ---------------------------------------------------------------------------


class TestRedirectSurfaceManager:
    def test_deny_cross_host_returns_3xx(self) -> None:
        redirect = HTTPResponse(
            status=302, headers={"location": "http://example.org/next"}
        )
        recorder = RequestRecorder(side_effects=[redirect])
        with PoolManager() as manager:
            with _record_class(recorder):
                with policy_scope(
                    PolicyProfile(
                        redirect_surface=RedirectSurface.DENY_REDIRECT
                    )
                ):
                    response = manager.urlopen(
                        "GET",
                        "http://example.com/",
                        retries=Retry(total=5, redirect=5),
                    )
        assert response.status == 302
        assert len(recorder.calls) == 1

    def test_force_get_cross_host_307(self) -> None:
        redirect = HTTPResponse(
            status=307, headers={"location": "http://example.org/next"}
        )
        recorder = RequestRecorder(
            side_effects=[redirect, HTTPResponse(status=200)]
        )
        with PoolManager() as manager:
            with _record_class(recorder):
                with policy_scope(
                    PolicyProfile(
                        redirect_surface=RedirectSurface.FORCE_GET_ON_3XX
                    )
                ):
                    manager.urlopen(
                        "POST",
                        "http://example.com/",
                        body=b"data",
                        retries=Retry(total=5, redirect=5),
                    )
        first, second = recorder.calls
        assert first["pool_host"] == "example.com"
        assert second["pool_host"] == "example.org"
        assert second["method"] == "GET"
        assert second["body"] is None

    def test_freeze_counts_across_hosts(self) -> None:
        def run(profile: PolicyProfile | None):
            redirect = HTTPResponse(
                status=302,
                headers={"location": "http://example.org/next"},
            )
            recorder = RequestRecorder(
                side_effects=[redirect, HTTPResponse(status=200)]
            )
            with PoolManager() as manager:
                with _record_class(recorder):
                    if profile is None:
                        response = manager.urlopen(
                            "GET",
                            "http://example.com/",
                            retries=Retry(total=3, redirect=3),
                        )
                    else:
                        with policy_scope(profile):
                            response = manager.urlopen(
                                "GET",
                                "http://example.com/",
                                retries=Retry(total=3, redirect=3),
                            )
            return response

        honor = run(None)
        assert honor.retries.total == 2
        assert honor.retries.redirect == 2

        frozen = run(
            PolicyProfile(retry_budget=RetryBudget.FREEZE_COUNTS)
        )
        assert frozen.retries.total == 3
        assert frozen.retries.redirect == 3

    def test_deny_redirect_does_not_block_failure_retry(self) -> None:
        recorder = RequestRecorder(
            side_effects=[OSError("reset"), HTTPResponse(status=200)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(
                        redirect_surface=RedirectSurface.DENY_REDIRECT
                    )
                ):
                    response = pool.urlopen(
                        "GET",
                        "/",
                        retries=Retry(total=3, connect=3),
                    )
        assert response.status == 200
        assert len(recorder.calls) == 2

    def test_forbid_status_retry_keeps_redirects(self) -> None:
        # status_force must not collapse into redirect_surface either.
        redirect = HTTPResponse(
            status=303, headers={"location": "http://example.org/next"}
        )
        recorder = RequestRecorder(
            side_effects=[redirect, HTTPResponse(status=200)]
        )
        with PoolManager() as manager:
            with _record_class(recorder):
                with policy_scope(
                    PolicyProfile(
                        status_force=StatusForce.FORBID_STATUS_RETRY
                    )
                ):
                    response = manager.urlopen(
                        "POST",
                        "http://example.com/",
                        body=b"data",
                        retries=Retry(total=5, redirect=5),
                    )
        assert response.status == 200
        assert len(recorder.calls) == 2
        assert recorder.calls[1]["method"] == "GET"


# ---------------------------------------------------------------------------
# status_force in Retry.is_retry and the pool status retry path
# ---------------------------------------------------------------------------


class TestStatusForce:
    def test_is_retry_matrix(self) -> None:
        retry = Retry(total=5, status_forcelist=[503])

        # honor
        assert retry.is_retry("GET", 503) is True
        assert retry.is_retry("GET", 500) is False
        assert retry.is_retry("GET", 429, has_retry_after=True) is True
        assert retry.is_retry("GET", 429) is False

        with policy_scope(
            PolicyProfile(status_force=StatusForce.IGNORE_FORCELIST)
        ):
            assert retry.is_retry("GET", 503) is False
            # Retry-After fork survives ignoring the forcelist.
            assert retry.is_retry("GET", 429, has_retry_after=True) is True

        with policy_scope(
            PolicyProfile(status_force=StatusForce.FORCE_ALL_5XX)
        ):
            assert retry.is_retry("GET", 500) is True
            assert retry.is_retry("GET", 599) is True
            assert retry.is_retry("GET", 418) is False

        with policy_scope(
            PolicyProfile(status_force=StatusForce.FORBID_STATUS_RETRY)
        ):
            assert retry.is_retry("GET", 503) is False
            assert retry.is_retry("GET", 429, has_retry_after=True) is False

    def test_honor_vs_ignore_forcelist_egress(self) -> None:
        def run(profile: PolicyProfile | None) -> tuple[int, int]:
            recorder = RequestRecorder(
                side_effects=[
                    HTTPResponse(status=503),
                    HTTPResponse(status=200),
                ]
            )
            with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
                with _record_instance(pool, recorder):
                    if profile is None:
                        response = pool.urlopen(
                            "GET",
                            "/",
                            retries=Retry(total=3, status_forcelist=[503]),
                        )
                    else:
                        with policy_scope(profile):
                            response = pool.urlopen(
                                "GET",
                                "/",
                                retries=Retry(
                                    total=3, status_forcelist=[503]
                                ),
                            )
            return response.status, len(recorder.calls)

        # honor: forcelist status is retried once.
        assert run(None) == (200, 2)
        # ignore: the 503 response is returned without recursion.
        assert run(
            PolicyProfile(status_force=StatusForce.IGNORE_FORCELIST)
        ) == (503, 1)

    def test_honor_vs_force_all_5xx_egress(self) -> None:
        # 500 is not in status_forcelist: the pin returns it immediately.
        honor_recorder = RequestRecorder(
            side_effects=[HTTPResponse(status=500)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, honor_recorder):
                response = pool.urlopen(
                    "GET", "/", retries=Retry(total=3)
                )
        assert response.status == 500
        assert len(honor_recorder.calls) == 1

        # force_all_5xx retries the same 500 into a success.
        force_recorder = RequestRecorder(
            side_effects=[HTTPResponse(status=500), HTTPResponse(status=200)]
        )
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, force_recorder):
                with policy_scope(
                    PolicyProfile(status_force=StatusForce.FORCE_ALL_5XX)
                ):
                    response = pool.urlopen(
                        "GET", "/", retries=Retry(total=3)
                    )
        assert response.status == 200
        assert len(force_recorder.calls) == 2

    def test_forbid_status_retry_egress(self) -> None:
        recorder = RequestRecorder(side_effects=[HTTPResponse(status=503)])
        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(
                        status_force=StatusForce.FORBID_STATUS_RETRY
                    )
                ):
                    response = pool.urlopen(
                        "GET",
                        "/",
                        retries=Retry(total=3, status_forcelist=[503]),
                    )
        assert response.status == 503
        assert len(recorder.calls) == 1

    def test_ignore_pushed_during_sleep_invalidates_pending_retry(self) -> None:
        # is_retry() already returned True and the hop was drained; a frame
        # pushed during the backoff sleep forces a recompute.
        recorder = RequestRecorder(side_effects=[HTTPResponse(status=503)])

        def push_ignore(*args: object) -> None:
            push_policy(
                PolicyProfile(status_force=StatusForce.IGNORE_FORCELIST)
            )

        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with _record_instance(pool, recorder):
                with patch.object(Retry, "sleep", push_ignore):
                    response = pool.urlopen(
                        "GET",
                        "/",
                        retries=Retry(
                            total=3,
                            status_forcelist=[503],
                            backoff_factor=1.0,
                        ),
                    )
        pop_policy()
        assert response.status == 503
        assert len(recorder.calls) == 1

    def test_late_push_after_is_retry_is_recomputed(self) -> None:
        # The first 503 is followed under honor. Before the second decision
        # the frame is pushed; the live re-evaluation then stops retrying.
        def side_effect(*args: object, **kwargs: object) -> HTTPResponse:
            side_effect.calls += 1
            if side_effect.calls == 2:
                push_policy(
                    PolicyProfile(
                        status_force=StatusForce.IGNORE_FORCELIST
                    )
                )
            return HTTPResponse(status=503)

        side_effect.calls = 0

        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with patch.object(pool, "_make_request", side_effect):
                response = pool.urlopen(
                    "GET",
                    "/",
                    retries=Retry(total=5, status_forcelist=[503]),
                )
        pop_policy()
        assert response.status == 503
        assert side_effect.calls == 2


# ---------------------------------------------------------------------------
# proxy_route
# ---------------------------------------------------------------------------


class TestProxyRoute:
    def test_honor_uses_proxy_pool_absolute_form_and_host(self) -> None:
        recorder = RequestRecorder()
        with ProxyManager("http://proxy.example:8080") as manager:
            with _record_class(recorder):
                response = manager.urlopen(
                    "GET", "http://target.example/path"
                )
        assert response.status == 200
        (call,) = recorder.calls
        assert call["pool_host"] == "proxy.example"
        assert call["pool_proxy"] is not None
        assert call["url"] == "http://target.example/path"
        assert call["headers"]["Host"] == "target.example"

    def test_bypass_connects_direct_with_origin_form(self) -> None:
        recorder = RequestRecorder()
        with ProxyManager("http://proxy.example:8080") as manager:
            proxy_before = manager.proxy
            with _record_class(recorder):
                with policy_scope(
                    PolicyProfile(proxy_route=ProxyRoute.BYPASS_PROXY)
                ):
                    response = manager.urlopen(
                        "GET", "http://target.example/path"
                    )
            # manager.proxy is never mutated.
            assert manager.proxy is proxy_before
        assert response.status == 200
        (call,) = recorder.calls
        assert call["pool_host"] == "target.example"
        assert call["pool_proxy"] is None
        assert call["url"] == "/path"
        assert "Host" not in {h for h in call["headers"]}

    def test_bypass_https_target_uses_direct_pool_without_tunnel(self) -> None:
        recorder = RequestRecorder()
        with ProxyManager("http://proxy.example:8080") as manager:
            with _record_class(recorder):
                with policy_scope(
                    PolicyProfile(proxy_route=ProxyRoute.BYPASS_PROXY)
                ):
                    response = manager.urlopen(
                        "GET", "https://target.example/path"
                    )
        assert response.status == 200
        (call,) = recorder.calls
        assert call["pool_host"] == "target.example"
        assert call["pool_proxy"] is None
        assert call["url"] == "/path"

    def test_bypass_is_isolated_from_cached_proxy_pool(self) -> None:
        recorder = RequestRecorder()
        with ProxyManager("http://proxy.example:8080") as manager:
            with _record_class(recorder):
                manager.urlopen("GET", "http://target.example/path")
                with policy_scope(
                    PolicyProfile(proxy_route=ProxyRoute.BYPASS_PROXY)
                ):
                    manager.urlopen(
                        "GET", "http://target.example/path"
                    )
                manager.urlopen("GET", "http://target.example/path")
        honor, bypass, honor_again = recorder.calls
        assert honor["pool_proxy"] is not None
        assert bypass["pool_proxy"] is None
        assert honor_again["pool_proxy"] is not None
        assert honor["pool_host"] == "proxy.example"
        assert bypass["pool_host"] == "target.example"

    def test_deny_tunnel_fails_for_https(self) -> None:
        recorder = RequestRecorder()
        with ProxyManager("http://proxy.example:8080") as manager:
            with _record_class(recorder):
                with policy_scope(
                    PolicyProfile(proxy_route=ProxyRoute.DENY_TUNNEL)
                ):
                    with pytest.raises(ProxyError):
                        manager.urlopen("GET", "https://target.example/")
        assert recorder.calls == []

    def test_deny_tunnel_read_at_prepare_proxy(self) -> None:
        from urllib3.connectionpool import HTTPSConnectionPool
        from urllib3.util.url import parse_url

        pool = HTTPSConnectionPool(
            "target.example",
            _proxy=parse_url("http://proxy.example:8080"),
        )
        try:
            with policy_scope(
                PolicyProfile(proxy_route=ProxyRoute.DENY_TUNNEL)
            ):
                with pytest.raises(ProxyError):
                    pool._prepare_proxy(conn=object())  # type: ignore[arg-type]
        finally:
            pool.close()

    def test_force_absolute_form_on_direct_manager(self) -> None:
        honor_recorder = RequestRecorder()
        with PoolManager() as manager:
            with _record_class(honor_recorder):
                manager.urlopen("GET", "http://target.example/path")
        assert honor_recorder.calls[0]["url"] == "/path"

        absolute_recorder = RequestRecorder()
        with PoolManager() as manager:
            with _record_class(absolute_recorder):
                with policy_scope(
                    PolicyProfile(
                        proxy_route=ProxyRoute.FORCE_ABSOLUTE_FORM
                    )
                ):
                    manager.urlopen("GET", "http://target.example/path")
        assert (
            absolute_recorder.calls[0]["url"]
            == "http://target.example/path"
        )

    def test_bypass_recomputed_on_redirect_hop(self) -> None:
        # Hop 1 is already in flight through the proxy when the frame is
        # pushed; the cross-host redirect recursion must re-evaluate and
        # egress directly instead of silently keeping the proxy route.
        def side_effect(
            pool: HTTPConnectionPool,
            conn: object,
            method: str,
            url: str,
            *args: object,
            **kwargs: object,
        ) -> HTTPResponse:
            side_effect.calls += 1
            side_effect.hosts.append(pool.host)
            if side_effect.calls == 1:
                push_policy(
                    PolicyProfile(proxy_route=ProxyRoute.BYPASS_PROXY)
                )
                response = HTTPResponse(
                    status=302,
                    headers={"location": "http://other.target.example/"},
                )
            else:
                response = HTTPResponse(status=200)
            response.retries = kwargs.get("retries")
            return response

        side_effect.calls = 0
        side_effect.hosts = []

        with ProxyManager("http://proxy.example:8080") as manager:
            with _record_class(side_effect):
                response = manager.urlopen(
                    "GET",
                    "http://target.example/",
                    retries=Retry(total=5, redirect=5),
                )
        pop_policy()
        assert response.status == 200
        assert side_effect.calls == 2
        # First hop used the proxy (decision predates the push); the
        # redirected hop went direct after re-evaluation.
        assert side_effect.hosts == [
            "proxy.example",
            "other.target.example",
        ]


# ---------------------------------------------------------------------------
# Late retry_budget push while a failure recursion is already in flight
# ---------------------------------------------------------------------------


class TestFlightRecompute:
    def test_deny_pushed_after_recursion_entered_fails_next_decision(
        self,
    ) -> None:
        def side_effect(*args: object, **kwargs: object) -> HTTPResponse:
            side_effect.calls += 1
            if side_effect.calls == 2:
                push_policy(
                    PolicyProfile(retry_budget=RetryBudget.DENY_RETRY)
                )
            raise OSError("reset")

        side_effect.calls = 0

        with HTTPConnectionPool(host="localhost", maxsize=1) as pool:
            with patch.object(pool, "_make_request", side_effect):
                with pytest.raises((ProtocolError, MaxRetryError)) as exc_info:
                    pool.urlopen(
                        "GET",
                        "/",
                        retries=Retry(total=5, connect=5),
                    )
        pop_policy()
        assert side_effect.calls == 2
        # The late frame is honored by the live decision: raw error rather
        # than another wrapped retry.
        assert isinstance(exc_info.value, ProtocolError)


# ---------------------------------------------------------------------------
# Convenience entry points funnel through the same urlopen decisions
# ---------------------------------------------------------------------------


class TestEntryPointConsistency:
    def test_request_and_encode_variants_share_redirect_surface(self) -> None:
        redirect = HTTPResponse(
            status=302, headers={"location": "http://example.org/next"}
        )
        recorder = RequestRecorder(side_effects=[redirect, redirect, redirect])
        with PoolManager() as manager:
            with _record_class(recorder):
                with policy_scope(
                    PolicyProfile(
                        redirect_surface=RedirectSurface.DENY_REDIRECT
                    )
                ):
                    for caller in (
                        lambda: manager.request(
                            "GET", "http://example.com/"
                        ),
                        lambda: manager.request_encode_url(
                            "GET", "http://example.com/"
                        ),
                        lambda: manager.request_encode_body(
                            "POST",
                            "http://example.com/",
                            fields={"k": "v"},
                            encode_multipart=False,
                        ),
                    ):
                        response = caller()
                        assert response.status == 302
        # Three entry points, zero recursive follows.
        assert len(recorder.calls) == 3

    def test_connection_from_url_reads_retry_budget(self) -> None:
        from urllib3 import connection_from_url

        recorder = RequestRecorder(
            side_effects=[OSError("reset") for _ in range(5)]
        )
        pool = connection_from_url("http://localhost/")
        try:
            with _record_instance(pool, recorder):
                with policy_scope(
                    PolicyProfile(retry_budget=RetryBudget.DENY_RETRY)
                ):
                    with pytest.raises(ProtocolError):
                        pool.urlopen(
                            "GET",
                            "/",
                            retries=Retry(total=5, connect=5),
                        )
        finally:
            pool.close()
        assert len(recorder.calls) == 1


# ---------------------------------------------------------------------------
# Two threads sharing one pool / one manager
# ---------------------------------------------------------------------------


class TestThreadIsolation:
    def test_two_threads_different_retry_budgets(self) -> None:
        pool = HTTPConnectionPool(host="localhost", maxsize=10)
        recorder = RequestRecorder(
            side_effects=[OSError("reset") for _ in range(50)]
        )
        results: dict[str, object] = {}
        barrier = threading.Barrier(2)

        def deny_worker() -> None:
            barrier.wait()
            with policy_scope(
                PolicyProfile(retry_budget=RetryBudget.DENY_RETRY)
            ):
                try:
                    pool.urlopen(
                        "GET",
                        "/",
                        retries=Retry(total=5, connect=5),
                    )
                except ProtocolError as e:
                    results["deny_error"] = e
                    results["deny_ident"] = threading.get_ident()

        def honor_worker() -> None:
            barrier.wait()
            try:
                pool.urlopen(
                    "GET",
                    "/",
                    retries=Retry(total=2, connect=2),
                )
            except MaxRetryError as e:
                results["honor_error"] = e
                results["honor_ident"] = threading.get_ident()

        threads = [
            threading.Thread(target=deny_worker),
            threading.Thread(target=honor_worker),
        ]
        # The patch is installed once, outside the threads, so concurrent
        # frame push/pop never races with mock restoration.
        with _record_instance(pool, recorder):
            try:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            finally:
                pool.close()

        assert isinstance(results.get("deny_error"), ProtocolError)
        assert isinstance(results.get("honor_error"), MaxRetryError)
        assert (
            recorder.attempts_for(results["deny_ident"]) == 1  # type: ignore[arg-type]
        )
        assert (
            recorder.attempts_for(results["honor_ident"]) == 3  # type: ignore[arg-type]
        )
        # Nothing leaked onto the worker/main stacks.
        assert current_policy() == DEFAULT_PROFILE

    def test_two_threads_different_proxy_routes(self) -> None:
        manager = ProxyManager("http://proxy.example:8080")
        recorder = RequestRecorder()
        barrier = threading.Barrier(2)

        def bypass_worker() -> None:
            barrier.wait()
            with policy_scope(
                PolicyProfile(proxy_route=ProxyRoute.BYPASS_PROXY)
            ):
                manager.urlopen("GET", "http://target.example/path")

        def honor_worker() -> None:
            barrier.wait()
            manager.urlopen("GET", "http://target.example/path")

        threads = [
            threading.Thread(target=bypass_worker),
            threading.Thread(target=honor_worker),
        ]
        with _record_class(recorder):
            try:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            finally:
                manager.clear()

        by_thread: dict[int, list[dict]] = {}
        for call in recorder.calls:
            by_thread.setdefault(call["thread"], []).append(call)
        groups = list(by_thread.values())
        assert len(groups) == 2
        proxy_group = [g for g in groups if g[0]["pool_proxy"] is not None]
        direct_group = [g for g in groups if g[0]["pool_proxy"] is None]
        assert len(proxy_group) == len(direct_group) == 1
        assert proxy_group[0][0]["pool_host"] == "proxy.example"
        assert proxy_group[0][0]["url"] == "http://target.example/path"
        assert direct_group[0][0]["pool_host"] == "target.example"
        assert direct_group[0][0]["url"] == "/path"
        assert urllib3.current_policy() == DEFAULT_PROFILE
