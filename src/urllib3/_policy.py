"""
Thread-local, nestable policy profiles for the egress decision points.

This module implements a per-thread stack of :class:`PolicyProfile` frames.
Each profile carries four independent dimensions:

* :class:`RetryBudget` -- what happens to connection/read/protocol failures
  and to the counters consumed by ``Retry.increment``.
* :class:`RedirectSurface` -- how 3xx responses are followed and whether the
  method is changed before the recursive hop.
* :class:`StatusForce` -- how ``Retry.is_retry`` treats ``status_forcelist``,
  ``Retry-After`` and 5xx responses.
* :class:`ProxyRoute` -- how ``ProxyManager``/connection-pool egress picks
  tunnel vs forwarding vs absolute URI form.

The stack is strictly thread-local (there is intentionally no process-wide
singleton) and it never mutates ``Retry`` instances, pool defaults or
``manager.proxy``.  An empty stack is exactly equivalent to the pinned
behavior: every egress point consults the stack live, at its own decision
point, so a profile pushed after a decision has flown only affects decisions
that are still pending -- nothing is silently decided from a snapshot taken
on entry.

Profiles nest: inner frames override outer frames per dimension, and popping
an inner frame reveals the outer one again.
"""

from __future__ import annotations

import contextlib
import enum
import threading
import typing
import weakref
from types import TracebackType

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from .connectionpool import ConnectionPool
    from .response import BaseHTTPResponse
    from .util.retry import Retry


# ---------------------------------------------------------------------------
# Policy dimensions
# ---------------------------------------------------------------------------


class RetryBudget(enum.Enum):
    """Retries on connection/read/protocol failures and counter accounting."""

    #: Follow ``Retry.increment`` and the total/connect/read counters as
    #: configured by the caller and the pool default.
    HONOR_RETRY = "honor_retry"
    #: Connection/read/protocol failures in this frame fail immediately; the
    #: error is re-raised without another recursive ``urlopen`` and without
    #: wrapping it in ``MaxRetryError``.
    DENY_RETRY = "deny_retry"
    #: ``is_retry``/redirect decisions still apply, but the counters observable
    #: on the ``Retry`` snapshots threaded through recursion never decrease
    #: below the snapshot seen when the frame started accounting for a chain.
    FREEZE_COUNTS = "freeze_counts"
    #: Give connection/read failures exactly one additional recursive attempt
    #: even if the incoming budget is exhausted; afterwards the outer budget
    #: resumes.
    ONE_MORE = "one_more"


class RedirectSurface(enum.Enum):
    """Following of 3xx responses and method changes across hops."""

    #: Follow redirects as pinned; only 303 changes the method to GET.
    HONOR_REDIRECT = "honor_redirect"
    #: Hand the 3xx response back to the caller even when ``redirect=True``
    #: and ``Retry.redirect`` is still positive.
    DENY_REDIRECT = "deny_redirect"
    #: Every followed 301/302/303/307/308 becomes GET with the body state
    #: cleared.
    FORCE_GET_ON_3XX = "force_get_on_3xx"
    #: Keep the method and the body even on a 303.
    PRESERVE_METHOD = "preserve_method"


class StatusForce(enum.Enum):
    """Status-code-forced retries as decided by ``Retry.is_retry``."""

    #: Honor ``status_forcelist`` and the ``Retry-After`` fork as pinned.
    HONOR_FORCELIST = "honor_forcelist"
    #: Never retry solely because the status is in ``status_forcelist``; the
    #: ``Retry-After`` fork still applies.
    IGNORE_FORCELIST = "ignore_forcelist"
    #: Treat every 500-599 status as retryable, even when the code is absent
    #: from ``status_forcelist``.
    FORCE_ALL_5XX = "force_all_5xx"
    #: Forbid every status-based retry (including ``Retry-After``). Only
    #: connection/read failures and redirects remain.
    FORBID_STATUS_RETRY = "forbid_status_retry"


class ProxyRoute(enum.Enum):
    """How the egress chooses proxies, tunnels and URI forms."""

    #: Honor CONNECT tunnels, forwarding and ``proxy_headers`` as pinned.
    HONOR_PROXY = "honor_proxy"
    #: Egress through ``ProxyManager`` must connect directly to the target
    #: host without going through the manager's proxy; ``manager.proxy`` is
    #: never modified.
    BYPASS_PROXY = "bypass_proxy"
    #: Even on a direct ``PoolManager`` the request line must use the absolute
    #: URI form.
    FORCE_ABSOLUTE_FORM = "force_absolute_form"
    #: Forbid CONNECT tunnels for HTTPS targets; fail loudly where the pin
    #: would have tunneled.
    DENY_TUNNEL = "deny_tunnel"


class PolicyProfile(typing.NamedTuple):
    """An immutable four-dimensional policy frame."""

    retry_budget: RetryBudget = RetryBudget.HONOR_RETRY
    redirect_surface: RedirectSurface = RedirectSurface.HONOR_REDIRECT
    status_force: StatusForce = StatusForce.HONOR_FORCELIST
    proxy_route: ProxyRoute = ProxyRoute.HONOR_PROXY


#: The default, do-nothing profile. An empty stack resolves to this value.
DEFAULT_PROFILE = PolicyProfile()

#: Convenience profiles overriding a single dimension.
DENY_RETRY_PROFILE = PolicyProfile(retry_budget=RetryBudget.DENY_RETRY)
FREEZE_COUNTS_PROFILE = PolicyProfile(retry_budget=RetryBudget.FREEZE_COUNTS)
ONE_MORE_PROFILE = PolicyProfile(retry_budget=RetryBudget.ONE_MORE)
DENY_REDIRECT_PROFILE = PolicyProfile(
    redirect_surface=RedirectSurface.DENY_REDIRECT
)
FORCE_GET_ON_3XX_PROFILE = PolicyProfile(
    redirect_surface=RedirectSurface.FORCE_GET_ON_3XX
)
PRESERVE_METHOD_PROFILE = PolicyProfile(
    redirect_surface=RedirectSurface.PRESERVE_METHOD
)
IGNORE_FORCELIST_PROFILE = PolicyProfile(status_force=StatusForce.IGNORE_FORCELIST)
FORCE_ALL_5XX_PROFILE = PolicyProfile(status_force=StatusForce.FORCE_ALL_5XX)
FORBID_STATUS_RETRY_PROFILE = PolicyProfile(
    status_force=StatusForce.FORBID_STATUS_RETRY
)
BYPASS_PROXY_PROFILE = PolicyProfile(proxy_route=ProxyRoute.BYPASS_PROXY)
FORCE_ABSOLUTE_FORM_PROFILE = PolicyProfile(
    proxy_route=ProxyRoute.FORCE_ABSOLUTE_FORM
)
DENY_TUNNEL_PROFILE = PolicyProfile(proxy_route=ProxyRoute.DENY_TUNNEL)


# ---------------------------------------------------------------------------
# Thread-local stack
# ---------------------------------------------------------------------------


class _PolicyFrame:
    """A pushed profile plus frame-scoped bookkeeping."""

    __slots__ = ("profile", "one_more_used")

    def __init__(self, profile: PolicyProfile) -> None:
        self.profile = profile
        # Whether the ONE_MORE frame has already granted its single attempt
        # for this thread.
        self.one_more_used = False


class _Tag(typing.NamedTuple):
    # Internal marker attached (through a thread-local table, never by
    # mutating the Retry object) to Retry snapshots produced while a
    # non-honor retry_budget frame is active.
    kind: str
    frame_id: int
    shadow: Retry | None = None
    baseline: Retry | None = None


class _LocalState(threading.local):
    def __init__(self) -> None:
        self.stack: list[_PolicyFrame] = []
        # Per-thread table mapping Retry snapshots to their policy tags.
        # Keying by object here is thread-isolated and therefore safe even
        # when two threads share the same pool-default Retry instance.
        self.tags: weakref.WeakKeyDictionary[Retry, _Tag] = weakref.WeakKeyDictionary()


_local = _LocalState()


def _stack() -> list[_PolicyFrame]:
    return _local.stack


def push_policy(profile: PolicyProfile = DEFAULT_PROFILE) -> _PolicyFrame:
    """Push a profile frame onto the current thread's stack.

    Returns the frame token, which may be passed to :func:`pop_policy` to
    # ensure the same frame is popped.
    """
    frame = _PolicyFrame(profile)
    _stack().append(frame)
    return frame


def pop_policy(expected: _PolicyFrame | None = None) -> PolicyProfile:
    """Pop the top profile frame from the current thread's stack."""
    stack = _stack()
    if not stack:
        raise IndexError("pop_policy() called on an empty policy stack")
    frame = stack.pop()
    if expected is not None and frame is not expected:
        # Put it back, the caller asked for a specific frame.
        stack.append(frame)
        raise RuntimeError("policy frame mismatch in pop_policy()")
    return frame.profile


@contextlib.contextmanager
def policy_scope(profile: PolicyProfile = DEFAULT_PROFILE) -> Iterator[None]:
    """Context manager pushing ``profile`` and guaranteeing pop on exit.

    The frame is popped even when the wrapped block raises, so nested
    middleware can never leak profiles onto the thread's stack.
    """
    frame = push_policy(profile)
    try:
        yield
    finally:
        pop_policy(frame)


def current_policy() -> PolicyProfile:
    """Return the read-only effective profile for the current thread.

    Dimensions resolve to the innermost frame overriding them; dimensions no
    frame overrides resolve to the honor values. An empty stack therefore
    yields :data:`DEFAULT_PROFILE` (identical to the pinned behavior).
    """
    stack = _stack()
    if not stack:
        return DEFAULT_PROFILE

    values: list[object | None] = [None, None, None, None]
    dimensions = ("retry_budget", "redirect_surface", "status_force", "proxy_route")
    honors = (
        RetryBudget.HONOR_RETRY,
        RedirectSurface.HONOR_REDIRECT,
        StatusForce.HONOR_FORCELIST,
        ProxyRoute.HONOR_PROXY,
    )

    for frame in reversed(stack):
        for i, name in enumerate(dimensions):
            if values[i] is None:
                value = getattr(frame.profile, name)
                if value is not honors[i]:
                    values[i] = value
        if all(value is not None for value in values):
            break

    return PolicyProfile(
        typing.cast(RetryBudget, values[0] or honors[0]),
        typing.cast(RedirectSurface, values[1] or honors[1]),
        typing.cast(StatusForce, values[2] or honors[2]),
        typing.cast(ProxyRoute, values[3] or honors[3]),
    )


# ---------------------------------------------------------------------------
# Dimension readers used by the egress points
# ---------------------------------------------------------------------------


def follow_redirects_allowed() -> bool:
    """Whether the current thread permits following a 3xx at this decision."""
    return current_policy().redirect_surface is not RedirectSurface.DENY_REDIRECT


def change_method_on_redirect(status_code: int) -> bool:
    """Whether the followed hop must be converted to GET with body cleared."""
    surface = current_policy().redirect_surface
    if surface is RedirectSurface.FORCE_GET_ON_3XX:
        return True
    if surface is RedirectSurface.PRESERVE_METHOD:
        return False
    # HONOR_REDIRECT (and DENY_REDIRECT, which never reaches the hop): only
    # 303 changes the method, per RFC 9110.
    return status_code == 303


def status_force_policy() -> StatusForce:
    return current_policy().status_force


def status_retry_decision(
    retry: Retry, method: str, status_code: int, has_retry_after: bool
) -> bool:
    """Policy-only half of ``Retry.is_retry``.

    Decides whether the status code itself is retryable under the active
    ``status_force`` profile. Counter accounting (``self.total``) is
    deliberately not consulted: this predicate is used to re-evaluate a
    pending status retry after the counters have already been incremented,
    where re-reading the budget would wrongly suppress the recursion the
    pin would have made.
    """
    if not retry._is_method_retryable(method):
        return False

    policy = status_force_policy()

    retry_after = bool(
        retry.respect_retry_after_header
        and has_retry_after
        and status_code in retry.RETRY_AFTER_STATUS_CODES
    )

    if policy is StatusForce.FORBID_STATUS_RETRY:
        return False
    if policy is StatusForce.IGNORE_FORCELIST:
        return retry_after
    if policy is StatusForce.FORCE_ALL_5XX:
        return retry_after or 500 <= status_code <= 599
    # honor_forcelist: the forcelist fork and the Retry-After fork.
    return bool(
        (retry.status_forcelist and status_code in retry.status_forcelist)
        or retry_after
    )


def current_proxy_route() -> ProxyRoute:
    return current_policy().proxy_route


# ---------------------------------------------------------------------------
# Retry counter accounting
# ---------------------------------------------------------------------------


def _active_frame(frame_id: int) -> _PolicyFrame | None:
    for frame in reversed(_stack()):
        if id(frame) == frame_id:
            return frame
    return None


def _frame_for(dimension: str, value: enum.Enum) -> _PolicyFrame | None:
    for frame in reversed(_stack()):
        if getattr(frame.profile, dimension) is value:
            return frame
    return None


_TYPE_COUNT = typing.Union[bool, int, None]


def _freeze_count(baseline: _TYPE_COUNT, current: _TYPE_COUNT) -> _TYPE_COUNT:
    # Counters only ever decrease through increment(); keep the baseline
    # value so observable counts never shrink, while the shadow Retry keeps
    # driving exhaustion decisions.
    if baseline is None or baseline is False:
        return baseline
    if current is None or current is False:
        return current
    return max(baseline, current)


def _freeze_increment(
    retries: Retry,
    method: str | None,
    url: str | None,
    response: BaseHTTPResponse | None,
    error: Exception | None,
    pool: ConnectionPool | None,
    stacktrace: TracebackType | None,
) -> Retry:
    tag = _local.tags.get(retries)
    if tag is not None and tag.kind == "freeze" and _active_frame(tag.frame_id):
        shadow = typing.cast("Retry", tag.shadow)
        baseline = typing.cast("Retry", tag.baseline)
        frame_id = tag.frame_id
    else:
        shadow = baseline = retries
        frame = _frame_for("retry_budget", RetryBudget.FREEZE_COUNTS)
        frame_id = id(frame) if frame is not None else -1

    # The shadow consumes the budget and decides exhaustion; the returned
    # snapshot keeps the frame-entry counts and only gains history.
    new_shadow = shadow.increment(
        method,
        url,
        response=response,
        error=error,
        _pool=pool,
        _stacktrace=stacktrace,
    )

    view = new_shadow.new(
        total=_freeze_count(baseline.total, new_shadow.total),
        connect=_freeze_count(baseline.connect, new_shadow.connect),
        read=_freeze_count(baseline.read, new_shadow.read),
        redirect=_freeze_count(baseline.redirect, new_shadow.redirect),
        status=_freeze_count(baseline.status, new_shadow.status),
        other=_freeze_count(baseline.other, new_shadow.other),
    )
    _local.tags[view] = _Tag(
        "freeze", frame_id, shadow=new_shadow, baseline=baseline
    )
    return view


def _one_more_increment(
    retries: Retry,
    method: str | None,
    url: str | None,
    error: Exception,
    pool: ConnectionPool | None,
    stacktrace: TracebackType | None,
) -> Retry:
    frame = _frame_for("retry_budget", RetryBudget.ONE_MORE)

    tag = _local.tags.get(retries)
    if (
        frame is not None
        and tag is not None
        and tag.kind == "grant"
        and _active_frame(tag.frame_id) is frame
    ):
        # The granted attempt failed. Resume the outer budget as if the grant
        # never consumed it.
        retries = typing.cast("Retry", tag.baseline)
        return retries.increment(
            method,
            url,
            error=error,
            _pool=pool,
            _stacktrace=stacktrace,
        )

    if frame is None or frame.one_more_used:
        return retries.increment(
            method,
            url,
            error=error,
            _pool=pool,
            _stacktrace=stacktrace,
        )

    frame.one_more_used = True
    # Exactly one further recursion: total is spent by the *next* failure
    # (which takes the tagged-grant branch), and type-specific counters are
    # left open so nothing can short-circuit before that recursion starts.
    grant = retries.new(
        total=1,
        connect=None,
        read=None,
        other=None,
        status=None,
        redirect=None,
    )
    _local.tags[grant] = _Tag("grant", id(frame), baseline=retries)
    return grant


def policy_increment(
    retries: Retry,
    method: str | None = None,
    url: str | None = None,
    response: BaseHTTPResponse | None = None,
    error: Exception | None = None,
    pool: ConnectionPool | None = None,
    stacktrace: TracebackType | None = None,
) -> Retry:
    """Apply the thread's effective ``retry_budget`` around ``Retry.increment``.

    The pin calls this at every increment site: connection/read/protocol
    failures (``error`` set), same-pool and cross-host redirects
    (``response`` with a redirect location), and status-forced retries.

    * ``DENY_RETRY`` only affects the failure path: the original error is
      re-raised immediately and no recursion happens.
    * ``FREEZE_COUNTS`` applies to every site; exhaustion is still driven by
      a shadow budget.
    * ``ONE_MORE`` only grants on the failure path, exactly once per frame.
    * Everything else delegates to plain ``Retry.increment``.
    """
    budget = current_policy().retry_budget

    if error is not None and budget is RetryBudget.DENY_RETRY:
        # Mirror the pinned retries=False path: re-raise the causing error
        # rather than wrapping it in MaxRetryError.
        from .util.util import reraise

        raise reraise(type(error), error, stacktrace)

    if budget is RetryBudget.FREEZE_COUNTS:
        return _freeze_increment(
            retries, method, url, response, error, pool, stacktrace
        )

    if error is not None and budget is RetryBudget.ONE_MORE:
        return _one_more_increment(retries, method, url, error, pool, stacktrace)

    return retries.increment(
        method,
        url,
        response=response,
        error=error,
        _pool=pool,
        _stacktrace=stacktrace,
    )


# ---------------------------------------------------------------------------
# Proxy route failures
# ---------------------------------------------------------------------------


def tunnel_denied_error(target_url: str | None) -> Exception:
    """Build the exception raised where the pin would have CONNECT-tunneled."""
    from .exceptions import ProxyError

    return ProxyError(
        f"CONNECT tunnel to {target_url!r} forbidden by the active "
        "proxy_route=deny_tunnel policy profile.",
        RuntimeError("deny_tunnel"),
    )


def proxy_bypass_unavailable_error() -> Exception:
    """Raised when bypass is requested on an already proxy-configured pool.

    Manager-level bypass routes through a fresh, direct pool instead. A raw
    proxy-configured connection pool cannot silently dial the target (its
    connections are created against the proxy), so fail loudly rather than
    pretend to honor the profile.
    """
    from .exceptions import ProxyError

    return ProxyError(
        "proxy_route=bypass_proxy cannot be honored by a connection pool "
        "constructed with a proxy; egress through PoolManager/ProxyManager "
        "so a direct pool can be selected.",
        RuntimeError("bypass_proxy"),
    )
