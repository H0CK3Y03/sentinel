"""Small async concurrency helpers.

Currently this module only contains :class:`RateLimiter`, used by the
orchestrator to throttle prompt dispatch when a manifest specifies a
maximum requests-per-second budget.
"""

from __future__ import annotations

import asyncio


class RateLimiter:
    """Serialise request starts to respect a maximum prompts-per-second budget.

    A value of ``0`` (or anything ≤ 0) disables rate limiting, which is the
    default for local backends that do not need throttling.
    """

    def __init__(self, rate_limit_rps: float) -> None:
        self._interval = 0.0 if rate_limit_rps <= 0 else 1.0 / rate_limit_rps
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def acquire(self) -> None:
        """Block until the caller is allowed to dispatch a new request."""
        if self._interval <= 0:
            return

        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            my_slot = max(self._next_allowed_at, now)
            self._next_allowed_at = my_slot + self._interval

        # Sleep outside the lock so concurrent callers can claim their slots
        # without waiting for the previous caller's sleep to finish.
        sleep_time = my_slot - loop.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
