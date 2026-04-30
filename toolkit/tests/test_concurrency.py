"""Tests for the small async concurrency helpers."""

from __future__ import annotations

import asyncio
import time

import pytest

from sentinel.concurrency import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_disabled_when_rps_zero() -> None:
    limiter = RateLimiter(rate_limit_rps=0)

    started = time.perf_counter()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.perf_counter() - started

    # No sleeping should happen when rate limiting is disabled.
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_rate_limiter_throttles_to_budget() -> None:
    # 50 requests per second -> ~20 ms minimum spacing per acquire().
    limiter = RateLimiter(rate_limit_rps=50)

    started = time.perf_counter()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.perf_counter() - started

    # 5 acquires at 50 rps -> at least 4 intervals of 20 ms == 0.08 s.
    # We allow a generous lower bound to absorb scheduler jitter.
    assert elapsed >= 0.06


@pytest.mark.asyncio
async def test_rate_limiter_serialises_concurrent_acquirers() -> None:
    limiter = RateLimiter(rate_limit_rps=20)  # 50 ms spacing

    started = time.perf_counter()
    await asyncio.gather(*(limiter.acquire() for _ in range(4)))
    elapsed = time.perf_counter() - started

    # 4 concurrent acquires at 20 rps -> at least 3 * 50 ms == 0.15 s.
    assert elapsed >= 0.12
