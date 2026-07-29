"""Bounded polling helpers.

Tests wait for an observable condition instead of sleeping for a guessed
duration: a passing run finishes as soon as the condition holds, and a broken
build fails with a description rather than hanging.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

DEFAULT_TIMEOUT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.001


async def wait_until(
    predicate: Callable[[], bool],
    *,
    description: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Yield to the loop until ``predicate`` holds, or fail with a message."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout_seconds}s waiting until {description}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def settle(cycles: int = 3) -> None:
    """Give already-scheduled callbacks a chance to run."""

    for _ in range(cycles):
        await asyncio.sleep(0)


__all__ = ["DEFAULT_TIMEOUT_SECONDS", "settle", "wait_until"]
