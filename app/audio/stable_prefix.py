"""Stable common-prefix calculation for rolling ASR hypotheses."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable


def longest_common_prefix(values: Iterable[str]) -> str:
    """Return the exact character prefix shared by every supplied string."""

    iterator = iter(values)
    try:
        prefix = next(iterator)
    except StopIteration:
        return ""
    for value in iterator:
        limit = min(len(prefix), len(value))
        index = 0
        while index < limit and prefix[index] == value[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def stable_common_prefix(hypotheses: Iterable[str], *, complete_words: bool = True) -> str:
    """Return text agreed on by all hypotheses, optionally at a word boundary.

    When every hypothesis is identical the complete hypothesis is stable. For
    differing hypotheses, incomplete final words are excluded so clients do
    not present a character fragment as committed text.
    """

    values = tuple(hypotheses)
    if not values:
        return ""
    prefix = longest_common_prefix(values)
    if not complete_words or all(value == prefix for value in values):
        return prefix
    if prefix and prefix[-1].isspace():
        return prefix.rstrip()
    boundary = max(
        (index for index, char in enumerate(prefix) if char.isspace()),
        default=-1,
    )
    return prefix[:boundary].rstrip() if boundary >= 0 else ""


class RollingStablePrefix:
    """Track a bounded hypothesis window and calculate its stable prefix."""

    __slots__ = ("_hypotheses",)

    def __init__(self, window_size: int = 3) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least two")
        self._hypotheses: deque[str] = deque(maxlen=window_size)

    def add(self, hypothesis: str) -> str:
        self._hypotheses.append(hypothesis)
        if len(self._hypotheses) < 2:
            return ""
        return stable_common_prefix(self._hypotheses)

    def reset(self) -> None:
        self._hypotheses.clear()
