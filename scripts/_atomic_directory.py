"""Shared crash-safe publication for fully prepared directories."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_directory(
    staging: Path,
    destination: Path,
    verify_existing: Callable[[Path], object],
) -> None:
    """Publish ``staging``, or verify a directory published by a concurrent winner.

    Only rename failures can be interpreted as a publication race. Once rename
    succeeds, failure to persist the destination entry in its parent propagates
    to the caller rather than being confused with an idempotent winner.
    """

    try:
        os.rename(staging, destination)
    except OSError:
        if destination.exists() and not destination.is_symlink():
            verify_existing(destination)
            return
        raise

    fsync_directory(destination.parent)
