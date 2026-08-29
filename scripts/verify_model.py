#!/usr/bin/env python3
"""Verify an immutable local model snapshot and its completion metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from download_model import ModelValidationError, verify_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="published model directory")
    parser.add_argument(
        "--repository", required=True, help="expected Hugging Face owner/name repository"
    )
    parser.add_argument(
        "--revision", required=True, help="expected full immutable 40-hex commit SHA"
    )
    return parser


def _fail(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        manifest = verify_model(arguments.model_dir, arguments.repository, arguments.revision)
    except (ModelValidationError, OSError) as exc:
        _fail(parser, str(exc))
    print(
        f"verified {len(manifest['files'])} files for "
        f"{manifest['repository']}@{manifest['revision']}"
    )


if __name__ == "__main__":
    main()
