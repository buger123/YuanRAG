"""Module entry point shim — ``python -m tests.eval``.

v2.0.32.0 — the user-facing command is ``python -m tests.eval``;
this module re-exports ``tests.eval.cli.main`` so that path works.
"""
from __future__ import annotations

from tests.eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main())