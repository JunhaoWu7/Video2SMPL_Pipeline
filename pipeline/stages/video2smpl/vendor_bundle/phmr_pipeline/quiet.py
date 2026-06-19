"""Pipeline integration: suppress verbose PromptHMR stdout when batch-processing."""

from __future__ import annotations

import os
import sys


def is_quiet() -> bool:
    return os.environ.get("VIDEO2SMPL_PROMPTHMR_QUIET", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def log(msg: str) -> None:
    if not is_quiet():
        print(msg, flush=True)


def log_warn(msg: str) -> None:
    """Non-fatal warnings (shown even in quiet mode, one line)."""
    print(f"WARN: {msg}", flush=True, file=sys.stderr)
