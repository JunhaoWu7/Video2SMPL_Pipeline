#!/usr/bin/env python3
"""
Backward-compatible entry for the external_smpl sub-stage.

Prefer: python run.py --stages external_smpl ...
Or:     python -m pipeline.stages.external_smpl.run ...
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages.external_smpl.run import build_parser, run  # noqa: E402

if __name__ == "__main__":
    run(build_parser().parse_args())
