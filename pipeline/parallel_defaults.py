"""Default parallel worker / GPU counts for pipeline stages."""

from __future__ import annotations

# API / filter parallelism (select, captions)
DEFAULT_STAGE_WORKERS = 8

# video2smpl: one worker process per GPU
DEFAULT_GPU_WORKERS = 8
