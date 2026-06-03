#!/usr/bin/env python3
"""Integration smoke test: vendored phmr_pipeline + Video2SMPL pipeline package coexist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Prove Video2SMPL pipeline is loaded first (same order as run.py)
import pipeline.manifest  # noqa: F401
from pipeline.stages.video2smpl.backends.prompthmr import run_prompthmr_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="/home/wujunhao/code/PromptHMR/data/examples/test_short.mp4",
    )
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--out", default="/tmp/test_prompthmr_out.npz")
    args = parser.parse_args()

    import pipeline as v2_pipeline

    print("=== PromptHMR in-process integration test ===")
    print(f"python: {sys.executable}")
    print(f"Video2SMPL pipeline pkg: {v2_pipeline.__file__}")

    class _A:
        max_frames = args.max_frames
        static_camera = True
        prompthmr_ckpt_root = "/data1/wjh/ckpt/PromptHMR"
        prompthmr_vendor = None

    run_prompthmr_sample(
        video_path=Path(args.video),
        output_npz=Path(args.out),
        text_prompt="person walking and moving arms in the scene",
        args=_A(),
    )
    print(f"OK: wrote {args.out}")
    print(f"Still Video2SMPL pipeline: {v2_pipeline.__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
