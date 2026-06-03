#!/usr/bin/env python3
"""Standalone PromptHMR clip runner (optional; main path uses in-process backend)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.stages.video2smpl.backends.prompthmr import run_prompthmr_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--text-prompt", required=True)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--static-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt-root", default="/data1/wjh/ckpt/PromptHMR")
    args = parser.parse_args()

    class _A:
        max_frames = args.max_frames
        static_camera = args.static_camera
        prompthmr_ckpt_root = args.ckpt_root
        prompthmr_vendor = None

    run_prompthmr_sample(
        video_path=Path(args.video),
        output_npz=Path(args.out_npz),
        text_prompt=args.text_prompt,
        args=_A(),
    )
    print(json.dumps({"ok": True, "out": args.out_npz}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
