from __future__ import annotations

import argparse
import os

from pipeline.stages.base import PipelineStage
from pipeline.llm_defaults import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_SELECT_VLM_MODEL,
    resolve_llm_api_key,
)
from pipeline.stages.select.filters.common import DEFAULT_SELECT_YOLO_PATH
from pipeline.parallel_defaults import DEFAULT_STAGE_WORKERS


class SelectStage(PipelineStage):
    """
    Select: step1/step2/step3 on ``video/``, ingest passes, mark stage complete.
    """

    name = "select"
    description = "Filter (step1/2/3 VLM) and ingest videos into manifest"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("select stage")
        group.add_argument(
            "--select-input-dir",
            type=str,
            default=None,
            help="Raw videos to ingest (default: <root_dir>/video).",
        )
        group.add_argument(
            "--select-symlink",
            action="store_true",
            help="Symlink instead of move into processed_trainable_data/<sample_id>/.",
        )
        group.add_argument(
            "--select-skip-filters",
            action="store_true",
            help="Skip step1/step2; step3 VLM still runs unless --select-skip-vlm.",
        )
        group.add_argument(
            "--select-skip-vlm",
            action="store_true",
            help="Skip step3 VLM (select will not be marked complete).",
        )
        group.add_argument(
            "--select-yolo-model",
            type=str,
            default=DEFAULT_SELECT_YOLO_PATH,
            help=f"YOLO weights for step2 (default: {DEFAULT_SELECT_YOLO_PATH}).",
        )
        group.add_argument("--select-min-duration", type=float, default=1.0)
        group.add_argument("--select-max-duration", type=float, default=120.0)
        group.add_argument("--select-min-side", type=int, default=240)
        group.add_argument("--select-step1-frames", type=int, default=12)
        group.add_argument("--select-step2-frames", type=int, default=16)
        group.add_argument(
            "--select-vlm-model",
            type=str,
            default=DEFAULT_SELECT_VLM_MODEL,
        )
        group.add_argument("--select-vlm-frames", type=int, default=6)
        group.add_argument("--select-vlm-max-side", type=int, default=512)
        group.add_argument(
            "--select-vlm-vision-detail",
            type=str,
            default="low",
            choices=("low", "high", "auto", "original"),
        )
        group.add_argument("--select-vlm-timeout", type=float, default=120.0)
        group.add_argument("--select-vlm-max-retries", type=int, default=2)
        group.add_argument(
            "--select-vlm-base-url",
            type=str,
            default=DEFAULT_LLM_BASE_URL,
        )
        group.add_argument("--select-vlm-http-referer", type=str, default="")
        group.add_argument("--select-vlm-x-title", type=str, default="video2smpl-select-vlm")
        group.add_argument(
            "--select-workers",
            type=int,
            default=DEFAULT_STAGE_WORKERS,
            help=f"Parallel filter workers per video (default: {DEFAULT_STAGE_WORKERS}). Use 1 for serial.",
        )

    def validate_args(self, args: argparse.Namespace) -> None:
        if not str(getattr(args, "source", "") or "").strip():
            raise ValueError('--source is required when running the "select" stage.')
        from pathlib import Path

        from pipeline.hub import resolve_select_input_dir

        root = Path(getattr(args, "root_dir", ".")).resolve()
        input_dir = resolve_select_input_dir(root, getattr(args, "select_input_dir", None))
        if not input_dir.is_dir():
            raise ValueError(
                f"Select input directory not found: {input_dir}. "
                f"Place videos under {root / 'video'} or pass --select-input-dir."
            )
        if not getattr(args, "select_skip_vlm", False):
            if not resolve_llm_api_key():
                raise ValueError(
                    "Step3 VLM requires TOKENROUTER_API_KEY or OPENAI_API_KEY "
                    "(or pass --select-skip-vlm)."
                )

    def run(self, args: argparse.Namespace) -> None:
        self.validate_args(args)
        from pipeline.stages.select.run import run as select_run

        select_run(args)
