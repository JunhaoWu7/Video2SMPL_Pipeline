from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.stages.base import PipelineStage


class Video2SmplStage(PipelineStage):
    name = "video2smpl"
    description = "Video -> SMPL for all caption-complete samples (required after captions)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("video2smpl stage")
        group.add_argument(
            "--weight_root",
            type=str,
            default="/data1/wjh/Video2SMPL",
            help="CameraHMR / SMPL / YOLO weights root; empty string uses third_party/.../data/",
        )
        group.add_argument("--max_frames", type=int, default=500)
        group.add_argument("--batch_size", type=int, default=32)
        group.add_argument("--person_idx", type=int, default=0)
        group.add_argument(
            "--set-floor",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Canonical DART floor alignment (default on). Use --no-set-floor to disable.",
        )
    def validate_args(self, args: argparse.Namespace) -> None:
        if not str(getattr(args, "source", "") or "").strip():
            raise ValueError('--source is required when running the "video2smpl" stage.')

        from pipeline.manifest import (
            captions_filled,
            load_manifest_list,
            manifest_path,
            resolve_video_rel,
            rows_caption_complete,
            rows_pending_smpl,
        )

        root = Path(getattr(args, "root_dir", ".")).resolve()
        mpath = manifest_path(root, getattr(args, "manifest_name", None))
        if not mpath.exists():
            raise ValueError(
                f"Manifest not found: {mpath}. Run the select stage first (--from-stage select), "
                "or ensure dataset_manifest.json exists with video_path per sample."
            )
        rows = load_manifest_list(mpath)
        if not rows:
            raise ValueError(f"Manifest is empty: {mpath}")
        if not rows_caption_complete(rows):
            raise ValueError(
                f"No caption-complete samples in {mpath}. "
                "Run the captions stage before video2smpl (caption, action_caption, "
                "robot_learnable, skill_category)."
            )
        missing_video = [
            str(r.get("sample_id", "?"))
            for r in rows_caption_complete(rows)
            if not resolve_video_rel(r)
        ]
        if missing_video:
            raise ValueError(
                f"{len(missing_video)} caption-complete sample(s) lack video_path "
                f"(e.g. {missing_video[:3]}). Run select first."
            )
        pending = rows_pending_smpl(rows)
        if not pending and not getattr(args, "overwrite", False):
            print(
                "video2smpl: all caption-complete samples already have smpl_path; nothing to do.",
                flush=True,
            )

    def run(self, args: argparse.Namespace) -> None:
        self.validate_args(args)
        from pipeline.stages.video2smpl.run import run as video2smpl_run

        video2smpl_run(args)
