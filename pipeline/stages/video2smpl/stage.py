from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.dataset_schema import (
    DEFAULT_HMR_BACKEND,
    HMR_BACKEND_CAMERAHMR,
    HMR_BACKEND_PROMPTHMR,
)
from pipeline.parallel_defaults import DEFAULT_GPU_WORKERS
from pipeline.stages.base import PipelineStage
from pipeline.stages.video2smpl.common import normalize_hmr_backend


class Video2SmplStage(PipelineStage):
    name = "video2smpl"
    description = "Video -> SMPL (default: PromptHMR world; optional: CameraHMR DART)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("video2smpl stage")
        group.add_argument(
            "--hmr-backend",
            type=str,
            default=DEFAULT_HMR_BACKEND,
            choices=[HMR_BACKEND_PROMPTHMR, HMR_BACKEND_CAMERAHMR],
            help=f"HMR backend (default: {HMR_BACKEND_PROMPTHMR}).",
        )
        group.add_argument(
            "--weight_root",
            type=str,
            default="/data1/wjh/Video2SMPL",
            help="CameraHMR / SMPL / YOLO weights (camerahmr backend only).",
        )
        group.add_argument(
            "--vendor_root",
            type=str,
            default="third_party",
            help="third_party root for CameraHMR extract_motion (camerahmr only).",
        )
        group.add_argument(
            "--prompthmr-vendor",
            type=str,
            default=None,
            help="PromptHMR vendor_bundle directory (prompthmr backend).",
        )
        group.add_argument(
            "--prompthmr-ckpt-root",
            type=str,
            default="/data1/wjh/ckpt/PromptHMR",
            help="PromptHMR checkpoint root for preflight validation.",
        )
        group.add_argument(
            "--static-camera",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="PromptHMR: assume fixed camera (default on).",
        )
        group.add_argument("--max_frames", type=int, default=500)
        group.add_argument("--batch_size", type=int, default=32)
        group.add_argument("--person_idx", type=int, default=0)
        group.add_argument("--smooth_window", type=int, default=5)
        group.add_argument(
            "--set-floor",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="CameraHMR DART floor alignment (camerahmr only).",
        )
        group.add_argument("--use_shape", action="store_true")
        group.add_argument(
            "--video2smpl-workers",
            type=int,
            default=0,
            help=f"Parallel workers (0=auto: {DEFAULT_GPU_WORKERS} GPUs; 1=serial).",
        )
        group.add_argument(
            "--video2smpl-gpus",
            type=str,
            default="auto",
            help=f'GPU list for workers (default: auto = first {DEFAULT_GPU_WORKERS} CUDA devices).',
        )

    def validate_args(self, args: argparse.Namespace) -> None:
        if not str(getattr(args, "source", "") or "").strip():
            raise ValueError('--source is required when running the "video2smpl" stage.')

        backend = normalize_hmr_backend(getattr(args, "hmr_backend", DEFAULT_HMR_BACKEND))

        from pipeline.manifest import (
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
                f"Manifest not found: {mpath}. Run the select stage first."
            )
        rows = load_manifest_list(mpath)
        if not rows:
            raise ValueError(f"Manifest is empty: {mpath}")
        if not rows_caption_complete(rows):
            raise ValueError(
                f"No caption-complete samples in {mpath}. Run captions before video2smpl."
            )
        missing_video = [
            str(r.get("sample_id", "?"))
            for r in rows_caption_complete(rows)
            if not resolve_video_rel(r)
        ]
        if missing_video:
            raise ValueError(
                f"{len(missing_video)} caption-complete sample(s) lack video_path."
            )

        if backend == HMR_BACKEND_PROMPTHMR:
            from pipeline.stages.video2smpl.prompthmr_weights import check_weights

            require_slam = not bool(getattr(args, "static_camera", True))
            ok, missing = check_weights(
                getattr(args, "prompthmr_vendor", None),
                getattr(args, "prompthmr_ckpt_root", None),
                require_slam=require_slam,
            )
            if not ok:
                raise FileNotFoundError(
                    "PromptHMR vendor/weights not ready:\n"
                    + "\n".join(missing)
                    + "\nRun: bash scripts/copy_prompthmr_vendor.sh"
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
