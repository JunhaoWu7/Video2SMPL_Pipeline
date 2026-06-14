"""Multiprocess worker for video2smpl (one sample per task, one GPU per process)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any


def process_video2smpl_task(task: dict[str, Any]) -> dict[str, Any]:
    """
    Run SMPL inference for one manifest row in an isolated process.

    ``CUDA_VISIBLE_DEVICES`` must be set before any torch import in this process.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(task["gpu_id"])

    from pipeline.dataset_schema import HMR_BACKEND_CAMERAHMR, HMR_BACKEND_PROMPTHMR, sample_paths
    from pipeline.manifest import build_video2smpl_row
    from pipeline.stages.video2smpl.backends.camerahmr import run_camerahmr_sample
    from pipeline.stages.video2smpl.backends.prompthmr import run_prompthmr_sample
    from pipeline.stages.video2smpl.common import extract_first_frame

    sample_id = str(task["sample_id"])
    backend = str(task["backend"])
    work_root = Path(task["work_root"])
    video_path = Path(task["video_path"])
    sample_train = work_root / "processed_trainable_data" / sample_id
    smpl_npz_path = Path(task["smpl_npz_path"])

    args = Namespace(**task["args"])
    try:
        if backend == HMR_BACKEND_CAMERAHMR:
            run_camerahmr_sample(
                video_path=video_path,
                output_npz=smpl_npz_path,
                args=args,
                vendor_root=Path(task["vendor_root"]) if task.get("vendor_root") else None,
            )
        else:
            run_prompthmr_sample(
                video_path=video_path,
                output_npz=smpl_npz_path,
                text_prompt=str(task.get("text_prompt") or ""),
                args=args,
                vendor_root=task.get("prompthmr_vendor"),
            )

        extract_first_frame(video_path, sample_train / "first_frame.jpg")
        paths = sample_paths(sample_id, video_path.name, hmr_backend=backend)
        row = build_video2smpl_row(
            sample_id=sample_id,
            original_video=str(task.get("original_video") or video_path.name),
            video_rel=paths["video_path"],
            first_frame_rel=paths["first_frame"],
            smpl_rel=paths["smpl_path"],
            smpl_backend=backend,
            source=str(task["manifest_source"]),
            link=str(task.get("link") or ""),
            old_row=task.get("old_row") or {},
        )
        return {
            "status": "ok",
            "sample_id": sample_id,
            "row": row,
            "mapping_item": task.get("mapping_item"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "sample_id": sample_id,
            "error": str(exc),
        }
