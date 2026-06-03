"""PromptHMR world-coordinate branch (default)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pipeline.dataset_schema import HMR_BACKEND_PROMPTHMR
from pipeline.stages.video2smpl.prompthmr_env import (
    apply_caption_text_patch,
    setup_prompthmr_runtime,
)

_PIPELINE: Any = None


def _get_pipeline(static_cam: bool, vendor_root: Optional[str], ckpt_root: Optional[str]) -> Any:
    global _PIPELINE
    setup_prompthmr_runtime(vendor_root=vendor_root, ckpt_root=ckpt_root)
    if _PIPELINE is None or getattr(_PIPELINE, "_static_cam", None) != static_cam:
        from phmr_pipeline.pipeline import Pipeline

        _PIPELINE = Pipeline(static_cam=static_cam)
        _PIPELINE._static_cam = static_cam  # type: ignore[attr-defined]
    return _PIPELINE


def _pick_primary_track(people: Dict[Any, Any]) -> Any:
    if not people:
        raise RuntimeError("PromptHMR: no tracks in results")
    return max(people.keys(), key=lambda k: len(people[k].get("frames", [])))


def _align_world_to_video_timeline(
    pose: np.ndarray,
    trans: np.ndarray,
    shape: np.ndarray,
    frame_ids: np.ndarray,
    num_video_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = int(num_video_frames)
    global_orient = np.zeros((T, 3), dtype=np.float32)
    body_pose = np.zeros((T, 63), dtype=np.float32)
    transl = np.zeros((T, 3), dtype=np.float32)
    frame_mask = np.zeros((T,), dtype=np.uint8)

    if pose.ndim != 2 or pose.shape[1] < 66:
        raise ValueError(f"Expected pose (N,>=66), got {pose.shape}")
    n = min(len(frame_ids), pose.shape[0])
    if shape.ndim == 1:
        betas_row = shape[:10].astype(np.float32)
    elif shape.ndim == 2:
        betas_row = shape[0, :10].astype(np.float32)
    else:
        betas_row = np.zeros(10, dtype=np.float32)
    betas = np.tile(betas_row[None, :], (T, 1))

    for i in range(n):
        fi = int(frame_ids[i])
        if fi < 0 or fi >= T:
            continue
        global_orient[fi] = pose[i, :3].astype(np.float32)
        body_pose[fi] = pose[i, 3:66].astype(np.float32)
        transl[fi] = trans[i, :3].astype(np.float32)
        frame_mask[fi] = 1

    return global_orient, body_pose, transl, betas, frame_mask


def run_prompthmr_sample(
    *,
    video_path: Path,
    output_npz: Path,
    text_prompt: str,
    args: Any,
    vendor_root: Optional[str] = None,
) -> None:
    prompt = (text_prompt or "").strip()
    if not prompt:
        raise ValueError("empty text_prompt")

    static_cam = bool(getattr(args, "static_camera", True))
    max_frame = int(getattr(args, "max_frames", 500))
    ckpt_root = getattr(args, "prompthmr_ckpt_root", None)

    apply_caption_text_patch(prompt)
    pipeline = _get_pipeline(static_cam, vendor_root, ckpt_root)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2smpl_phmr_") as tmp:
        results = pipeline.__call__(
            str(video_path),
            str(Path(tmp) / "phmr_out"),
            static_cam=static_cam,
            save_only_essential=True,
            max_frame=max_frame if max_frame > 0 else None,
        )

    people = results.get("people") or {}
    tid = _pick_primary_track(people)
    world = people[tid].get("smplx_world")
    if not world:
        raise RuntimeError(f"PromptHMR: missing smplx_world for track {tid}")

    pose = np.asarray(world["pose"], dtype=np.float32)
    trans = np.asarray(world["trans"], dtype=np.float32)
    shape = np.asarray(world["shape"], dtype=np.float32)
    frame_ids = np.asarray(people[tid]["frames"], dtype=np.int64)

    num_video_frames = min(len(pipeline.images), max_frame) if max_frame > 0 else len(pipeline.images)
    go, bp, tr, betas, frame_mask = _align_world_to_video_timeline(
        pose, trans, shape, frame_ids, num_video_frames
    )

    np.savez(
        output_npz,
        global_orient=go,
        body_pose=bp,
        trans=tr,
        shape=betas,
        frame_mask=frame_mask,
        track_id=np.array([int(tid)], dtype=np.int32),
        coord_note=np.bytes_(b"prompthmr_world_smplx_aa_v1"),
        smpl_backend=np.bytes_(HMR_BACKEND_PROMPTHMR),
        text_prompt_used=np.bytes_(prompt.encode("utf-8")[:512]),
    )
