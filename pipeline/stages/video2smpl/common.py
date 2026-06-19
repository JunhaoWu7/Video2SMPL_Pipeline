"""Shared helpers for video2smpl backends."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_first_frame(video_path: Path, output_jpg: Path) -> None:
    import cv2

    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame from video: {video_path}")
    cv2.imwrite(str(output_jpg), frame)


def ensure_sample_video(video_path: Path, target_video: Path) -> None:
    """
    Ensure the trainable sample directory contains the source video.

    The normal select path already places each video under
    processed_trainable_data/<sample_id>/. Pre-built manifests may still point
    at video/*.mp4; video2smpl standardizes the manifest path after SMPL
    extraction, so the target video must exist before saving that path.
    """
    src = Path(video_path).resolve()
    dst = Path(target_video)
    if dst.exists():
        if dst.resolve() == src:
            return
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        rel_src = os.path.relpath(src, start=dst.parent.resolve())
        os.symlink(rel_src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def parse_sample_id_numeric(sample_id: str) -> Optional[int]:
    if sample_id.isdigit():
        return int(sample_id)
    return None


def max_sample_id_from_dirs(bases: List[Path], id_width: int) -> int:
    m = 0
    for base in bases:
        if not base.exists():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if name.isdigit() and len(name) == id_width:
                m = max(m, int(name))
    return m


def load_id_mapping(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("items") or [])


def resolve_manifest_link(args) -> str:
    if getattr(args, "link", None) is not None:
        return str(args.link)
    return str(getattr(args, "default_link", "") or "")


def normalize_hmr_backend(raw: str) -> str:
    from pipeline.dataset_schema import (
        DEFAULT_HMR_BACKEND,
        HMR_BACKEND_CAMERAHMR,
        HMR_BACKEND_PROMPTHMR,
        VALID_HMR_BACKENDS,
    )

    b = (raw or DEFAULT_HMR_BACKEND).strip().lower()
    if b in ("camera", "camerahmr", "dart", "canonical"):
        return HMR_BACKEND_CAMERAHMR
    if b in ("prompt", "prompthmr", "phmr"):
        return HMR_BACKEND_PROMPTHMR
    if b not in VALID_HMR_BACKENDS:
        raise ValueError(
            f"Unknown --hmr-backend {raw!r}; use {HMR_BACKEND_PROMPTHMR!r} or {HMR_BACKEND_CAMERAHMR!r}"
        )
    return b
