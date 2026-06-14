from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

from pipeline.llm_defaults import DEFAULT_LLM_BASE_URL, DEFAULT_SELECT_VLM_MODEL

SelectFilterOutcome = Literal["passed", "deferred", "rejected"]

# Shared with PromptHMR pretrain layout (step2 person detection).
DEFAULT_SELECT_YOLO_PATH = "/data1/wjh/ckpt/PromptHMR/pretrain/yolo11x.pt"


@dataclass(frozen=True)
class SelectFilterConfig:
    min_duration_s: float = 1.0
    max_duration_s: float = 120.0
    min_side_px: int = 240
    step1_sample_frames: int = 12
    step2_sample_frames: int = 16
    static_frame_ratio_reject: float = 0.95
    motion_max_reject: float = 0.01
    motion_mean_defer: float = 0.015
    yolo_model: str = DEFAULT_SELECT_YOLO_PATH
    yolo_conf: float = 0.25
    vlm_model: str = DEFAULT_SELECT_VLM_MODEL
    vlm_frames: int = 6
    vlm_max_side: int = 512
    vlm_vision_detail: str = "low"
    vlm_timeout: float = 120.0
    vlm_max_retries: int = 2
    vlm_base_url: str = DEFAULT_LLM_BASE_URL
    vlm_http_referer: str = ""
    vlm_x_title: str = "video2smpl-select-vlm"


@dataclass(frozen=True)
class SelectFilterResult:
    status: SelectFilterOutcome


def combine_filter_status(
    step1: SelectFilterOutcome,
    step2: Optional[SelectFilterOutcome],
) -> SelectFilterOutcome:
    if step1 == "rejected":
        return "rejected"
    if step2 is None:
        return step1
    if step2 == "rejected":
        return "rejected"
    if step1 == "deferred" or step2 == "deferred":
        return "deferred"
    return "passed"


def uniform_frame_indices(n_total: int, num_frames: int) -> List[int]:
    if n_total <= 0:
        return []
    k = max(1, min(num_frames, n_total))
    if k == 1:
        return [0]
    return [int(round(i * (n_total - 1) / (k - 1))) for i in range(k)]


def read_video_meta(video_path: str) -> tuple[int, int, int, float]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for select filters (pip install opencv-python)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"cannot open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 and n_frames > 0:
            fps = 30.0
        return width, height, n_frames, fps
    finally:
        cap.release()


def read_frames_at_indices(video_path: str, indices: Sequence[int]) -> List["cv2.Mat"]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for select filters")
    if not indices:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"cannot open video: {video_path}")
    frames = []
    try:
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def resolve_yolo_model(model_path: str | None = None) -> str:
    """
    Resolve YOLO weights for step2.

    Order: explicit file -> VIDEO2SMPL_SELECT_YOLO -> DEFAULT_SELECT_YOLO_PATH.
    """
    raw = (model_path or DEFAULT_SELECT_YOLO_PATH).strip()
    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())

    env = os.environ.get("VIDEO2SMPL_SELECT_YOLO", "").strip()
    if env:
        env_path = Path(env).expanduser()
        if env_path.is_file():
            return str(env_path.resolve())

    default = Path(DEFAULT_SELECT_YOLO_PATH)
    if default.is_file():
        return str(default.resolve())

    raise FileNotFoundError(
        f"YOLO weights not found: {raw}. "
        f"Expected {DEFAULT_SELECT_YOLO_PATH} or set VIDEO2SMPL_SELECT_YOLO."
    )


def grayscale_mean_abs_diff(frame_a, frame_b) -> float:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for select filters")
    g1 = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype("float32")
    g2 = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype("float32")
    return float(abs(g1 - g2).mean() / 255.0)
