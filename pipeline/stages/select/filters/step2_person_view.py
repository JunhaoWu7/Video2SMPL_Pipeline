from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

from pipeline.stages.select.filters.common import (
    SelectFilterConfig,
    SelectFilterOutcome,
    SelectFilterResult,
    read_frames_at_indices,
    read_video_meta,
    resolve_yolo_model,
    uniform_frame_indices,
)

ViewHint = Literal["third_person", "first_person", "uncertain", "no_person"]

_YOLO_MODEL = None
_YOLO_MODEL_PATH: Optional[str] = None


@dataclass(frozen=True)
class _FramePersonStats:
    person_count: int
    main_area_ratio: float
    main_cy: float
    main_aspect: float


def _get_yolo(model_path: str):
    global _YOLO_MODEL, _YOLO_MODEL_PATH
    if _YOLO_MODEL is not None and _YOLO_MODEL_PATH == model_path:
        return _YOLO_MODEL
    from ultralytics import YOLO

    _YOLO_MODEL = YOLO(model_path)
    _YOLO_MODEL_PATH = model_path
    return _YOLO_MODEL


def _largest_person_box(result, frame_w: int, frame_h: int) -> Optional[Tuple[float, float, float, float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    frame_area = float(max(frame_w * frame_h, 1))
    best = None
    best_area = 0.0
    for box in boxes:
        cls_id = int(box.cls.item())
        if cls_id != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    if best is None:
        return None
    return best


def _stats_for_frame(result, frame_w: int, frame_h: int) -> _FramePersonStats:
    boxes = result.boxes
    person_count = 0
    if boxes is not None:
        person_count = sum(1 for box in boxes if int(box.cls.item()) == 0)

    main = _largest_person_box(result, frame_w, frame_h)
    if main is None:
        return _FramePersonStats(
            person_count=person_count,
            main_area_ratio=0.0,
            main_cy=0.0,
            main_aspect=0.0,
        )

    x1, y1, x2, y2 = main
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    frame_area = float(max(frame_w * frame_h, 1))
    area_ratio = (bw * bh) / frame_area
    cy = ((y1 + y2) * 0.5) / float(max(frame_h, 1))
    aspect = bh / bw
    return _FramePersonStats(
        person_count=person_count,
        main_area_ratio=area_ratio,
        main_cy=cy,
        main_aspect=aspect,
    )


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _view_hint(area_p50: float, cy_p50: float, aspect_p50: float) -> ViewHint:
    if area_p50 <= 0:
        return "no_person"
    if area_p50 > 0.75 and cy_p50 < 0.4:
        return "first_person"
    if 0.15 <= area_p50 <= 0.70 and cy_p50 > 0.45 and 1.2 <= aspect_p50 <= 4.0:
        return "third_person"
    return "uncertain"


def run_step2_person_view(video_path: Path, cfg: SelectFilterConfig) -> SelectFilterResult:
    path = str(video_path.resolve())
    try:
        width, height, n_frames, _fps = read_video_meta(path)
    except OSError:
        return SelectFilterResult(status="rejected")

    indices = uniform_frame_indices(n_frames, cfg.step2_sample_frames)
    frames = read_frames_at_indices(path, indices)
    if not frames:
        return SelectFilterResult(status="rejected")

    model = _get_yolo(resolve_yolo_model(cfg.yolo_model))
    per_frame: List[_FramePersonStats] = []
    for frame in frames:
        h, w = frame.shape[:2]
        results = model.predict(frame, conf=cfg.yolo_conf, verbose=False, classes=[0])
        per_frame.append(_stats_for_frame(results[0], w, h))

    visible_frames = sum(1 for s in per_frame if s.person_count >= 1)
    person_visible_ratio = visible_frames / len(per_frame)
    person_counts = [float(s.person_count) for s in per_frame]
    p95_person_count = int(round(_percentile(person_counts, 0.95)))
    multi_person_ratio = sum(1 for s in per_frame if s.person_count >= 2) / len(per_frame)

    areas = [s.main_area_ratio for s in per_frame if s.person_count >= 1]
    main_area_p50 = _percentile(areas, 0.5) if areas else 0.0
    cys = [s.main_cy for s in per_frame if s.person_count >= 1]
    main_cy_p50 = _percentile(cys, 0.5) if cys else 0.0
    aspects = [s.main_aspect for s in per_frame if s.person_count >= 1]
    main_aspect_p50 = _percentile(aspects, 0.5) if aspects else 0.0
    view = _view_hint(main_area_p50, main_cy_p50, main_aspect_p50)

    if person_visible_ratio < 0.5 or view == "no_person":
        return SelectFilterResult(status="rejected")
    if main_area_p50 < 0.08:
        return SelectFilterResult(status="rejected")
    if p95_person_count >= 2 and multi_person_ratio >= 0.3:
        return SelectFilterResult(status="rejected")

    if (
        person_visible_ratio >= 0.75
        and p95_person_count <= 1
        and 0.15 <= main_area_p50 <= 0.70
        and view == "third_person"
    ):
        return SelectFilterResult(status="passed")

    return SelectFilterResult(status="deferred")
