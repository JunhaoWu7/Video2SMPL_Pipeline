from __future__ import annotations

from pathlib import Path

from pipeline.stages.select.filters.common import (
    SelectFilterConfig,
    SelectFilterOutcome,
    SelectFilterResult,
    grayscale_mean_abs_diff,
    read_frames_at_indices,
    read_video_meta,
    uniform_frame_indices,
)


def run_step1_basic(video_path: Path, cfg: SelectFilterConfig) -> SelectFilterResult:
    path = str(video_path.resolve())
    try:
        width, height, n_frames, fps = read_video_meta(path)
    except OSError:
        return SelectFilterResult(status="rejected")

    if width <= 0 or height <= 0 or n_frames <= 0:
        return SelectFilterResult(status="rejected")

    min_side = min(width, height)
    if min_side < cfg.min_side_px:
        return SelectFilterResult(status="rejected")

    duration_s = n_frames / fps if fps > 0 else 0.0
    if duration_s < cfg.min_duration_s or duration_s > cfg.max_duration_s:
        return SelectFilterResult(status="rejected")

    indices = uniform_frame_indices(n_frames, cfg.step1_sample_frames)
    frames = read_frames_at_indices(path, indices)
    if len(frames) < 2:
        return SelectFilterResult(status="deferred")

    diffs = [
        grayscale_mean_abs_diff(frames[i], frames[i + 1]) for i in range(len(frames) - 1)
    ]
    motion_mean = sum(diffs) / len(diffs)
    motion_max = max(diffs)
    static_eps = cfg.motion_max_reject
    static_frame_ratio = sum(1 for d in diffs if d < static_eps) / len(diffs)

    if static_frame_ratio >= cfg.static_frame_ratio_reject and motion_max < cfg.motion_max_reject:
        return SelectFilterResult(status="rejected")

    if motion_mean < cfg.motion_mean_defer:
        return SelectFilterResult(status="deferred")

    return SelectFilterResult(status="passed")
