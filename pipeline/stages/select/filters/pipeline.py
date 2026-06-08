from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pipeline.stages.select.filters.common import (
    SelectFilterConfig,
    SelectFilterOutcome,
    SelectFilterResult,
    combine_filter_status,
)
from pipeline.stages.select.filters.step1_basic import run_step1_basic
from pipeline.stages.select.filters.step2_person_view import run_step2_person_view
from pipeline.stages.select.filters.step3_vlm import run_step3_vlm


def run_select_filters(
    video_path: Path,
    *,
    cfg: Optional[SelectFilterConfig] = None,
    skip_step1_step2: bool = False,
    skip_step2: bool = False,
    skip_step3: bool = False,
    vlm_client: Any = None,
) -> SelectFilterResult:
    """
    Run select pre-ingest filters on a source file under ``video/``.

    Step1 + step2 may yield ``passed`` or ``deferred``; both proceed to step3.
    Step3 yields only ``passed`` or ``rejected``. Rejected samples are discarded.
    """
    config = cfg or SelectFilterConfig()
    step1_status: SelectFilterOutcome = "passed"
    step2_status: Optional[SelectFilterOutcome] = None

    if not skip_step1_step2:
        step1 = run_step1_basic(video_path, config)
        if step1.status == "rejected":
            return SelectFilterResult(status="rejected")
        step1_status = step1.status

        if not skip_step2:
            step2 = run_step2_person_view(video_path, config)
            if step2.status == "rejected":
                return SelectFilterResult(status="rejected")
            step2_status = step2.status

    if skip_step3:
        if skip_step1_step2:
            return SelectFilterResult(status="passed")
        return SelectFilterResult(status=combine_filter_status(step1_status, step2_status))

    if vlm_client is None:
        raise ValueError("step3 VLM requires vlm_client unless --select-skip-vlm is set.")

    step3 = run_step3_vlm(video_path, config, client=vlm_client)
    return step3
