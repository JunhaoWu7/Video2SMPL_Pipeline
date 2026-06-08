#!/usr/bin/env python3
"""Unit checks for select step3 VLM prefilter logic and weight resolution."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.stages.select.filters.common import DEFAULT_SELECT_YOLO_PATH, resolve_yolo_model
from pipeline.stages.select.filters.pipeline import run_select_filters
from pipeline.stages.select.filters.step3_vlm import evaluate_vlm_prefilter


def test_evaluate_pass() -> None:
    out = evaluate_vlm_prefilter(
        {
            "third_person_view": True,
            "person_visible": True,
            "has_discernible_action": True,
            "reject_reason": "",
        }
    )
    assert out.status == "passed"


def test_evaluate_reject_bool() -> None:
    out = evaluate_vlm_prefilter(
        {
            "third_person_view": False,
            "person_visible": True,
            "has_discernible_action": True,
            "reject_reason": "",
        }
    )
    assert out.status == "rejected"


def test_evaluate_reject_reason() -> None:
    out = evaluate_vlm_prefilter(
        {
            "third_person_view": True,
            "person_visible": True,
            "has_discernible_action": True,
            "reject_reason": "idle",
        }
    )
    assert out.status == "rejected"


def test_yolo_resolve() -> None:
    p = Path(DEFAULT_SELECT_YOLO_PATH)
    resolved = resolve_yolo_model()
    assert resolved == str(p.resolve()), resolved


def test_pipeline_requires_client_for_step3() -> None:
    try:
        run_select_filters(Path("/tmp/nonexistent.mp4"), skip_step1_step2=True, skip_step3=False)
    except ValueError as e:
        assert "vlm_client" in str(e)
    else:
        raise AssertionError("expected ValueError when step3 enabled without client")


def main() -> int:
    test_evaluate_pass()
    test_evaluate_reject_bool()
    test_evaluate_reject_reason()
    test_yolo_resolve()
    test_pipeline_requires_client_for_step3()
    print("select step3 unit checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
