"""Step3: lightweight VLM fine-grained view + action-existence check."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pipeline.stages.select.filters.common import SelectFilterConfig, SelectFilterResult

_HARD_REJECT_REASONS = frozenset(
    {
        "multi_person",
        "heavy_occlusion",
        "idle",
        "first_person",
        "camera_motion_only",
        "no_person",
        "no_action",
    }
)


def build_vlm_prefilter_prompt(num_frames: int) -> str:
    return f"""You review frames from a short human motion video for a robot-learning dataset.

You are given {num_frames} still frames sampled uniformly over time.

Output ONLY a JSON object with exactly these fields:
{{
  "third_person_view": true,
  "person_visible": true,
  "has_discernible_action": true,
  "reject_reason": ""
}}

Field rules:
- ``third_person_view`` (boolean): true if filmed in third-person with a visible human subject (not selfie / first-person POV).
- ``person_visible`` (boolean): true if a person is clearly visible in enough frames (not mostly off-screen or fully occluded).
- ``has_discernible_action`` (boolean): true if there is a discernible body motion skill (walk, reach, grasp, turn, etc.), not idle standing/sitting, pure conversation, or camera-only motion.
- ``reject_reason`` (string): empty when acceptable; otherwise one of: multi_person, heavy_occlusion, idle, first_person, camera_motion_only, no_person, no_action.

Reply with raw JSON only. No markdown fences, no extra keys."""


def normalize_vlm_prefilter_output(raw: dict[str, Any]) -> dict[str, Any]:
    def _bool(val: Any) -> Optional[bool]:
        if isinstance(val, bool):
            return val
        s = str(val or "").strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        return None

    return {
        "third_person_view": _bool(raw.get("third_person_view")),
        "person_visible": _bool(raw.get("person_visible")),
        "has_discernible_action": _bool(
            raw.get("has_discernible_action", raw.get("has_action"))
        ),
        "reject_reason": str(raw.get("reject_reason", "")).strip().lower(),
    }


def evaluate_vlm_prefilter(norm: dict[str, Any]) -> SelectFilterResult:
    reason = norm.get("reject_reason") or ""
    if reason in _HARD_REJECT_REASONS:
        return SelectFilterResult(status="rejected")

    required = (
        norm.get("third_person_view"),
        norm.get("person_visible"),
        norm.get("has_discernible_action"),
    )
    if any(v is False for v in required):
        return SelectFilterResult(status="rejected")
    if any(v is None for v in required):
        return SelectFilterResult(status="rejected")
    return SelectFilterResult(status="passed")


def _is_vlm_content_blocked(exc: BaseException) -> bool:
    """True when the provider refuses to score frames (e.g. Gemini PROHIBITED_CONTENT)."""
    msg = str(exc).lower()
    if "prohibited_content" in msg or "prompt_blocked" in msg:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        code = str(err.get("code", "")).lower()
        text = str(err.get("message", "")).lower()
        if code == "prompt_blocked" or "prohibited_content" in text:
            return True
    return False


def call_vlm_prefilter(
    client: Any,
    *,
    model: str,
    data_urls: list[str],
    vision_detail: str,
    num_frames: int,
) -> dict[str, Any]:
    from generate_sequence_captions import extract_json_object

    user_prompt = build_vlm_prefilter_prompt(num_frames)
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for url in data_urls:
        user_content.append(
            {"type": "image_url", "image_url": {"url": url, "detail": vision_detail}}
        )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict video-quality gate for human action clips. "
                        "Answer with JSON only."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=256,
            temperature=0.0,
        )
    except Exception as exc:
        if _is_vlm_content_blocked(exc):
            raise ValueError("VLM content blocked (PROHIBITED_CONTENT)") from exc
        raise
    text = resp.choices[0].message.content
    if not text:
        raise ValueError("Empty VLM response for select step3.")
    return normalize_vlm_prefilter_output(extract_json_object(text))


def run_step3_vlm(
    video_path: Path,
    cfg: SelectFilterConfig,
    *,
    client: Any,
) -> SelectFilterResult:
    from generate_sequence_captions import sample_frames_from_video

    data_urls, _used = sample_frames_from_video(
        video_path.resolve(),
        cfg.vlm_frames,
        cfg.vlm_max_side,
    )
    if not data_urls:
        return SelectFilterResult(status="rejected")

    vlm_called = False
    try:
        vlm_called = True
        norm = call_vlm_prefilter(
            client,
            model=cfg.vlm_model,
            data_urls=data_urls,
            vision_detail=cfg.vlm_vision_detail,
            num_frames=len(data_urls),
        )
    except ValueError as exc:
        if "content blocked" in str(exc).lower():
            return SelectFilterResult(status="rejected", vlm_called=vlm_called)
        raise
    out = evaluate_vlm_prefilter(norm)
    return SelectFilterResult(status=out.status, vlm_called=vlm_called)


def create_select_vlm_client(cfg: SelectFilterConfig) -> Any:
    """Build OpenAI-compatible client (same env/keys as captions stage)."""
    import argparse

    from generate_sequence_captions import create_openai_client

    ns = argparse.Namespace(
        timeout=cfg.vlm_timeout,
        max_retries=cfg.vlm_max_retries,
        base_url=cfg.vlm_base_url,
        http_referer=cfg.vlm_http_referer,
        x_title=cfg.vlm_x_title,
    )
    client, _timeout, _base = create_openai_client(ns)
    from pipeline.llm_defaults import resolve_llm_api_key

    if not resolve_llm_api_key():
        raise RuntimeError(
            "Step3 VLM requires TOKENROUTER_API_KEY or OPENAI_API_KEY in the environment."
        )
    return client
