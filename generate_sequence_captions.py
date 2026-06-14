#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Fill caption fields and robot-skill labels in the unified pipeline manifest (in-place by default).

Writes per clip: ``caption``, ``action_caption``, ``robot_learnable``, ``skill_category``.

Reads a JSON list (e.g. dataset_manifest.json), resolves video_path / rgb_path / first_frame
under --pipeline-root, calls a vision API, and writes both fields back to the same manifest.

Usage:
  export TOKENROUTER_API_KEY=...   # or OPENAI_API_KEY
  pip install openai pillow httpx

    python generate_sequence_captions.py \
    --manifest examples/training/dataset_manifest.json \
    --pipeline-root examples/training \
    --model google/gemini-3.1-flash-image-preview \
    --workers 8 \
    --resume

  # 并行（默认 8 路；注意供应商速率限制，报错多时改成 --workers 2）
  python generate_sequence_captions.py ... --workers 8

  python generate_sequence_captions.py --dry-run \\
    --manifest examples/training/train_stage4_empty_text.json \\
    --pipeline-root examples/training

  # Slow proxies: raise --timeout (default 600s)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def pick_frame_indices(n_total: int, num_frames: int) -> list[int]:
    if n_total <= 0:
        return []
    k = min(num_frames, n_total)
    if k == 1:
        return [0]
    return [int(round(i * (n_total - 1) / (k - 1))) for i in range(k)]


def image_to_data_url(path: Path, max_side: int) -> str:
    if Image is None:
        raw = path.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        mime = "image/jpeg"
        if path.suffix.lower() == ".png":
            mime = "image/png"
        return f"data:{mime};base64,{b64}"

    im = Image.open(path).convert("RGB")
    w, h = im.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _loads_json_object(cand: str) -> dict[str, Any]:
    """Parse JSON object; tolerate raw newlines/tabs inside strings (common VLM output)."""
    for strict in (True, False):
        try:
            obj = json.loads(cand, strict=strict)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("not a JSON object", cand, 0)


def sanitize_caption_text(text: str) -> str:
    """Collapse raw newlines/tabs in VLM strings into single spaces."""
    return " ".join(str(text or "").split())


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return _loads_json_object(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = text[start : end + 1]
        try:
            return _loads_json_object(cand)
        except json.JSONDecodeError:
            pass
    raise ValueError("Model output is not a valid JSON object.")


def normalize_caption_output(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Expect JSON with caption, action_caption, robot_learnable, skill_category.
    Tolerates legacy keys for text fields only.
    """
    try:
        from pipeline.manifest import (
            VALID_SKILL_CATEGORIES,
            normalize_skill_category,
            parse_robot_learnable,
        )
    except ImportError:
        VALID_SKILL_CATEGORIES = frozenset(
            {"manipulation", "locomotion", "loco-manipulation"}
        )

        def normalize_skill_category(val: Any) -> str:
            s = str(val or "").strip().lower().replace("_", "-").replace(" ", "-")
            if s in ("locomanipulation", "loco-manip"):
                return "loco-manipulation"
            return s if s in VALID_SKILL_CATEGORIES else ""

        def parse_robot_learnable(val: Any) -> bool | None:
            if isinstance(val, bool):
                return val
            s = str(val or "").strip().lower()
            if s in ("true", "yes", "1"):
                return True
            if s in ("false", "no", "0"):
                return False
            return None

    caption = sanitize_caption_text(raw.get("caption", ""))
    if not caption:
        caption = sanitize_caption_text(raw.get("description", ""))
    if not caption:
        caption = sanitize_caption_text(raw.get("task_caption", ""))

    action = sanitize_caption_text(raw.get("action_caption", ""))
    if not action:
        action = sanitize_caption_text(raw.get("action_label", ""))
    if not action:
        action = sanitize_caption_text(raw.get("action", ""))

    skill = normalize_skill_category(
        raw.get("skill_category", raw.get("motion_category", raw.get("category", "")))
    )
    learnable = parse_robot_learnable(
        raw.get("robot_learnable", raw.get("is_robot_learnable", raw.get("learnable")))
    )

    return {
        "caption": caption,
        "action_caption": action,
        "skill_category": skill,
        "robot_learnable": learnable,
    }


def validate_caption_output(norm: dict[str, Any]) -> None:
    validate_caption_output_partial(
        norm,
        ("caption", "action_caption", "robot_learnable", "skill_category"),
    )


def validate_caption_output_partial(
    norm: dict[str, Any],
    required_fields: Sequence[str],
) -> None:
    if "caption" in required_fields and not norm.get("caption"):
        raise ValueError("Model JSON missing non-empty caption.")
    if "action_caption" in required_fields and not norm.get("action_caption"):
        raise ValueError("Model JSON missing non-empty action_caption.")
    if "skill_category" in required_fields:
        if norm.get("skill_category") not in (
            "manipulation",
            "locomotion",
            "loco-manipulation",
        ):
            raise ValueError(
                f"Invalid skill_category: {norm.get('skill_category')!r}; "
                "expected manipulation | locomotion | loco-manipulation."
            )
    if "robot_learnable" in required_fields and norm.get("robot_learnable") is None:
        raise ValueError("Model JSON missing boolean robot_learnable.")


def captions_from_output(output: dict[str, Any]) -> tuple[str, str, bool, str]:
    norm = normalize_caption_output(output)
    validate_caption_output(norm)
    return (
        norm["caption"],
        norm["action_caption"],
        bool(norm["robot_learnable"]),
        str(norm["skill_category"]),
    )


def caption_missing_fields_for_row(row: dict[str, Any]) -> list[str]:
    try:
        from pipeline.manifest import caption_missing_fields

        return list(caption_missing_fields(row))
    except ImportError:
        missing: list[str] = []
        if not str(row.get("caption", "")).strip():
            missing.append("caption")
        if not str(row.get("action_caption", "")).strip():
            missing.append("action_caption")
        if str(row.get("skill_category", "")).strip() not in (
            "manipulation",
            "locomotion",
            "loco-manipulation",
        ):
            missing.append("skill_category")
        if row.get("robot_learnable") is None:
            missing.append("robot_learnable")
        return missing


def existing_caption_hints(row: dict[str, Any]) -> dict[str, str]:
    """Non-empty caption-stage values already stored on the manifest row."""
    try:
        from pipeline.manifest import (
            VALID_SKILL_CATEGORIES,
            get_action_caption,
            get_caption,
            normalize_skill_category,
            parse_robot_learnable,
        )
    except ImportError:
        hints: dict[str, str] = {}
        cap = str(row.get("caption", "")).strip()
        if cap:
            hints["caption"] = cap
        act = str(row.get("action_caption", "")).strip()
        if act:
            hints["action_caption"] = act
        sc = str(row.get("skill_category", "")).strip()
        if sc in ("manipulation", "locomotion", "loco-manipulation"):
            hints["skill_category"] = sc
        if isinstance(row.get("robot_learnable"), bool):
            hints["robot_learnable"] = "true" if row["robot_learnable"] else "false"
        return hints

    hints: dict[str, str] = {}
    cap = get_caption(row)
    if cap:
        hints["caption"] = cap
    act = get_action_caption(row)
    if act:
        hints["action_caption"] = act
    sc = normalize_skill_category(row.get("skill_category"))
    if sc in VALID_SKILL_CATEGORIES:
        hints["skill_category"] = sc
    learnable = parse_robot_learnable(row.get("robot_learnable"))
    if learnable is not None:
        hints["robot_learnable"] = "true" if learnable else "false"
    return hints


_CAPTION_JSON_FIELD_EXAMPLES: dict[str, str] = {
    "caption": '"A person is picking up a water cup."',
    "action_caption": '"picking up a water cup"',
    "robot_learnable": "true",
    "skill_category": '"manipulation"',
}


def load_manifest_list(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a JSON list, got {type(raw).__name__}")
    return [x for x in raw if isinstance(x, dict)]


def resolve_under_root(root: Path, rel: str) -> Path:
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    return (root / rel).resolve()


def ndarray_rgb_to_data_url(rgb: Any, max_side: int) -> str:
    if Image is None:
        raise RuntimeError("Pillow is required to encode video frames; pip install pillow")
    im = Image.fromarray(rgb).convert("RGB")
    w, h = im.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def sample_frames_from_video(video_path: Path, num_frames: int, max_side: int) -> tuple[list[str], list[int]]:
    if cv2 is None:
        return [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return [], []
    idxs = pick_frame_indices(n, num_frames)
    data_urls: list[str] = []
    used: list[int] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, bgr = cap.read()
        if not ret or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        data_urls.append(ndarray_rgb_to_data_url(rgb, max_side))
        used.append(int(fi))
    cap.release()
    return data_urls, used


def build_data_urls_for_manifest_row(
    root: Path, row: dict[str, Any], num_frames: int, max_side: int
) -> tuple[list[str], str, list[int]]:
    try:
        from pipeline.manifest import resolve_video_rel

        video_rel = resolve_video_rel(row)
    except ImportError:
        video_rel = str(row.get("video_path", "")).strip()

    ff_rel = str(row.get("first_frame", "")).strip()
    video_abs = resolve_under_root(root, video_rel) if video_rel else None
    ff_abs = resolve_under_root(root, ff_rel) if ff_rel else None

    if video_abs and video_abs.is_file():
        urls, used = sample_frames_from_video(video_abs, num_frames, max_side)
        if urls:
            return urls, "video_path", used

    if ff_abs and ff_abs.is_file():
        return [image_to_data_url(ff_abs, max_side)], "first_frame", [0]

    return [], "none", []


def build_user_prompt_manifest(
    row: dict[str, Any],
    num_sampled_frames: int,
    caption_lang: str,
    missing_fields: Sequence[str] | None = None,
) -> str:
    sid = str(row.get("sample_id", "")).strip()
    orig = str(row.get("original_video", "")).strip()
    src = str(row.get("source", "")).strip()
    typ = str(row.get("type", "video")).strip()

    if missing_fields is None:
        missing_fields = caption_missing_fields_for_row(row)
    missing = list(missing_fields)

    lang_line = {
        "en": "Write ``caption`` in natural English.",
        "zh": "``caption`` 请用自然、流畅的简体中文。",
        "bilingual": "``caption`` in natural English.",
    }.get(caption_lang, "Write ``caption`` in natural English.")

    action_lang = {
        "en": "``action_caption``: one short English phrase (roughly 3–8 words), verb-led, naming the same single action as ``caption``.",
        "zh": "``action_caption``：与 ``caption`` 同一动作的简短中文短语（约 3–12 字），动词开头。",
        "bilingual": "``action_caption``: short English phrase for the same single action as ``caption``.",
    }.get(caption_lang, "``action_caption``: short English phrase, verb-led.")

    field_rules = """Field rules:
- ``caption``: **one** natural-language sentence describing the **single most salient** human action in this clip (who + what they are doing). Examples: "A person is picking up a water cup." / "一个人在拿起一个水杯。" Describe **one** dominant action only — do **not** list multiple actions, do **not** narrate a sequence of different skills, and do **not** chain clauses with "then/and/随后/接着". If several motions appear, pick the most prominent one and ignore the rest.
- ``action_caption``: a concise label for that **same single** action; no full sentences; no bullet lists; no multi-action phrases.
- ``robot_learnable`` (**required JSON boolean**): you **must** output the literal `true` or `false` (not a string, not omitted). true if a humanoid / mobile manipulator could **plausibly learn and reproduce** the main action from this clip (locomotion, manipulation, or both). false if the clip is not suitable, e.g. idle standing/sitting with no skill, pure conversation, camera-only motion, heavy occlusion, multi-person chaos, dance/acrobat stunts beyond typical robot skills, or action too ambiguous to label.
- ``skill_category`` (string): assign **exactly one** category that best matches the **primary** skill in the clip:
  - ``manipulation``: mostly upper-body / hand–object interaction; base largely stationary.
  - ``locomotion``: mostly whole-body displacement (walk, run, step, turn, climb stairs) with little object interaction.
  - ``loco-manipulation``: clear **combination** (walk while carrying, approach object while stepping, locomote then manipulate in one clip).
  Pick the single best fit; do not output multiple categories."""

    json_lines = ",\n  ".join(
        f'"{name}": {_CAPTION_JSON_FIELD_EXAMPLES[name]}' for name in missing
    )
    json_block = "{\n  " + json_lines + "\n}"

    known = existing_caption_hints(row)
    known_block = ""
    if known and len(missing) < 4:
        known_lines = "\n".join(
            f"- {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in known.items()
        )
        known_block = f"""
Already provided labels (use as trusted context; **do not** repeat or change them in your JSON output):
{known_lines}

Fill in **only** the missing fields listed below. Your JSON must contain **only** those keys.
"""

    output_intro = (
        "Output ONLY a JSON object with exactly these four fields:"
        if len(missing) == 4
        else f"Output ONLY a JSON object with exactly these {len(missing)} field(s):"
    )

    return f"""You analyze frames from a short human **action clip** (single person, third-person view).
Identify the **one most salient** action in the clip for robot imitation learning review.

Sample id: `{sid}`
Original filename hint: `{orig}`
Dataset / batch label (provenance): `{src}`
Media type: `{typ}`

You are given {num_sampled_frames} still frames sampled uniformly over time from the clip.
{known_block}
{output_intro}
{json_block}

{field_rules}
- {lang_line}
- {action_lang}

Reply with raw JSON only. No markdown fences, no extra keys, no commentary.
Every listed key MUST be present with the correct JSON type (strings for text fields, boolean for ``robot_learnable``)."""


def build_caption_system_message(required_fields: Sequence[str]) -> str:
    keys = ", ".join(required_fields)
    return (
        "You label short human action clips for robot learning datasets. "
        "For each clip, describe exactly one most salient action in natural language; "
        "avoid multi-action or sequential descriptions. "
        f"Reply with a single JSON object containing exactly these keys: {keys}. "
        "robot_learnable must be a JSON boolean true or false, never omitted or quoted as a string."
    )


def build_caption_retry_suffix(
    attempt: int,
    missing_fields: Sequence[str],
    last_err: Exception,
) -> str:
    keys = ", ".join(missing_fields)
    return (
        f"\n\nRETRY {attempt}: Your previous reply was rejected ({last_err}). "
        f"Return ONLY one valid JSON object with exactly these keys: {keys}. "
        "robot_learnable must be boolean true/false. "
        "No markdown, no prose outside JSON."
    )


def format_seconds(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def call_with_heartbeat(fn: Any, label: str, interval_sec: float) -> Any:
    stop_event = threading.Event()

    def _ticker() -> None:
        start = time.time()
        while not stop_event.wait(interval_sec):
            elapsed = time.time() - start
            msg = f"  [waiting] {label} ... elapsed={format_seconds(elapsed)}"
            print(msg, flush=True)
            try:
                from pipeline.stage_timing import stage_progress_update

                stage_progress_update(item=label, note=f"waiting elapsed={format_seconds(elapsed)}")
            except ImportError:
                pass

    t = threading.Thread(target=_ticker, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        stop_event.set()
        t.join(timeout=0.2)


def _should_retry_without_json_mode(exc: Exception) -> bool:
    """True when provider likely rejected response_format=json_object (common for vision VLMs)."""
    err = str(exc).lower()
    markers = (
        "response_format",
        "json_object",
        "unsupported",
        "not supported",
        "upstream rejected",
        "upstream_error",
    )
    if any(m in err for m in markers):
        return True
    if "error code: 400" in err or "status code: 400" in err:
        return True
    status = getattr(exc, "status_code", None)
    return status == 400


def call_openai_caption_with_prompt(
    client: Any,
    model: str,
    user_prompt: str,
    data_urls: list[str],
    vision_detail: str,
    *,
    required_fields: Sequence[str] | None = None,
    temperature: float = 0.0,
    use_json_mode: bool = False,
) -> dict[str, Any]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_prompt},
    ]
    for url in data_urls:
        user_content.append(
            {"type": "image_url", "image_url": {"url": url, "detail": vision_detail}}
        )

    if required_fields is None:
        required_fields = ("caption", "action_caption", "robot_learnable", "skill_category")

    create_kw: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": build_caption_system_message(required_fields),
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 512,
        "temperature": float(temperature),
    }
    if use_json_mode:
        create_kw["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**create_kw)
    except Exception as exc:
        if use_json_mode and _should_retry_without_json_mode(exc):
            create_kw.pop("response_format", None)
            resp = client.chat.completions.create(**create_kw)
        else:
            raise
    text = resp.choices[0].message.content
    if not text:
        raise ValueError("Empty model response.")
    raw = extract_json_object(text)
    norm = normalize_caption_output(raw)
    if len(required_fields) == 4:
        validate_caption_output(norm)
    else:
        validate_caption_output_partial(norm, required_fields)
    return norm


def caption_one_sample(
    client: Any,
    args: argparse.Namespace,
    pipeline_root: Path,
    idx: int,
    row: dict[str, Any],
    *,
    missing_fields: Sequence[str],
) -> tuple[int, str, str, str, str, bool | None, str, str | None, str, tuple[str, ...]]:
    """
    Run vision API for one manifest row. idx is 1-based list position.
    Returns (idx, sample_id, status, caption, action_caption, robot_learnable,
             skill_category, error_message, frame_source, filled_fields).
    status: ok | error | no_frames
    """
    sid = str(row.get("sample_id", f"row{idx}"))
    data_urls, src, _ = build_data_urls_for_manifest_row(
        pipeline_root, row, args.num_frames, args.max_side
    )
    if not data_urls:
        return (idx, sid, "no_frames", "", "", None, "", None, src, tuple(missing_fields))

    user_prompt = build_user_prompt_manifest(
        row,
        len(data_urls),
        args.caption_lang,
        missing_fields=missing_fields,
    )
    parse_retries = max(0, int(getattr(args, "caption_parse_retries", 2)))
    temperature = float(getattr(args, "caption_temperature", 0.0))
    use_json_mode = bool(getattr(args, "json_mode", False)) and not bool(
        getattr(args, "no_json_mode", False)
    )
    try:
        output: dict[str, Any] | None = None
        last_err: Exception | None = None
        for attempt in range(1 + parse_retries):
            prompt = user_prompt
            if attempt > 0 and last_err is not None:
                prompt += build_caption_retry_suffix(attempt, missing_fields, last_err)
            try:
                output = call_with_heartbeat(
                    lambda p=prompt: call_openai_caption_with_prompt(
                        client,
                        args.model,
                        p,
                        data_urls,
                        args.vision_detail,
                        required_fields=missing_fields,
                        temperature=temperature,
                        use_json_mode=use_json_mode,
                    ),
                    label=sid if attempt == 0 else f"{sid} (retry {attempt})",
                    interval_sec=max(3.0, float(args.heartbeat_sec)),
                )
                break
            except Exception as exc:
                last_err = exc
                if attempt >= parse_retries:
                    raise
        if output is None:
            raise RuntimeError("caption API returned no output")
        norm = normalize_caption_output(output)
        hints = existing_caption_hints(row)
        caption = norm["caption"] if "caption" in missing_fields else hints.get("caption", "")
        action_caption = (
            norm["action_caption"]
            if "action_caption" in missing_fields
            else hints.get("action_caption", "")
        )
        if "robot_learnable" in missing_fields:
            robot_learnable = bool(norm["robot_learnable"])
        elif "robot_learnable" in hints:
            robot_learnable = hints["robot_learnable"] == "true"
        else:
            robot_learnable = None
        skill_category = (
            str(norm["skill_category"])
            if "skill_category" in missing_fields
            else hints.get("skill_category", "")
        )
        return (
            idx,
            sid,
            "ok",
            caption,
            action_caption,
            robot_learnable,
            skill_category,
            None,
            src,
            tuple(missing_fields),
        )
    except Exception as e:
        return (idx, sid, "error", "", "", None, "", str(e), src, tuple(missing_fields))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fill caption, action_caption, robot_learnable, skill_category in manifest (in-place default)."
        )
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Unified manifest JSON (e.g. dataset_manifest.json).",
    )
    p.add_argument(
        "--pipeline-root",
        type=Path,
        required=True,
        help="Training root (same as run.py --root_dir); video_path / rgb_path / first_frame resolve here.",
    )
    p.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Default: same as --manifest (in-place update).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (default: TokenRouter google/gemini-3.1-flash-image-preview).",
    )
    p.add_argument(
        "--vision-detail",
        type=str,
        default="high",
        choices=("low", "high", "auto", "original"),
        help="Per-image vision fidelity.",
    )
    p.add_argument("--num-frames", type=int, default=16, help="Frames sampled per video.")
    p.add_argument("--max-side", type=int, default=768, help="Resize so max(w,h) <= this before base64.")
    p.add_argument(
        "--caption-lang",
        choices=("en", "zh", "bilingual"),
        default="en",
    )
    p.add_argument("--sleep", type=float, default=0.5, help="Throttle pause after each finished sample (split across workers).")
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel API calls (threads, default: 8). Use 1 for strictly sequential behavior.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated no-op (kept for old scripts). Skipping already-captioned rows is always on unless --force-recaption.",
    )
    p.add_argument(
        "--force-recaption",
        action="store_true",
        help="Re-call the API for all four fields and overwrite any pre-filled values.",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not call API; print plan only.")
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request timeout (seconds).",
    )
    p.add_argument("--max-retries", type=int, default=2, help="Retries on transient HTTP/API failures.")
    p.add_argument(
        "--caption-parse-retries",
        type=int,
        default=2,
        help="Extra attempts when model JSON fails validation (total tries = 1 + this value).",
    )
    p.add_argument(
        "--caption-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for caption API (0 = most deterministic).",
    )
    p.add_argument(
        "--json-mode",
        action="store_true",
        help="Request response_format=json_object (off by default; many vision VLMs reject it).",
    )
    p.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Force-disable JSON mode (default is already off for vision APIs).",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (default: TokenRouter). Env OPENAI_BASE_URL / TOKENROUTER_BASE_URL override.",
    )
    p.add_argument(
        "--http-referer",
        type=str,
        default="",
        help="Optional HTTP-Referer for OpenRouter.",
    )
    p.add_argument(
        "--x-title",
        type=str,
        default="video2smpl-manifest-captions",
        help="Optional X-Title for OpenRouter.",
    )
    p.add_argument(
        "--heartbeat-sec",
        type=float,
        default=15.0,
        help="Seconds between waiting logs during each API request.",
    )
    p.add_argument(
        "--no-drop-invalid-skill-category",
        action="store_true",
        help=(
            "Keep samples when skill_category validation fails after retries. "
            "Default: delete manifest row + processed_trainable_data/<id>/ and renumber."
        ),
    )
    return p.parse_args()


def create_openai_client(args: argparse.Namespace) -> tuple[Any, float, str]:
    from pipeline.llm_defaults import resolve_llm_api_key, resolve_llm_base_url

    if OpenAI is None:
        raise RuntimeError("Install openai: pip install openai")
    api_key = resolve_llm_api_key()
    if not api_key:
        raise RuntimeError(
            "Set TOKENROUTER_API_KEY (recommended) or OPENAI_API_KEY in the environment."
        )

    t = float(args.timeout)
    if httpx is not None:
        timeout_cfg = httpx.Timeout(
            connect=min(120.0, t),
            read=t,
            write=min(600.0, max(t, 120.0)),
            pool=min(120.0, t),
        )
    else:
        timeout_cfg = t

    client_kw: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout_cfg,
        "max_retries": int(args.max_retries),
    }
    base_url = resolve_llm_base_url(str(getattr(args, "base_url", None) or ""))
    client_kw["base_url"] = base_url
    default_headers: dict[str, str] = {}
    http_referer = (os.environ.get("OPENROUTER_HTTP_REFERER") or args.http_referer).strip()
    x_title = (os.environ.get("OPENROUTER_X_TITLE") or args.x_title).strip()
    if http_referer:
        default_headers["HTTP-Referer"] = http_referer
    if x_title:
        default_headers["X-Title"] = x_title
    if default_headers:
        client_kw["default_headers"] = default_headers
    return OpenAI(**client_kw), t, base_url


def is_invalid_skill_category_error(err: str | None) -> bool:
    """True when caption JSON validation failed on skill_category after retries."""
    return bool(err and "Invalid skill_category" in err)


def _save_mapping_file(
    path: Path,
    work_root: Path,
    id_width: int,
    items: list[dict[str, str]],
) -> None:
    items = sorted(
        items,
        key=lambda it: int(it["sample_id"]) if str(it.get("sample_id", "")).isdigit() else 0,
    )
    for i, item in enumerate(items, start=1):
        item["seq_index"] = str(i)
    path.write_text(
        json.dumps(
            {
                "root_dir": str(work_root),
                "id_width": id_width,
                "count": len(items),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def drop_invalid_skill_category_samples(
    pipeline_root: Path,
    rows: list[dict[str, Any]],
    drop_ids: set[str],
    *,
    manifest_path: Path,
    mapping_name: str = "sample_id_to_source.json",
    id_width: int = 6,
) -> int:
    """
    Remove manifest rows and ``processed_trainable_data/<id>/`` for failed captions.

    Renumbers remaining samples to contiguous ``000001``..``N``. Mutates *rows* in place.
    """
    if not drop_ids:
        return 0

    from pipeline.dataset_schema import sample_dir_rel
    from pipeline.manifest import save_manifest
    from pipeline.sample_renumber import renumber_sample_ids

    work_root = pipeline_root.resolve()
    mapping_path = work_root / mapping_name

    for sid in sorted(drop_ids):
        sample_dir = work_root / sample_dir_rel(sid)
        if sample_dir.is_dir():
            shutil.rmtree(sample_dir)
            print(f"  dropped sample_id={sid} (removed {sample_dir})", flush=True)
        else:
            print(f"  dropped sample_id={sid} (no on-disk dir)", flush=True)

    kept = [r for r in rows if str(r.get("sample_id", "")).strip() not in drop_ids]

    mapping_items: list[dict[str, str]] = []
    if mapping_path.is_file():
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        id_width = int(data.get("id_width") or id_width)
        mapping_items = [
            dict(it)
            for it in (data.get("items") or [])
            if str(it.get("sample_id", "")).strip() not in drop_ids
        ]

    new_rows, new_mapping, id_remap = renumber_sample_ids(
        work_root,
        kept,
        mapping_items,
        id_width=id_width,
    )
    changed = sum(1 for old, new in id_remap.items() if old != new)
    if changed:
        last_id = f"{len(new_rows):0{id_width}d}" if new_rows else "0"
        print(
            f"  Renumbered after drop: {changed} dir(s) -> contiguous 000001..{last_id}",
            flush=True,
        )

    rows.clear()
    rows.extend(new_rows)
    save_manifest(manifest_path, new_rows)
    if mapping_path.is_file() or new_mapping:
        _save_mapping_file(mapping_path, work_root, id_width, new_mapping)

    return len(drop_ids)


def row_captions_complete(row: dict[str, Any]) -> bool:
    try:
        from pipeline.manifest import captions_filled

        return captions_filled(row)
    except ImportError:
        return bool(str(row.get("caption", "")).strip()) and bool(
            str(row.get("action_caption", "")).strip()
        )


def hydrate_rows_from_disk(rows: list[dict[str, Any]], disk_path: Path) -> None:
    """Merge caption fields from on-disk manifest (same or separate output path)."""
    if not disk_path.is_file():
        return
    try:
        prev_list = load_manifest_list(disk_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return
    prev_by_id = {str(r.get("sample_id", "")): r for r in prev_list}
    for r in rows:
        sid = str(r.get("sample_id", ""))
        old = prev_by_id.get(sid)
        if not old:
            continue
        for key in ("caption", "action_caption", "skill_category"):
            val = str(old.get(key, "")).strip()
            if val and not str(r.get(key, "")).strip():
                r[key] = old[key]
        if r.get("robot_learnable") is None and isinstance(old.get("robot_learnable"), bool):
            r["robot_learnable"] = old["robot_learnable"]


def main() -> int:
    from pipeline.llm_defaults import DEFAULT_CAPTIONS_MODEL

    args = parse_args()
    if not getattr(args, "model", None):
        args.model = DEFAULT_CAPTIONS_MODEL

    try:
        manifest_path = args.manifest.resolve(strict=False)
    except OSError as e:
        print(f"Invalid --manifest path: {e}", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"--manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    pipeline_root = args.pipeline_root.resolve()
    out_path = args.output_manifest
    if out_path is None:
        out_path = manifest_path
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    in_place = out_path.resolve() == manifest_path.resolve()

    try:
        rows = load_manifest_list(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Failed to read manifest: {e}", file=sys.stderr)
        return 1
    hydrate_rows_from_disk(rows, out_path)
    if not in_place:
        hydrate_rows_from_disk(rows, manifest_path)

    if args.dry_run:
        print(f"Manifest samples: {len(rows)} (pipeline root: {pipeline_root})")
        for row in rows[:8]:
            sid = row.get("sample_id", "?")
            urls, src, used = build_data_urls_for_manifest_row(
                pipeline_root, row, args.num_frames, args.max_side
            )
            missing = caption_missing_fields_for_row(row)
            print(
                f"  [dry-run] sample_id={sid} frame_source={src} n_images={len(urls)} "
                f"video_frame_idx={used} captions_done={row_captions_complete(row)} "
                f"missing={missing or 'none'}"
            )
        if len(rows) > 8:
            print(f"  ... and {len(rows) - 8} more")
        return 0

    try:
        client, t, base_url = create_openai_client(args)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(
        f"OpenAI client: timeout read={t}s, max_retries={args.max_retries}"
        + (f", base_url={base_url}" if base_url else ""),
        flush=True,
    )

    workers = max(1, int(args.workers))
    print(f"Parallel workers: {workers}", flush=True)

    total = len(rows)
    run_start = time.time()
    ok_count = 0
    err_count = 0

    all_fields = ("caption", "action_caption", "robot_learnable", "skill_category")
    pending: list[tuple[int, dict[str, Any], tuple[str, ...]]] = []
    for idx, row in enumerate(rows, start=1):
        sid = str(row.get("sample_id", f"row{idx}"))
        if row_captions_complete(row) and not args.force_recaption:
            print(f"[{idx}/{total}] skip (caption stage complete): {sid}", flush=True)
            continue
        if args.force_recaption:
            missing_fields: tuple[str, ...] = all_fields
        else:
            missing_fields = tuple(caption_missing_fields_for_row(row))
        if not missing_fields:
            print(f"[{idx}/{total}] skip (nothing missing): {sid}", flush=True)
            continue
        if missing_fields != all_fields:
            print(
                f"[{idx}/{total}] partial fill {sid}: missing={list(missing_fields)}",
                flush=True,
            )
        pending.append((idx, row, missing_fields))

    if not pending:
        print("Nothing to caption (all skipped or empty pending list).", flush=True)
        return 0

    try:
        from pipeline.stage_timing import stage_progress_set_total

        stage_progress_set_total(len(pending))
    except ImportError:
        pass

    file_lock = threading.Lock()
    done_count = 0
    drop_ids: set[str] = set()
    drop_on_bad_skill = not bool(getattr(args, "no_drop_invalid_skill_category", False))
    sleep_piece = float(args.sleep) / float(workers) if args.sleep > 0 else 0.0

    def _submit(
        item: tuple[int, dict[str, Any], tuple[str, ...]],
    ) -> tuple[int, str, str, str, str, bool | None, str, str | None, str, tuple[str, ...]]:
        idx, row, missing_fields = item
        return caption_one_sample(
            client,
            args,
            pipeline_root,
            idx,
            row,
            missing_fields=missing_fields,
        )

    def _apply_captions_to_row(
        row: dict[str, Any],
        *,
        caption: str,
        action_caption: str,
        robot_learnable: bool | None,
        skill_category: str,
        filled_fields: Sequence[str],
        force_full: bool,
    ) -> None:
        try:
            if force_full:
                from pipeline.manifest import apply_captions_update

                updated = apply_captions_update(
                    row,
                    caption=caption,
                    action_caption=action_caption,
                    robot_learnable=bool(robot_learnable),
                    skill_category=skill_category,
                )
            else:
                from pipeline.manifest import apply_captions_partial_update

                kwargs: dict[str, Any] = {}
                if "caption" in filled_fields:
                    kwargs["caption"] = caption
                if "action_caption" in filled_fields:
                    kwargs["action_caption"] = action_caption
                if "robot_learnable" in filled_fields and robot_learnable is not None:
                    kwargs["robot_learnable"] = robot_learnable
                if "skill_category" in filled_fields:
                    kwargs["skill_category"] = skill_category
                updated = apply_captions_partial_update(row, **kwargs)
            row.clear()
            row.update(updated)
        except ImportError:
            if force_full or "caption" in filled_fields:
                row["caption"] = caption
            if force_full or "action_caption" in filled_fields:
                row["action_caption"] = action_caption
            row.pop("text", None)
            if force_full or "robot_learnable" in filled_fields:
                row["robot_learnable"] = robot_learnable
            if force_full or "skill_category" in filled_fields:
                row["skill_category"] = skill_category

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_submit, item): item for item in pending}
        for fut in as_completed(future_map):
            (
                idx,
                sid,
                status,
                caption,
                action_caption,
                robot_learnable,
                skill_category,
                err,
                src,
                filled_fields,
            ) = fut.result()
            seq_elapsed = time.time() - run_start
            with file_lock:
                done_count += 1
                try:
                    from pipeline.stage_timing import stage_progress_update

                    stage_progress_update(
                        done=done_count,
                        total=len(pending),
                        item=sid,
                        note=status,
                    )
                except ImportError:
                    pass
                row = rows[idx - 1]
                if status == "no_frames":
                    err_count += 1
                    print(
                        f"[done {done_count}/{len(pending)}] [{idx}/{total}] WARN: no frames sample_id={sid}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif status == "error":
                    err_count += 1
                    if drop_on_bad_skill and is_invalid_skill_category_error(err):
                        drop_ids.add(sid)
                        print(
                            f"[done {done_count}/{len(pending)}] [{idx}/{total}] ERROR {sid}: {err} "
                            f"-> drop",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        print(
                            f"[done {done_count}/{len(pending)}] [{idx}/{total}] ERROR {sid}: {err}",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    ok_count += 1
                    _apply_captions_to_row(
                        row,
                        caption=caption,
                        action_caption=action_caption,
                        robot_learnable=robot_learnable,
                        skill_category=skill_category,
                        filled_fields=filled_fields,
                        force_full=bool(args.force_recaption),
                    )
                    note = ""
                    if src == "first_frame":
                        note = " | note:first_frame_only"
                    print(
                        f"[done {done_count}/{len(pending)}] [{idx}/{total}] ok: {sid} | "
                        f"action={action_caption!r} | learnable={robot_learnable} | "
                        f"skill={skill_category} | "
                        f"wall={format_seconds(seq_elapsed)} | ok={ok_count} err={err_count}{note}",
                        flush=True,
                    )

                out_path.write_text(
                    json.dumps(rows, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            if sleep_piece > 0:
                time.sleep(sleep_piece)

    dropped = 0
    if drop_ids:
        if args.dry_run:
            print(
                f"[dry-run] would drop {len(drop_ids)} sample(s) (invalid skill_category): "
                f"{', '.join(sorted(drop_ids))}",
                flush=True,
            )
        elif drop_on_bad_skill:
            print(
                f"Dropping {len(drop_ids)} sample(s) with invalid skill_category...",
                flush=True,
            )
            dropped = drop_invalid_skill_category_samples(
                pipeline_root,
                rows,
                drop_ids,
                manifest_path=out_path,
            )

    print(
        f"Done. total_rows={len(rows)}, captioned_ok={ok_count}, err={err_count}, "
        f"dropped_invalid_skill_category={dropped}, "
        f"elapsed={format_seconds(time.time() - run_start)} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
