#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Fill caption fields and robot-skill labels in the unified pipeline manifest (in-place by default).

Writes per clip: ``caption``, ``action_caption``, ``robot_learnable``, ``skill_category``.

Reads a JSON list (e.g. dataset_manifest.json), resolves video_path / rgb_path / first_frame
under --pipeline-root, calls a vision API, and writes both fields back to the same manifest.

Usage:
  export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY
  pip install openai pillow httpx

    python generate_sequence_captions.py \
    --manifest examples/training/dataset_manifest.json \
    --pipeline-root examples/training \
    --model google/gemini-2.5-flash-lite \
    --workers 8 \
    --resume

  # 并行（默认 4 路；注意供应商速率限制，报错多时改成 --workers 2）
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
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = text[start : end + 1]
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj
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

    caption = str(raw.get("caption", "")).strip()
    if not caption:
        caption = str(raw.get("description", "")).strip()
    if not caption:
        caption = str(raw.get("task_caption", "")).strip()

    action = str(raw.get("action_caption", "")).strip()
    if not action:
        action = str(raw.get("action_label", "")).strip()
    if not action:
        action = str(raw.get("action", "")).strip()

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
    if not norm.get("caption") or not norm.get("action_caption"):
        raise ValueError("Model JSON missing non-empty caption or action_caption.")
    if norm.get("skill_category") not in (
        "manipulation",
        "locomotion",
        "loco-manipulation",
    ):
        raise ValueError(
            f"Invalid skill_category: {norm.get('skill_category')!r}; "
            "expected manipulation | locomotion | loco-manipulation."
        )
    if norm.get("robot_learnable") is None:
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
) -> str:
    sid = str(row.get("sample_id", "")).strip()
    orig = str(row.get("original_video", "")).strip()
    src = str(row.get("source", "")).strip()
    typ = str(row.get("type", "video")).strip()

    lang_line = {
        "en": "Write the description in English.",
        "zh": "描述请使用简体中文。",
        "bilingual": "Write the description in English.",
    }.get(caption_lang, "Write the description in English.")

    action_lang = {
        "en": "action_caption must be a short English phrase (roughly 3–8 words), verb-led.",
        "zh": "action_caption 用简短中文动作短语（约 3–12 字），动词开头。",
        "bilingual": "caption in English; action_caption as a short English phrase.",
    }.get(caption_lang, "action_caption: short English phrase, verb-led.")

    return f"""You analyze frames from a short human **action clip** (single person, third-person view).
Assume the clip shows one coherent motion segment suitable for robot imitation learning review.

Sample id: `{sid}`
Original filename hint: `{orig}`
Dataset / batch label (provenance): `{src}`
Media type: `{typ}`

You are given {num_sampled_frames} still frames sampled uniformly over time from the clip.

Output ONLY a JSON object with exactly these four fields:
{{
  "caption": "1-2 sentences describing the visible motion, pose changes, and scene interaction.",
  "action_caption": "Short action phrase naming the main motion (e.g. 'walking forward', 'picking up box').",
  "robot_learnable": true,
  "skill_category": "manipulation"
}}

Field rules:
- ``caption``: 1–2 sentences; do not focus on clothing unless needed for the action.
- ``action_caption``: concise action label for this clip; no full sentences; no bullet lists.
- ``robot_learnable`` (boolean): true if a humanoid / mobile manipulator could **plausibly learn and reproduce** the main action from this clip (locomotion, manipulation, or both). false if the clip is not suitable, e.g. idle standing/sitting with no skill, pure conversation, camera-only motion, heavy occlusion, multi-person chaos, dance/acrobat stunts beyond typical robot skills, or action too ambiguous to label.
- ``skill_category`` (string): assign **exactly one** category that best matches the **primary** skill in the clip:
  - ``manipulation``: mostly upper-body / hand–object interaction; base largely stationary.
  - ``locomotion``: mostly whole-body displacement (walk, run, step, turn, climb stairs) with little object interaction.
  - ``loco-manipulation``: clear **combination** (walk while carrying, approach object while stepping, locomote then manipulate in one clip).
  Pick the single best fit; do not output multiple categories.
- {lang_line}
- {action_lang}

Reply with raw JSON only. No markdown fences, no extra keys."""


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
            print(
                f"  [waiting] {label} ... elapsed={format_seconds(elapsed)}",
                flush=True,
            )

    t = threading.Thread(target=_ticker, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        stop_event.set()
        t.join(timeout=0.2)


def call_openai_caption_with_prompt(
    client: Any,
    model: str,
    user_prompt: str,
    data_urls: list[str],
    vision_detail: str,
) -> dict[str, Any]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_prompt},
    ]
    for url in data_urls:
        user_content.append(
            {"type": "image_url", "image_url": {"url": url, "detail": vision_detail}}
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You label short human action clips for robot learning datasets: "
                    "descriptions, robot learnability, and skill category."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    text = resp.choices[0].message.content
    if not text:
        raise ValueError("Empty model response.")
    raw = extract_json_object(text)
    norm = normalize_caption_output(raw)
    validate_caption_output(norm)
    return norm


def caption_one_sample(
    client: Any,
    args: argparse.Namespace,
    pipeline_root: Path,
    idx: int,
    row: dict[str, Any],
) -> tuple[int, str, str, str, str, bool | None, str, str | None, str]:
    """
    Run vision API for one manifest row. idx is 1-based list position.
    Returns (idx, sample_id, status, caption, action_caption, robot_learnable,
             skill_category, error_message, frame_source).
    status: ok | error | no_frames
    """
    sid = str(row.get("sample_id", f"row{idx}"))
    data_urls, src, _ = build_data_urls_for_manifest_row(
        pipeline_root, row, args.num_frames, args.max_side
    )
    if not data_urls:
        return (idx, sid, "no_frames", "", "", None, "", None, src)

    user_prompt = build_user_prompt_manifest(row, len(data_urls), args.caption_lang)
    try:
        output = call_with_heartbeat(
            lambda: call_openai_caption_with_prompt(
                client,
                args.model,
                user_prompt,
                data_urls,
                args.vision_detail,
            ),
            label=sid,
            interval_sec=max(3.0, float(args.heartbeat_sec)),
        )
        caption, action_caption, robot_learnable, skill_category = captions_from_output(output)
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
        )
    except Exception as e:
        return (idx, sid, "error", "", "", None, "", str(e), src)


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
        default="openai/gpt-4o",
        help="Model ID (e.g. OpenRouter route openai/gpt-4o).",
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
        default=4,
        help="Parallel API calls (threads). Use 1 for strictly sequential behavior. If you hit 429/rate limits, lower this.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated no-op (kept for old scripts). Skipping already-captioned rows is always on unless --force-recaption.",
    )
    p.add_argument(
        "--force-recaption",
        action="store_true",
        help="Call the API even when all caption-stage fields are already filled.",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not call API; print plan only.")
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request timeout (seconds).",
    )
    p.add_argument("--max-retries", type=int, default=2, help="Retries on transient failures.")
    p.add_argument(
        "--base-url",
        type=str,
        default="http://47.94.22.126/v1",
        help="API base URL; env OPENAI_BASE_URL overrides.",
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
    return p.parse_args()


def create_openai_client(args: argparse.Namespace) -> tuple[Any, float, str]:
    if OpenAI is None:
        raise RuntimeError("Install openai: pip install openai")
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY).")

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
    base_url = (os.environ.get("OPENAI_BASE_URL") or args.base_url or "").strip().rstrip("/")
    if base_url:
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
    args = parse_args()

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
            print(
                f"  [dry-run] sample_id={sid} frame_source={src} n_images={len(urls)} "
                f"video_frame_idx={used} captions_done={row_captions_complete(row)}"
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

    pending: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(rows, start=1):
        sid = str(row.get("sample_id", f"row{idx}"))
        if row_captions_complete(row) and not args.force_recaption:
            print(f"[{idx}/{total}] skip (caption stage complete): {sid}", flush=True)
            continue
        pending.append((idx, row))

    if not pending:
        print("Nothing to caption (all skipped or empty pending list).", flush=True)
        return 0

    file_lock = threading.Lock()
    done_count = 0
    sleep_piece = float(args.sleep) / float(workers) if args.sleep > 0 else 0.0

    def _submit(
        item: tuple[int, dict[str, Any]],
    ) -> tuple[int, str, str, str, str, bool | None, str, str | None, str]:
        idx, row = item
        return caption_one_sample(client, args, pipeline_root, idx, row)

    def _apply_captions_to_row(
        row: dict[str, Any],
        caption: str,
        action_caption: str,
        robot_learnable: bool,
        skill_category: str,
    ) -> None:
        try:
            from pipeline.manifest import apply_captions_update

            updated = apply_captions_update(
                row,
                caption=caption,
                action_caption=action_caption,
                robot_learnable=robot_learnable,
                skill_category=skill_category,
            )
            row.clear()
            row.update(updated)
        except ImportError:
            row["caption"] = caption
            row["action_caption"] = action_caption
            row.pop("text", None)
            row["robot_learnable"] = robot_learnable
            row["skill_category"] = skill_category

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_submit, item): item for item in pending}
        for fut in as_completed(future_map):
            idx, sid, status, caption, action_caption, robot_learnable, skill_category, err, src = (
                fut.result()
            )
            seq_elapsed = time.time() - run_start
            with file_lock:
                done_count += 1
                row = rows[idx - 1]
                if status == "no_frames":
                    err_count += 1
                    row.setdefault("caption", "")
                    row.setdefault("action_caption", "")
                    row.setdefault("skill_category", "")
                    row.setdefault("robot_learnable", None)
                    print(
                        f"[done {done_count}/{len(pending)}] [{idx}/{total}] WARN: no frames sample_id={sid}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif status == "error":
                    err_count += 1
                    row.setdefault("caption", "")
                    row.setdefault("action_caption", "")
                    row.setdefault("skill_category", "")
                    row.setdefault("robot_learnable", None)
                    print(
                        f"[done {done_count}/{len(pending)}] [{idx}/{total}] ERROR {sid}: {err}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    ok_count += 1
                    _apply_captions_to_row(
                        row,
                        caption,
                        action_caption,
                        bool(robot_learnable),
                        skill_category,
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

    print(
        f"Done. total_rows={total}, captioned_ok={ok_count}, err={err_count}, "
        f"elapsed={format_seconds(time.time() - run_start)} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
