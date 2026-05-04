#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Fill `text` in Video2SMPL pipeline manifests using an OpenAI-compatible vision API.

Reads a JSON list (e.g. train_stage4_empty_text.json), resolves rgb_path / first_frame under
--pipeline-root, uniformly samples frames from rgb.mp4 (fallback: first_frame), and writes
train_stage5_with_text.json (or --output-manifest) with each row's `text` set from model output.

Usage:
  export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY
  pip install openai pillow httpx

    python generate_sequence_captions.py \
    --manifest examples/training/train_stage4_empty_text.json \
    --pipeline-root examples/training \
    --output-manifest examples/training/train_stage5_with_text.json \
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
    """Expect {\"description\": \"...\"}; tolerate plain task_caption if model misbehaves."""
    desc = str(raw.get("description", "")).strip()
    if not desc:
        desc = str(raw.get("task_caption", "")).strip()
    return {"description": desc, "caption": desc}


def manifest_text_from_output(output: dict[str, Any]) -> str:
    for key in ("description", "caption"):
        v = str(output.get(key, "")).strip()
        if v:
            return v
    return ""


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
    rgb_rel = str(row.get("rgb_path", "")).strip()
    ff_rel = str(row.get("first_frame", "")).strip()
    rgb_abs = resolve_under_root(root, rgb_rel) if rgb_rel else None
    ff_abs = resolve_under_root(root, ff_rel) if ff_rel else None

    if rgb_abs and rgb_abs.is_file():
        urls, used = sample_frames_from_video(rgb_abs, num_frames, max_side)
        if urls:
            return urls, "video", used

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

    return f"""You analyze frames from a video clip of human motion (third-person/only one person).

Sample id: `{sid}`
Original filename hint: `{orig}`
Dataset / batch label (provenance): `{src}`
Media type: `{typ}`

You are given {num_sampled_frames} still frames sampled uniformly over time from the clip.

Describe in 1–2 short sentences what the person does: motion, pose changes, and clear interactions with objects or the scene. Do not focus on clothing or looks unless necessary for the action. Stick to what is visible.

Output ONLY a JSON object with exactly one field:
{{
  "description": "1-2 sentences."
}}

Rules:
- Single string value only; no bullet lists or extra keys.
- {lang_line}

Reply with raw JSON only. No markdown fences, no extra keys, no extra text."""


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
                "content": "You describe human motion in video frames accurately and concisely.",
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    text = resp.choices[0].message.content
    if not text:
        return {"description": "", "caption": ""}
    raw = extract_json_object(text)
    return normalize_caption_output(raw)


def caption_one_sample(
    client: Any,
    args: argparse.Namespace,
    pipeline_root: Path,
    idx: int,
    row: dict[str, Any],
) -> tuple[int, str, str, str, str | None, str]:
    """
    Run vision API for one manifest row. idx is 1-based list position.
    Returns (idx, sample_id, status, text, error_message, frame_source).
    status: ok | error | no_frames
    """
    sid = str(row.get("sample_id", f"row{idx}"))
    data_urls, src, _ = build_data_urls_for_manifest_row(
        pipeline_root, row, args.num_frames, args.max_side
    )
    if not data_urls:
        return (idx, sid, "no_frames", "", None, src)

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
        text = manifest_text_from_output(output)
        return (idx, sid, "ok", text, None, src)
    except Exception as e:
        return (idx, sid, "error", "", str(e), src)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video2SMPL manifest captions via OpenAI-compatible vision API."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Stage-4 JSON list (e.g. train_stage4_empty_text.json).",
    )
    p.add_argument(
        "--pipeline-root",
        type=Path,
        required=True,
        help="Training root (same as run_pipeline --root_dir); rgb_path / first_frame resolve here.",
    )
    p.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Default: <manifest_dir>/train_stage5_with_text.json",
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
        help="Call the API for every row even if `text` is already non-empty (e.g. full regenerate).",
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


def hydrate_rows_from_output_manifest(rows: list[dict[str, Any]], out_path: Path) -> None:
    """
    If output JSON already exists, copy non-empty `text` into matching `sample_id` rows.
    Runs whenever out_path exists (no flag required), so reruns do not re-bill filled samples.
    """
    if not out_path.is_file():
        return
    try:
        prev_list = load_manifest_list(out_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return
    prev_by_id = {str(r.get("sample_id", "")): r for r in prev_list}
    for r in rows:
        sid = str(r.get("sample_id", ""))
        old = prev_by_id.get(sid)
        if not old:
            continue
        tx = str(old.get("text", "")).strip()
        if tx:
            r["text"] = old["text"]


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
        out_path = manifest_path.parent / "train_stage5_with_text.json"
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        rows = load_manifest_list(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Failed to read manifest: {e}", file=sys.stderr)
        return 1
    hydrate_rows_from_output_manifest(rows, out_path)

    if args.dry_run:
        print(f"Manifest samples: {len(rows)} (pipeline root: {pipeline_root})")
        for row in rows[:8]:
            sid = row.get("sample_id", "?")
            urls, src, used = build_data_urls_for_manifest_row(
                pipeline_root, row, args.num_frames, args.max_side
            )
            print(
                f"  [dry-run] sample_id={sid} frame_source={src} n_images={len(urls)} "
                f"video_frame_idx={used} text_filled={bool(str(row.get('text','')).strip())}"
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
        if str(row.get("text", "")).strip() and not args.force_recaption:
            print(f"[{idx}/{total}] skip (already has text): {sid}", flush=True)
            continue
        pending.append((idx, row))

    if not pending:
        print("Nothing to caption (all skipped or empty pending list).", flush=True)
        return 0

    file_lock = threading.Lock()
    done_count = 0
    sleep_piece = float(args.sleep) / float(workers) if args.sleep > 0 else 0.0

    def _submit(item: tuple[int, dict[str, Any]]) -> tuple[int, str, str, str, str | None, str]:
        idx, row = item
        return caption_one_sample(client, args, pipeline_root, idx, row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_submit, item): item for item in pending}
        for fut in as_completed(future_map):
            idx, sid, status, text, err, src = fut.result()
            seq_elapsed = time.time() - run_start
            with file_lock:
                done_count += 1
                row = rows[idx - 1]
                if status == "no_frames":
                    err_count += 1
                    row.setdefault("text", "")
                    print(
                        f"[done {done_count}/{len(pending)}] [{idx}/{total}] WARN: no frames sample_id={sid}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif status == "error":
                    err_count += 1
                    row.setdefault("text", "")
                    print(
                        f"[done {done_count}/{len(pending)}] [{idx}/{total}] ERROR {sid}: {err}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    ok_count += 1
                    row["text"] = text
                    note = ""
                    if src == "first_frame":
                        note = " | note:first_frame_only"
                    print(
                        f"[done {done_count}/{len(pending)}] [{idx}/{total}] ok: {sid} | "
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
