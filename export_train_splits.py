#!/usr/bin/env python3
"""
Export unified train/val/test manifests from dataset_manifest.json.

All dataset types share the same row schema after pipeline processing; this script
filters ready rows and writes splits/*.json for downstream training (text fields
only — no T5 / embedding export).

Examples:
  python export_train_splits.py --root-dir examples/training
  python export_train_splits.py --root-dir /data/batch_a --source-filter batch_a,batch_b
  python export_train_splits.py --root-dir /data/mix --require-captions --train-ratio 0.9
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.dataset_schema import (  # noqa: E402
    SPLITS_DIR,
    filter_train_ready,
    layout_tree_text,
)
from pipeline.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_NAME,
    STAGE_CAPTIONS,
    STAGE_VIDEO2SMPL,
    load_manifest_list,
    manifest_path,
)


def _parse_sources(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _split_rows(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        return [], [], []
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = 0
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export unified splits/train|val|test.json from dataset_manifest.json"
    )
    p.add_argument("--root-dir", type=Path, required=True, help="Dataset root (same as pipeline --root_dir)")
    p.add_argument("--manifest-name", type=str, default=DEFAULT_MANIFEST_NAME)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Default: <root-dir>/{SPLITS_DIR}",
    )
    p.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help="Comma-separated manifest ``source`` values to include (multi-dataset mix).",
    )
    p.add_argument(
        "--require-captions",
        action="store_true",
        help="Require non-empty caption and action_caption.",
    )
    p.add_argument(
        "--require-stages",
        type=str,
        default=f"{STAGE_VIDEO2SMPL}",
        help=f"Comma-separated stages_completed (default: {STAGE_VIDEO2SMPL}).",
    )
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--list-layout", action="store_true", help="Print canonical directory tree and exit.")
    args = p.parse_args(argv)

    if args.list_layout:
        print(layout_tree_text(str(args.root_dir.resolve())))
        return 0

    root = args.root_dir.resolve()
    mpath = manifest_path(root, args.manifest_name)
    if not mpath.is_file():
        print(f"Manifest not found: {mpath}", file=sys.stderr)
        return 1

    rows = load_manifest_list(mpath)
    require_stages = [s.strip() for s in args.require_stages.split(",") if s.strip()]
    ready = filter_train_ready(
        rows,
        require_captions=args.require_captions,
        require_stages=require_stages or None,
        sources=_parse_sources(args.source_filter),
    )

    out_dir = args.output_dir or (root / SPLITS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = _split_rows(ready, args.train_ratio, args.val_ratio, args.seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(part, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "root_dir": str(root),
        "manifest": str(mpath),
        "manifest_total": len(rows),
        "train_ready": len(ready),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "output_dir": str(out_dir),
        "require_captions": args.require_captions,
        "require_stages": require_stages,
        "source_filter": _parse_sources(args.source_filter),
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nUnified layout:\n{layout_tree_text(str(root))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
