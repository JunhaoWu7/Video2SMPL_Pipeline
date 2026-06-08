#!/usr/bin/env python3
"""
Top-level pipeline orchestrator.

Multi-dataset hub (recommended for production):
  Ingest raw videos with select (moves each clip into processed_trainable_data/<id>/):

  python run.py --dataset humanvid --select-input-dir /path/to/raw_videos
  python run.py --hub-root /data1/wjh/HumanRetarget --dataset sports

Each dataset subfolder has the same internal layout; see doc/data_layout.md.

Single-root mode (dev / examples):
  python run.py --root_dir examples/training --source my_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.hub import (  # noqa: E402
    DEFAULT_HUB_ROOT,
    hub_layout_text,
    init_dataset_layout,
    list_datasets,
    resolve_dataset_root,
    resolve_root_from_args,
)
from pipeline.registry import (  # noqa: E402
    DEFAULT_STAGE_ORDER,
    PIPELINE_STAGE_ORDER,
    STAGE_REGISTRY,
    list_stage_names,
    resolve_stages_to_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video2SMPL multi-stage pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Hub layout:\n"
            + hub_layout_text(DEFAULT_HUB_ROOT)
            + "\nAvailable stages: "
            + ", ".join(f"{n} ({STAGE_REGISTRY[n].description})" for n in sorted(STAGE_REGISTRY))
            + f"\nPipeline order: {', '.join(PIPELINE_STAGE_ORDER)}"
        ),
    )
    parser.add_argument(
        "--hub-root",
        type=str,
        default=DEFAULT_HUB_ROOT,
        help=f"Parent directory for multiple datasets (default: {DEFAULT_HUB_ROOT}).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset subfolder under --hub-root; sets root_dir=<hub-root>/<dataset>.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names; run pipeline for each in turn.",
    )
    parser.add_argument(
        "--init-dataset",
        type=str,
        default=None,
        metavar="NAME",
        help="Create <hub-root>/<NAME>/ with standard subdirs and exit.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List dataset subfolders under --hub-root and exit.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help='Manifest "source" label (default: same as --dataset when using hub mode).',
    )
    parser.add_argument(
        "--mapping_name",
        type=str,
        default="sample_id_to_source.json",
        help="Sample id <-> original video mapping under root_dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process samples already registered (select / video2smpl).",
    )
    parser.add_argument("--link", type=str, default=None)
    parser.add_argument("--default_link", type=str, default="")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="examples/training",
        help="Single-dataset root (ignored if --dataset / --datasets is set).",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=(
            "Comma-separated stage names to run. "
            f"Default: {','.join(DEFAULT_STAGE_ORDER)}. "
            f"Choices: {', '.join(list_stage_names())}"
        ),
    )
    parser.add_argument(
        "--from-stage",
        dest="from_stage",
        type=str,
        default=None,
        help="Start from this stage (skip earlier stages in the full/sub chain).",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print registered stages and exit.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="dataset_manifest.json",
        help="Single manifest under root_dir; all stages update this file in place.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--id_width", type=int, default=6)

    for stage in STAGE_REGISTRY.values():
        stage.add_arguments(parser)

    return parser


def _parse_csv(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _apply_root_and_source(args: argparse.Namespace, dataset_name: str | None) -> None:
    """Mutate args.root_dir and default args.source for one dataset run."""
    if dataset_name:
        args.root_dir = str(resolve_dataset_root(args.hub_root, dataset_name))
        if not str(getattr(args, "source", "") or "").strip():
            args.source = dataset_name
    else:
        args.root_dir = str(Path(args.root_dir).expanduser().resolve())


def _run_pipeline_for_args(args: argparse.Namespace) -> None:
    stage_names = _parse_csv(args.stages)
    to_run = resolve_stages_to_run(stage_names, args.from_stage)
    print(f"Pipeline root: {Path(args.root_dir).resolve()}")
    print(f"Manifest source label: {args.source}")
    print(f"Stages to run: {', '.join(to_run)}")
    for name in to_run:
        stage = STAGE_REGISTRY[name]
        print(f"\n=== Stage: {name} ({args.dataset or 'single-root'}) ===")
        stage.validate_args(args)
        stage.run(args)
    print(f"\nFinished: {', '.join(to_run)} @ {args.root_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_stages:
        for name in sorted(STAGE_REGISTRY):
            st = STAGE_REGISTRY[name]
            print(f"  {name}: {st.description}")
        print(f"\nPipeline order: {', '.join(PIPELINE_STAGE_ORDER)}")
        return 0

    hub = Path(args.hub_root).expanduser().resolve()

    if args.list_datasets:
        names = list_datasets(hub)
        print(f"Hub root: {hub}")
        print(f"Datasets ({len(names)}):")
        for n in names:
            print(f"  {n}  ->  {hub / n}")
        if not names:
            print("  (none — use --init-dataset <name> to scaffold)")
        return 0

    if args.init_dataset:
        name = args.init_dataset.strip()
        root = init_dataset_layout(resolve_dataset_root(hub, name), dataset_name=name, id_width=args.id_width)
        print(f"Initialized dataset layout: {root}")
        print(f"Place videos in: {root / 'video'}")
        print(f"Then ingest: python run.py --dataset {name} --from-stage select")
        return 0

    dataset_list = _parse_csv(args.datasets)
    if dataset_list:
        for name in dataset_list:
            run_args = argparse.Namespace(**vars(args))
            run_args.dataset = name
            _apply_root_and_source(run_args, name)
            _run_pipeline_for_args(run_args)
        return 0

    single_dataset = str(args.dataset).strip() if args.dataset else None
    if single_dataset:
        _apply_root_and_source(args, single_dataset)
    else:
        args.root_dir = str(Path(args.root_dir).expanduser().resolve())
        if not str(getattr(args, "source", "") or "").strip():
            parser.error("Provide --source, or use --dataset <name> (defaults source to dataset name).")

    try:
        _run_pipeline_for_args(args)
    except ValueError as e:
        parser.error(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
