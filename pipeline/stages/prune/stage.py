from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.stages.base import PipelineStage


class PruneStage(PipelineStage):
    name = "prune"
    description = "Delete robot_learnable=false samples after captions (required before video2smpl)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("prune stage")
        group.add_argument(
            "--prune-dry-run",
            action="store_true",
            help="List samples that would be removed without deleting files or updating manifest.",
        )

    def validate_args(self, args: argparse.Namespace) -> None:
        from pipeline.manifest import load_manifest_list, manifest_path, parse_robot_learnable

        root = Path(args.root_dir).resolve()
        mpath = manifest_path(root, getattr(args, "manifest_name", None))
        if not mpath.exists():
            raise ValueError(f"Manifest not found: {mpath}. Run captions before prune.")
        rows = load_manifest_list(mpath)
        if not rows:
            raise ValueError(f"Manifest is empty: {mpath}")
        if not any(parse_robot_learnable(r.get("robot_learnable")) is not None for r in rows):
            raise ValueError(
                "No robot_learnable labels in manifest. Run the captions stage before prune."
            )

    def run(self, args: argparse.Namespace) -> None:
        from pipeline.stages.prune.run import run as prune_run

        ns = argparse.Namespace(**vars(args))
        ns.dry_run = bool(getattr(args, "prune_dry_run", False))
        prune_run(ns)
