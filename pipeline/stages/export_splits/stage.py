from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.manifest import DEFAULT_MANIFEST_NAME, manifest_path, smpl_filled
from pipeline.stages.base import PipelineStage


class ExportSplitsStage(PipelineStage):
    name = "export_splits"
    description = "Write splits/*.json by skill_category (auto after video2smpl; not inside SMPL code)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("export_splits stage")
        group.add_argument(
            "--export-splits-output-dir",
            type=str,
            default=None,
            help="Override output dir (default: <root_dir>/splits).",
        )
        group.add_argument(
            "--export-splits-dry-run",
            action="store_true",
            help="Print per-skill counts only; do not write JSON.",
        )

    def validate_args(self, args: argparse.Namespace) -> None:
        from pipeline.manifest import captions_filled, load_manifest_list

        root = Path(args.root_dir).resolve()
        mpath = manifest_path(root, getattr(args, "manifest_name", DEFAULT_MANIFEST_NAME))
        if not mpath.exists():
            raise ValueError(f"Manifest not found: {mpath}. Run video2smpl first.")
        rows = load_manifest_list(mpath)
        ready = [r for r in rows if captions_filled(r) and smpl_filled(r)]
        if not ready:
            raise ValueError(
                f"No SMPL-ready rows in {mpath}. Complete video2smpl before export_splits."
            )

    def run(self, args: argparse.Namespace) -> None:
        root = Path(args.root_dir).resolve()
        manifest_name = getattr(args, "manifest_name", DEFAULT_MANIFEST_NAME)

        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from export_skill_splits import main as export_main

        argv = [
            "export_skill_splits",
            "--root-dir",
            str(root),
            "--manifest-name",
            manifest_name,
        ]
        out_dir = getattr(args, "export_splits_output_dir", None)
        if out_dir:
            argv.extend(["--output-dir", str(out_dir)])
        if getattr(args, "export_splits_dry_run", False):
            argv.append("--dry-run")

        old_argv = sys.argv
        try:
            sys.argv = argv
            exit_code = export_main()
        finally:
            sys.argv = old_argv

        if exit_code != 0:
            raise SystemExit(exit_code)
