from __future__ import annotations

import argparse

from pipeline.stages.base import PipelineStage


class ExternalSmplStage(PipelineStage):
    name = "external_smpl"
    description = "External SMPL files -> same smooth/canonical chain as video2smpl (separate run folder)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("external_smpl stage")
        group.add_argument(
            "--external_smpl_dir",
            type=str,
            default=None,
            help="Directory of external SMPL (.npz/.pt/.pth); required when running this stage.",
        )
        group.add_argument("--glob", type=str, default="*.npz")
        group.add_argument("--start_id", type=int, default=1)
        group.add_argument("--set_floor", action="store_true")
        group.add_argument("--fx", type=float, default=1000.0)
        group.add_argument("--fy", type=float, default=1000.0)
        group.add_argument("--cx", type=float, default=0.0)
        group.add_argument("--cy", type=float, default=0.0)
        group.add_argument(
            "--self_check_confirm",
            action="store_true",
            help="Confirm external SMPL conventions were manually verified.",
        )
        group.add_argument(
            "--check_only",
            action="store_true",
            help="Precheck only; write report and exit.",
        )

    def validate_args(self, args: argparse.Namespace) -> None:
        if not getattr(args, "external_smpl_dir", None):
            raise ValueError('--external_smpl_dir is required when running the "external_smpl" stage.')

    def run(self, args: argparse.Namespace) -> None:
        self.validate_args(args)
        from pipeline.stages.external_smpl.run import run as external_smpl_run

        external_smpl_run(args)
