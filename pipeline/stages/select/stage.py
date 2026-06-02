from __future__ import annotations

import argparse

from pipeline.stages.base import PipelineStage


class SelectStage(PipelineStage):
    """
    Stage 1: ingest clips into ``processed_trainable_data/<sample_id>/`` and register paths.
    """

    name = "select"
    description = "Ingest videos into per-sample dirs and register video_path (filter TBD)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("select stage")
        group.add_argument(
            "--select-input-dir",
            type=str,
            default=None,
            help="Raw videos to ingest (recursive); required for select.",
        )
        group.add_argument(
            "--select-symlink",
            action="store_true",
            help="Symlink instead of move into processed_trainable_data/<sample_id>/.",
        )

    def validate_args(self, args: argparse.Namespace) -> None:
        if not str(getattr(args, "source", "") or "").strip():
            raise ValueError('--source is required when running the "select" stage.')
        if not str(getattr(args, "select_input_dir", "") or "").strip():
            raise ValueError(
                "--select-input-dir is required for select "
                "(folder of videos to move into processed_trainable_data/<sample_id>/)."
            )

    def run(self, args: argparse.Namespace) -> None:
        self.validate_args(args)
        from pipeline.stages.select.run import run as select_run

        select_run(args)
