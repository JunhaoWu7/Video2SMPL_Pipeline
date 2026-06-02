from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Iterable, Optional


class PipelineStage(ABC):
    """One executable sub-stage in the top-level pipeline."""

    name: str
    description: str

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register CLI flags for this stage on the shared parser."""

    @abstractmethod
    def run(self, args: argparse.Namespace) -> None:
        """Execute this stage."""

    def validate_args(self, args: argparse.Namespace) -> None:
        """Optional pre-run checks; override when a stage needs extra validation."""
