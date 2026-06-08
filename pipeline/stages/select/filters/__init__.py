"""Select-stage pre-ingest filters (step1 basic + step2 person/view)."""

from pipeline.stages.select.filters.pipeline import run_select_filters

__all__ = ["run_select_filters"]
