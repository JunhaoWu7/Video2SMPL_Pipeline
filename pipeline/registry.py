from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pipeline.stages.base import PipelineStage
from pipeline.stages.captions.stage import CaptionsStage
from pipeline.stages.export_splits.stage import ExportSplitsStage
from pipeline.stages.external_smpl.stage import ExternalSmplStage
from pipeline.stages.prune.stage import PruneStage
from pipeline.stages.select.stage import SelectStage
from pipeline.stages.video2smpl.stage import Video2SmplStage

# Canonical main-chain order (fixed).
PIPELINE_STAGE_ORDER: List[str] = [
    "select",
    "captions",
    "prune",
    "video2smpl",
    "export_splits",
]

# Default CLI chain = full pipeline
DEFAULT_STAGE_ORDER: List[str] = list(PIPELINE_STAGE_ORDER)

# Backward-compatible alias
FULL_STAGE_ORDER: List[str] = PIPELINE_STAGE_ORDER

STAGE_REGISTRY: Dict[str, PipelineStage] = {
    SelectStage.name: SelectStage(),
    CaptionsStage.name: CaptionsStage(),
    PruneStage.name: PruneStage(),
    Video2SmplStage.name: Video2SmplStage(),
    ExportSplitsStage.name: ExportSplitsStage(),
    ExternalSmplStage.name: ExternalSmplStage(),
}


def list_stage_names() -> List[str]:
    return list(STAGE_REGISTRY.keys())


def get_stage(name: str) -> PipelineStage:
    key = name.strip()
    if key not in STAGE_REGISTRY:
        known = ", ".join(sorted(STAGE_REGISTRY))
        raise ValueError(f'Unknown stage "{name}". Available: {known}')
    return STAGE_REGISTRY[key]


def _canonicalize_stage_list(names: Sequence[str]) -> List[str]:
    """Reorder user-provided stages to canonical pipeline order."""
    requested = []
    seen = set()
    for raw in names:
        n = raw.strip()
        if not n or n in seen:
            continue
        get_stage(n)
        requested.append(n)
        seen.add(n)
    return [s for s in PIPELINE_STAGE_ORDER if s in seen]


def resolve_stages_to_run(
    stages: Optional[Sequence[str]],
    from_stage: Optional[str],
) -> List[str]:
    """
    Resolve stages to execute in canonical order.

    - No ``--stages``: full chain through ``export_splits``.
    - ``--stages``: subset, reordered to canonical order (not user list order).
    - ``--from-stage``: drop earlier stages. Without ``--stages``, run from that
      stage through the end of the full chain (e.g. ``--from-stage captions`` ->
      ``captions, video2smpl``).
    """
    if stages:
        selected = _canonicalize_stage_list(stages)
        if not selected:
            raise ValueError("No valid stage names in --stages.")
    else:
        selected = list(PIPELINE_STAGE_ORDER)

    if from_stage:
        start = from_stage.strip()
        get_stage(start)
        if start not in PIPELINE_STAGE_ORDER:
            raise ValueError(f'--from-stage "{start}" is not part of the main pipeline chain.')

        if not stages:
            idx = PIPELINE_STAGE_ORDER.index(start)
            selected = PIPELINE_STAGE_ORDER[idx:]
        elif start not in selected:
            tail = PIPELINE_STAGE_ORDER[PIPELINE_STAGE_ORDER.index(start) :]
            raise ValueError(
                f'--from-stage "{start}" is not in the --stages list {selected}. '
                f'Include it or omit --stages to run {",".join(tail)}.'
            )
        else:
            idx = selected.index(start)
            selected = selected[idx:]

    return selected


def register_stage(stage: PipelineStage) -> None:
    """Register an additional stage at runtime (for future extensions)."""
    STAGE_REGISTRY[stage.name] = stage
