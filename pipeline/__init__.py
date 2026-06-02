"""Video2SMPL pipeline package: multi-stage orchestration and sub-stages."""

__all__ = [
    "DEFAULT_STAGE_ORDER",
    "PIPELINE_STAGE_ORDER",
    "STAGE_REGISTRY",
    "get_stage",
    "list_stage_names",
    "resolve_stages_to_run",
]


def __getattr__(name: str):
    if name in __all__:
        from pipeline import registry as _registry

        return getattr(_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
