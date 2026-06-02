__all__ = ["SelectStage"]


def __getattr__(name: str):
    if name == "SelectStage":
        from pipeline.stages.select.stage import SelectStage

        return SelectStage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
