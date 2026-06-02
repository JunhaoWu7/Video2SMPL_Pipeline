__all__ = ["ExternalSmplStage"]


def __getattr__(name: str):
    if name == "ExternalSmplStage":
        from pipeline.stages.external_smpl.stage import ExternalSmplStage

        return ExternalSmplStage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
