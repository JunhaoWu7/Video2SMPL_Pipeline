__all__ = ["Video2SmplStage"]


def __getattr__(name: str):
    if name == "Video2SmplStage":
        from pipeline.stages.video2smpl.stage import Video2SmplStage

        return Video2SmplStage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
