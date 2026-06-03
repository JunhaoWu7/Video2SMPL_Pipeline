"""video2smpl HMR backends (prompthmr default, camerahmr optional)."""

from pipeline.stages.video2smpl.backends.camerahmr import run_camerahmr_sample
from pipeline.stages.video2smpl.backends.prompthmr import run_prompthmr_sample

__all__ = ["run_camerahmr_sample", "run_prompthmr_sample"]
