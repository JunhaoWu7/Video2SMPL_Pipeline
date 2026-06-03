"""CameraHMR + DART canonical branch (legacy optional backend)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _setup_vendor_imports(vendor_root: Path) -> None:
    root_str = str(vendor_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _group_consecutive_frame_ids(frame_ids: torch.Tensor) -> List[List[int]]:
    if frame_ids.numel() == 0:
        return []
    grouped: List[List[int]] = []
    current = [int(frame_ids[0].item())]
    for value in frame_ids[1:]:
        idx = int(value.item())
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            grouped.append(current)
            current = [idx]
    grouped.append(current)
    return grouped


def _canonical_smpl_to_npz_dict(
    smpl_params_canonical: Dict[str, torch.Tensor],
    person_idx: int,
    num_persons: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, val in smpl_params_canonical.items():
        if val is None:
            continue
        t = val.detach().cpu()
        if num_persons > 1:
            t = t[person_idx]
        out[key] = t.float().numpy().astype(np.float32)
    return out


def _betas_np_from_smoothed(
    smpl_smooth: Dict[str, torch.Tensor],
    T_canonical: int,
) -> np.ndarray:
    b = smpl_smooth.get("betas")
    if b is None:
        return np.zeros((T_canonical, 10), dtype=np.float32)
    t = b.detach().cpu().float()
    if t.ndim == 1:
        if t.numel() != 10:
            raise ValueError(f"Expected betas (10,), got {tuple(t.shape)}")
        row = t.numpy().astype(np.float32)
        return np.tile(row[None, :], (T_canonical, 1))
    if t.ndim != 2 or t.shape[1] != 10:
        raise ValueError(f"Expected betas (T,10), got {tuple(t.shape)}")
    arr = t.numpy().astype(np.float32)
    if arr.shape[0] == T_canonical:
        return arr
    if arr.shape[0] > T_canonical:
        return arr[:T_canonical]
    pad = np.tile(arr[-1:], (T_canonical - arr.shape[0], 1))
    return np.concatenate([arr, pad], axis=0)


def _smooth_smpl_for_one_person(
    person_smpl_params: Dict[str, torch.Tensor],
    frame_mask: Optional[torch.Tensor],
    smooth_window: int,
    echo_module: Any,
) -> Dict[str, torch.Tensor]:
    seq_len = person_smpl_params["global_orient"].shape[0]
    if frame_mask is None:
        missing_groups: List[List[int]] = []
    else:
        mask = frame_mask[:seq_len].bool()
        missing_frame_ids = torch.where(~mask)[0]
        missing_groups = _group_consecutive_frame_ids(missing_frame_ids)

    betas = person_smpl_params.get("betas")
    smpl_dict = {
        "global_orient": person_smpl_params["global_orient"],
        "body_pose": person_smpl_params["body_pose"],
        "transl": person_smpl_params["transl"],
        "betas": betas if betas is not None else None,
    }
    smpl_6d = echo_module.smpl_dict_to_rot6d(smpl_dict)
    if missing_groups:
        smpl_6d = echo_module.linear_interpolate_frame_ids(smpl_6d, missing_groups)
    if smooth_window > 0:
        smpl_6d = echo_module.smooth_motion_rep(smpl_6d, kernel_size=smooth_window, sigma=1.0)
    return echo_module.rot6d_to_smpl_dict(smpl_6d)


def run_camerahmr_sample(
    *,
    video_path: Path,
    output_npz: Path,
    args: Any,
    vendor_root: Path,
) -> None:
    """Run CameraHMR pipeline for one clip and write ``smpl_canonical.npz``."""
    _setup_vendor_imports(vendor_root)
    from extract_motion import MotionExtractor  # type: ignore
    from scripts.data_processors.motion_alignment.retarget_mogen_db import (  # type: ignore
        rot6d_to_smpl_dict,
        smpl_dict_to_rot6d,
        smooth_motion_rep,
    )
    from scripts.data_processors.motion_alignment.seq_utils import (  # type: ignore
        linear_interpolate_frame_ids,
    )

    class EchoModule:
        pass

    EchoModule.smooth_motion_rep = staticmethod(smooth_motion_rep)
    EchoModule.smpl_dict_to_rot6d = staticmethod(smpl_dict_to_rot6d)
    EchoModule.rot6d_to_smpl_dict = staticmethod(rot6d_to_smpl_dict)
    EchoModule.linear_interpolate_frame_ids = staticmethod(linear_interpolate_frame_ids)

    extractor = MotionExtractor(
        device=torch.device(args.device) if args.device else None,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        use_shape=args.use_shape,
        overwrite=args.overwrite,
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2smpl_cam_") as tmp:
        bbox_path = Path(tmp) / "bbox.pt"
        bbx_xyxy, conf, frame_mask = extractor.extract_bbox(
            video_path=str(video_path),
            output_path=None,
            overwrite=args.overwrite,
        )
        torch.save(
            {
                "bbx_xyxy": bbx_xyxy.detach().cpu(),
                "bbx_conf": conf.detach().cpu(),
                "frame_mask": frame_mask.detach().cpu(),
            },
            bbox_path,
        )
        smpl_data = extractor.extract_smpl(
            video_path=str(video_path),
            bbox_path=str(bbox_path),
            output_path=None,
            overwrite=args.overwrite,
        )
        post_res = extractor.post_process(
            smpl_data=smpl_data,
            smooth_window=args.smooth_window,
            set_floor=args.set_floor,
            frame_mask=frame_mask,
            use_shape=args.use_shape,
        )

        person_idx = args.person_idx
        num_persons = int(smpl_data["smpl_params_incam"]["global_orient"].shape[0])
        smpl_incam = {
            k: v[person_idx].detach().cpu() for k, v in smpl_data["smpl_params_incam"].items()
        }
        smpl_smooth = _smooth_smpl_for_one_person(
            person_smpl_params=smpl_incam,
            frame_mask=frame_mask.detach().cpu(),
            smooth_window=args.smooth_window,
            echo_module=EchoModule,
        )

        canon_np = _canonical_smpl_to_npz_dict(
            post_res["smpl_params_canonical"], person_idx, num_persons
        )
        T_canon = int(canon_np["global_orient"].shape[0])
        canon_np["betas"] = _betas_np_from_smoothed(smpl_smooth, T_canon)
        np.savez(
            output_npz,
            **canon_np,
            intrinsic=smpl_data["intrinsic"].detach().cpu().numpy(),
            frame_mask=frame_mask.detach().cpu().numpy(),
            bbox_xyxy=bbx_xyxy.detach().cpu().numpy(),
            bbox_conf=conf.detach().cpu().numpy(),
            set_floor=np.array([int(args.set_floor)], dtype=np.int32),
            coord_note=np.bytes_("canonical_dart_smpl_axis_angle"),
            smpl_backend=np.bytes_("camerahmr"),
        )
