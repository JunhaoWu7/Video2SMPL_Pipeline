import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import smplx
import torch

_AXIS_ANGLE_TO_MATRIX_FN = None


def _setup_vendor_imports(vendor_root: Path) -> None:
    vendor_root = vendor_root.resolve()
    root_str = str(vendor_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    camera_hmr_root = vendor_root / "extract_motion" / "CameraHMR"
    camera_hmr_scripts = camera_hmr_root / "scripts"
    for p in (camera_hmr_root, camera_hmr_scripts):
        p_str = str(p.resolve())
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _make_new_run_dir(root_dir: Path, run_prefix: str = "external_smpl_run_") -> Path:
    runs_root = root_dir / "external_smpl_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{run_prefix}{stamp}"
    idx = 1
    while run_dir.exists():
        run_dir = runs_root / f"{run_prefix}{stamp}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _load_external_smpl(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        out = {k: data[k] for k in data.files}
        data.close()
        return out
    if path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict in torch file: {path}")
        if "smpl_params_incam" in obj and isinstance(obj["smpl_params_incam"], dict):
            d = obj["smpl_params_incam"]
            out = {k: d.get(k) for k in ("global_orient", "body_pose", "transl", "betas")}
            if "intrinsic" in obj:
                out["intrinsic"] = obj["intrinsic"]
            if "frame_mask" in obj:
                out["frame_mask"] = obj["frame_mask"]
            return out
        if "smpl_params" in obj and isinstance(obj["smpl_params"], dict):
            d = obj["smpl_params"]
            out = {k: d.get(k) for k in ("global_orient", "body_pose", "transl", "betas")}
            if "intrinsic" in obj:
                out["intrinsic"] = obj["intrinsic"]
            if "frame_mask" in obj:
                out["frame_mask"] = obj["frame_mask"]
            return out
        return obj
    raise ValueError(f"Unsupported file extension for external SMPL: {path}")


def _as_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if x is None:
        raise ValueError("Required field is missing in external SMPL data")
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _to_rotmat_aa_or_mat(x: torch.Tensor, is_body_pose: bool) -> torch.Tensor:
    # Support either axis-angle (..., 3) or rotation matrix (..., 3, 3).
    if x.ndim >= 3 and x.shape[-2:] == (3, 3):
        return x.to(torch.float32)
    if x.shape[-1] != 3:
        raise ValueError(f"Unexpected rotation shape: {tuple(x.shape)}")
    if is_body_pose and x.ndim == 2 and x.shape[1] == 69:
        x = x.reshape(x.shape[0], 23, 3)
    if _AXIS_ANGLE_TO_MATRIX_FN is None:
        raise RuntimeError("axis_angle_to_matrix is not initialized. Call run() entrypoint instead of importing helpers directly.")
    return _AXIS_ANGLE_TO_MATRIX_FN(x.to(torch.float32))


def _normalize_smpl_dict(raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    global_orient_raw = raw.get("global_orient")
    body_pose_raw = raw.get("body_pose")
    transl_raw = raw.get("transl")
    betas_raw = raw.get("betas")
    if betas_raw is None:
        raise ValueError("betas is required and must be provided as (10,) or (T,10)")

    global_orient_t = _as_tensor(global_orient_raw)
    body_pose_t = _as_tensor(body_pose_raw)
    transl_t = _as_tensor(transl_raw)
    betas_t = _as_tensor(betas_raw)

    global_orient = _to_rotmat_aa_or_mat(global_orient_t, is_body_pose=False)
    body_pose = _to_rotmat_aa_or_mat(body_pose_t, is_body_pose=True)

    if global_orient.ndim == 4 and global_orient.shape[1] == 1:
        global_orient = global_orient[:, 0]
    if global_orient.ndim != 3 or global_orient.shape[-2:] != (3, 3):
        raise ValueError(f"global_orient must become (T,3,3), got {tuple(global_orient.shape)}")

    if body_pose.ndim == 4 and body_pose.shape[1] == 23:
        pass
    elif body_pose.ndim == 3 and body_pose.shape[1] == 69:
        body_pose = body_pose.reshape(body_pose.shape[0], 23, 3)
        if _AXIS_ANGLE_TO_MATRIX_FN is None:
            raise RuntimeError("axis_angle_to_matrix is not initialized. Call run() entrypoint instead of importing helpers directly.")
        body_pose = _AXIS_ANGLE_TO_MATRIX_FN(body_pose)
    elif body_pose.ndim == 3 and body_pose.shape[1] == 23 and body_pose.shape[2] == 3:
        if _AXIS_ANGLE_TO_MATRIX_FN is None:
            raise RuntimeError("axis_angle_to_matrix is not initialized. Call run() entrypoint instead of importing helpers directly.")
        body_pose = _AXIS_ANGLE_TO_MATRIX_FN(body_pose)
    else:
        raise ValueError(f"body_pose must become (T,23,3,3), got {tuple(body_pose.shape)}")

    if transl_t.ndim != 2 or transl_t.shape[1] != 3:
        raise ValueError(f"transl must be (T,3), got {tuple(transl_t.shape)}")
    if transl_t.shape[0] != global_orient.shape[0]:
        raise ValueError("transl and global_orient must share the same frame length")
    if body_pose.shape[0] != global_orient.shape[0]:
        raise ValueError("body_pose and global_orient must share the same frame length")

    if betas_t.ndim == 1:
        pass
    elif betas_t.ndim == 2:
        betas_t = betas_t[0]
    else:
        raise ValueError(f"betas must be (10,) or (T,10), got {tuple(betas_t.shape)}")

    return {
        "global_orient": global_orient.to(torch.float32).cpu(),
        "body_pose": body_pose.to(torch.float32).cpu(),
        "transl": transl_t.to(torch.float32).cpu(),
        "betas": betas_t.to(torch.float32).cpu(),
    }


def _resolve_intrinsic(raw: Dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    if raw.get("intrinsic") is not None:
        intrinsic = _as_tensor(raw["intrinsic"])
        if intrinsic.shape[-2:] == (3, 3):
            return intrinsic.reshape(3, 3).to(torch.float32)
    return torch.tensor(
        [
            [args.fx, 0.0, args.cx],
            [0.0, args.fy, args.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _resolve_frame_mask(raw: Dict[str, Any], num_frames: int) -> torch.Tensor:
    mask = raw.get("frame_mask")
    if mask is None:
        return torch.ones(num_frames, dtype=torch.bool)
    mask_t = _as_tensor(mask, dtype=torch.float32).flatten()
    if mask_t.numel() < num_frames:
        pad = torch.ones(num_frames - mask_t.numel(), dtype=torch.float32)
        mask_t = torch.cat([mask_t, pad], dim=0)
    return (mask_t[:num_frames] > 0.5).bool()


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


def _smooth_smpl_for_one_person(
    person_smpl_params: Dict[str, torch.Tensor],
    frame_mask: Optional[torch.Tensor],
    smooth_window: int,
    echo_module,
) -> Dict[str, torch.Tensor]:
    seq_len = person_smpl_params["global_orient"].shape[0]
    if frame_mask is None:
        missing_groups: List[List[int]] = []
    else:
        mask = frame_mask[:seq_len].bool()
        missing_frame_ids = torch.where(~mask)[0]
        missing_groups = _group_consecutive_frame_ids(missing_frame_ids)

    smpl_dict = {
        "global_orient": person_smpl_params["global_orient"],
        "body_pose": person_smpl_params["body_pose"],
        "transl": person_smpl_params["transl"],
        "betas": person_smpl_params.get("betas"),
    }
    smpl_6d = echo_module.smpl_dict_to_rot6d(smpl_dict)
    if missing_groups:
        smpl_6d = echo_module.linear_interpolate_frame_ids(smpl_6d, missing_groups)
    if smooth_window > 0:
        smpl_6d = echo_module.smooth_motion_rep(smpl_6d, kernel_size=smooth_window, sigma=1.0)
    return echo_module.rot6d_to_smpl_dict(smpl_6d)


def _process_one_external(
    in_file: Path,
    out_dir: Path,
    sample_id: str,
    smooth_window: int,
    set_floor: bool,
    use_shape: bool,
    device: torch.device,
    smpl_model,
    echo_module,
    process_hmr_motion_fn,
    collect_motion_rep_dart_fn,
    mat3x3_to_axis_angle_fn,
    joint_num: int,
    default_intrinsic: torch.Tensor,
) -> Dict[str, Any]:
    raw = _load_external_smpl(in_file)
    smpl_incam = _normalize_smpl_dict(raw)
    num_frames = smpl_incam["global_orient"].shape[0]
    frame_mask = _resolve_frame_mask(raw, num_frames)
    intrinsic = _resolve_intrinsic(
        raw,
        argparse.Namespace(
            fx=float(default_intrinsic[0, 0].item()),
            fy=float(default_intrinsic[1, 1].item()),
            cx=float(default_intrinsic[0, 2].item()),
            cy=float(default_intrinsic[1, 2].item()),
        ),
    )

    smpl_smooth = _smooth_smpl_for_one_person(
        person_smpl_params=smpl_incam,
        frame_mask=frame_mask,
        smooth_window=smooth_window,
        echo_module=echo_module,
    )

    global_orient_mat = smpl_smooth["global_orient"].to(device)
    body_pose_mat = smpl_smooth["body_pose"].to(device)
    transl = smpl_smooth["transl"].to(device)
    betas = smpl_smooth["betas"]
    betas_device = betas.to(device)
    seq_len = global_orient_mat.shape[0]

    smpl_results = smpl_model(
        global_orient=global_orient_mat,
        body_pose=body_pose_mat,
        transl=transl,
        betas=betas_device,
    )
    joints = smpl_results.joints[:, :joint_num]

    global_orient_aa = mat3x3_to_axis_angle_fn(global_orient_mat.reshape(seq_len, 3, 3))
    body_pose_aa = mat3x3_to_axis_angle_fn(body_pose_mat.reshape(seq_len * 23, 3, 3)).reshape(seq_len, 69)

    smpl_params_aa = {
        "global_orient": global_orient_aa.to(joints.device),
        "body_pose": body_pose_aa.to(joints.device),
        "transl": transl.to(joints.device),
        "betas": betas_device.to(joints.device),
    }

    hmr_motion = collect_motion_rep_dart_fn(smpl_params_aa, joints)
    processed, joints_canonical = process_hmr_motion_fn(
        hmr_motion,
        intrinsic.to(joints.device),
        to_cpu=False,
        set_floor=set_floor,
        collect_local_motion=True,
        use_shape=use_shape,
    )

    sample_raw = out_dir / "CameraHMR_smpl_results" / sample_id
    sample_smooth = out_dir / "CameraHMR_smpl_results_smoothed" / sample_id
    sample_raw.mkdir(parents=True, exist_ok=True)
    sample_smooth.mkdir(parents=True, exist_ok=True)

    # Keep a raw checkpoint for traceability.
    torch.save(
        {
            "smpl_params_incam": smpl_incam,
            "intrinsic": intrinsic.cpu(),
            "frame_mask": frame_mask.cpu(),
            "source_file": str(in_file),
        },
        sample_raw / "smpl_raw.pt",
    )

    post_res = {
        "motion": processed["motion"].detach().cpu(),
        "extrinsic": processed["extrinsic"].detach().cpu(),
        "intrinsic": intrinsic.detach().cpu(),
        "joints_canonical": joints_canonical.detach().cpu(),
        "smpl_params": {
            "global_orient": global_orient_aa.detach().cpu(),
            "body_pose": body_pose_aa.detach().cpu(),
            "betas": betas.detach().cpu(),
        },
        "num_frames": int(num_frames),
        "set_floor": bool(set_floor),
        "use_shape": bool(use_shape),
        "source_file": str(in_file),
    }
    torch.save(post_res, sample_smooth / "motion_postprocess.pt")

    np.savez(
        sample_smooth / "smpls_smoothed_group.npz",
        global_orient=smpl_smooth["global_orient"].detach().cpu().numpy(),
        body_pose=smpl_smooth["body_pose"].detach().cpu().numpy(),
        transl=smpl_smooth["transl"].detach().cpu().numpy(),
        betas=betas.detach().cpu().numpy(),
        intrinsic=intrinsic.detach().cpu().numpy(),
        frame_mask=frame_mask.detach().cpu().numpy(),
        bbox_xyxy=np.zeros((num_frames, 4), dtype=np.float32),
        bbox_conf=np.ones((num_frames,), dtype=np.float32),
        set_floor=np.array([int(set_floor)], dtype=np.int32),
    )

    return {
        "sample_id": sample_id,
        "external_smpl_file": str(in_file),
        "num_frames": num_frames,
        "smpl_npz": f"CameraHMR_smpl_results_smoothed/{sample_id}/smpls_smoothed_group.npz",
        "motion_postprocess": f"CameraHMR_smpl_results_smoothed/{sample_id}/motion_postprocess.pt",
    }


def _precheck_external_file(in_file: Path) -> Dict[str, Any]:
    raw = _load_external_smpl(in_file)
    required = ("global_orient", "body_pose", "transl", "betas")
    missing = [k for k in required if raw.get(k) is None]
    if missing:
        raise ValueError(f"[{in_file.name}] missing required keys: {missing}")

    # Reuse normalization checks to guarantee format compatibility before processing.
    smpl = _normalize_smpl_dict(raw)
    num_frames = int(smpl["global_orient"].shape[0])
    has_intrinsic = raw.get("intrinsic") is not None
    has_frame_mask = raw.get("frame_mask") is not None
    return {
        "file": str(in_file),
        "num_frames": num_frames,
        "has_intrinsic": has_intrinsic,
        "has_frame_mask": has_frame_mask,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process external SMPL with the same transform+smoothing chain as CameraHMR pipeline."
    )
    parser.add_argument("--root_dir", type=str, default="examples/training")
    parser.add_argument("--vendor_root", type=str, default="third_party")
    parser.add_argument(
        "--external_smpl_dir",
        type=str,
        required=True,
        help="Directory containing external SMPL files (.npz/.pt/.pth).",
    )
    parser.add_argument("--glob", type=str, default="*.npz", help="Glob pattern under external_smpl_dir.")
    parser.add_argument("--id_width", type=int, default=6)
    parser.add_argument("--start_id", type=int, default=1, help="First sample id for this run folder.")
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--set_floor", action="store_true")
    parser.add_argument("--use_shape", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fx", type=float, default=1000.0)
    parser.add_argument("--fy", type=float, default=1000.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument(
        "--self_check_confirm",
        action="store_true",
        help=(
            "Mandatory safety confirmation. Set this flag only after you manually verify "
            "external SMPL conventions (axis direction, units, camera intrinsic, frame semantics)."
        ),
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Run compatibility precheck only, write report, and exit without generating outputs.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if args.id_width < 1:
        raise ValueError("--id_width must be >= 1")
    if args.start_id < 0:
        raise ValueError("--start_id must be >= 0")
    if not args.self_check_confirm:
        raise ValueError(
            "--self_check_confirm is required before processing. "
            "Please manually validate external SMPL conventions first (see README)."
        )

    root_dir = Path(args.root_dir).resolve()
    vendor_root = Path(args.vendor_root).resolve()
    external_smpl_dir = Path(args.external_smpl_dir).resolve()
    if not external_smpl_dir.exists():
        raise FileNotFoundError(f"external_smpl_dir not found: {external_smpl_dir}")

    _setup_vendor_imports(vendor_root)
    from scripts.data_processors.motion_alignment.retarget_mogen_db import (  # type: ignore
        rot6d_to_smpl_dict,
        smooth_motion_rep,
        smpl_dict_to_rot6d,
    )
    from common.constants import JOINT_NUM  # type: ignore
    from scripts.data_processors.motion_alignment.seq_utils import linear_interpolate_frame_ids  # type: ignore
    from scripts.data_processors.motion_alignment.retarget_motion import process_hmr_motion  # type: ignore
    from scripts.data_processors.smpl.motion_rep import collect_motion_rep_DART  # type: ignore
    from scripts.data_processors.smpl.rotation_transform import mat3x3_to_axis_angle, axis_angle_to_matrix  # type: ignore

    global _AXIS_ANGLE_TO_MATRIX_FN
    _AXIS_ANGLE_TO_MATRIX_FN = axis_angle_to_matrix

    class EchoModule:
        smooth_motion_rep = staticmethod(smooth_motion_rep)
        smpl_dict_to_rot6d = staticmethod(smpl_dict_to_rot6d)
        rot6d_to_smpl_dict = staticmethod(rot6d_to_smpl_dict)
        linear_interpolate_frame_ids = staticmethod(linear_interpolate_frame_ids)

    smpl_model_path = vendor_root / "extract_motion" / "CameraHMR" / "data" / "models" / "SMPL" / "SMPL_NEUTRAL.pkl"
    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL model not found: {smpl_model_path}")

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    smpl_model = smplx.SMPLLayer(model_path=str(smpl_model_path), num_betas=10).to(device)
    default_intrinsic = torch.tensor(
        [
            [args.fx, 0.0, args.cx],
            [0.0, args.fy, args.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    inputs = sorted([p for p in external_smpl_dir.glob(args.glob) if p.is_file() and p.suffix.lower() in {".npz", ".pt", ".pth"}])
    if not inputs:
        raise FileNotFoundError(f"No external SMPL files found in {external_smpl_dir} with glob: {args.glob}")

    precheck_reports: List[Dict[str, Any]] = []
    for in_file in inputs:
        precheck_reports.append(_precheck_external_file(in_file))

    run_dir = _make_new_run_dir(root_dir)
    with open(run_dir / "precheck_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "external_smpl_dir": str(external_smpl_dir),
                "glob": args.glob,
                "count": len(precheck_reports),
                "items": precheck_reports,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.check_only:
        print(f"Run folder: {run_dir}")
        print(f"Precheck only. Checked files: {len(precheck_reports)}")
        print(f"Precheck report written to: {run_dir / 'precheck_report.json'}")
        return

    outputs: List[Dict[str, Any]] = []
    for i, in_file in enumerate(inputs):
        sample_id = f"{args.start_id + i:0{args.id_width}d}"
        result = _process_one_external(
            in_file=in_file,
            out_dir=run_dir,
            sample_id=sample_id,
            smooth_window=args.smooth_window,
            set_floor=args.set_floor,
            use_shape=args.use_shape,
            device=device,
            smpl_model=smpl_model,
            echo_module=EchoModule,
            process_hmr_motion_fn=process_hmr_motion,
            collect_motion_rep_dart_fn=collect_motion_rep_DART,
            mat3x3_to_axis_angle_fn=mat3x3_to_axis_angle,
            joint_num=JOINT_NUM,
            default_intrinsic=default_intrinsic,
        )
        outputs.append(result)

    mapping = {
        "root_dir": str(root_dir),
        "run_dir": str(run_dir),
        "external_smpl_dir": str(external_smpl_dir),
        "glob": args.glob,
        "count": len(outputs),
        "items": outputs,
    }
    with open(run_dir / "external_smpl_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Run folder: {run_dir}")
    print(f"Processed external SMPL files: {len(outputs)}")
    print(f"Mapping written to: {run_dir / 'external_smpl_mapping.json'}")


if __name__ == "__main__":
    run(build_parser().parse_args())
