import os
import sys
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from scripts.data_processors.smpl.rotation_transform import rot6d_to_mat3x3, axis_angle_to_mat3x3, mat3x3_to_axis_angle, mat3x3_to_rot6d
from scripts.data_processors.smpl.motion_rep import collect_motion_rep_DART, motion_rep_to_SMPL
from scripts.data_processors.smpl.transforms import get_transform_DART

from ..smpl.constants import JOINT_NUM

smplx_root = torch.load(os.path.join(_SCRIPT_DIR, "smplx_root.pt"), weights_only=True)

def is_rotation_matrix(R):
    identity_matrix = torch.eye(3)
    R_transpose = torch.transpose(R, 0, 1)
    is_orthogonal = torch.allclose(torch.matmul(R, R_transpose), identity_matrix, atol=1e-6)
    
    det_R = torch.det(R)
    is_det_one = torch.isclose(det_R, torch.tensor(1.0), atol=1e-6)
    
    return is_orthogonal and is_det_one

def apply_rotation(smpl_params, R):
    """
    Update SMPL parameters based on the rotation matrix R.

    Args:
        smpl_params (dict): Dictionary containing SMPL parameters ('global_orient', 'transl', etc.).
        R (torch.Tensor): Rotation matrix of shape (3, 3).

    Returns:
        smpl_params (dict): Updated SMPL parameters.
    """
    N = smpl_params['global_orient'].shape[0]  # Number of frames
    device = smpl_params['global_orient'].device

    # Convert global_orient from axis-angle to 3x3 matrix
    global_orient_mat = axis_angle_to_mat3x3(smpl_params['global_orient'].view(-1, 3))  # Shape: (N, 3, 3)

    # Adjust the global orientation by the computed rotation
    adjusted_global_orient_mat = torch.matmul(R[None,], global_orient_mat)  # Shape: (N, 3, 3)

    # Convert adjusted global_orient back to axis-angle
    smpl_params['global_orient'] = mat3x3_to_axis_angle(adjusted_global_orient_mat)  # Shape: (N, 3)

    # Adjust the translation by rotating
    smpl_params['transl'] += smplx_root.to(device)
    smpl_params['transl'] = torch.matmul(R[None,], smpl_params['transl'][..., None]).squeeze(-1)
    smpl_params['transl'] -= smplx_root.to(device)

    return smpl_params

def canonicalize_motion(smpl_params, joints, set_floor=False, debug=False, smpl_model=None, use_shape=False):
    # Get transformation and update smpl_params
    R_inv = get_transform_DART(joints)
    aligned_smpl_params = apply_rotation(smpl_params, R_inv)
    joints_base = torch.matmul(R_inv[None, None, :, :], joints.unsqueeze(-1)).squeeze(-1)
    
    delta_transl = -joints_base[0, 0:1]  # fetch the pelvis joint from first frame, Shape: (1,3)
    if set_floor:   
        # For gravity axis (Z), set the minimum z-coordinate to 0 as the floor
        delta_transl[0, 2] = - torch.min(joints_base[..., 2])
    joints = joints_base + delta_transl[None,]
    aligned_smpl_params['transl'] += delta_transl

    # Convert to motion representation
    # return without betas
    motion = collect_motion_rep_DART(aligned_smpl_params, joints)  # 276-dim representation
    if use_shape:
        motion = motion[:, :-10]

    # Detached copy for export (axis-angle global_orient/body_pose, canonical transl, betas)
    smpl_params_canonical = {
        k: (v.detach().clone() if torch.is_tensor(v) else v)
        for k, v in aligned_smpl_params.items()
    }
    return motion, joints[:-1], R_inv, delta_transl, smpl_params_canonical

def process_hmr_motion(hmr_motion, intrinsic, to_cpu=True, set_floor=False, collect_local_motion=False, use_shape=False):
    new_data = {}
    device = hmr_motion.device
    seq_len = hmr_motion.shape[0]
    # Step1: hmr -> amass
    R_motionx_to_amass = torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float32, device=device)
    smpl_params, original_joints = motion_rep_to_SMPL(hmr_motion, equal_length=True)
    joints_amass = torch.matmul(R_motionx_to_amass[None, None, :, :], original_joints.unsqueeze(-1)).squeeze(-1)

    # return without betas
    smpl_params_amass = apply_rotation(smpl_params, R_motionx_to_amass) 
    
    if collect_local_motion and use_shape:
        local_motion = collect_motion_rep_DART(smpl_params_amass, joints_amass)[:, -(JOINT_NUM*6+18+10):]
    elif collect_local_motion and not use_shape:
        local_motion = collect_motion_rep_DART(smpl_params_amass, joints_amass)[:, -(JOINT_NUM*6+18):]

    # Step2: amass -> dart
    aligned_motion, joints_canonical, R_inv, delta_transl, smpl_params_canonical = canonicalize_motion(
        smpl_params_amass, joints_amass, set_floor=set_floor, debug=False, use_shape=use_shape
    )
    new_data['motion'] = aligned_motion.detach()
    new_data['smpl_params_canonical'] = smpl_params_canonical
    if collect_local_motion:
        new_data['motion'] = torch.cat([new_data['motion'], local_motion], dim=-1)
    # rotation
    extrinsic_R = torch.tensor([[1.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0],
                              [0.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    extrinsic_R = mat3x3_to_rot6d(torch.matmul(R_inv, extrinsic_R)[None,]).repeat(seq_len, 1)
    # translation
    extrinsic_T = - torch.matmul(delta_transl, R_inv)[:, [0,2,1]]
    extrinsic_T[0, 2] *= -1
    extrinsic_T = extrinsic_T.repeat(seq_len, 1)
    
    extrinsic = torch.cat([extrinsic_R, extrinsic_T], dim=-1)
    new_data['extrinsic'] = extrinsic.detach()
    new_data['intrinsic'] = intrinsic.detach()
    if to_cpu:
        cpu_data = {}
        for k, v in new_data.items():
            if k == 'smpl_params_canonical':
                cpu_data[k] = {
                    kk: vv.cpu() if torch.is_tensor(vv) else vv
                    for kk, vv in v.items()
                }
            else:
                cpu_data[k] = v.cpu() if torch.is_tensor(v) else v
        new_data = cpu_data
    
    return new_data, joints_canonical

'''
Pytorch3D coordinate system:
            Y     Z
            |    /
            |   /
            |  /
            | /
X<----------  
'''

# used for amass coordinate system
def perspective_projection(points, cam_extrinsic=None, cam_intrinsics=None):
    """
    Perform perspective projection in batch.

    Args:
        points (torch.Tensor): 3D points, shape (B, N, 3)
        cam_extrinsic (torch.Tensor): Camera extrinsic matrix, shape (B, 9)
        cam_intrinsics (torch.Tensor): Camera intrinsic matrix, shape (3, 3)

    Returns:
        torch.Tensor: Projected 2D points, shape (B, N, 2)
    """
    bs = points.shape[0]
    if cam_extrinsic is None:
        rotation = torch.tensor([[[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]]], dtype=torch.float32, device=points.device).repeat(bs, 1, 1)
        translation = torch.zeros(bs, 3, device=points.device)
    else:
        rotation, translation = cam_extrinsic.split([6, 3], dim=-1)  # (B, 6), (B, 3)
        rotation = rot6d_to_mat3x3(rotation)    # (B, 3, 3)
    if cam_intrinsics is None:
        cam_intrinsics = torch.tensor([[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]], dtype=torch.float32, device=points.device)
    points = torch.matmul(points, rotation) + translation[:, None, :]
    points[..., 2] = - points[..., 2]
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = torch.matmul(projected_points, cam_intrinsics[None,].transpose(-1, -2))[..., :2]
    return projected_points