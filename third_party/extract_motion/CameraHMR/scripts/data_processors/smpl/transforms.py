import copy

import torch
from .rotation_transform import mat3x3_to_axis_angle, axis_angle_to_mat3x3
from .constants import face_joint_indx, JOINT_NUM
from .motion_rep import collect_motion_rep_DART, motion_rep_to_SMPL


'''         Z
            |
            |
            |
            |
            -----------> X
          /
        /
      / 
    Y
'''
def get_transform_DART(joints):
    """
    Compute the translation and rotation to align the SMPL-X model output to the canonical coordinate frame.

    Args:
        joints (torch.Tensor): SMPL-X joints of shape (seq_len, joints_num, 3).

    Returns:
        delta_transl (torch.Tensor): Translation vector of shape (3,).
        root_quat_init (torch.Tensor): Rotation quaternion of shape (1, 4).
    """
    # Indices of the relevant joints (ensure these are correct for SMPL-X)
    pelvis_index = 0      # Pelvis (root joint)
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx    # Right hip, Left hip, Right Shoulder, Left Shoulder

    device = joints.device

    # First frame joints positions
    joints_0 = joints[0]  # Shape: (joints_num, 3)

    # Step 1: Compute Rotation Quaternion

    # Compute x_axis (from left hip to right hip, projected onto xy-plane)
    x_axis = (joints_0[r_hip] - joints_0[l_hip])    # Shape: (3,)
    x_axis[2] = 0   # Project to the xy-plane (set z-component to zero)
    x_axis = x_axis / torch.norm(x_axis)  # Normalize

    # z_axis is pointing upwards (inverse gravity direction)
    z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

    # Compute y_axis as the cross product of z_axis and x_axis
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / torch.norm(y_axis)  # Normalize

    # Build rotation matrix R (from world frame to canonical frame)
    R_inv = torch.stack([x_axis, y_axis, z_axis], dim=1).T  # Shape: (3, 3)

    return R_inv


def update_smpl_params(smpl_params, R_inv):
    """
    Update SMPL parameters to align the model output to the canonical coordinate frame.

    Args:
        smpl_params (dict): Dictionary containing SMPL parameters ('global_orient', 'transl', etc.).
        R_inv (torch.Tensor): Rotation matrix of shape (3, 3).

    Returns:
        smpl_params (dict): Updated SMPL parameters.
    """
    N = smpl_params['global_orient'].shape[0]  # Number of frames

    # Convert global_orient from axis-angle to 3x3 matrix
    global_orient_mat = axis_angle_to_mat3x3(smpl_params['global_orient'].view(-1, 3))  # Shape: (N, 3, 3)

    # Adjust the global orientation by the computed rotation
    adjusted_global_orient_mat = torch.matmul(R_inv[None,], global_orient_mat)  # Shape: (N, 3, 3)

    # Convert adjusted global_orient back to axis-angle
    smpl_params['global_orient'] = mat3x3_to_axis_angle(adjusted_global_orient_mat)  # Shape: (N, 3)

    # Adjust the translation by rotating
    smpl_params['transl'] = torch.matmul(R_inv[None, :, :], smpl_params['transl'].unsqueeze(-1)).squeeze(-1)  # Shape: (N, 3)

    return smpl_params


def process_motion(smpl_params, smpl_model, joints, orient_align):
    if orient_align:
        # Get transformation and update smpl_params
        R_inv = get_transform_DART(joints)
        aligned_smpl_params = update_smpl_params(smpl_params, R_inv)

        # Apply SMPL Model again for newly aligned data
        model_output = smpl_model(**aligned_smpl_params)
        verts = model_output.vertices
        joints = model_output.joints[:, :JOINT_NUM]  # fetch only 22 body joints.
        delta_transl = -joints[0, 0:1]  # fetch the pelvis joint from first frame, Shape: (1,3)

        # Add transl finally
        verts += delta_transl[None,]
        joints += delta_transl[None,]
        aligned_smpl_params['transl'] += delta_transl
        motion = collect_motion_rep_DART(aligned_smpl_params, joints)   # 276-dim representation
        return motion, verts[:-1], joints[:-1]
    else:
        # Convert to motion representation
        motion = collect_motion_rep_DART(smpl_params, joints)   # 276-dim representation

        return motion