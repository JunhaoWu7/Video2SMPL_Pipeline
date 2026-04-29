import torch
import torchgeometry as tgm
from torch.nn import functional as F

def mat3x3_to_rot6d(R):
    # rot6d takes the first two columns of R
    return R[..., :, :2].reshape(R.shape[0], 6)

def rot6d_to_mat3x3(rot6d):
    """
    Convert 6d rotation representation to 3x3 rotation matrix.
    Shape:
        - Input: :Torch:`(N, 6)`
        - Output: :Torch:`(N, 3, 3)`
    """
    rot6d = rot6d.view(-1, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix
    return rot_mat
    
def axis_angle_to_rot6d(angle_axis):
    """Convert 3d vector of axis-angle rotation to 6d rotation representation.
    Shape:
        - Input: :Torch:`(N, 3)`
        - Output: :Torch:`(N, 6)`
    """
    rot_mat = tgm.angle_axis_to_rotation_matrix(angle_axis)  # 4x4 rotation matrix
    rot6d = rot_mat[:, :3, :2]
    rot6d = rot6d.reshape(-1, 6)

    return rot6d

def rot6d_to_axis_angle(rot6d):
    """Convert 6d rotation representation to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(N, 6)`
        - Output: :Torch:`(N, 3)`
    """
    batch_size = rot6d.shape[0]

    rot6d = rot6d.view(batch_size, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    # Avoid torchgeometry.rotation_matrix_to_angle_axis (compat bug with torch>=2.1).
    axis_angle = mat3x3_to_axis_angle(rot_mat)
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def axis_angle_to_mat3x3(angle_axis):
    """
    Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.
    Shape:
        - Input: :Torch:`(N, 3)`
        - Output: :Torch:`(N, 3, 3)`
    """
    rot_mat = tgm.angle_axis_to_rotation_matrix(angle_axis)  # 4x4 rotation matrix

    return rot_mat[:, :3, :3]

def mat3x3_to_axis_angle(rot_mat):
    """
    Convert 3x3 rotation matrix to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(N, 3, 3)`
        - Output: :Torch:`(N, 3)`
    """
    # Pure PyTorch rotation matrix -> axis-angle conversion.
    # rot_mat: (N, 3, 3)
    R = rot_mat.float()
    eps = 1e-6

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (N,)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)  # (N,)
    sin_theta = torch.sin(theta)  # (N,)

    # axis numerator: (R - R^T) / 2
    axis_num = torch.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
        dim=-1,
    )  # (N,3)

    denom = 2.0 * sin_theta  # (N,)
    axis = torch.zeros_like(axis_num)  # (N,3)
    mask = denom.abs() > eps  # (N,)
    axis[mask] = axis_num[mask] / denom[mask].unsqueeze(-1)

    axis_angle = axis * theta.unsqueeze(-1)  # (N,3)

    # Handle near-zero sin(theta): either theta ~ 0 (axis-angle ~ 0) or theta ~ pi (axis from diagonal).
    mask_small = ~mask
    if mask_small.any():
        mask_pi = mask_small & (cos_theta < 0)
        if mask_pi.any():
            # axis components from diagonal entries
            ax = torch.sqrt(torch.clamp((R[..., 0, 0] + 1.0) / 2.0, min=0.0))
            ay = torch.sqrt(torch.clamp((R[..., 1, 1] + 1.0) / 2.0, min=0.0))
            az = torch.sqrt(torch.clamp((R[..., 2, 2] + 1.0) / 2.0, min=0.0))
            # recover signs from off-diagonals
            ax = torch.where((R[..., 2, 1] - R[..., 1, 2]) >= 0, ax, -ax)
            ay = torch.where((R[..., 0, 2] - R[..., 2, 0]) >= 0, ay, -ay)
            az = torch.where((R[..., 1, 0] - R[..., 0, 1]) >= 0, az, -az)
            axis_pi = torch.stack([ax, ay, az], dim=-1)
            axis_angle_pi = axis_pi * torch.pi
            axis_angle[mask_pi] = axis_angle_pi[mask_pi]

        # theta ~ 0 already gives near-zero axis_angle; keep as-is (axis_angle is zeros here).

    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def quaternion_to_axis_angle(quaternion):
    """
    Convert 4d quaternion to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(..., 4)`
        - Output: :Torch:`(..., 3)`
    """
    angle_axis = tgm.quaternion_to_angle_axis(quaternion)
    return angle_axis

def axis_angle_to_quaternion(angle_axis):
    """
    Convert 3d vector of axis-angle rotation to 4d quaternion.
    Shape:
        - Input: :Torch:`(..., 3)`
        - Output: :Torch:`(..., 4)`
    """
    quaternion = tgm.angle_axis_to_quaternion(angle_axis)
    return quaternion

def quaternion_to_rot6d(quaternion):
    """
    Convert 4d quaternion to 6d rotation representation.
    Shape:
        - Input: :Torch:`(N, 4)`
        - Output: :Torch:`(N, 6)`
    """

    return axis_angle_to_rot6d(quaternion_to_axis_angle(quaternion))

def rot6d_to_quaternion(rot6d):
    """
    Convert 6d rotation representation to 4d quaternion.
    Shape:
        - Input: :Torch:`(N, 4)`
        - Output: :Torch:`(N, 6)`
    """

    return axis_angle_to_quaternion(rot6d_to_axis_angle(rot6d))