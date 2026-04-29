import argparse
import torch
import smplx

from scripts.data_processors.smpl.rotation_transform import mat3x3_to_rot6d, rot6d_to_mat3x3
from core.constants import SMPL_MODEL_PATH 
from ..smpl.constants import NUM_PARAMS

smpl_model = smplx.SMPLLayer(model_path=SMPL_MODEL_PATH, num_betas=10).cuda()

def smpl_dict_to_rot6d(smpl_dict):
    seq_len = smpl_dict['body_pose'].shape[0]
    global_orient_6d = mat3x3_to_rot6d(smpl_dict['global_orient'].reshape(seq_len, 3, 3))  # (seq_len, 6)
    body_pose_6d = mat3x3_to_rot6d(smpl_dict['body_pose'].reshape(seq_len*NUM_PARAMS, 3, 3)).reshape(seq_len, -1)  # (seq_len, 21*6)
    motion_rep = torch.cat([global_orient_6d, body_pose_6d, smpl_dict['transl']], dim=-1)
    return motion_rep

def rot6d_to_smpl_dict(motion_rep):
    seq_len = motion_rep.shape[0]
    global_orient = rot6d_to_mat3x3(motion_rep[:, :6].contiguous()).reshape(seq_len, 1, 3, 3)
    body_pose = rot6d_to_mat3x3(motion_rep[:, 6:-3].contiguous()).reshape(seq_len, NUM_PARAMS, 3, 3)
    transl = motion_rep[:, -3:]
    smpl_dict = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'transl': transl
    }
    return smpl_dict

def gaussian_kernel(kernel_size: int, sigma: float):
    x = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    gauss = torch.exp(-x**2 / (2*sigma**2))
    return gauss / gauss.sum()

def smooth_motion_rep(motion_rep, kernel_size: int, sigma: float):
    assert kernel_size % 2 == 1, 'kernel_size must be odd'
    data_dim = motion_rep.shape[-1]
    padding = (kernel_size - 1) // 2
    kernel = gaussian_kernel(kernel_size, sigma).to(motion_rep.device)[None, None, :].repeat(data_dim, 1, 1)
    motion_rep_smoothed = torch.nn.functional.conv1d(motion_rep.transpose(0, 1).unsqueeze(0), kernel, padding=padding, groups=data_dim)
    motion_rep_smoothed = motion_rep_smoothed.squeeze(0).transpose(0, 1)
    motion_rep_smoothed[:padding] = motion_rep[:padding]
    motion_rep_smoothed[-padding:] = motion_rep[-padding:]
    return motion_rep_smoothed