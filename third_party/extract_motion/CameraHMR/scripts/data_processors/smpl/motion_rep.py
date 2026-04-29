import torch

from .constants import JOINT_NUM
from .rotation_transform import axis_angle_to_rot6d, rot6d_to_axis_angle, axis_angle_to_mat3x3, mat3x3_to_rot6d, rot6d_to_mat3x3, mat3x3_to_axis_angle

def collect_motion_rep_DART(smpl_params, joints, debug=False):
    # follow DART: (https://arxiv.org/pdf/2410.05260), final motion: (seq_len, 276)
    seq_len = smpl_params['transl'].shape[0]
    # global_orient & angle velocity of global_orient
    global_orient_aa = smpl_params['global_orient'].view(seq_len, 3)  # (seq_len, 3)
    rot_R = axis_angle_to_mat3x3(global_orient_aa)  # (seq_len, 3, 3)
    rot_vel = torch.matmul(rot_R[1:], rot_R[:-1].transpose(-1, -2))  # (seq_len-1, 3, 3)
    # Use Rot6D as representation
    rot_6D = mat3x3_to_rot6d(rot_R)  # Shape: (seq_len, 6)
    rot_vel_6D = mat3x3_to_rot6d(rot_vel)  # Shape: (seq_len-1, 6)
    
    # translation & velocity of translation
    trans = smpl_params['transl']  # Shape: (seq_len, 3)
    trans_vel = smpl_params['transl'][1:] - smpl_params['transl'][:-1]  # Shape: (seq_len-1, 3)

    # joints & velocity of joints 
    joints = joints.reshape(seq_len, -1)  # Shape: (seq_len, joints_num*3)
    joints_vel = (joints[1:] - joints[:-1])  # Shape: (seq_len-1, joints_num*3)

    # body poses axis-angle -> Rot6D
    body_poses = smpl_params['body_pose'].reshape(-1, 3)  # Shape: (seq_len*(joints_num-1), 3)
    body_poses = axis_angle_to_rot6d(body_poses).reshape(seq_len, -1)  # Shape: (seq_len, (joints_num-1)*6)

    if debug:
        results = {
            "body_pose": body_poses,                 # body_pose in SMPL (seq_len, (joints_num-1)*6)
            "joints_positions": joints,              # 3D global joints (seq_len, joints_num*3) 
            "joints_velocity": joints_vel,           # velocity of 3D global joints (seq_len-1, joints_num*3)
            "orient_rotation": rot_6D,               # global_orient, (seq_len, 6)
            "orient_velocity": rot_vel_6D,           # velocity of global_orient, (seq_len-1, 6)
            "root_positions": trans,                 # transl, (seq_len, 3)
            "root_velocity": trans_vel,              # velocity of transl, (seq_len-1, 3)
        }

    if smpl_params.get('betas') is not None:
        seq_len_ = len(body_poses[:-1])
        motion = torch.cat([body_poses[:-1], joints[:-1], joints_vel, rot_6D[:-1], rot_vel_6D, trans[:-1], trans_vel, smpl_params['betas'][:seq_len_]], dim=-1)

    else:
        motion = torch.cat([body_poses[:-1], joints[:-1], joints_vel, rot_6D[:-1], rot_vel_6D, trans[:-1], trans_vel], dim=-1)
    return motion


def motion_rep_to_SMPL(motion, recover_from_velocity=False, equal_length=False, fetch_local_motion=True):
    if motion.shape[1] == JOINT_NUM*12+12:    # global motion
        idx_bias = 0
    elif motion.shape[1] == JOINT_NUM*18+30:    # local motion
        if fetch_local_motion:
            idx_bias = JOINT_NUM*6+18
        else:
            idx_bias = 0
    elif motion.shape[1] == JOINT_NUM*12+12+10:    # global motion with betas
        idx_bias = 0
    
    elif motion.shape[1] == JOINT_NUM*18+30+10:    # local motion with betas
        if fetch_local_motion:
            idx_bias = JOINT_NUM*6+18
        else:
            idx_bias = 0
    else:
        raise ValueError(f"get unexpected motion shape: {motion.shape}")
    
    seq_len = motion.shape[0]
    body_poses = motion[:, :(JOINT_NUM-1)*6]    # (seq_len-1, (joints_num-1)*6)
    body_poses = rot6d_to_axis_angle(body_poses.reshape(-1, 6)).reshape(seq_len, -1)    # (seq_len-1, (joints_num-1)*3)
    joints = motion[:, (idx_bias+JOINT_NUM*6-6):(idx_bias+JOINT_NUM*9-6)].reshape(seq_len, -1, 3)    # (seq_len-1, joints_num, 3) 
    joints_vel = motion[:, (idx_bias+JOINT_NUM*9-6):(idx_bias+JOINT_NUM*12-6)].reshape(seq_len, -1, 3)    # (seq_len-1, joints_num, 3)  
    # fetch the global or local motion
    global_orient = motion[:, (idx_bias+JOINT_NUM*12-6):(idx_bias+JOINT_NUM*12)]    # (seq_len-1, 6)
    global_orient = rot6d_to_axis_angle(global_orient)    # (seq_len-1, 3)
    trans = motion[:, (idx_bias+JOINT_NUM*12+6):(idx_bias+JOINT_NUM*12+9)]    # (seq_len-1, 3)
    trans_vel = motion[:, (idx_bias+JOINT_NUM*12+9):(idx_bias+JOINT_NUM*12+12)]
    betas = motion[:, idx_bias+JOINT_NUM*12+12: idx_bias+JOINT_NUM*12+12+10]

    smpl_data = {
        'global_orient': global_orient, 
        'body_pose': body_poses,
        'transl': trans,
        'betas': betas
    }

    if recover_from_velocity or equal_length:
        seq_end = seq_len + 1 if equal_length else seq_len
        # recover the global_orient seq from velocity
        R_first = rot6d_to_mat3x3(motion[0:1, (idx_bias+JOINT_NUM*12-6):(idx_bias+JOINT_NUM*12)])  # (1, 6)
        R_vel = rot6d_to_mat3x3(motion[:, (idx_bias+JOINT_NUM*12):(idx_bias+JOINT_NUM*12+6)])  # (seq_len-1, 6)
        # Recover global orientation by cumulative multiplication of velocities
        R_rec = [R_first]
        for i in range(1, seq_end):
            R_curr = torch.matmul(R_vel[i-1:i], R_rec[i-1])  # (1,3,3) x (1,3,3)
            R_rec.append(R_curr)
        R_rec = torch.cat(R_rec, dim=0) # (seq_len, 3, 3)

        # Similarly, recover translations:
        trans_first_frame = trans[0:1]  # (1,3)
        trans_recovered = [trans_first_frame]
        for i in range(1, seq_end):
            trans_recovered.append(trans_recovered[i-1] + trans_vel[i-1:i])
        trans_recovered = torch.cat(trans_recovered, dim=0)  # (seq_len, 3)

        # recover the joints sequence
        joints_recovered = [joints[0:1]]
        for i in range(1, seq_end):
            joints_recovered.append(joints_recovered[i-1] + joints_vel[i-1:i])
        joints_recovered = torch.cat(joints_recovered, dim=0)  # (seq_len, joints_num, 3)
        
        smpl_data['global_orient'] = mat3x3_to_axis_angle(R_rec)  # (seq_len, 3)
        smpl_data['transl'] = trans_recovered
        if equal_length:
            last_frame_body_pose = smpl_data['body_pose'][-1:]
            smpl_data['body_pose'] = torch.cat([smpl_data['body_pose'], last_frame_body_pose], dim=0)
        joints = joints_recovered

    return smpl_data, joints