from core.constants import SMPL_MODEL_PATH

JOINT_NUM = 24  # 24 for smpl, 22 for smplx
NUM_PARAMS = JOINT_NUM - 1

face_joint_indx = [2, 1, 17, 16]  # （r_hip, l_hip, sdr_r, sdr_l）
