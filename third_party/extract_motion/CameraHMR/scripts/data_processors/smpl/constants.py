import os

JOINT_NUM = 24 # 24 for smpl, 22 for smplx
NUM_PARAMS = JOINT_NUM - 1

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
_CAMERAHMR_ROOT = os.path.dirname(os.path.dirname(_SCRIPTS_DIR))
SMPL_MODEL_PATH = os.path.join(_CAMERAHMR_ROOT, 'data', 'models', 'SMPL', 'SMPL_NEUTRAL.pkl')

face_joint_indx = [2, 1, 17, 16]  # （r_hip, l_hip, sdr_r, sdr_l）