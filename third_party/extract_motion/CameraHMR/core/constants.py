import os

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_CAMERAHMR_ROOT = os.path.dirname(os.path.dirname(_CORE_DIR))
_DATA_ROOT = os.path.join(_CAMERAHMR_ROOT, 'data')

def _p(path):
    return os.path.join(_DATA_ROOT, path)

CHECKPOINT_PATH=_p('pretrained-models/camerahmr_checkpoint_cleaned.ckpt')
CAM_MODEL_CKPT=_p('pretrained-models/cam_model_cleaned.ckpt')
SMPL_MEAN_PARAMS_FILE=_p('smpl_mean_params.npz')
SMPL_MODEL_PATH=_p('models/SMPL/SMPL_NEUTRAL.pkl')
DETECTRON_CKPT=_p('pretrained-models/model_final_f05665.pkl')
DETECTRON_CFG=_p('core/utils/cascade_mask_rcnn_vitdet_h_75ep.py')
TRANSFORMER_DECODER={'depth': 6,
                    'heads': 8,
                    'mlp_dim': 1024,
                    'dim_head': 64,
                    'dropout': 0.0,
                    'emb_dropout': 0.0,
                    'norm': 'layer',
                    'context_dim': 1280}

IMAGE_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
NUM_POSE_PARAMS = 23
NUM_PARAMS_SMPL = 24
NUM_BETAS = 10

smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                    7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

SMPL_to_J19 = _p('train-eval-utils/SMPL_to_J19.pkl')
SMPL_MODEL_DIR=_p('models/SMPL')