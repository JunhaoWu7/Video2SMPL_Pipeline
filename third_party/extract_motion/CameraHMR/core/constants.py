import os

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_CAMERAHMR_ROOT = os.path.dirname(_CORE_DIR)
_DATA_ROOT = os.path.join(_CAMERAHMR_ROOT, 'data')

# 优先从该目录加载扁平放置的权重（与 pipeline/run_pipeline.py --weight_root 一致）
# 设环境变量 VIDEO2SMPL_WEIGHT_ROOT="" 可强制只用仓库内 third_party/.../data/
_LOCAL_ENV = os.environ.get("VIDEO2SMPL_WEIGHT_ROOT")
if _LOCAL_ENV is None:
    _LOCAL_WEIGHT_ROOT = "/data1/wjh/Video2SMPL"
else:
    _LOCAL_WEIGHT_ROOT = _LOCAL_ENV.strip()


def _data_p(path):
    return os.path.join(_DATA_ROOT, path)


def _root_p(path):
    return os.path.join(_CAMERAHMR_ROOT, path)


def _weight(filename: str, fallback_under_data: str) -> str:
    """若本地根目录下存在同名文件则使用，否则回退到 CameraHMR/data/..."""
    if _LOCAL_WEIGHT_ROOT:
        local = os.path.join(_LOCAL_WEIGHT_ROOT, filename)
        if os.path.isfile(local):
            return local
    return _data_p(fallback_under_data)


CHECKPOINT_PATH = _weight("camerahmr_checkpoint_cleaned.ckpt", "pretrained-models/camerahmr_checkpoint_cleaned.ckpt")
CAM_MODEL_CKPT = _weight("cam_model_cleaned.ckpt", "pretrained-models/cam_model_cleaned.ckpt")
SMPL_MEAN_PARAMS_FILE = _weight("smpl_mean_params.npz", "smpl_mean_params.npz")
SMPL_MODEL_PATH = _weight("SMPL_NEUTRAL.pkl", "models/SMPL/SMPL_NEUTRAL.pkl")
DETECTRON_CKPT = _weight("model_final_f05665.pkl", "pretrained-models/model_final_f05665.pkl")
YOLO_WEIGHT_PATH = _weight("yolov8x.pt", "yolo/yolov8x.pt")
DETECTRON_CFG=_root_p('core/utils/cascade_mask_rcnn_vitdet_h_75ep.py')
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

SMPL_to_J19 = _data_p('train-eval-utils/SMPL_to_J19.pkl')
SMPL_MODEL_DIR = _data_p('models/SMPL')