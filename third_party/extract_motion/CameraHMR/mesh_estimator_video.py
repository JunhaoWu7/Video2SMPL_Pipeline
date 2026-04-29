import tempfile
import cv2
import os
import json
import torch
import smplx
import trimesh
import numpy as np
from glob import glob
from torchvision.transforms import Normalize

from core.camerahmr_model import CameraHMR
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH
from core.datasets.dataset import Dataset_video
from core.utils import recursive_to
from core.cam_model.fl_net import FLNet
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS


def read_frames_from_bytes(data_bytes):
    # Create a temporary file to store the video data
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(data_bytes)  # Write bytes to the temp file
        temp_file_path = temp_file.name  # Get the temp file path

    # Read frames from the temporary video file
    return read_frames(temp_file_path)

def read_frames(video_path, max_frames=300):
    """
    Reads frames from a video file.
    MODIFIED: Added a `max_frames` parameter to stop reading after a certain number of frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    frame_count = 0
    
    while cap.isOpened():
        if max_frames is not None and frame_count >= max_frames:
            print(f"INFO: Reached max_frames limit ({max_frames}). Stopped reading frames from {os.path.basename(video_path)}.")
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        frame_count += 1
        
    cap.release()
    return frames, fps

def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

class HumanMeshEstimator:
    def __init__(self, smpl_model_path=SMPL_MODEL_PATH):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = self.init_model()
        self.smpl_model = smplx.SMPLLayer(model_path=smpl_model_path, num_betas=NUM_BETAS).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT)['state_dict']
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
        model = model.to(self.device)
        model.eval()
        return model
        
    def convert_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]  # Each has shape [32]

        # Calculate tz without expanding
        tz = 2. * focal_length / (bbox_height * s)  # Shape [32]
        
        # Remove unsqueeze(-1) to maintain shape [32]
        cx = 2. * (bbox_center[..., 0] - img_w / 2.) / (s * bbox_height)  # Shape [32]
        cy = 2. * (bbox_center[..., 1] - img_h / 2.) / (s * bbox_height)  # Shape [32]
        
        # Stack directly since all tensors are [32]
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)  # Shape [32, 3]
        return cam_t

    def get_output_trans(self, pred_cam, batch):
        img_h, img_w = batch['img_size'].unbind(-1)  # Handle batch dimension
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][..., 0, 0]
        )
        return cam_trans

    def get_output_mesh(self, params):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices

        return pred_vertices, pred_keypoints_3d
    
    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        fl_h = (img_w * img_w + img_h * img_h) ** 0.5
        cam_int = np.array([[fl_h, 0, img_w/2], [0, fl_h, img_h / 2], [0, 0, 1]]).astype(np.float32)
        return cam_int

    def remove_pelvis_rotation(self, smpl):
        """We don't trust the body orientation coming out of bedlam_cliff, so we're just going to zero it out."""
        smpl.body_pose[0][0][:] = np.zeros(3)

    def process_images(self, img_cv2_list, bbox_center_list, bbox_scale_list, cam_int_list, batch_size):
        """Process a batch of images and return SMPL parameters"""
        dataset = Dataset_video(img_cv2_list, bbox_center_list, bbox_scale_list, cam_int_list)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_smpl_params = {
            'global_orient': [], 'body_pose': [], 'betas': [], 'transl': []
        }
        all_verts = []
        all_keypoints = []
        
        # import ipdb; ipdb.set_trace()
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length = self.model(batch)
                
                # Get mesh outputs
                output_cam_trans = self.get_output_trans(out_cam, batch).float()
                output_verts, output_keypoints = self.get_output_mesh(out_smpl_params)

                # Collect parameters
                for k in ['global_orient', 'body_pose', 'betas']:
                    all_smpl_params[k].append(out_smpl_params[k])
                all_smpl_params['transl'].append(output_cam_trans)
                all_verts.append(output_verts)
                all_keypoints.append(output_keypoints)
                
        # Concatenate results
        for k in all_smpl_params.keys():
            all_smpl_params[k] = torch.cat(all_smpl_params[k], dim=0).cpu()
        all_verts = torch.cat(all_verts, dim=0)
        all_keypoints = torch.cat(all_keypoints, dim=0).cpu()
        return all_smpl_params, focal_length.cpu(), all_verts.cpu(), all_keypoints.cpu()

    def run_on_images(self, video_path, bbox_path, out_path, batch_size=64, save_verts=False):
        # Load video frames
        if isinstance(video_path, str):
            frames, fps = read_frames(video_path)
        else:
            frames, fps = read_frames_from_bytes(video_path)
        width, height = frames[0].shape[1], frames[0].shape[0]

        # Load bbox results -> (num_frames, num_humans, 4) 
        try:
            bbox_results = torch.load(bbox_path, weights_only=True)
            bbox_xyxy = bbox_results["bbx_xyxy"].transpose(0, 1).cpu().numpy()
            original_num_frames = bbox_xyxy.shape[0]
            num_frames_to_load = min(original_num_frames, 300)
            bbox_xyxy = bbox_xyxy[:num_frames_to_load, :, :]  # (num_humans, num_frames, 4)
            
            
            num_frames, num_humans = bbox_xyxy.shape[0], bbox_xyxy.shape[1]

            # Prepare inputs
            bbox_scale_list = (bbox_xyxy[..., 2:4] - bbox_xyxy[..., 0:2]) / 200.0 
            bbox_center_list = (bbox_xyxy[..., 2:4] + bbox_xyxy[..., 0:2]) / 2.0
            cam_int_list = [self.get_cam_intrinsics(frame) for frame in frames]

            # Process all frames
            all_smpl_params, focal_length, all_verts = self.process_images(
                frames, bbox_center_list, bbox_scale_list, cam_int_list, batch_size
            )

            # Reshape results to (num_humans, num_frames, ...)
            def chunk_first_axis(tensor, person_num):
                return tensor.reshape(-1, person_num, *tensor.shape[1:])

            for k in all_smpl_params:
                all_smpl_params[k] = chunk_first_axis(all_smpl_params[k], num_humans).transpose(0, 1)
            all_verts = chunk_first_axis(all_verts, num_humans).transpose(0, 1)
            cam_t = all_smpl_params['transl'][:, :, None, :]
            
            # Save results
            final_dict = {
                'smpl_params_incam': all_smpl_params,
                'focal_length': focal_length[0].repeat(num_frames),
                'width': width * torch.ones(num_frames, dtype=torch.int32),
                'height': height * torch.ones(num_frames, dtype=torch.int32),
                'fps': fps
            }
            if save_verts:
                final_dict['verts'] = all_verts
                final_dict['cam_t'] = cam_t
            torch.save(final_dict, out_path)
        except Exception as e:
            print(f"[ERROR] {e}")
