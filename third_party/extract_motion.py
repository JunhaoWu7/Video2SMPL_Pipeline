"""
Motion Extraction Pipeline for EchoMotion.

Extracts SMPL motion parameters from video and converts to EchoMotion motion representation.
Output format: dual motion_w_smpl_motion.pt containing both raw SMPL params and processed motion.

Pipeline:
    1. Bbox Extraction (YOLO tracking)
    2. CameraHMR Processing (SMPL parameter estimation)
    3. Post-processing (smoothing, canonicalization, coordinate transform)

Usage:
    python3 extract_motion.py --video_path dataset/demo/videos/demo_video_1.mp4 --output_path outputs/extract_motion/motion_1.pt --use_shape --visualize --overwrite
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

import torch
import numpy as np
import cv2
import smplx

_CAMERAHMR_ROOT = os.path.join(os.path.dirname(__file__), 'extract_motion', 'CameraHMR')
if _CAMERAHMR_ROOT not in sys.path:
    sys.path.insert(0, _CAMERAHMR_ROOT)

_CAMERAHMR_SCRIPTS = os.path.join(_CAMERAHMR_ROOT, 'scripts')
if _CAMERAHMR_SCRIPTS not in sys.path:
    sys.path.insert(0, _CAMERAHMR_SCRIPTS)

SMPL_MODEL_PATH = os.path.join(_CAMERAHMR_ROOT, 'data', 'models', 'SMPL', 'SMPL_NEUTRAL.pkl')

from common.constants import JOINT_NUM
from dataset.utils import load_motion
from vis_motion.pyrender_checker import motion_vis_during_validation

from mesh_estimator_video import HumanMeshEstimator, read_frames
from bbox_preprocess.tracker import Tracker
from scripts.data_processors.smpl.rotation_transform import mat3x3_to_axis_angle
from scripts.data_processors.smpl.motion_rep import collect_motion_rep_DART
from scripts.data_processors.motion_alignment.retarget_motion import process_hmr_motion
from scripts.data_processors.motion_alignment.retarget_mogen_db import (
    smooth_motion_rep, smpl_dict_to_rot6d, rot6d_to_smpl_dict
)
from scripts.data_processors.motion_alignment.seq_utils import (
    get_frame_id_list_from_mask, linear_interpolate_frame_ids
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionExtractor:
    """
    Extract SMPL motion from video and convert to EchoMotion representation.
    """
    
    def __init__(
        self,
        smpl_model_path: str = SMPL_MODEL_PATH,
        device: Optional[torch.device] = None,
        max_frames: int = 300,
        batch_size: int = 32,
        use_shape: bool = False,
        overwrite: bool = False,
    ):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.use_shape = use_shape
        self.overwrite = overwrite
        
        logger.info(f"Initializing MotionExtractor on device: {self.device}, use_shape: {self.use_shape}, overwrite: {self.overwrite}")
        
        self.mesh_estimator = HumanMeshEstimator(smpl_model_path=smpl_model_path)
        self.tracker = Tracker()
        
        SMPL_MODEL_PATH = os.path.join(_CAMERAHMR_ROOT, 'data', 'models', 'SMPL', 'SMPL_NEUTRAL.pkl')
        logger.info(f"Loading SMPL model from: {SMPL_MODEL_PATH}")
        self.smpl_model = smplx.SMPLLayer(model_path=SMPL_MODEL_PATH, num_betas=10).to(self.device)
        
    def extract_bbox(self, video_path: str, output_path: Optional[str] = None, overwrite: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 1: Extract bounding boxes using YOLO tracking.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save bbox results
            overwrite: Whether to overwrite existing results
            
        Returns:
            bbx_xyxy: (F, 4) bounding box coordinates
            conf: (F,) detection confidence scores
            mask: (F,) frame mask indicating valid detections
        """
        if output_path and os.path.exists(output_path) and not overwrite:
            logger.info(f"Loading existing bbox from: {output_path}")
            results = torch.load(output_path, weights_only=True)
            return results["bbx_xyxy"], results["bbx_conf"], results["frame_mask"]
        
        logger.info(f"Extracting bboxes from: {video_path}")
        bbx_xyxy, conf, mask = self.tracker.get_one_track(video_path)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                "bbx_xyxy": bbx_xyxy.detach().cpu(),
                "bbx_conf": conf.detach().cpu(),
                "frame_mask": mask.detach().cpu()
            }, output_path)
            logger.info(f"Saved bbox to: {output_path}")
        
        return bbx_xyxy, conf, mask
    
    def extract_smpl(
        self,
        video_path: str,
        bbox_path: str,
        output_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Stage 2: Run CameraHMR to extract SMPL parameters.
        
        Args:
            video_path: Path to input video
            bbox_path: Path to bbox results
            output_path: Optional path to save SMPL results
            
        Returns:
            Dictionary containing SMPL parameters and metadata
        """
        if output_path and os.path.exists(output_path) and not overwrite:
            logger.info(f"Loading existing SMPL results from: {output_path}")
            return torch.load(output_path, weights_only=True)
        elif output_path and os.path.exists(output_path) and overwrite:
            logger.info(f"Overwrite mode: Re-running SMPL extraction...")
        
        logger.info(f"Running CameraHMR on: {video_path}")
        
        frames, fps = read_frames(video_path, max_frames=self.max_frames)
        logger.info(f"Loaded {len(frames)} frames from video")
        
        width, height = frames[0].shape[1], frames[0].shape[0]
        logger.info(f"Video resolution: {width}x{height}")
        
        bbox_results = torch.load(bbox_path, weights_only=True)
        bbx_xyxy = bbox_results["bbx_xyxy"].cpu()
        
        if bbx_xyxy.dim() == 2:
            bbx_xyxy = bbx_xyxy.unsqueeze(0)
        
        bbx_xyxy = bbx_xyxy.transpose(0, 1).numpy()
        
        original_num_frames = bbx_xyxy.shape[0]
        num_frames = min(original_num_frames, self.max_frames)
        bbx_xyxy = bbx_xyxy[:num_frames, :, :]
        
        num_humans = bbx_xyxy.shape[1]
        
        bbox_scale = (bbx_xyxy[..., 2:4] - bbx_xyxy[..., 0:2]) / 200.0
        bbox_center = (bbx_xyxy[..., 2:4] + bbx_xyxy[..., 0:2]) / 2.0
        cam_int_list = [self.mesh_estimator.get_cam_intrinsics(frame) for frame in frames]
        
        logger.info(f"Running mesh_estimator.process_images with {num_frames} frames, {num_humans} humans, batch_size={self.batch_size}")
        
        all_smpl_params, focal_length, all_verts, all_keypoints = self.mesh_estimator.process_images(
            frames[:num_frames], bbox_center, bbox_scale, cam_int_list, self.batch_size
        )
        
        
        def chunk_first_axis(tensor, person_num):
            return tensor.reshape(-1, person_num, *tensor.shape[1:])
        
        for k in all_smpl_params:
            all_smpl_params[k] = chunk_first_axis(all_smpl_params[k], num_humans).transpose(0, 1)
        all_verts = chunk_first_axis(all_verts, num_humans).transpose(0, 1)
        all_keypoints = chunk_first_axis(all_keypoints, num_humans).transpose(0, 1)
        
        focal = focal_length[0].item()
        intrinsic = torch.tensor(
            [[focal, 0.0, width / 2.0],
             [0.0, focal, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32
        )
        
        results = {
            'smpl_params_incam': all_smpl_params,
            'focal_length': focal_length[0].repeat(num_frames),
            'width': width * torch.ones(num_frames, dtype=torch.int32),
            'height': height * torch.ones(num_frames, dtype=torch.int32),
            'fps': fps,
            'verts': all_verts,
            'joints': all_keypoints,
            'cam_t': all_smpl_params['transl'][:, :, None, :],
            'num_frames': num_frames,
            'num_humans': num_humans,
            'intrinsic': intrinsic,
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(results, output_path)
            logger.info(f"Saved SMPL results to: {output_path}")
        
        return results
    
    def post_process(
        self, 
        smpl_data: Dict[str, Any], 
        smooth_window: int = 5, 
        set_floor: bool = True,
        frame_mask: Optional[torch.Tensor] = None,
        use_shape: bool = False,
    ) -> Dict[str, Any]:
        """
        Stage 3: Post-process SMPL data - smoothing and coordinate transformation.
        
        Uses process_hmr_motion for coordinate transform and canonicalization.
        
        Args:
            smpl_data: Dictionary containing SMPL parameters from CameraHMR
            smooth_window: Window size for smoothing (0 to disable)
            set_floor: Whether to set floor alignment
            frame_mask: Optional tensor indicating valid frames for interpolation
            use_shape: Whether to include shape parameters in motion representation
            
        Returns:
            Dictionary containing processed motion in EchoMotion format
        """
        logger.info("Post-processing motion data...")
        
        smpl_params = smpl_data['smpl_params_incam']
        
        num_persons = smpl_params['global_orient'].shape[0]
        device = smpl_params['global_orient'].device
        
        if 'intrinsic' in smpl_data and smpl_data['intrinsic'] is not None:
            intrinsic = smpl_data['intrinsic'].to(device)
            logger.info(f"Using pre-computed intrinsic from CameraHMR output")
        else:
            focal = smpl_data.get('focal_length')
            width = smpl_data.get('width')
            height = smpl_data.get('height')
            if focal is not None and width is not None and height is not None:
                if isinstance(focal, torch.Tensor):
                    focal = focal[0].item() if focal.dim() > 0 else focal.item()
                if isinstance(width, torch.Tensor):
                    width = width[0].item() if width.dim() > 0 else width.item()
                if isinstance(height, torch.Tensor):
                    height = height[0].item() if height.dim() > 0 else height.item()
                intrinsic = torch.tensor(
                    [[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                    device=device
                )
                logger.info(f"Constructed intrinsic from CameraHMR: focal={focal}, resolution={width}x{height}")
            else:
                intrinsic = torch.tensor([[1000, 0, 512], [0, 1000, 512], [0, 0, 1]], 
                                        dtype=torch.float32, device=device)
                logger.warning("Could not get focal length from CameraHMR, using fixed intrinsic")
        
        all_motions = []
        all_joints_canonical = []
        all_extrinsics = []
        all_smpl_params = []
        
        for person_idx in range(num_persons):
            person_smpl_params = {k: v[person_idx] for k, v in smpl_params.items()}
            
            seq_len_person = person_smpl_params['global_orient'].shape[0]
            NUM_PARAMS = JOINT_NUM - 1
            
            if frame_mask is not None:
                if len(frame_mask) > seq_len_person:
                    frame_mask = frame_mask[:seq_len_person]
                elif len(frame_mask) < seq_len_person:
                    pad_len = seq_len_person - len(frame_mask)
                    frame_mask = torch.cat([frame_mask, torch.ones(pad_len, dtype=torch.bool, device=device)])
                missing_frame_id_list = get_frame_id_list_from_mask(~frame_mask.bool())
            else:
                missing_frame_id_list = []

            betas = person_smpl_params.get('betas')
            if betas is not None:
                betas = betas[:seq_len_person] if betas.dim() > 1 else betas
            
            smpl_params_for_smoothing = {
                'body_pose': person_smpl_params['body_pose'][:, :NUM_PARAMS].to(device), # (seq_len, 23, 3, 3)
                'global_orient': person_smpl_params['global_orient'].to(device), # (seq_len, 1, 3, 3)
                'transl': person_smpl_params['transl'].to(device), # (seq_len, 3)
                'betas': betas.to(device) if betas is not None else None, # (10, )
            }
            
            # run interpolation and smoothness for the above smpl_params
            smpl_params_6d = smpl_dict_to_rot6d(smpl_params_for_smoothing)
            
            if missing_frame_id_list:
                smpl_params_6d = linear_interpolate_frame_ids(smpl_params_6d, missing_frame_id_list)
            
            if smooth_window > 0:
                smpl_params_6d = smooth_motion_rep(smpl_params_6d, kernel_size=smooth_window, sigma=1.0)
            
            smpl_params_smooth = rot6d_to_smpl_dict(smpl_params_6d)
            
            all_smpl_params.append(smpl_params_smooth)
        
        for person_idx in range(num_persons):
            smpl_params_smooth = all_smpl_params[person_idx]
            betas = smpl_params_smooth.get('betas')
            
            seq_len = smpl_params_smooth['global_orient'].shape[0]
            device = self.device
            
            global_orient_mat = smpl_params_smooth['global_orient'].to(device)
            body_pose_mat = smpl_params_smooth['body_pose'].to(device)
            transl = smpl_params_smooth['transl'].to(device)
            betas_device = betas.to(device) if betas is not None else None
            
            smpl_params_mat = {
                'global_orient': global_orient_mat,
                'body_pose': body_pose_mat,
                'transl': transl,
                'betas': betas_device
            }
            
            smpl_results = self.smpl_model(**smpl_params_mat)
            joints = smpl_results.joints[:, :JOINT_NUM]
            
            global_orient_aa = mat3x3_to_axis_angle(global_orient_mat.reshape(seq_len, 3, 3))
            body_pose_aa = mat3x3_to_axis_angle(body_pose_mat.reshape(seq_len * 23, 3, 3)).reshape(seq_len, 69)
            
            joints_device = joints.device
            smpl_params_aa = {
                'global_orient': global_orient_aa.to(joints_device),
                'body_pose': body_pose_aa.to(joints_device),
                'transl': smpl_params_smooth['transl'].to(joints_device),
                'betas': betas.to(joints_device) if betas is not None else None
            }
            
            hmr_motion = collect_motion_rep_DART(smpl_params_aa, joints)
            
            processed, joints_canonical_person = process_hmr_motion(
                hmr_motion, 
                intrinsic, 
                to_cpu=False, 
                set_floor=set_floor, 
                collect_local_motion=True, 
                use_shape=use_shape
            )
            
            motion_person = processed['motion']
            extrinsic_person = processed['extrinsic']
            
            all_motions.append(motion_person)
            all_joints_canonical.append(joints_canonical_person)
            all_extrinsics.append(extrinsic_person)
        
        if num_persons == 1:
            motion = all_motions[0]
            joints_canonical = all_joints_canonical[0]
            extrinsic = all_extrinsics[0]
        else:
            motion = torch.stack(all_motions, dim=0)
            joints_canonical = torch.stack(all_joints_canonical, dim=0)
            extrinsic = torch.stack(all_extrinsics, dim=0)
        
        smpl_params_0 = all_smpl_params[0] if num_persons >= 1 else None
        betas_final = smpl_params_0.get('betas') if smpl_params_0 else None
        
        result = {
            'motion': motion.detach().cpu(),
            'extrinsic': extrinsic.detach().cpu(),
            'intrinsic': intrinsic.detach().cpu() if intrinsic is not None else None,
            'joints_canonical': joints_canonical.detach().cpu(),
            'smpl_params': {
                'global_orient': smpl_params['global_orient'].detach().cpu(),
                'body_pose': smpl_params['body_pose'].detach().cpu(),
                'betas': betas_final.detach().cpu() if betas_final is not None else None,
            },
            'num_frames': joints_canonical.shape[0] if joints_canonical.dim() == 2 else joints_canonical.shape[1],
            'set_floor': set_floor,
            'use_shape': use_shape,
        }
        
        return result
    
    def extract(
        self,
        video_path: str,
        output_path: str,
        bbox_output: Optional[str] = None,
        smpl_output: Optional[str] = None,
        smooth_window: int = 5,
        set_floor: bool = True,
        overwrite: bool = False,
        person_idx: int = 0,
        use_shape: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Full motion extraction pipeline.
        
        Args:
            video_path: Path to input video
            output_path: Path to save final motion (.pt)
            bbox_output: Optional path to save bbox intermediate results
            smpl_output: Optional path to save SMPL intermediate results
            smooth_window: Smoothing window size (0 to disable)
            set_floor: Whether to set floor alignment
            overwrite: Whether to overwrite existing files
            person_idx: Index of person to extract (for multi-person videos)
            use_shape: Whether to include shape parameters in motion representation
            
        Returns:
            Dictionary containing final motion data
        """
        if os.path.exists(output_path) and not overwrite:
            logger.info(f"Loading existing motion from: {output_path}")
            return torch.load(output_path, weights_only=True)
        
        use_shape = use_shape if use_shape is not None else self.use_shape
        
        bbox_path = bbox_output or output_path.replace('.pt', '_bbox.pt')
        smpl_path = smpl_output or output_path.replace('.pt', '_smpl.pt')
        
        bbx_xyxy, conf, mask = self.extract_bbox(video_path, bbox_path, overwrite)
        
        smpl_data = self.extract_smpl(video_path, bbox_path, smpl_path, overwrite)
        
        if person_idx > 0:
            for k in smpl_data['smpl_params_incam']:
                smpl_data['smpl_params_incam'][k] = smpl_data['smpl_params_incam'][k][person_idx:person_idx+1]
        
        result = self.post_process(
            smpl_data, 
            smooth_window=smooth_window, 
            set_floor=set_floor,
            frame_mask=mask,
            use_shape=use_shape,
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if overwrite and os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Removed existing output file: {output_path}")
        
        torch.save(result, output_path)
        logger.info(f"Saved final motion to: {output_path}")
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Extract SMPL motion from video for EchoMotion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output motion (.pt)')
    parser.add_argument('--bbox_output', type=str, default=None, help='Path to save bbox intermediate results')
    parser.add_argument('--smpl_output', type=str, default=None, help='Path to save SMPL intermediate results')
    parser.add_argument('--smooth_window', type=int, default=5, help='Smoothing window size (0 to disable)')
    parser.add_argument('--set_floor', action='store_true', help='Set floor alignment')
    parser.add_argument('--person_idx', type=int, default=0, help='Person index for multi-person videos')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--max_frames', type=int, default=300, help='Maximum number of frames to process')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for CameraHMR')
    parser.add_argument('--use_shape', action='store_true', help='Include shape parameters (betas) in motion representation')
    parser.add_argument('--visualize', action='store_true', help='Visualize the extracted motion')
    parser.add_argument('--vis_output', type=str, default=None, help='Path to save visualization video')
    parser.add_argument('--vis_height', type=int, default=None, help='Height of visualization output (default: use input video height)')
    parser.add_argument('--vis_width', type=int, default=None, help='Width of visualization output (default: use input video width)')
    parser.add_argument('--vis_fps', type=int, default=24, help='FPS of visualization output')
    
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    device = torch.device(args.device) if args.device else None
    
    extractor = MotionExtractor(
        device=device,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        use_shape=args.use_shape if hasattr(args, 'use_shape') else False,
        overwrite=args.overwrite,
    )
    
    result = extractor.extract(
        video_path=args.video_path,
        output_path=args.output_path,
        bbox_output=args.bbox_output,
        smpl_output=args.smpl_output,
        smooth_window=args.smooth_window,
        set_floor=args.set_floor,
        person_idx=args.person_idx,
        overwrite=args.overwrite,
    )
    
    if args.visualize:
        logger.info("Visualizing extracted motion...")
        motion_data = result['motion'].to(device)
        
        import cv2
        cap = cv2.VideoCapture(args.video_path)
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        logger.info(f"Input video resolution: {video_width}x{video_height}")
        
        vis_height = args.vis_height if args.vis_height else video_height
        vis_width = args.vis_width if args.vis_width else video_width
        
        vis_output_path = args.vis_output or args.output_path.replace('.pt', '_vis.mp4')
        motion_gt_dict, motion_gt = load_motion(motion_path=args.output_path, 
                                                fetch_local_motion=True, 
                                                init_canonical=False, 
                                                use_shape=True, 
                                                max_frame_len=242)
        
        motion_vis_during_validation(
            motion_data=motion_gt,
            output_dir=None,
            H=vis_height,
            W=vis_width,
            fps=20,
            motion_name=None,
            device=motion_data.device,
            verbose=True,
            output_video_path=vis_output_path,
            zero_trans=False,
            is_6d=True,
            use_shape=args.use_shape,
        )
        
        logger.info(f"Visualization saved to: {vis_output_path}")


if __name__ == '__main__':
    main()
