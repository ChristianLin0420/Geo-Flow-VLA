"""
Camera Coordinate Transformations for Geo-Flow VLA.

Provides utilities to transform between robot and camera coordinate frames
for trajectory visualization on MoGe-2 generated point clouds.

Coordinate Conventions:
- Robot frame: X forward, Y left, Z up (right-hand rule, common in robotics)
- Camera frame (OpenCV): X right, Y down, Z forward (used by MoGe-2)

The transformation T_cam_robot converts points from robot frame to camera frame.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Dict, Optional, Union


# ============================================================================
# Default Camera Extrinsics per Dataset
# These are approximate values based on typical simulation camera setups
# ============================================================================

# Camera extrinsic matrices: T_cam_robot (4x4)
# Transforms points FROM robot frame TO camera frame
# [R | t]
# [0 | 1]

def get_default_camera_extrinsics(dataset: str, camera: str = "default") -> np.ndarray:
    """
    Get default camera extrinsic matrix for a dataset.
    
    Args:
        dataset: Dataset name ('libero', 'rlbench', 'calvin')
        camera: Camera name (dataset-specific)
        
    Returns:
        4x4 transformation matrix T_cam_robot
    """
    
    if dataset.lower() == "libero":
        # LIBERO uses Robosuite's agentview camera
        # Camera is positioned above and in front of the robot, looking down
        # Approximate: camera is ~1.5m above table, looking down at ~45 degrees
        if camera in ["agentview", "default"]:
            # Rotation: camera looks down at workspace
            # Robot Z-up -> Camera Z-forward
            # Robot X-forward -> Camera X-right (with inversion)
            # Robot Y-left -> Camera Y-down (with inversion)
            R = np.array([
                [1.0,  0.0,  0.0],   # Camera X = Robot X (right)
                [0.0,  0.0, -1.0],   # Camera Y = -Robot Z (down)
                [0.0,  1.0,  0.0],   # Camera Z = Robot Y (forward/depth)
            ])
            t = np.array([0.0, 1.2, 0.5])  # Camera position in robot frame
        elif camera == "eye_in_hand":
            # Wrist camera - attached to end-effector, looking forward
            R = np.eye(3)
            t = np.array([0.0, 0.0, 0.0])  # Relative to EE
        else:
            R = np.eye(3)
            t = np.zeros(3)
            
    elif dataset.lower() == "rlbench":
        # RLBench uses CoppeliaSim cameras
        # front_rgb camera is positioned in front of robot
        if camera in ["front_rgb", "default"]:
            # Front camera: positioned in front, looking at robot workspace
            R = np.array([
                [-1.0,  0.0,  0.0],  # Camera X = -Robot X (flip)
                [0.0,  0.0, -1.0],   # Camera Y = -Robot Z (down)
                [0.0, -1.0,  0.0],   # Camera Z = -Robot Y (depth)
            ])
            t = np.array([0.0, -1.5, 1.0])  # Camera in front of robot
        elif camera == "left_shoulder_rgb":
            R = np.array([
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0],
                [1.0,  0.0,  0.0],
            ])
            t = np.array([1.0, 0.0, 1.0])
        elif camera == "right_shoulder_rgb":
            R = np.array([
                [0.0,  1.0,  0.0],
                [0.0,  0.0, -1.0],
                [-1.0, 0.0,  0.0],
            ])
            t = np.array([-1.0, 0.0, 1.0])
        elif camera == "wrist_rgb":
            R = np.eye(3)
            t = np.zeros(3)
        else:
            R = np.eye(3)
            t = np.zeros(3)
            
    elif dataset.lower() == "calvin":
        # CALVIN uses top-down camera
        if camera in ["top", "default"]:
            # Top camera: looking straight down at table
            R = np.array([
                [1.0,  0.0,  0.0],   # Camera X = Robot X
                [0.0, -1.0,  0.0],   # Camera Y = -Robot Y (flip)
                [0.0,  0.0, -1.0],   # Camera Z = -Robot Z (looking down)
            ])
            t = np.array([0.0, 0.0, 1.5])  # Camera above workspace
        elif camera == "wrist":
            R = np.eye(3)
            t = np.zeros(3)
        else:
            R = np.eye(3)
            t = np.zeros(3)
    else:
        # Default: identity transform
        R = np.eye(3)
        t = np.zeros(3)
    
    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def transform_points_robot_to_camera(
    points: Union[np.ndarray, Tensor],
    T_cam_robot: Union[np.ndarray, Tensor],
) -> Union[np.ndarray, Tensor]:
    """
    Transform points from robot frame to camera frame.
    
    Args:
        points: (N, 3) points in robot frame
        T_cam_robot: (4, 4) transformation matrix
        
    Returns:
        (N, 3) points in camera frame
    """
    is_tensor = isinstance(points, Tensor)
    
    if is_tensor:
        device = points.device
        dtype = points.dtype
        points_np = points.cpu().numpy()
        T_np = T_cam_robot.cpu().numpy() if isinstance(T_cam_robot, Tensor) else T_cam_robot
    else:
        points_np = points
        T_np = T_cam_robot
    
    # Convert to homogeneous coordinates
    N = points_np.shape[0]
    points_homo = np.hstack([points_np, np.ones((N, 1))])  # (N, 4)
    
    # Apply transformation
    points_cam = (T_np @ points_homo.T).T[:, :3]  # (N, 3)
    
    if is_tensor:
        return torch.from_numpy(points_cam).to(device=device, dtype=dtype)
    return points_cam


def transform_trajectory_to_camera_frame(
    trajectory: Union[np.ndarray, Tensor],
    dataset: str,
    camera: str = "default",
    T_cam_robot: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tensor]:
    """
    Transform robot action trajectory to camera frame for visualization.
    
    Args:
        trajectory: (T, 3) or (T, D) trajectory where first 3 dims are XYZ positions
        dataset: Dataset name for default extrinsics
        camera: Camera name for default extrinsics  
        T_cam_robot: Optional custom extrinsic matrix (overrides defaults)
        
    Returns:
        Transformed trajectory in camera frame
    """
    if T_cam_robot is None:
        T_cam_robot = get_default_camera_extrinsics(dataset, camera)
    
    is_tensor = isinstance(trajectory, Tensor)
    
    if is_tensor:
        device = trajectory.device
        dtype = trajectory.dtype
        traj_np = trajectory.cpu().numpy()
    else:
        traj_np = trajectory
    
    # Extract position (first 3 dims) and other dimensions
    positions = traj_np[:, :3]  # (T, 3)
    other_dims = traj_np[:, 3:] if traj_np.shape[1] > 3 else None
    
    # Transform positions
    positions_cam = transform_points_robot_to_camera(positions, T_cam_robot)
    
    # Reconstruct trajectory
    if other_dims is not None:
        result = np.hstack([positions_cam, other_dims])
    else:
        result = positions_cam
    
    if is_tensor:
        return torch.from_numpy(result).to(device=device, dtype=dtype)
    return result


def scale_trajectory_to_scene(
    trajectory: np.ndarray,
    scene_points: np.ndarray,
    scale_factor: float = 0.3,
) -> np.ndarray:
    """
    Scale trajectory to fit within scene bounds (fallback when transform is inaccurate).
    
    Args:
        trajectory: (T, 3) trajectory points
        scene_points: (N, 3) scene point cloud
        scale_factor: How much of scene bounds to use (0.3 = 30%)
        
    Returns:
        Scaled trajectory
    """
    # Compute scene bounds
    scene_center = scene_points.mean(axis=0)
    scene_extent = np.abs(scene_points - scene_center).max()
    
    # Compute trajectory bounds
    traj_center = trajectory.mean(axis=0)
    traj_extent = np.abs(trajectory - traj_center).max() + 1e-6
    
    # Normalize and rescale
    traj_normalized = (trajectory - traj_center) / traj_extent
    traj_scaled = traj_normalized * scene_extent * scale_factor + scene_center
    
    return traj_scaled


# ============================================================================
# Dataset-Camera Configuration
# ============================================================================

DATASET_CAMERA_CONFIG = {
    "libero": {
        "default_camera": "agentview",
        "cameras": ["agentview", "eye_in_hand"],
    },
    "rlbench": {
        "default_camera": "front_rgb",
        "cameras": ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"],
    },
    "calvin": {
        "default_camera": "top",
        "cameras": ["top", "wrist"],
    },
}


def get_dataset_camera(dataset: str) -> str:
    """Get default camera name for a dataset."""
    config = DATASET_CAMERA_CONFIG.get(dataset.lower(), {})
    return config.get("default_camera", "default")
