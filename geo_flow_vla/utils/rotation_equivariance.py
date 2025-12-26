"""
3D Rotation Equivariance Utilities for Geo-Flow VLA.

Provides SO(3) and SE(3) augmentations for:
- Point cloud data from MoGe-2
- Robot action trajectories (position + rotation)
- Consistent transformations across modalities

References:
    - Rotation representations: Zhou et al., "On the Continuity of Rotation Representations"
    - Equivariant networks: Thomas et al., "Tensor Field Networks"
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def random_so3_rotation(
    batch_size: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Sample random rotation matrices uniformly from SO(3).
    
    Uses the QR decomposition method for uniform sampling.
    
    Args:
        batch_size: Number of rotation matrices to sample
        device: Target device
        dtype: Data type
        
    Returns:
        Rotation matrices of shape (batch_size, 3, 3)
    """
    # Sample random matrix from standard normal
    random_matrix = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
    
    # QR decomposition gives orthogonal matrix Q
    q, r = torch.linalg.qr(random_matrix)
    
    # Ensure proper rotation (det = +1, not -1)
    # Multiply by sign of diagonal of R to ensure determinant is positive
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    sign = torch.sign(d)
    
    # Handle zero diagonal elements
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    
    # Apply sign correction
    q = q * sign.unsqueeze(-2)
    
    return q


def axis_angle_to_rotation_matrix(axis_angle: Tensor) -> Tensor:
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axis_angle: Axis-angle vectors, shape (..., 3)
        
    Returns:
        Rotation matrices, shape (..., 3, 3)
    """
    # Compute angle (norm of axis-angle vector)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    
    # Normalize to get axis (handle zero rotation)
    axis = axis_angle / (angle + 1e-8)
    
    # Rodrigues' rotation formula
    angle = angle.squeeze(-1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    # Skew-symmetric matrix K
    K = skew_symmetric(axis)
    
    # R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    I = I.expand(*axis_angle.shape[:-1], 3, 3)
    
    cos_a = cos_a.unsqueeze(-1).unsqueeze(-1)
    sin_a = sin_a.unsqueeze(-1).unsqueeze(-1)
    
    R = I + sin_a * K + (1 - cos_a) * torch.bmm(
        K.view(-1, 3, 3), K.view(-1, 3, 3)
    ).view(*K.shape)
    
    return R


def rotation_matrix_to_axis_angle(R: Tensor) -> Tensor:
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        R: Rotation matrices, shape (..., 3, 3)
        
    Returns:
        Axis-angle vectors, shape (..., 3)
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    
    # Angle from trace: cos(θ) = (tr(R) - 1) / 2
    trace = torch.diagonal(R_flat, dim1=-2, dim2=-1).sum(-1)
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    
    # Axis from skew-symmetric part: [R - R^T] = 2*sin(θ)*K
    skew = (R_flat - R_flat.transpose(-1, -2)) / 2
    
    # Extract axis components
    axis = torch.stack([
        skew[:, 2, 1],
        skew[:, 0, 2],
        skew[:, 1, 0],
    ], dim=-1)
    
    # Normalize axis
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis = axis / (axis_norm + 1e-8)
    
    # Scale by angle
    axis_angle = axis * angle.unsqueeze(-1)
    
    return axis_angle.reshape(*batch_shape, 3)


def skew_symmetric(v: Tensor) -> Tensor:
    """
    Compute skew-symmetric matrix from 3D vector.
    
    Args:
        v: Vectors, shape (..., 3)
        
    Returns:
        Skew-symmetric matrices, shape (..., 3, 3)
    """
    batch_shape = v.shape[:-1]
    v = v.reshape(-1, 3)
    
    zero = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
    
    K = torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=-1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=-1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=-1),
    ], dim=-2)
    
    return K.reshape(*batch_shape, 3, 3)


def rotate_point_cloud(
    points: Tensor,
    rotation: Tensor,
    center: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply rotation to point cloud.
    
    Args:
        points: Point cloud, shape (B, N, 3) or (B, 3, H, W)
        rotation: Rotation matrices, shape (B, 3, 3)
        center: Optional rotation center, shape (B, 3). Defaults to centroid.
        
    Returns:
        Rotated point cloud, same shape as input
    """
    is_image_format = points.dim() == 4  # (B, 3, H, W)
    
    if is_image_format:
        B, C, H, W = points.shape
        assert C == 3, f"Expected 3 channels for XYZ, got {C}"
        # Reshape to (B, N, 3)
        points = points.permute(0, 2, 3, 1).reshape(B, H * W, 3)
    
    B, N, _ = points.shape
    
    # Compute center if not provided
    if center is None:
        center = points.mean(dim=1, keepdim=True)  # (B, 1, 3)
    else:
        center = center.unsqueeze(1)  # (B, 1, 3)
    
    # Center, rotate, uncenter
    centered = points - center
    rotated = torch.bmm(centered, rotation.transpose(-1, -2))
    result = rotated + center
    
    if is_image_format:
        result = result.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    
    return result


def rotate_actions(
    actions: Tensor,
    rotation: Tensor,
    action_format: str = "pos_rot_grip",
) -> Tensor:
    """
    Apply rotation transformation to robot actions.
    
    Handles different action formats consistently with point cloud rotation.
    
    Args:
        actions: Action tensor, shape (B, T, action_dim) or (B, action_dim)
        rotation: Rotation matrices, shape (B, 3, 3)
        action_format: Format string describing action layout:
            - "pos_rot_grip": [x,y,z, rx,ry,rz, gripper] (7D)
            - "pos_quat_grip": [x,y,z, qw,qx,qy,qz, gripper] (8D)
            - "delta_pos_rot_grip": Delta actions in same format
            
    Returns:
        Rotated actions, same shape as input
    """
    has_time_dim = actions.dim() == 3
    
    if has_time_dim:
        B, T, D = actions.shape
        actions_flat = actions.reshape(B * T, D)
        rotation_expanded = rotation.unsqueeze(1).expand(B, T, 3, 3).reshape(B * T, 3, 3)
    else:
        B, D = actions.shape
        actions_flat = actions
        rotation_expanded = rotation
    
    if action_format in ["pos_rot_grip", "delta_pos_rot_grip"]:
        assert D == 7, f"Expected 7D actions for {action_format}, got {D}"
        
        pos = actions_flat[:, :3]  # (BT, 3)
        rot = actions_flat[:, 3:6]  # (BT, 3) axis-angle
        grip = actions_flat[:, 6:7]  # (BT, 1)
        
        # Rotate position
        pos_rotated = torch.bmm(pos.unsqueeze(1), rotation_expanded.transpose(-1, -2)).squeeze(1)
        
        # Rotate orientation: R_new = R_aug @ R_old
        R_old = axis_angle_to_rotation_matrix(rot)
        R_new = torch.bmm(rotation_expanded, R_old)
        rot_rotated = rotation_matrix_to_axis_angle(R_new)
        
        actions_rotated = torch.cat([pos_rotated, rot_rotated, grip], dim=-1)
        
    elif action_format == "pos_quat_grip":
        assert D == 8, f"Expected 8D actions for {action_format}, got {D}"
        
        pos = actions_flat[:, :3]
        quat = actions_flat[:, 3:7]  # (w, x, y, z)
        grip = actions_flat[:, 7:8]
        
        # Rotate position
        pos_rotated = torch.bmm(pos.unsqueeze(1), rotation_expanded.transpose(-1, -2)).squeeze(1)
        
        # Rotate quaternion
        R_old = quaternion_to_rotation_matrix(quat)
        R_new = torch.bmm(rotation_expanded, R_old)
        quat_rotated = rotation_matrix_to_quaternion(R_new)
        
        actions_rotated = torch.cat([pos_rotated, quat_rotated, grip], dim=-1)
    else:
        raise ValueError(f"Unknown action format: {action_format}")
    
    if has_time_dim:
        actions_rotated = actions_rotated.reshape(B, T, D)
    
    return actions_rotated


def quaternion_to_rotation_matrix(quat: Tensor) -> Tensor:
    """
    Convert quaternion (w, x, y, z) to rotation matrix.
    
    Args:
        quat: Quaternions, shape (..., 4)
        
    Returns:
        Rotation matrices, shape (..., 3, 3)
    """
    batch_shape = quat.shape[:-1]
    quat = quat.reshape(-1, 4)
    
    # Normalize quaternion
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Build rotation matrix
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1),
    ], dim=-2)
    
    return R.reshape(*batch_shape, 3, 3)


def rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    
    Uses Shepperd's method for numerical stability.
    
    Args:
        R: Rotation matrices, shape (..., 3, 3)
        
    Returns:
        Quaternions, shape (..., 4)
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Initialize output
    quat = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    
    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1) * 2
    quat[mask, 0] = 0.25 * s
    quat[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    quat[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    quat[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
    
    # Case 2: R[0,0] is largest diagonal
    mask = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s = torch.sqrt(1 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]) * 2
    quat[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    quat[mask, 1] = 0.25 * s
    quat[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
    quat[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
    
    # Case 3: R[1,1] is largest diagonal
    mask = (~mask) & (R[:, 1, 1] > R[:, 2, 2])
    s = torch.sqrt(1 + R[mask, 1, 1] - R[mask, 0, 0] - R[mask, 2, 2]) * 2
    quat[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    quat[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
    quat[mask, 2] = 0.25 * s
    quat[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
    
    # Case 4: R[2,2] is largest diagonal
    mask = ~mask
    s = torch.sqrt(1 + R[mask, 2, 2] - R[mask, 0, 0] - R[mask, 1, 1]) * 2
    quat[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
    quat[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
    quat[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
    quat[mask, 3] = 0.25 * s
    
    # Normalize
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    
    return quat.reshape(*batch_shape, 4)


class SE3Augmentation(nn.Module):
    """
    SE(3) data augmentation for 3D robotics data.
    
    Applies consistent random rotations and translations to:
    - Point clouds from MoGe-2
    - Robot action trajectories
    
    Ensures equivariance: augmenting input and actions together
    preserves the semantic meaning of the demonstration.
    """

    def __init__(
        self,
        rotation_range: float = 0.0,  # Radians, 0 = full SO(3)
        translation_range: float = 0.1,  # Meters
        apply_to_actions: bool = True,
        action_format: str = "pos_rot_grip",
    ) -> None:
        """
        Args:
            rotation_range: If > 0, limit rotations to this range around z-axis.
                           If 0, sample uniformly from full SO(3).
            translation_range: Maximum translation in each axis (meters)
            apply_to_actions: Whether to transform actions consistently
            action_format: Format of action vectors
        """
        super().__init__()
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.apply_to_actions = apply_to_actions
        self.action_format = action_format

    def forward(
        self,
        points: Tensor,
        actions: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply SE(3) augmentation.
        
        Args:
            points: Point cloud, shape (B, 3, H, W) or (B, N, 3)
            actions: Optional actions, shape (B, T, action_dim)
            
        Returns:
            If actions is None: augmented points
            Otherwise: (augmented_points, augmented_actions)
        """
        B = points.shape[0]
        device = points.device
        
        # Sample rotation
        if self.rotation_range > 0:
            # Z-axis rotation only
            angles = (torch.rand(B, device=device) * 2 - 1) * self.rotation_range
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            zeros = torch.zeros_like(angles)
            ones = torch.ones_like(angles)
            
            rotation = torch.stack([
                torch.stack([cos_a, -sin_a, zeros], dim=-1),
                torch.stack([sin_a, cos_a, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ], dim=-2)
        else:
            # Full SO(3) rotation
            rotation = random_so3_rotation(B, device=device, dtype=points.dtype)
        
        # Sample translation
        translation = (torch.rand(B, 3, device=device) * 2 - 1) * self.translation_range
        
        # Apply to points
        points_aug = rotate_point_cloud(points, rotation)
        
        # Add translation
        if points.dim() == 4:  # (B, 3, H, W)
            points_aug = points_aug + translation.view(B, 3, 1, 1)
        else:  # (B, N, 3)
            points_aug = points_aug + translation.unsqueeze(1)
        
        if actions is None:
            return points_aug
        
        if self.apply_to_actions:
            actions_aug = rotate_actions(actions, rotation, self.action_format)
            # Add translation to position component
            if actions.dim() == 3:  # (B, T, D)
                actions_aug[:, :, :3] = actions_aug[:, :, :3] + translation.unsqueeze(1)
            else:  # (B, D)
                actions_aug[:, :3] = actions_aug[:, :3] + translation
        else:
            actions_aug = actions
        
        return points_aug, actions_aug

