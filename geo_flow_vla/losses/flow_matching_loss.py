"""
Conditional Flow Matching (CFM) Loss for Geo-Flow VLA.

Implements the flow matching objective for training the DiT policy.
The policy learns to predict the velocity field that transforms noise
into action trajectories.

Loss:
    L_CFM = E_{t, x_0, x_1}[||v_θ(x_t, t, c) - (x_1 - x_0)||²]
    
Where:
    x_0 ~ N(0, I) is noise
    x_1 is ground truth action trajectory
    x_t = (1-t)x_0 + t*x_1 is interpolation
    c is conditioning (state, goal)

References:
    - Lipman et al., "Flow Matching for Generative Modeling" (2023)
    - Tong et al., "Improving and Generalizing Flow-Based Generative Models" (2023)
"""

from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


def compute_cfm_loss(
    model: nn.Module,
    ground_truth: Tensor,
    state: Tensor,
    goal: Tensor,
    noise: Optional[Tensor] = None,
    schedule: Literal["linear", "cosine", "ot"] = "linear",
) -> Tensor:
    """
    Compute Conditional Flow Matching loss.
    
    Args:
        model: Policy model predicting velocity field
        ground_truth: Target action trajectory (B, T, action_dim)
        state: State conditioning (B, state_dim)
        goal: Goal conditioning (B, latent_dim)
        noise: Optional pre-sampled noise
        schedule: Interpolation schedule
        
    Returns:
        CFM loss scalar
    """
    B = ground_truth.shape[0]
    device = ground_truth.device
    
    # Sample noise (x_0)
    if noise is None:
        noise = torch.randn_like(ground_truth)
    
    # Sample random timesteps in (0, 1)
    t = torch.rand(B, device=device)
    
    # Interpolate based on schedule
    if schedule == "linear":
        # Linear: x_t = (1-t)x_0 + t*x_1
        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * ground_truth
        target_v = ground_truth - noise
        
    elif schedule == "cosine":
        # Cosine: smoother transitions at boundaries
        t_cos = 0.5 * (1 - torch.cos(t * math.pi))
        t_expanded = t_cos.view(B, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * ground_truth
        
        # Velocity for cosine schedule
        dt_cos = 0.5 * math.pi * torch.sin(t * math.pi)
        target_v = dt_cos.view(B, 1, 1) * (ground_truth - noise)
        
    elif schedule == "ot":
        # Optimal Transport: variance-preserving
        t_expanded = t.view(B, 1, 1)
        cos_t = torch.cos(0.5 * math.pi * t_expanded)
        sin_t = torch.sin(0.5 * math.pi * t_expanded)
        x_t = cos_t * noise + sin_t * ground_truth
        
        # OT velocity
        target_v = 0.5 * math.pi * (-sin_t * noise + cos_t * ground_truth)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    # Predict velocity
    pred_v = model(x_t, t, state, goal)
    
    # MSE loss
    loss = F.mse_loss(pred_v, target_v)
    
    return loss


class FlowMatchingLoss(nn.Module):
    """
    Conditional Flow Matching loss module.
    
    Provides:
    - Multiple interpolation schedules
    - Optional weighting by timestep
    - Detailed metrics for logging
    """

    def __init__(
        self,
        schedule: Literal["linear", "cosine", "ot"] = "linear",
        weight_by_timestep: bool = False,
        min_timestep: float = 1e-5,
        max_timestep: float = 1.0 - 1e-5,
    ) -> None:
        """
        Args:
            schedule: Interpolation schedule type
            weight_by_timestep: Weight loss by 1/t for importance sampling
            min_timestep: Minimum timestep value
            max_timestep: Maximum timestep value
        """
        super().__init__()
        
        self.schedule = schedule
        self.weight_by_timestep = weight_by_timestep
        self.min_t = min_timestep
        self.max_t = max_timestep

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample random timesteps."""
        t = torch.rand(batch_size, device=device)
        t = t * (self.max_t - self.min_t) + self.min_t
        return t

    def interpolate(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Interpolate between noise and data.
        
        Args:
            x_0: Noise samples (B, ...)
            x_1: Data samples (B, ...)
            t: Timesteps (B,)
            
        Returns:
            Tuple of (interpolated samples, target velocity)
        """
        B = x_0.shape[0]
        t_expanded = t.view(B, *([1] * (x_0.dim() - 1)))
        
        if self.schedule == "linear":
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            v_target = x_1 - x_0
            
        elif self.schedule == "cosine":
            t_cos = 0.5 * (1 - torch.cos(t * math.pi))
            t_cos = t_cos.view(B, *([1] * (x_0.dim() - 1)))
            x_t = (1 - t_cos) * x_0 + t_cos * x_1
            
            dt_cos = 0.5 * math.pi * torch.sin(t * math.pi)
            dt_cos = dt_cos.view(B, *([1] * (x_0.dim() - 1)))
            v_target = dt_cos * (x_1 - x_0)
            
        elif self.schedule == "ot":
            cos_t = torch.cos(0.5 * math.pi * t_expanded)
            sin_t = torch.sin(0.5 * math.pi * t_expanded)
            x_t = cos_t * x_0 + sin_t * x_1
            v_target = 0.5 * math.pi * (-sin_t * x_0 + cos_t * x_1)
            
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        
        return x_t, v_target

    def forward(
        self,
        model: nn.Module,
        ground_truth: Tensor,
        state: Tensor,
        goal: Tensor,
        noise: Optional[Tensor] = None,
        return_metrics: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Compute CFM loss with optional metrics.
        
        Args:
            model: Policy model
            ground_truth: Target actions (B, T, action_dim)
            state: State conditioning (B, state_dim)
            goal: Goal conditioning (B, latent_dim)
            noise: Optional noise samples
            return_metrics: Return detailed metrics
            
        Returns:
            Dictionary with loss and optional metrics
        """
        B = ground_truth.shape[0]
        device = ground_truth.device
        
        # Sample noise and timesteps
        if noise is None:
            noise = torch.randn_like(ground_truth)
        t = self.sample_timesteps(B, device)
        
        # Interpolate
        x_t, v_target = self.interpolate(noise, ground_truth, t)
        
        # Predict velocity
        v_pred = model(x_t, t, state, goal)
        
        # Compute per-sample loss
        per_sample_loss = F.mse_loss(v_pred, v_target, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=(1, 2))  # (B,)
        
        # Optional timestep weighting
        if self.weight_by_timestep:
            weights = 1.0 / (t + 0.1)  # Avoid division by zero
            weights = weights / weights.sum() * B
            loss = (per_sample_loss * weights).mean()
        else:
            loss = per_sample_loss.mean()
        
        result = {"loss": loss}
        
        if return_metrics:
            with torch.no_grad():
                # Per-timestep loss (for debugging)
                early_mask = t < 0.33
                mid_mask = (t >= 0.33) & (t < 0.67)
                late_mask = t >= 0.67
                
                if early_mask.any():
                    result["loss_early"] = per_sample_loss[early_mask].mean()
                if mid_mask.any():
                    result["loss_mid"] = per_sample_loss[mid_mask].mean()
                if late_mask.any():
                    result["loss_late"] = per_sample_loss[late_mask].mean()
                
                # Velocity magnitude
                result["v_pred_norm"] = v_pred.norm(dim=-1).mean()
                result["v_target_norm"] = v_target.norm(dim=-1).mean()
        
        return result


class WeightedFlowMatchingLoss(FlowMatchingLoss):
    """
    Flow Matching loss with learned timestep weighting.
    
    Useful for focusing on difficult timesteps during training.
    """

    def __init__(
        self,
        num_timestep_bins: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.num_bins = num_timestep_bins
        
        # Learnable per-bin weights
        self.log_weights = nn.Parameter(torch.zeros(num_timestep_bins))

    def get_weights(self, t: Tensor) -> Tensor:
        """Get weights for given timesteps."""
        # Bin timesteps
        bins = (t * self.num_bins).long().clamp(0, self.num_bins - 1)
        
        # Get log weights and convert to weights
        log_w = self.log_weights[bins]
        weights = F.softmax(log_w, dim=0) * t.shape[0]
        
        return weights

    def forward(
        self,
        model: nn.Module,
        ground_truth: Tensor,
        state: Tensor,
        goal: Tensor,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Compute weighted loss."""
        B = ground_truth.shape[0]
        device = ground_truth.device
        
        noise = torch.randn_like(ground_truth)
        t = self.sample_timesteps(B, device)
        
        x_t, v_target = self.interpolate(noise, ground_truth, t)
        v_pred = model(x_t, t, state, goal)
        
        per_sample_loss = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=(1, 2))
        weights = self.get_weights(t)
        
        loss = (per_sample_loss * weights).mean()
        
        return {
            "loss": loss,
            "unweighted_loss": per_sample_loss.mean(),
        }

