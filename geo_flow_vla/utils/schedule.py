"""
Noise Schedules for Conditional Flow Matching.

Implements various schedules for the flow matching process:
- Linear: σ(t) = 1 - t (standard interpolation)
- Cosine: σ(t) = cos(πt/2) (smoother transitions)
- Optimal Transport: Variance-preserving schedule

References:
    - Lipman et al., "Flow Matching for Generative Modeling" (2023)
    - Tong et al., "Improving and Generalizing Flow-Based Generative Models" (2023)
"""

from abc import ABC, abstractmethod
from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BaseSchedule(ABC, nn.Module):
    """Abstract base class for flow matching schedules."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sigma(self, t: Tensor) -> Tensor:
        """Compute noise level at time t.
        
        Args:
            t: Time values in [0, 1], shape (B,) or (B, 1, ...)
            
        Returns:
            Noise level sigma(t), same shape as t
        """
        pass

    @abstractmethod
    def dsigma_dt(self, t: Tensor) -> Tensor:
        """Compute derivative of sigma w.r.t. time.
        
        Args:
            t: Time values in [0, 1]
            
        Returns:
            d(sigma)/dt at time t
        """
        pass

    def interpolate(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Interpolate between noise x0 and data x1 at time t.
        
        For linear interpolation: x_t = (1-t)*x0 + t*x1
        
        Args:
            x0: Noise samples, shape (B, ...)
            x1: Data samples (ground truth), shape (B, ...)
            t: Time values in [0, 1], shape (B,) or broadcastable
            
        Returns:
            Interpolated samples x_t, shape (B, ...)
        """
        # Ensure t has correct shape for broadcasting
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        
        return (1 - t) * x0 + t * x1

    def target_velocity(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute target velocity field for CFM training.
        
        For conditional flow matching: v* = x1 - x0
        
        Args:
            x0: Noise samples
            x1: Data samples
            
        Returns:
            Target velocity field
        """
        return x1 - x0


class LinearSchedule(BaseSchedule):
    """
    Linear noise schedule: σ(t) = 1 - t
    
    Properties:
        - σ(0) = 1 (full noise at t=0)
        - σ(1) = 0 (no noise at t=1)
        - Constant velocity interpolation
    """

    def __init__(self, sigma_min: float = 1e-4) -> None:
        """
        Args:
            sigma_min: Minimum noise level to avoid numerical issues
        """
        super().__init__()
        self.sigma_min = sigma_min

    def sigma(self, t: Tensor) -> Tensor:
        return torch.clamp(1.0 - t, min=self.sigma_min)

    def dsigma_dt(self, t: Tensor) -> Tensor:
        return -torch.ones_like(t)


class CosineSchedule(BaseSchedule):
    """
    Cosine noise schedule: σ(t) = cos(πt/2)
    
    Properties:
        - Smoother transitions at endpoints
        - σ(0) = 1, σ(1) = 0
        - Derivative: dσ/dt = -π/2 * sin(πt/2)
    """

    def __init__(self, sigma_min: float = 1e-4) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.pi_half = torch.pi / 2

    def sigma(self, t: Tensor) -> Tensor:
        return torch.clamp(torch.cos(self.pi_half * t), min=self.sigma_min)

    def dsigma_dt(self, t: Tensor) -> Tensor:
        return -self.pi_half * torch.sin(self.pi_half * t)


class OptimalTransportSchedule(BaseSchedule):
    """
    Optimal Transport (variance-preserving) schedule.
    
    Uses the geodesic interpolation that preserves variance:
        x_t = cos(πt/2) * x0 + sin(πt/2) * x1
    
    This corresponds to the optimal transport path between
    the noise and data distributions on the probability simplex.
    """

    def __init__(self, sigma_min: float = 1e-4) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.pi_half = torch.pi / 2

    def sigma(self, t: Tensor) -> Tensor:
        # Coefficient for noise component
        return torch.clamp(torch.cos(self.pi_half * t), min=self.sigma_min)

    def dsigma_dt(self, t: Tensor) -> Tensor:
        return -self.pi_half * torch.sin(self.pi_half * t)

    def interpolate(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Geodesic interpolation for OT schedule."""
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        
        cos_t = torch.cos(self.pi_half * t)
        sin_t = torch.sin(self.pi_half * t)
        
        return cos_t * x0 + sin_t * x1

    def target_velocity(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Compute velocity for OT schedule (depends on t)."""
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        
        cos_t = torch.cos(self.pi_half * t)
        sin_t = torch.sin(self.pi_half * t)
        
        # d/dt [cos(πt/2)*x0 + sin(πt/2)*x1]
        return self.pi_half * (-sin_t * x0 + cos_t * x1)


def get_schedule(
    name: Literal["linear", "cosine", "ot"],
    sigma_min: float = 1e-4,
) -> BaseSchedule:
    """
    Factory function to get a noise schedule by name.
    
    Args:
        name: Schedule type - "linear", "cosine", or "ot" (optimal transport)
        sigma_min: Minimum noise level
        
    Returns:
        Initialized schedule instance
        
    Raises:
        ValueError: If schedule name is not recognized
    """
    schedules = {
        "linear": LinearSchedule,
        "cosine": CosineSchedule,
        "ot": OptimalTransportSchedule,
    }
    
    if name not in schedules:
        raise ValueError(
            f"Unknown schedule '{name}'. Available: {list(schedules.keys())}"
        )
    
    return schedules[name](sigma_min=sigma_min)


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    eps: float = 1e-5,
) -> Tensor:
    """
    Sample random timesteps uniformly from (eps, 1-eps).
    
    Args:
        batch_size: Number of timesteps to sample
        device: Target device
        eps: Small offset to avoid boundary issues
        
    Returns:
        Tensor of shape (batch_size,) with values in (eps, 1-eps)
    """
    return torch.rand(batch_size, device=device) * (1 - 2 * eps) + eps


def get_sigmas_karras(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Karras et al. sigmas for deterministic sampling.
    
    From "Elucidating the Design Space of Diffusion-Based Generative Models"
    
    Args:
        num_steps: Number of sampling steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level  
        rho: Schedule curvature parameter
        device: Target device
        
    Returns:
        Tensor of sigma values, shape (num_steps + 1,)
    """
    ramp = torch.linspace(0, 1, num_steps + 1, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    
    return sigmas

