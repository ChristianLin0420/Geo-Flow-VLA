"""
State and Action Normalization for Diffusion Policy Training.

Proper normalization is critical for stable diffusion/flow matching training.
This module provides running statistics computation and normalization utilities.

References:
    - Chi et al., "Diffusion Policy" (2023) - Uses Gaussian normalization
"""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class StateActionNormalizer(nn.Module):
    """
    Gaussian normalization for states and actions.
    
    Computes running mean and std during data collection,
    then normalizes to zero-mean, unit-variance for training.
    
    Critical for Flow Matching stability:
    - Actions normalized to similar scale as noise
    - Prevents gradient explosion/vanishing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 16,
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            state_dim: Dimension of state vectors
            action_dim: Dimension of single-step action
            action_horizon: Number of action steps (for action chunks)
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.eps = eps
        
        # Running statistics for states
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("state_count", torch.zeros(1))
        self.register_buffer("state_m2", torch.zeros(state_dim))  # For Welford's algorithm
        
        # Running statistics for actions (flattened action chunk)
        action_chunk_dim = action_dim * action_horizon
        self.register_buffer("action_mean", torch.zeros(action_chunk_dim))
        self.register_buffer("action_std", torch.ones(action_chunk_dim))
        self.register_buffer("action_count", torch.zeros(1))
        self.register_buffer("action_m2", torch.zeros(action_chunk_dim))
        
        # Per-dimension statistics (for individual action dimensions)
        self.register_buffer("action_dim_mean", torch.zeros(action_dim))
        self.register_buffer("action_dim_std", torch.ones(action_dim))
        
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether normalizer has been fitted to data."""
        return self._fitted

    def fit(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> None:
        """
        Fit normalizer to a batch of data.
        
        Uses Welford's online algorithm for numerically stable
        mean and variance computation.
        
        Args:
            states: State vectors, shape (N, state_dim)
            actions: Action chunks, shape (N, action_horizon, action_dim)
        """
        # Validate shapes
        assert states.shape[-1] == self.state_dim, \
            f"State dim mismatch: {states.shape[-1]} vs {self.state_dim}"
        assert actions.shape[-1] == self.action_dim, \
            f"Action dim mismatch: {actions.shape[-1]} vs {self.action_dim}"
        assert actions.shape[-2] == self.action_horizon, \
            f"Action horizon mismatch: {actions.shape[-2]} vs {self.action_horizon}"
        
        # Flatten batch dimensions
        states = states.reshape(-1, self.state_dim)
        actions_flat = actions.reshape(-1, self.action_horizon * self.action_dim)
        
        # Update state statistics using Welford's algorithm
        self._update_stats(
            states,
            self.state_mean,
            self.state_m2,
            self.state_count,
        )
        
        # Update action statistics
        self._update_stats(
            actions_flat,
            self.action_mean,
            self.action_m2,
            self.action_count,
        )
        
        # Also compute per-dimension action stats
        actions_per_dim = actions.reshape(-1, self.action_dim)
        self._update_per_dim_stats(actions_per_dim)
        
        self._fitted = True

    def _update_stats(
        self,
        data: Tensor,
        mean: Tensor,
        m2: Tensor,
        count: Tensor,
    ) -> None:
        """Update running statistics using Welford's algorithm."""
        n = data.shape[0]
        
        for i in range(n):
            x = data[i]
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2

    def _update_per_dim_stats(self, actions: Tensor) -> None:
        """Update per-dimension action statistics."""
        self.action_dim_mean = actions.mean(dim=0)
        self.action_dim_std = actions.std(dim=0).clamp(min=self.eps)

    def finalize(self) -> None:
        """Finalize statistics computation (compute std from M2)."""
        if self.state_count > 1:
            self.state_std = torch.sqrt(self.state_m2 / (self.state_count - 1)).clamp(min=self.eps)
        
        if self.action_count > 1:
            self.action_std = torch.sqrt(self.action_m2 / (self.action_count - 1)).clamp(min=self.eps)
        
        self._fitted = True

    def fit_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        num_samples: Optional[int] = None,
        batch_size: int = 256,
    ) -> None:
        """
        Fit normalizer to an entire dataset.
        
        Args:
            dataset: PyTorch dataset returning (rgb, depth, proprio, instruction, actions)
            num_samples: Number of samples to use (None = all)
            batch_size: Batch size for processing
        """
        from torch.utils.data import DataLoader, Subset
        
        if num_samples is not None:
            indices = torch.randperm(len(dataset))[:num_samples].tolist()
            dataset = Subset(dataset, indices)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        all_actions = []
        
        for batch in loader:
            # Expect batch to have actions as last element
            actions = batch[-1]  # (B, T, action_dim)
            all_actions.append(actions)
        
        all_actions = torch.cat(all_actions, dim=0)
        
        # Compute statistics
        self.action_mean = all_actions.reshape(-1, self.action_horizon * self.action_dim).mean(dim=0)
        self.action_std = all_actions.reshape(-1, self.action_horizon * self.action_dim).std(dim=0).clamp(min=self.eps)
        
        self.action_dim_mean = all_actions.reshape(-1, self.action_dim).mean(dim=0)
        self.action_dim_std = all_actions.reshape(-1, self.action_dim).std(dim=0).clamp(min=self.eps)
        
        self._fitted = True

    def normalize_state(self, state: Tensor) -> Tensor:
        """
        Normalize state to zero-mean, unit-variance.
        
        Args:
            state: State vector, shape (..., state_dim)
            
        Returns:
            Normalized state, same shape
        """
        return (state - self.state_mean) / self.state_std

    def denormalize_state(self, state: Tensor) -> Tensor:
        """
        Denormalize state back to original scale.
        
        Args:
            state: Normalized state, shape (..., state_dim)
            
        Returns:
            Original-scale state, same shape
        """
        return state * self.state_std + self.state_mean

    def normalize_action(self, action: Tensor) -> Tensor:
        """
        Normalize action chunk to zero-mean, unit-variance.
        
        Args:
            action: Action chunk, shape (..., action_horizon, action_dim)
                   or flattened shape (..., action_horizon * action_dim)
            
        Returns:
            Normalized action, same shape
        """
        original_shape = action.shape
        is_chunked = action.shape[-2:] == (self.action_horizon, self.action_dim)
        
        if is_chunked:
            action = action.reshape(*action.shape[:-2], -1)
        
        normalized = (action - self.action_mean) / self.action_std
        
        if is_chunked:
            normalized = normalized.reshape(original_shape)
        
        return normalized

    def denormalize_action(self, action: Tensor) -> Tensor:
        """
        Denormalize action back to original scale.
        
        Args:
            action: Normalized action chunk
            
        Returns:
            Original-scale action, same shape
        """
        original_shape = action.shape
        is_chunked = len(action.shape) >= 2 and action.shape[-2:] == (self.action_horizon, self.action_dim)
        
        if is_chunked:
            action = action.reshape(*action.shape[:-2], -1)
        
        denormalized = action * self.action_std + self.action_mean
        
        if is_chunked:
            denormalized = denormalized.reshape(original_shape)
        
        return denormalized

    def normalize_action_per_dim(self, action: Tensor) -> Tensor:
        """
        Normalize action using per-dimension statistics.
        
        Useful when actions are not chunked.
        
        Args:
            action: Action, shape (..., action_dim)
            
        Returns:
            Normalized action
        """
        return (action - self.action_dim_mean) / self.action_dim_std

    def denormalize_action_per_dim(self, action: Tensor) -> Tensor:
        """Denormalize per-dimension normalized action."""
        return action * self.action_dim_std + self.action_dim_mean

    def save(self, path: Union[str, Path]) -> None:
        """
        Save normalizer statistics to file.
        
        Args:
            path: Path to save file (.pt or .json)
        """
        path = Path(path)
        
        stats = {
            "state_mean": self.state_mean.cpu().numpy().tolist(),
            "state_std": self.state_std.cpu().numpy().tolist(),
            "action_mean": self.action_mean.cpu().numpy().tolist(),
            "action_std": self.action_std.cpu().numpy().tolist(),
            "action_dim_mean": self.action_dim_mean.cpu().numpy().tolist(),
            "action_dim_std": self.action_dim_std.cpu().numpy().tolist(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
        }
        
        if path.suffix == ".pt":
            torch.save(stats, path)
        else:
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load normalizer statistics from file.
        
        Args:
            path: Path to saved statistics
        """
        path = Path(path)
        
        if path.suffix == ".pt":
            stats = torch.load(path)
        else:
            with open(path, "r") as f:
                stats = json.load(f)
        
        self.state_mean = torch.tensor(stats["state_mean"])
        self.state_std = torch.tensor(stats["state_std"])
        self.action_mean = torch.tensor(stats["action_mean"])
        self.action_std = torch.tensor(stats["action_std"])
        self.action_dim_mean = torch.tensor(stats["action_dim_mean"])
        self.action_dim_std = torch.tensor(stats["action_dim_std"])
        
        self._fitted = True

    def get_stats_dict(self) -> Dict[str, Tensor]:
        """Get statistics as dictionary."""
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "action_dim_mean": self.action_dim_mean,
            "action_dim_std": self.action_dim_std,
        }

    def __repr__(self) -> str:
        return (
            f"StateActionNormalizer("
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"action_horizon={self.action_horizon}, "
            f"fitted={self._fitted})"
        )

