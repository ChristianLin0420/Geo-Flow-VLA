"""
Base Dataset Interface for Manipulation Benchmarks.

Provides a unified interface for LIBERO, RLBench, CALVIN, and future datasets.
All datasets return standardized trajectory data for training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np


@dataclass
class TrajectoryData:
    """
    Standardized trajectory data structure.
    
    All manipulation datasets return data in this format for
    consistent training across benchmarks.
    """
    
    # Visual observations
    rgb: Tensor                    # (T, C, H, W) or (C, H, W) for single frame
    depth: Optional[Tensor]        # (T, 1, H, W) or None
    
    # Robot state
    proprio: Tensor                # (T, proprio_dim) - joint positions, gripper state, etc.
    
    # Language instruction
    instruction: str               # Natural language task description
    instruction_embedding: Optional[Tensor]  # (embed_dim,) pre-computed embedding
    
    # Actions
    actions: Tensor                # (T, action_dim) - ground truth actions
    
    # Metadata
    task_name: str                 # Task identifier
    episode_id: int                # Episode number
    
    # Optional additional data
    point_cloud: Optional[Tensor] = None  # (T, N, 3) or pre-computed from MoGe
    success: bool = True           # Whether episode was successful
    
    def to(self, device: torch.device) -> "TrajectoryData":
        """Move all tensors to device."""
        return TrajectoryData(
            rgb=self.rgb.to(device),
            depth=self.depth.to(device) if self.depth is not None else None,
            proprio=self.proprio.to(device),
            instruction=self.instruction,
            instruction_embedding=self.instruction_embedding.to(device) if self.instruction_embedding is not None else None,
            actions=self.actions.to(device),
            task_name=self.task_name,
            episode_id=self.episode_id,
            point_cloud=self.point_cloud.to(device) if self.point_cloud is not None else None,
            success=self.success,
        )


class BaseManipulationDataset(ABC, Dataset):
    """
    Abstract base class for manipulation datasets.
    
    Provides:
    - Unified interface for trajectory data
    - Action chunking with configurable horizon
    - Data augmentation hooks
    - Caching utilities
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        action_horizon: int = 16,
        action_dim: int = 7,
        image_size: int = 224,
        chunk_overlap: int = 8,
        cache_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        """
        Args:
            data_root: Root directory for dataset
            split: Data split ("train", "val", "test")
            action_horizon: Number of action steps per sample
            action_dim: Dimension of action vector
            image_size: Target image size
            chunk_overlap: Overlap between consecutive action chunks
            cache_dir: Optional directory for caching processed data
            transform: Optional transform to apply to images
        """
        self.data_root = Path(data_root)
        self.split = split
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.image_size = image_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform
        
        # To be populated by subclasses
        self.episodes: List[Dict[str, Any]] = []
        self.task_names: List[str] = []
        
        # Build index for efficient sampling
        self._sample_index: List[Tuple[int, int]] = []  # (episode_idx, start_frame)

    @abstractmethod
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """
        Load episode metadata from disk.
        
        Returns:
            List of episode dictionaries with paths and metadata
        """
        pass

    @abstractmethod
    def _load_episode_data(
        self,
        episode: Dict[str, Any],
        start_idx: int,
        length: int,
    ) -> TrajectoryData:
        """
        Load actual data for an episode chunk.
        
        Args:
            episode: Episode metadata dict
            start_idx: Starting frame index
            length: Number of frames to load
            
        Returns:
            TrajectoryData for the chunk
        """
        pass

    def _build_sample_index(self) -> None:
        """Build index mapping sample idx to (episode, frame) pairs."""
        self._sample_index = []
        
        for ep_idx, episode in enumerate(self.episodes):
            ep_length = episode.get("length", 100)
            step = self.action_horizon - self.chunk_overlap
            
            for start in range(0, ep_length - self.action_horizon + 1, step):
                self._sample_index.append((ep_idx, start))
        
        print(f"Built sample index with {len(self._sample_index)} samples "
              f"from {len(self.episodes)} episodes")

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, str, Tensor]:
        """
        Get a training sample.
        
        Returns:
            Tuple of (rgb, depth, proprio, instruction, actions)
            - rgb: (C, H, W) or (T, C, H, W)
            - depth: (1, H, W) or (T, 1, H, W) or None
            - proprio: (proprio_dim,) or (T, proprio_dim)
            - instruction: str
            - actions: (action_horizon, action_dim)
        """
        ep_idx, start_frame = self._sample_index[idx]
        episode = self.episodes[ep_idx]
        
        data = self._load_episode_data(
            episode,
            start_frame,
            self.action_horizon,
        )
        
        # Apply transforms
        rgb = data.rgb
        if self.transform is not None:
            rgb = self.transform(rgb)
        
        # Return in standard format
        return (
            rgb,
            data.depth if data.depth is not None else torch.zeros(1),
            data.proprio,
            data.instruction,
            data.actions,
        )

    def get_trajectory(self, episode_idx: int) -> TrajectoryData:
        """Get full trajectory for an episode."""
        episode = self.episodes[episode_idx]
        return self._load_episode_data(episode, 0, episode["length"])

    @property
    def num_tasks(self) -> int:
        """Number of unique tasks."""
        return len(set(self.task_names))

    @property
    def proprio_dim(self) -> int:
        """Dimension of proprioceptive state."""
        # Default, override in subclasses
        return 7

    def get_task_episodes(self, task_name: str) -> List[int]:
        """Get episode indices for a specific task."""
        return [
            i for i, ep in enumerate(self.episodes)
            if ep.get("task_name") == task_name
        ]

    def get_normalizer_stats(self) -> Dict[str, Tensor]:
        """
        Compute normalization statistics from dataset.
        
        Returns:
            Dictionary with action mean/std
        """
        all_actions = []
        
        for ep in self.episodes:
            data = self._load_episode_data(ep, 0, ep["length"])
            all_actions.append(data.actions)
        
        all_actions = torch.cat(all_actions, dim=0)
        
        return {
            "action_mean": all_actions.mean(dim=0),
            "action_std": all_actions.std(dim=0),
        }


class ChunkedTrajectoryDataset(BaseManipulationDataset):
    """
    Dataset wrapper that returns action chunks.
    
    Given a base dataset, this wrapper handles:
    - Slicing trajectories into overlapping chunks
    - Proper padding at trajectory boundaries
    - Efficient memory usage with lazy loading
    """

    def __init__(
        self,
        base_dataset: BaseManipulationDataset,
        chunk_size: int = 16,
        overlap: int = 8,
        pad_mode: str = "last",
    ) -> None:
        """
        Args:
            base_dataset: Underlying trajectory dataset
            chunk_size: Number of steps per chunk
            overlap: Overlap between consecutive chunks
            pad_mode: How to pad short sequences ("last", "zero", "repeat")
        """
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.pad_mode = pad_mode
        
        # Build chunk index
        self._chunks: List[Tuple[int, int]] = []
        self._build_chunk_index()

    def _build_chunk_index(self) -> None:
        """Build index of (episode_idx, start_frame) for each chunk."""
        step = self.chunk_size - self.overlap
        
        for ep_idx, episode in enumerate(self.base_dataset.episodes):
            ep_length = episode.get("length", 100)
            
            for start in range(0, ep_length, step):
                self._chunks.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        ep_idx, start = self._chunks[idx]
        episode = self.base_dataset.episodes[ep_idx]
        ep_length = episode.get("length", 100)
        
        # Calculate actual available length
        available = min(self.chunk_size, ep_length - start)
        
        # Load chunk
        data = self.base_dataset._load_episode_data(episode, start, available)
        
        # Pad if necessary
        if available < self.chunk_size:
            data = self._pad_chunk(data, self.chunk_size - available)
        
        return (
            data.rgb,
            data.depth if data.depth is not None else torch.zeros(1),
            data.proprio,
            data.instruction,
            data.actions,
        )

    def _pad_chunk(self, data: TrajectoryData, pad_length: int) -> TrajectoryData:
        """Pad trajectory data to full chunk size."""
        if self.pad_mode == "last":
            # Repeat last frame/action
            rgb_pad = data.rgb[-1:].repeat(pad_length, 1, 1, 1)
            action_pad = data.actions[-1:].repeat(pad_length, 1)
            proprio_pad = data.proprio[-1:].repeat(pad_length, 1)
        elif self.pad_mode == "zero":
            # Zero padding
            rgb_pad = torch.zeros(pad_length, *data.rgb.shape[1:])
            action_pad = torch.zeros(pad_length, data.actions.shape[-1])
            proprio_pad = torch.zeros(pad_length, data.proprio.shape[-1])
        else:
            raise ValueError(f"Unknown pad_mode: {self.pad_mode}")
        
        return TrajectoryData(
            rgb=torch.cat([data.rgb, rgb_pad], dim=0),
            depth=torch.cat([data.depth, torch.zeros(pad_length, *data.depth.shape[1:])], dim=0) if data.depth is not None else None,
            proprio=torch.cat([data.proprio, proprio_pad], dim=0),
            instruction=data.instruction,
            instruction_embedding=data.instruction_embedding,
            actions=torch.cat([data.actions, action_pad], dim=0),
            task_name=data.task_name,
            episode_id=data.episode_id,
            success=data.success,
        )

