"""
LIBERO Dataset Implementation for Geo-Flow VLA.

LIBERO is a benchmark for lifelong robot learning with 130 tasks
across 4 task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long.

Reference:
    Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning" NeurIPS 2023
    GitHub: https://github.com/Lifelong-Robot-Learning/LIBERO
    HuggingFace: https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
from torch import Tensor
import numpy as np
import h5py

from .base_dataset import BaseManipulationDataset, TrajectoryData

logger = logging.getLogger(__name__)


# LIBERO task suite definitions
# Reference: https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
LIBERO_SUITES = {
    "spatial": ["libero_spatial"],   # 10 tasks focusing on spatial relationships
    "object": ["libero_object"],     # 10 tasks focusing on object manipulation
    "goal": ["libero_goal"],         # 10 tasks focusing on goal-conditioned behavior
    "10": ["libero_10"],             # 10 test tasks for lifelong learning evaluation
    "90": ["libero_90"],             # 90 diverse pretraining tasks
    "100": ["libero_90", "libero_10"],  # All 100 tasks (LIBERO-100)
    "all": [                         # All evaluation suites (30 tasks)
        "libero_spatial",
        "libero_object", 
        "libero_goal",
    ],
    "full": [                        # Everything (130 tasks)
        "libero_spatial",
        "libero_object", 
        "libero_goal",
        "libero_10",
        "libero_90",
    ],
}


class LIBERODataset(BaseManipulationDataset):
    """
    PyTorch Dataset for LIBERO benchmark.
    
    Loads demonstrations from HDF5 files and returns standardized
    trajectory data for policy training.
    
    LIBERO HDF5 structure:
        data/demo_X/
            actions: (T, 7) - 3D pos + 3D rot + gripper
            obs/agentview_rgb: (T, 128, 128, 3)
            obs/eye_in_hand_rgb: (T, 128, 128, 3)
            obs/ee_pos: (T, 3) - end-effector position
            obs/ee_ori: (T, 3) - end-effector orientation
            obs/joint_states: (T, 7) - joint positions
            obs/gripper_states: (T, 2) - gripper state
            robot_states: (T, 9) - full robot state
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        suite: str = "all",
        split: str = "train",
        action_horizon: int = 16,
        image_size: int = 224,
        chunk_overlap: int = 8,
        use_eye_in_hand: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Any] = None,
        max_episodes_per_task: Optional[int] = None,
    ) -> None:
        """
        Args:
            data_root: Root directory containing LIBERO data
            suite: Task suite - "spatial", "object", "goal", "10", "90", "100", or "all"
            split: Data split
            action_horizon: Action chunk size
            image_size: Target image resolution
            chunk_overlap: Overlap between action chunks
            use_eye_in_hand: Whether to use eye-in-hand camera
            cache_dir: Optional cache directory
            transform: Optional image transforms
            max_episodes_per_task: Limit episodes per task (for debugging)
        """
        super().__init__(
            data_root=data_root,
            split=split,
            action_horizon=action_horizon,
            action_dim=7,  # LIBERO uses 7D actions
            image_size=image_size,
            chunk_overlap=chunk_overlap,
            cache_dir=cache_dir,
            transform=transform,
        )
        
        self.suite = suite
        self.use_eye_in_hand = use_eye_in_hand
        self.max_episodes_per_task = max_episodes_per_task
        
        # Validate suite
        if suite not in LIBERO_SUITES:
            raise ValueError(f"Unknown suite '{suite}'. Available: {list(LIBERO_SUITES.keys())}")
        
        # Load episodes
        self.episodes = self._load_episodes()
        self._build_sample_index()
        
        logger.info(f"Loaded LIBERO {suite} with {len(self.episodes)} episodes, {len(self)} samples")

    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load episode metadata from LIBERO HDF5 files."""
        episodes = []
        
        suite_dirs = LIBERO_SUITES[self.suite]
        
        for suite_name in suite_dirs:
            suite_path = self.data_root / suite_name
            
            if not suite_path.exists():
                logger.warning(f"Suite directory not found: {suite_path}")
                continue
            
            # LIBERO structure: each HDF5 file is a task with multiple demos
            hdf5_files = sorted(suite_path.glob("*.hdf5"))
            
            if not hdf5_files:
                # Try subdirectory structure
                hdf5_files = sorted(suite_path.glob("*/*.hdf5"))
            
            for hdf5_file in hdf5_files:
                task_name = hdf5_file.stem.replace("_demo", "")
                
                if task_name not in self.task_names:
                    self.task_names.append(task_name)
                
                try:
                    with h5py.File(hdf5_file, 'r') as f:
                        data_group = f['data']
                        
                        # Each HDF5 contains multiple demos
                        demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                        demo_keys = sorted(demo_keys, key=lambda x: int(x.split('_')[1]))
                        
                        for i, demo_key in enumerate(demo_keys):
                            if self.max_episodes_per_task and i >= self.max_episodes_per_task:
                                break
                            
                            demo = data_group[demo_key]
                            length = demo['actions'].shape[0]
                            
                            episodes.append({
                                "path": hdf5_file,
                                "demo_key": demo_key,
                                "task_name": task_name,
                                "suite": suite_name,
                                "length": length,
                                "episode_id": i,
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to load {hdf5_file}: {e}")
        
        return episodes

    def _load_episode_data(
        self,
        episode: Dict[str, Any],
        start_idx: int,
        length: int,
    ) -> TrajectoryData:
        """Load actual trajectory data from HDF5 file."""
        path = episode["path"]
        demo_key = episode["demo_key"]
        
        with h5py.File(path, 'r') as f:
            demo = f['data'][demo_key]
            obs = demo['obs']
            
            end_idx = min(start_idx + length, episode["length"])
            actual_length = end_idx - start_idx
            
            # Load RGB image (agentview camera)
            rgb = torch.from_numpy(
                obs['agentview_rgb'][start_idx:end_idx]
            ).float() / 255.0  # (T, H, W, C)
            rgb = rgb.permute(0, 3, 1, 2)  # (T, C, H, W)
            
            # Optionally add eye-in-hand view
            if self.use_eye_in_hand and 'eye_in_hand_rgb' in obs:
                eih_rgb = torch.from_numpy(
                    obs['eye_in_hand_rgb'][start_idx:end_idx]
                ).float() / 255.0
                eih_rgb = eih_rgb.permute(0, 3, 1, 2)
                rgb = torch.cat([rgb, eih_rgb], dim=1)  # Concatenate channels
            
            # Load proprioception: ee_pos (3) + ee_ori (3) + gripper (2) = 8D
            proprio_parts = []
            
            if 'ee_pos' in obs:
                proprio_parts.append(
                    torch.from_numpy(obs['ee_pos'][start_idx:end_idx]).float()
                )
            if 'ee_ori' in obs:
                proprio_parts.append(
                    torch.from_numpy(obs['ee_ori'][start_idx:end_idx]).float()
                )
            if 'gripper_states' in obs:
                proprio_parts.append(
                    torch.from_numpy(obs['gripper_states'][start_idx:end_idx]).float()
                )
            elif 'ee_states' in obs:
                # ee_states contains gripper info
                proprio_parts.append(
                    torch.from_numpy(obs['ee_states'][start_idx:end_idx]).float()
                )
            
            if proprio_parts:
                proprio = torch.cat(proprio_parts, dim=-1)
            else:
                # Fallback: use robot state
                proprio = torch.from_numpy(
                    demo['robot_states'][start_idx:end_idx]
                ).float()
            
            # Load actions (7D: xyz delta + axis-angle delta + gripper)
            actions = torch.from_numpy(
                demo['actions'][start_idx:end_idx]
            ).float()
            
            # Get instruction from task name
            # LIBERO uses descriptive task names as instructions
            instruction = episode["task_name"].replace("_", " ")
        
        # Resize images if needed
        if rgb.shape[-1] != self.image_size or rgb.shape[-2] != self.image_size:
            rgb = torch.nn.functional.interpolate(
                rgb,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
        
        return TrajectoryData(
            rgb=rgb,
            depth=None,  # LIBERO doesn't provide depth in standard format
            proprio=proprio,
            instruction=instruction,
            instruction_embedding=None,  # Computed later if needed
            actions=actions,
            task_name=episode["task_name"],
            episode_id=episode["episode_id"],
            success=True,  # LIBERO demos are successful by design
        )

    @property
    def proprio_dim(self) -> int:
        """Dimension of proprioceptive state in LIBERO."""
        return 8  # ee_pos(3) + ee_ori(3) + gripper(2)

    def get_task_instruction(self, task_name: str) -> str:
        """Get natural language instruction for a task."""
        return task_name.replace("_", " ")

    @staticmethod
    def get_suite_tasks(suite: str) -> List[str]:
        """Get list of task suite directories."""
        if suite not in LIBERO_SUITES:
            raise ValueError(f"Unknown suite: {suite}")
        return LIBERO_SUITES[suite]


class LIBEROSingleTaskDataset(LIBERODataset):
    """
    LIBERO dataset for a single task.
    
    Useful for task-specific fine-tuning or analysis.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        task_name: str,
        **kwargs,
    ) -> None:
        """
        Args:
            data_root: Root LIBERO directory
            task_name: Specific task to load
            **kwargs: Additional arguments for LIBERODataset
        """
        self.target_task = task_name
        super().__init__(data_root=data_root, suite="all", **kwargs)
        
        # Filter to single task
        self.episodes = [ep for ep in self.episodes if ep["task_name"] == task_name]
        self._build_sample_index()
        
        if len(self.episodes) == 0:
            raise ValueError(f"No episodes found for task: {task_name}")


def create_libero_dataloaders(
    data_root: Union[str, Path],
    suite: str = "all",
    batch_size: int = 32,
    num_workers: int = 8,
    action_horizon: int = 16,
    image_size: int = 224,
    val_ratio: float = 0.1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for LIBERO.
    
    Args:
        data_root: Path to LIBERO data
        suite: Task suite
        batch_size: Batch size
        num_workers: Number of data loading workers
        action_horizon: Action chunk size
        image_size: Image resolution
        val_ratio: Fraction of data for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    full_dataset = LIBERODataset(
        data_root=data_root,
        suite=suite,
        split="train",
        action_horizon=action_horizon,
        image_size=image_size,
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
