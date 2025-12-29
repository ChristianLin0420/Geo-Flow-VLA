"""
RLBench Dataset Implementation.

RLBench is a large-scale benchmark for robot learning with 100+ tasks
in simulation using CoppeliaSim.

This implementation supports the HuggingFace RLBench-18-Tasks dataset:
    https://huggingface.co/datasets/hqfang/rlbench-18-tasks

Reference:
    James et al., "RLBench: The Robot Learning Benchmark" IEEE RA-L 2020
    GitHub: https://github.com/stepjam/RLBench
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import pickle
import io

import torch
from torch import Tensor
import numpy as np
from PIL import Image

from .base_dataset import BaseManipulationDataset, TrajectoryData


class RLBenchObservation:
    """
    Minimal RLBench Observation class for unpickling low_dim_obs.pkl files.
    
    This allows loading RLBench data without the full rlbench/pyrep dependencies.
    """
    def __init__(self):
        # Arm and gripper state
        self.joint_velocities = None
        self.joint_positions = None
        self.joint_forces = None
        self.gripper_open = None
        self.gripper_pose = None
        self.gripper_matrix = None
        self.gripper_touch_forces = None
        self.gripper_joint_positions = None
        
        # Task state
        self.task_low_dim_state = None
        self.misc = {}
        
    def __setstate__(self, state):
        """Allow pickle to restore any attributes."""
        self.__dict__.update(state)
    
    def __getstate__(self):
        return self.__dict__


class RLBenchUnpickler(pickle.Unpickler):
    """
    Custom unpickler that substitutes RLBench classes with our minimal versions.
    
    This avoids the need to install rlbench/pyrep which require CoppeliaSim.
    """
    
    # Map of (module, class) -> replacement class
    CLASS_MAPPING = {
        ('rlbench.backend.observation', 'Observation'): RLBenchObservation,
        ('rlbench.observation_config', 'ObservationConfig'): dict,
        ('pyrep.objects.dummy', 'Dummy'): dict,
        ('pyrep.objects.vision_sensor', 'VisionSensor'): dict,
    }
    
    def find_class(self, module: str, name: str):
        """Override class lookup to substitute missing RLBench classes."""
        key = (module, name)
        
        if key in self.CLASS_MAPPING:
            return self.CLASS_MAPPING[key]
        
        # For any other rlbench/pyrep class, return a generic dict-like class
        if module.startswith('rlbench') or module.startswith('pyrep'):
            return RLBenchObservation
        
        # Default behavior for other classes
        return super().find_class(module, name)


def rlbench_load_pickle(filepath: Path) -> Any:
    """Load a pickle file that may contain RLBench objects."""
    with open(filepath, 'rb') as f:
        return RLBenchUnpickler(f).load()

logger = logging.getLogger(__name__)


# RLBench-18-Tasks from HuggingFace (verified from repo)
RLBENCH_18_TASKS = [
    "close_jar",
    "insert_onto_square_peg",
    "light_bulb_in",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_shape_in_shape_sorter",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "reach_and_drag",
    "slide_block_to_color_target",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]

# Task categories
RLBENCH_TASK_CATEGORIES = {
    "easy": [
        "close_jar",
        "light_bulb_in",
        "open_drawer",
        "push_buttons",
        "put_money_in_safe",
        "turn_tap",
    ],
    "medium": [
        "insert_onto_square_peg",
        "meat_off_grill",
        "place_cups",
        "put_groceries_in_cupboard",
        "stack_blocks",
        "stack_cups",
    ],
    "hard": [
        "place_shape_in_shape_sorter",
        "place_wine_at_rack_location",
        "put_item_in_drawer",
        "reach_and_drag",
        "slide_block_to_color_target",
        "sweep_to_dustpan_of_size",
    ],
    "all": RLBENCH_18_TASKS,
}


class RLBenchDataset(BaseManipulationDataset):
    """
    PyTorch Dataset for RLBench benchmark.
    
    Supports the HuggingFace RLBench-18-Tasks dataset format:
        data_root/
            train/
                task_name/
                    all_variations/
                        episodes/
                            episode0/
                                front_rgb/
                                low_dim_obs.pkl
                                variation_descriptions.pkl
            val/
            test/
    
    Action format: 8D (3D position + 4D quaternion + 1D gripper)
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        tasks: Optional[List[str]] = None,
        split: str = "train",
        action_horizon: int = 16,
        image_size: int = 224,
        camera: str = "front_rgb",
        cache_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            data_root: Root directory for RLBench data (e.g., ./data/rlbench)
            tasks: List of tasks to load (None = all available, or 'easy', 'medium', 'hard')
            split: Data split ('train', 'val', 'test')
            action_horizon: Action chunk size
            image_size: Image resolution
            camera: Camera view to use ('front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb')
            cache_dir: Cache directory
            transform: Image transforms
        """
        super().__init__(
            data_root=data_root,
            split=split,
            action_horizon=action_horizon,
            action_dim=8,  # RLBench uses 8D actions (pos + quat + gripper)
            image_size=image_size,
            cache_dir=cache_dir,
            transform=transform,
        )
        
        self.camera = camera
        
        # Resolve task list
        if tasks is None:
            self.tasks = RLBENCH_18_TASKS
        elif isinstance(tasks, str):
            if tasks in RLBENCH_TASK_CATEGORIES:
                self.tasks = RLBENCH_TASK_CATEGORIES[tasks]
            else:
                self.tasks = [tasks]
        else:
            resolved_tasks = []
            for t in tasks:
                if t in RLBENCH_TASK_CATEGORIES:
                    resolved_tasks.extend(RLBENCH_TASK_CATEGORIES[t])
                else:
                    resolved_tasks.append(t)
            self.tasks = list(set(resolved_tasks))
        
        # Build path to split directory
        self.split_dir = self.data_root / split
        
        # Check if data exists
        if not self.split_dir.exists():
            logger.warning(
                f"RLBench data not found at {self.split_dir}. "
                "Please download the dataset with: "
                "python -m geo_flow_vla.data.download.download_rlbench --task all --output ./data/rlbench"
            )
            self.episodes = []
            return
        
        self.episodes = self._load_episodes()
        self._build_sample_index()
        
        logger.info(f"Loaded RLBench {split} with {len(self.episodes)} episodes from {len(self.task_names)} tasks")

    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load episode metadata from RLBench data directory."""
        episodes = []
        
        # RLBench HuggingFace dataset structure:
        # split/task_name/all_variations/episodes/episode0/
        #   front_rgb/, left_shoulder_rgb/, right_shoulder_rgb/, wrist_rgb/
        #   low_dim_obs.pkl
        #   variation_descriptions.pkl
        
        for task_name in self.tasks:
            task_dir = self.split_dir / task_name
            
            if not task_dir.exists():
                logger.warning(f"Task not found: {task_name}")
                continue
            
            self.task_names.append(task_name)
            
            # Check for all_variations structure (HuggingFace format)
            all_var_dir = task_dir / "all_variations" / "episodes"
            if all_var_dir.exists():
                episode_base = all_var_dir
            else:
                # Also support direct episode structure
                episode_base = task_dir
            
            # Find all episode directories
            episode_dirs = sorted(episode_base.glob("episode*"))
            
            for ep_dir in episode_dirs:
                # Check for required files
                low_dim_path = ep_dir / "low_dim_obs.pkl"
                camera_dir = ep_dir / self.camera
                
                if not low_dim_path.exists():
                    logger.debug(f"Missing low_dim_obs.pkl in {ep_dir}")
                    continue
                
                if not camera_dir.exists():
                    # Try alternative camera
                    for alt_camera in ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]:
                        alt_camera_dir = ep_dir / alt_camera
                        if alt_camera_dir.exists():
                            camera_dir = alt_camera_dir
                            break
                    else:
                        logger.debug(f"No camera images found in {ep_dir}")
                        continue
                
                # Get episode length from image files
                rgb_files = sorted(camera_dir.glob("*.png")) + sorted(camera_dir.glob("*.jpg"))
                length = len(rgb_files)
                
                if length == 0:
                    continue
                
                # Get episode ID
                ep_id_str = ep_dir.name.replace("episode", "")
                try:
                    ep_id = int(ep_id_str)
                except ValueError:
                    ep_id = hash(ep_dir.name) % 10000
                
                # Try to get task description
                desc_path = ep_dir / "variation_descriptions.pkl"
                description = task_name.replace("_", " ")
                if desc_path.exists():
                    try:
                        # Use custom unpickler for RLBench compatibility
                        desc_data = rlbench_load_pickle(desc_path)
                        if isinstance(desc_data, list) and len(desc_data) > 0:
                            description = desc_data[0]
                        elif isinstance(desc_data, str):
                            description = desc_data
                    except Exception:
                        pass
                
                episodes.append({
                    "path": ep_dir,
                    "task_name": task_name,
                    "length": length,
                    "episode_id": ep_id,
                    "description": description,
                    "camera_dir": camera_dir,
                })
        
        return episodes

    def _load_episode_data(
        self,
        episode: Dict[str, Any],
        start_idx: int,
        length: int,
    ) -> TrajectoryData:
        """Load actual trajectory data from RLBench episode directory."""
        path = episode["path"]
        camera_dir = episode["camera_dir"]
        
        end_idx = min(start_idx + length, episode["length"])
        actual_length = end_idx - start_idx
        
        try:
            # Load low_dim_obs.pkl for actions and proprioception
            # Use custom unpickler to avoid requiring full rlbench/pyrep installation
            low_dim_path = path / "low_dim_obs.pkl"
            low_dim_data = rlbench_load_pickle(low_dim_path)
            
            # Handle different data structures for low_dim_obs
            # The pickle contains a wrapper object with _observations attribute
            if hasattr(low_dim_data, '_observations'):
                # RLBench Demo wrapper - extract the observations list
                observations = low_dim_data._observations
            elif isinstance(low_dim_data, (list, tuple)):
                observations = low_dim_data
            elif isinstance(low_dim_data, np.ndarray):
                observations = list(low_dim_data)
            elif hasattr(low_dim_data, '__iter__') and not isinstance(low_dim_data, (str, dict, RLBenchObservation)):
                observations = list(low_dim_data)
            else:
                # Single observation - repeat it for all frames
                observations = [low_dim_data] * episode["length"]
            
            num_obs = len(observations)
            
            # Load RGB images
            rgb_files = sorted(camera_dir.glob("*.png")) + sorted(camera_dir.glob("*.jpg"))
            rgb_files = rgb_files[start_idx:end_idx]
            
            rgb_frames = []
            for img_path in rgb_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW
                    rgb_frames.append(img_tensor)
                except Exception:
                    # Use black image on failure
                    rgb_frames.append(torch.zeros(3, self.image_size, self.image_size))
            
            rgb = torch.stack(rgb_frames) if rgb_frames else torch.zeros(actual_length, 3, self.image_size, self.image_size)
            
            # Extract actions and proprioception from low_dim_obs
            # RLBench low_dim_obs format varies, handle different structures
            actions_list = []
            proprio_list = []
            
            def get_obs_attr(obs, attr_name, default=None):
                """Safely get attribute from observation (handles class and dict)."""
                val = None
                if hasattr(obs, attr_name):
                    val = getattr(obs, attr_name)
                elif isinstance(obs, dict) and attr_name in obs:
                    val = obs[attr_name]
                
                if val is None:
                    return default
                
                # Convert to numpy and flatten to 1D
                if isinstance(val, (int, float)):
                    return val
                
                arr = np.array(val)
                # Flatten multi-dimensional arrays to 1D
                if arr.ndim > 1:
                    arr = arr.flatten()
                return arr
            
            for i in range(start_idx, end_idx):
                if i >= num_obs:
                    # Pad with last observation if needed
                    obs_idx = num_obs - 1
                else:
                    obs_idx = i
                
                obs = observations[obs_idx]
                
                # Extract gripper pose as action (3D pos + 4D quat + 1D gripper)
                gripper_pose = get_obs_attr(obs, 'gripper_pose', np.zeros(7))
                gripper_open = get_obs_attr(obs, 'gripper_open', 1.0)
                
                # Ensure gripper_pose is the right shape
                if not isinstance(gripper_pose, np.ndarray):
                    gripper_pose = np.array(gripper_pose)
                if gripper_pose.shape != (7,):
                    gripper_pose = np.zeros(7)
                
                # Ensure gripper_open is scalar
                if isinstance(gripper_open, np.ndarray):
                    gripper_open = float(gripper_open.flatten()[0]) if gripper_open.size > 0 else 1.0
                
                action = np.concatenate([gripper_pose, [float(gripper_open)]])
                actions_list.append(action)
                
                # Extract proprioception (joint positions)
                proprio = get_obs_attr(obs, 'joint_positions', np.zeros(7))
                
                if not isinstance(proprio, np.ndarray):
                    proprio = np.array(proprio)
                
                # Flatten if needed
                proprio = proprio.flatten()
                
                # Pad proprio to 8D to match action dim
                if len(proprio) < 8:
                    proprio = np.concatenate([proprio, np.zeros(8 - len(proprio))])
                elif len(proprio) > 8:
                    proprio = proprio[:8]
                
                proprio_list.append(proprio)
            
            actions = torch.tensor(np.array(actions_list), dtype=torch.float32)
            proprio = torch.tensor(np.array(proprio_list), dtype=torch.float32)
            
            # Placeholder depth (RLBench has depth but may not always be available)
            depth = torch.zeros(actual_length, 1, self.image_size, self.image_size)
            
            # Try to load depth if available
            depth_dir = path / self.camera.replace("_rgb", "_depth")
            if depth_dir.exists():
                depth_files = sorted(depth_dir.glob("*.png")) + sorted(depth_dir.glob("*.npy"))
                depth_files = depth_files[start_idx:end_idx]
                
                depth_frames = []
                for depth_path in depth_files:
                    try:
                        if depth_path.suffix == '.npy':
                            d = np.load(depth_path)
                        else:
                            d = np.array(Image.open(depth_path))
                        
                        # Ensure 2D array for depth
                        if d.ndim == 3:
                            d = d[:, :, 0]  # Take first channel
                        
                        # Convert to tensor and resize using torch interpolation
                        d_tensor = torch.from_numpy(d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                        if d_tensor.shape[-2] != self.image_size or d_tensor.shape[-1] != self.image_size:
                            d_tensor = torch.nn.functional.interpolate(
                                d_tensor, 
                                size=(self.image_size, self.image_size), 
                                mode='nearest'
                            )
                        d_tensor = d_tensor.squeeze(0)  # Remove batch dim, keep channel dim
                        
                        # Normalize depth
                        if d_tensor.max() > 0:
                            d_tensor = d_tensor / d_tensor.max()
                        
                        depth_frames.append(d_tensor)
                    except Exception:
                        # Skip problematic depth frames, use zeros
                        depth_frames.append(torch.zeros(1, self.image_size, self.image_size))
                
                if depth_frames:
                    depth = torch.stack(depth_frames)
            
            return TrajectoryData(
                rgb=rgb,
                depth=depth,
                proprio=proprio,
                instruction=episode["description"],
                instruction_embedding=None,
                actions=actions,
                task_name=episode["task_name"],
                episode_id=episode["episode_id"],
                success=True,  # RLBench demos are successful
            )
            
        except Exception as e:
            logger.warning(f"Failed to load episode {episode['episode_id']}: {e}")
            # Return dummy data on failure
            return TrajectoryData(
                rgb=torch.zeros(actual_length, 3, self.image_size, self.image_size),
                depth=torch.zeros(actual_length, 1, self.image_size, self.image_size),
                proprio=torch.zeros(actual_length, 8),
                instruction=episode["task_name"].replace("_", " "),
                instruction_embedding=None,
                actions=torch.zeros(actual_length, 8),
                task_name=episode["task_name"],
                episode_id=episode["episode_id"],
                success=True,
            )

    @property
    def proprio_dim(self) -> int:
        """Dimension of proprioceptive state in RLBench."""
        return 8  # Joint positions + gripper

    @staticmethod
    def get_available_tasks() -> List[str]:
        """Get list of all RLBench-18-Tasks."""
        return RLBENCH_18_TASKS

    @staticmethod
    def get_task_categories() -> Dict[str, List[str]]:
        """Get task categories."""
        return RLBENCH_TASK_CATEGORIES
