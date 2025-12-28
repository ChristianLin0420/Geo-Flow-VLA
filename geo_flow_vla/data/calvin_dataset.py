"""
CALVIN Dataset Implementation for Geo-Flow VLA.

CALVIN is a benchmark for learning long-horizon language-conditioned tasks
in a simulated kitchen environment.

Reference:
    Mees et al., "CALVIN: A Benchmark for Language-conditioned Policy Learning 
    for Long-Horizon Robot Manipulation Tasks" IEEE RA-L 2022
    GitHub: https://github.com/mees/calvin

This implementation supports LeRobot format (parquet files) from HuggingFace.
Source: https://huggingface.co/fywang
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import json
import io

import torch
from torch import Tensor
import numpy as np
from PIL import Image

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from .base_dataset import BaseManipulationDataset, TrajectoryData

logger = logging.getLogger(__name__)


# CALVIN environment configurations
CALVIN_ENVS = {
    "A": "environment_A",  # Training environment
    "B": "environment_B",  # Transfer environment
    "C": "environment_C",  # Transfer environment
    "D": "environment_D",  # Transfer environment
    "ABC": ["environment_A", "environment_B", "environment_C"],  # Multi-env training
    "ABCD": ["environment_A", "environment_B", "environment_C", "environment_D"],
}

# CALVIN task categories
CALVIN_TASKS = [
    "rotate_red_block_right",
    "rotate_red_block_left",
    "rotate_blue_block_right",
    "rotate_blue_block_left",
    "push_red_block_right",
    "push_red_block_left",
    "push_blue_block_right",
    "push_blue_block_left",
    "move_slider_left",
    "move_slider_right",
    "open_drawer",
    "close_drawer",
    "lift_red_block_table",
    "lift_red_block_slider",
    "lift_red_block_drawer",
    "lift_blue_block_table",
    "lift_blue_block_slider",
    "lift_blue_block_drawer",
    "place_in_slider",
    "place_in_drawer",
    "stack_block",
    "unstack_block",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "push_into_drawer",
]


class CALVINDataset(BaseManipulationDataset):
    """
    PyTorch Dataset for CALVIN benchmark (LeRobot format).
    
    CALVIN provides long-horizon language-conditioned demonstrations
    with rich language annotations.
    
    LeRobot format features:
        - observation.images.top: (200, 200, 3) RGB from top camera
        - observation.images.wrist: (84, 84, 3) RGB from wrist camera  
        - observation.state: (15,) proprioceptive state
        - action: (7,) action vector (3D pos delta + 3D rot delta + gripper)
    
    Action format: 7D (3D position delta + 3D rotation delta + 1D gripper)
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        env: str = "D",
        split: str = "train",
        action_horizon: int = 16,
        image_size: int = 224,
        use_wrist_camera: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            data_root: Root directory for CALVIN data (LeRobot format)
            env: Environment configuration ("D", "ABC", "ABCD", "debug")
            split: Data split ("train", "validation")
            action_horizon: Action chunk size
            image_size: Image resolution (images will be resized)
            use_wrist_camera: Whether to use wrist camera in addition to top
            cache_dir: Cache directory
            transform: Image transforms
        """
        super().__init__(
            data_root=data_root,
            split=split,
            action_horizon=action_horizon,
            action_dim=7,  # CALVIN uses 7D actions
            image_size=image_size,
            cache_dir=cache_dir,
            transform=transform,
        )
        
        self.env = env
        self.use_wrist_camera = use_wrist_camera
        
        if not HAS_PYARROW:
            raise ImportError(
                "pyarrow is required for CALVIN dataset. "
                "Install with: pip install pyarrow"
            )
        
        # Load dataset metadata
        self.meta = self._load_metadata()
        
        if self.meta is None:
            logger.warning(
                f"CALVIN data not found at {self.data_root}. "
                "Please download using: python -m geo_flow_vla.data.download.download_calvin"
            )
            self.episodes = []
            return
        
        # Load episode information
        self.episodes = self._load_episodes()
        self._build_sample_index()
        
        logger.info(f"Loaded CALVIN {env} with {len(self.episodes)} episodes, {len(self)} samples")

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load dataset metadata from info.json."""
        meta_path = self.data_root / "meta" / "info.json"
        
        if not meta_path.exists():
            return None
        
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None

    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load episode metadata from LeRobot format."""
        episodes = []
        
        if self.meta is None:
            return episodes
        
        # Load episode info from episodes.jsonl
        episodes_file = self.data_root / "meta" / "episodes.jsonl"
        
        if not episodes_file.exists():
            # Fallback: scan data directory for parquet files
            return self._scan_parquet_files()
        
        try:
            with open(episodes_file, 'r') as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    ep_info = json.loads(line)
                    episode_index = ep_info.get("episode_index", line_idx)
                    length = ep_info.get("length", 0)
                    
                    # Calculate chunk and file path
                    chunks_size = self.meta.get("chunks_size", 1000)
                    chunk_idx = episode_index // chunks_size
                    
                    parquet_path = self.data_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"
                    
                    if parquet_path.exists() and length > self.action_horizon:
                        # Get task info if available
                        task_index = ep_info.get("task_index", 0)
                        
                        episodes.append({
                            "path": parquet_path,
                            "episode_index": episode_index,
                            "length": length,
                            "task_index": task_index,
                            "instruction": ep_info.get("task", "perform the manipulation task"),
                        })
                        
        except Exception as e:
            logger.warning(f"Failed to load episodes.jsonl: {e}")
            return self._scan_parquet_files()
        
        return episodes

    def _scan_parquet_files(self) -> List[Dict[str, Any]]:
        """Fallback: Scan data directory for parquet files."""
        episodes = []
        data_dir = self.data_root / "data"
        
        if not data_dir.exists():
            return episodes
        
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        
        for pf in parquet_files:
            try:
                # Read parquet to get length
                table = pq.read_table(pf)
                length = len(table)
                
                if length > self.action_horizon:
                    # Extract episode index from filename
                    ep_idx = int(pf.stem.split("_")[-1])
                    
                    episodes.append({
                        "path": pf,
                        "episode_index": ep_idx,
                        "length": length,
                        "task_index": 0,
                        "instruction": "perform the manipulation task",
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to read {pf}: {e}")
        
        return episodes

    def _decode_image(self, img_data: Any) -> np.ndarray:
        """
        Decode image data from LeRobot parquet format.
        
        LeRobot can store images as:
        - PIL.Image objects
        - Bytes (compressed JPEG/PNG)
        - Dict with 'bytes' or 'path' keys
        - Numpy arrays (rare)
        """
        if isinstance(img_data, Image.Image):
            # PIL Image - convert to numpy
            return np.array(img_data)
        
        elif isinstance(img_data, (bytes, bytearray)):
            # Compressed image bytes - decode with PIL
            return np.array(Image.open(io.BytesIO(img_data)))
        
        elif isinstance(img_data, dict):
            # Dict format - check for 'bytes' or 'path'
            if 'bytes' in img_data:
                return np.array(Image.open(io.BytesIO(img_data['bytes'])))
            elif 'path' in img_data:
                return np.array(Image.open(img_data['path']))
            else:
                raise ValueError(f"Unknown dict format: {img_data.keys()}")
        
        elif isinstance(img_data, np.ndarray):
            if img_data.dtype == object:
                # Object array - recursively decode first element
                if img_data.size == 1:
                    return self._decode_image(img_data.item())
                else:
                    # Array of objects - decode each
                    return np.array([self._decode_image(x) for x in img_data.flat])
            else:
                # Regular numpy array
                return img_data
        
        else:
            # Try direct conversion as fallback
            return np.array(img_data)

    def _load_episode_data(
        self,
        episode: Dict[str, Any],
        start_idx: int,
        length: int,
    ) -> TrajectoryData:
        """Load actual trajectory data from LeRobot parquet file."""
        path = episode["path"]
        
        end_idx = min(start_idx + length, episode["length"])
        actual_length = end_idx - start_idx
        
        try:
            # Read parquet file
            table = pq.read_table(path)
            df = table.to_pandas()
            
            # Slice to requested range
            df_slice = df.iloc[start_idx:end_idx]
            
            # Load RGB image from top camera
            # LeRobot stores images in various formats
            rgb_data = df_slice["observation.images.top"].values
            
            # Stack images into tensor - handle LeRobot format
            rgb_list = []
            for img in rgb_data:
                arr = self._decode_image(img)
                # Ensure we have the right shape (H, W, C)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)  # Grayscale to RGB
                rgb_list.append(torch.from_numpy(arr.copy()).float())
            
            rgb = torch.stack(rgb_list)  # (T, H, W, C)
            rgb = rgb / 255.0  # Normalize to [0, 1]
            rgb = rgb.permute(0, 3, 1, 2)  # (T, C, H, W)
            
            # Resize images if needed
            if rgb.shape[-1] != self.image_size or rgb.shape[-2] != self.image_size:
                rgb = torch.nn.functional.interpolate(
                    rgb,
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False,
                )
            
            # Load proprioception (15D state)
            proprio_data = df_slice["observation.state"].values
            proprio_list = []
            for s in proprio_data:
                if isinstance(s, np.ndarray) and s.dtype != object:
                    proprio_list.append(torch.from_numpy(s.copy()).float())
                else:
                    proprio_list.append(torch.from_numpy(np.array(s, dtype=np.float32)).float())
            proprio = torch.stack(proprio_list)  # (T, 15)
            
            # Load actions (7D)
            action_data = df_slice["action"].values
            action_list = []
            for a in action_data:
                if isinstance(a, np.ndarray) and a.dtype != object:
                    action_list.append(torch.from_numpy(a.copy()).float())
                else:
                    action_list.append(torch.from_numpy(np.array(a, dtype=np.float32)).float())
            actions = torch.stack(action_list)  # (T, 7)
            
            return TrajectoryData(
                rgb=rgb,
                depth=None,  # CALVIN doesn't provide depth in LeRobot format
                proprio=proprio,
                instruction=episode.get("instruction", "perform the manipulation task"),
                instruction_embedding=None,
                actions=actions,
                task_name=f"calvin_task_{episode.get('task_index', 0)}",
                episode_id=episode["episode_index"],
                success=True,
            )
            
        except Exception as e:
            logger.warning(f"Failed to load episode data from {path}: {e}")
            # Return dummy data on failure
            return TrajectoryData(
                rgb=torch.zeros(actual_length, 3, self.image_size, self.image_size),
                depth=None,
                proprio=torch.zeros(actual_length, 15),
                instruction="perform the manipulation task",
                instruction_embedding=None,
                actions=torch.zeros(actual_length, 7),
                task_name="calvin_task",
                episode_id=episode["episode_index"],
                success=True,
            )

    @property
    def proprio_dim(self) -> int:
        """Dimension of proprioceptive state in CALVIN."""
        return 15  # Full robot state in LeRobot format

    @staticmethod
    def get_available_tasks() -> List[str]:
        """Get list of CALVIN tasks."""
        return CALVIN_TASKS

    @staticmethod
    def get_available_envs() -> List[str]:
        """Get list of CALVIN environments."""
        return list(CALVIN_ENVS.keys())
