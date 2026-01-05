"""
Inference wrapper for Geo-Flow-VLA trained policy.

Loads Phase 1 (world model) and Phase 2 (policy) checkpoints
and provides action prediction interface for evaluation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class GeoFlowVLAPolicy:
    """
    Unified policy wrapper for Geo-Flow-VLA evaluation.
    
    Combines:
    - DualEncoder: RGB → state embedding (DINOv2-G + MoGe-2)
    - FBWorldModel: State → goal embedding (frozen from Phase 1)
    - DiffusionPolicy: Generates action chunks via flow matching
    """
    
    def __init__(
        self,
        world_model_path: Optional[Union[str, Path]] = None,
        policy_path: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,  # Deprecated, for backward compat
        config_path: Optional[str] = None,
        device: str = "cuda",
        action_horizon: int = 16,
        action_exec_horizon: int = 8,  # How many actions to execute before re-planning
    ):
        """
        Args:
            world_model_path: Direct path to world model checkpoint (e.g., phase1/world_model.pth)
            policy_path: Direct path to policy checkpoint (e.g., phase2/policy.pth or phase2/best.pt)
            checkpoint_dir: [DEPRECATED] Directory containing phase1/ and phase2/ subdirectories
            config_path: Path to config yaml (optional, loads from checkpoint if not provided)
            device: Torch device
            action_horizon: Full action chunk size
            action_exec_horizon: Actions to execute before replanning
        """
        self.device = torch.device(device)
        self.action_horizon = action_horizon
        self.action_exec_horizon = action_exec_horizon
        
        # Handle paths - support both new explicit paths and legacy checkpoint_dir
        if world_model_path and policy_path:
            # New explicit path mode
            self.world_model_path = Path(world_model_path)
            self.policy_path = Path(policy_path)
            self.checkpoint_dir = None
            logger.info("Using explicit checkpoint paths")
        elif checkpoint_dir:
            # Legacy mode: derive paths from checkpoint_dir
            self.checkpoint_dir = Path(checkpoint_dir)
            self.world_model_path = self.checkpoint_dir / "phase1" / "world_model.pth"
            self.policy_path = self._find_policy_path(self.checkpoint_dir / "phase2")
            logger.info(f"Using legacy checkpoint_dir mode: {checkpoint_dir}")
        else:
            raise ValueError(
                "Must provide either (world_model_path, policy_path) or checkpoint_dir"
            )
        
        # Load config
        if config_path:
            self.cfg = OmegaConf.load(config_path)
        else:
            self.cfg = self._load_config_from_checkpoint()
        
        # Infer architecture from checkpoint, then build and load models
        self._policy_state_dict = self._load_policy_state_dict()
        self._build_models()
        self._load_checkpoints()
        
        # Action buffer for temporal smoothing
        self.action_buffer = None
        self.buffer_idx = 0
        
        # Inference settings from config
        training_cfg = self.cfg.get("training", {})
        phase2_cfg = training_cfg.get("phase2", {}) if training_cfg else {}
        self.num_inference_steps = phase2_cfg.get("num_inference_steps", 50) if phase2_cfg else 50
        
        logger.info(f"GeoFlowVLAPolicy initialized")
        logger.info(f"  World model: {self.world_model_path}")
        logger.info(f"  Policy: {self.policy_path}")
        logger.info(f"  Using num_inference_steps={self.num_inference_steps}")
    
    def _find_policy_path(self, phase2_dir: Path) -> Path:
        """Find policy checkpoint in phase2 directory."""
        for name in ["policy.pth", "best.pt", "latest.pt"]:
            path = phase2_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"No policy checkpoint found in {phase2_dir}")
    
    def _load_config_from_checkpoint(self) -> DictConfig:
        """Load config from policy checkpoint."""
        # Try loading from policy checkpoint if it's a .pt file with config
        if self.policy_path.suffix == ".pt" and self.policy_path.exists():
            ckpt = torch.load(self.policy_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "config" in ckpt:
                return OmegaConf.create(ckpt["config"])
        
        # If policy is .pth (state dict only), try to find config in .pt checkpoints
        policy_dir = self.policy_path.parent
        for ckpt_name in ["best.pt", "latest.pt"]:
            ckpt_path = policy_dir / ckpt_name
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict) and "config" in ckpt:
                        logger.info(f"Loaded config from {ckpt_path}")
                        return OmegaConf.create(ckpt["config"])
                except Exception as e:
                    logger.warning(f"Failed to load config from {ckpt_path}: {e}")
        
        # Fallback: try to find config.yaml in same/parent directory
        config_candidates = [
            self.policy_path.parent / "config.yaml",
            self.policy_path.parent.parent / "config.yaml",
            self.world_model_path.parent / "config.yaml",
            self.world_model_path.parent.parent / "config.yaml",
        ]
        if self.checkpoint_dir:
            config_candidates.append(self.checkpoint_dir / "config.yaml")
        
        for cfg_path in config_candidates:
            if cfg_path.exists():
                logger.info(f"Loaded config from {cfg_path}")
                return OmegaConf.load(cfg_path)
        
        # Last resort: try to find default config from package
        try:
            import geo_flow_vla
            package_dir = Path(geo_flow_vla.__file__).parent
            default_configs = [
                package_dir / "configs" / "config.yaml",
                package_dir / "configs" / "libero_config.yaml",
            ]
            for cfg_path in default_configs:
                if cfg_path.exists():
                    logger.warning(f"Using default package config: {cfg_path}")
                    return OmegaConf.load(cfg_path)
        except Exception as e:
            logger.debug(f"Could not load package config: {e}")
        
        raise FileNotFoundError(
            f"No config found. Please provide --config explicitly.\n"
            f"Searched .pt files in {policy_dir} and config.yaml in: {config_candidates}"
        )
    
    def _load_policy_state_dict(self) -> Optional[Dict]:
        """Load policy state dict to infer architecture dimensions."""
        if not self.policy_path.exists():
            return None
        
        ckpt = torch.load(self.policy_path, map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if "policy_state_dict" in ckpt:
                return ckpt["policy_state_dict"]
            elif "state_dict" in ckpt:
                return ckpt["state_dict"]
            elif "blocks.0.qkv.weight" in ckpt:
                # Direct state dict (policy.pth format)
                return ckpt
        
        return None
    
    def _infer_action_dim_from_world_model(self, action_horizon: int) -> Optional[int]:
        """Infer action_dim from world model checkpoint."""
        if not self.world_model_path.exists():
            return None
        
        try:
            wm_ckpt = torch.load(self.world_model_path, map_location="cpu", weights_only=False)
            
            # forward_net.action_encoder.0.weight has shape (hidden_dim, action_dim * action_horizon)
            if "forward_net.action_encoder.0.weight" in wm_ckpt:
                weight_shape = wm_ckpt["forward_net.action_encoder.0.weight"].shape
                total_action_dim = weight_shape[1]  # action_dim * action_horizon
                action_dim = total_action_dim // action_horizon
                logger.info(f"Inferred action_dim={action_dim} from world model checkpoint "
                           f"(total={total_action_dim}, horizon={action_horizon})")
                return action_dim
        except Exception as e:
            logger.warning(f"Failed to infer action_dim from world model: {e}")
        
        return None
    
    def _infer_policy_dims(self, state_dict: Dict) -> Tuple[int, int, int]:
        """Infer hidden_dim, num_layers, num_heads from policy state dict."""
        hidden_dim = 512
        num_layers = 6
        num_heads = 8
        
        if state_dict is None:
            return hidden_dim, num_layers, num_heads
        
        # Infer hidden_dim from blocks.0.qkv.weight shape: (3*hidden_dim, hidden_dim)
        if "blocks.0.qkv.weight" in state_dict:
            qkv_shape = state_dict["blocks.0.qkv.weight"].shape
            hidden_dim = qkv_shape[1]
            logger.info(f"Inferred hidden_dim={hidden_dim} from checkpoint")
        
        # Count number of transformer blocks
        block_ids = set()
        for key in state_dict.keys():
            if key.startswith("blocks."):
                parts = key.split(".")
                if len(parts) > 1:
                    try:
                        block_ids.add(int(parts[1]))
                    except ValueError:
                        pass
        if block_ids:
            num_layers = len(block_ids)
            logger.info(f"Inferred num_layers={num_layers} from checkpoint")
        
        # Infer num_heads from qkv weight (3*hidden_dim where head_dim is typically 64)
        # num_heads = hidden_dim / head_dim
        if hidden_dim in [512, 768, 1024]:
            # Common head_dim is 64
            num_heads = hidden_dim // 64
            logger.info(f"Inferred num_heads={num_heads} from checkpoint")
        
        return hidden_dim, num_layers, num_heads
        
    def _build_models(self) -> None:
        """Initialize model architectures."""
        from geo_flow_vla.models.dual_encoder import DualEncoder
        from geo_flow_vla.models.world_model import FBWorldModel
        from geo_flow_vla.models.diffusion_policy import DiffusionPolicy
        
        cfg = self.cfg
        
        # Dual Encoder
        self.dual_encoder = DualEncoder(
            semantic_dim=cfg.model.get("semantic_dim", 1536),
            geometric_dim=cfg.model.get("geometric_dim", 256),
            output_dim=cfg.model.get("state_dim", 512),
            dinov2_model=cfg.model.get("dinov2_model", "dinov2_vitg14"),
            moge_model=cfg.model.get("moge_model", "Ruicheng/moge-2-vitl-normal"),
            use_mock_moge=cfg.model.get("use_mock_moge", False),
            freeze_backbones=True,
        ).to(self.device)
        
        # Get model config - use correct paths matching training
        state_dim = cfg.model.get("state_dim", 512)
        action_horizon = cfg.model.get("action_horizon", 16)
        
        # Infer action_dim from world model checkpoint (takes precedence over config)
        inferred_action_dim = self._infer_action_dim_from_world_model(action_horizon)
        action_dim = inferred_action_dim if inferred_action_dim else cfg.model.get("action_dim", 7)
        
        # Get FB world model config
        fb_cfg = cfg.model.get("fb", {}) if hasattr(cfg.model, "get") else {}
        if not fb_cfg and hasattr(cfg.model, "fb"):
            fb_cfg = dict(cfg.model.fb)
        latent_dim = fb_cfg.get("latent_dim", 256) if fb_cfg else cfg.model.get("latent_dim", 256)
        fb_hidden_dim = fb_cfg.get("hidden_dim", 512) if fb_cfg else 512
        num_residual_blocks = fb_cfg.get("num_residual_blocks", 2) if fb_cfg else 2
        
        # World Model (frozen) - use explicit parameters matching training
        self.world_model = FBWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            latent_dim=latent_dim,
            hidden_dim=fb_hidden_dim,
            num_residual_blocks=num_residual_blocks,
        ).to(self.device)
        
        logger.info(f"WorldModel: state_dim={state_dim}, action_dim={action_dim}, "
                    f"action_horizon={action_horizon}, latent_dim={latent_dim}")
        
        # Diffusion Policy - infer dimensions from checkpoint first
        inferred_hidden, inferred_layers, inferred_heads = self._infer_policy_dims(self._policy_state_dict)
        
        # Get DiT config
        dit_cfg = cfg.model.get("dit", {}) if hasattr(cfg.model, "get") else {}
        if not dit_cfg and hasattr(cfg.model, "dit"):
            dit_cfg = dict(cfg.model.dit)
        
        # Use inferred values as primary, fall back to config
        hidden_dim = inferred_hidden if self._policy_state_dict else (
            dit_cfg.get("hidden_dim", 512) if dit_cfg else 512
        )
        num_layers = inferred_layers if self._policy_state_dict else (
            dit_cfg.get("num_layers", 6) if dit_cfg else 6
        )
        num_heads = inferred_heads if self._policy_state_dict else (
            dit_cfg.get("num_heads", 8) if dit_cfg else 8
        )
        dropout = dit_cfg.get("dropout", 0.1) if dit_cfg else 0.1
        
        self.policy = DiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        ).to(self.device)
        
        logger.info(f"DiffusionPolicy: hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}")
        
        # Set all to eval mode
        self.dual_encoder.eval()
        self.world_model.eval()
        self.policy.eval()
        
    def _load_checkpoints(self) -> None:
        """Load trained weights."""
        # Load world model
        if self.world_model_path.exists():
            self.world_model.load_state_dict(
                torch.load(self.world_model_path, map_location=self.device, weights_only=False)
            )
            logger.info(f"✓ Loaded world model from {self.world_model_path}")
        else:
            raise FileNotFoundError(f"World model not found: {self.world_model_path}")
        
        # Load policy - reuse already loaded state dict if available
        if self._policy_state_dict is not None:
            # Move tensors to device
            policy_sd = {k: v.to(self.device) for k, v in self._policy_state_dict.items()}
            self.policy.load_state_dict(policy_sd)
            logger.info(f"✓ Loaded policy from {self.policy_path}")
        else:
            raise FileNotFoundError(f"Policy not found or invalid: {self.policy_path}")
    
    @torch.no_grad()
    def predict(
        self,
        rgb: np.ndarray,
        instruction: Optional[str] = None,
        proprio: Optional[np.ndarray] = None,
        use_buffer: bool = True,
    ) -> np.ndarray:
        """
        Predict action(s) from RGB observation.
        
        Args:
            rgb: RGB image (H, W, 3), uint8 or float [0,1]
            instruction: Language instruction (optional, for future use)
            proprio: Proprioceptive state (optional, for future use)
            use_buffer: Use action buffer for temporal smoothing
            
        Returns:
            Action array of shape (action_dim,)
        """
        # Use buffered action if available
        if use_buffer and self.action_buffer is not None:
            if self.buffer_idx < min(len(self.action_buffer), self.action_exec_horizon):
                action = self.action_buffer[self.buffer_idx]
                self.buffer_idx += 1
                return action
        
        # Preprocess image
        rgb_tensor = self._preprocess_image(rgb)
        
        # Encode observation → state
        state = self.dual_encoder(rgb_tensor)  # (1, state_dim)
        
        # Get goal embedding from world model
        goal = self.world_model.get_goal_embedding(state)  # (1, latent_dim)
        
        # Generate action chunk via diffusion/flow matching
        action_chunk = self.policy.sample(
            state=state,
            goal=goal,
            num_steps=self.num_inference_steps,
        )  # (1, action_horizon, action_dim)
        
        # Store in buffer
        self.action_buffer = action_chunk[0].cpu().numpy()  # (action_horizon, action_dim)
        self.buffer_idx = 1  # Return first action now
        
        return self.action_buffer[0]  # (action_dim,)
    
    @torch.no_grad()
    def predict_chunk(
        self,
        rgb: np.ndarray,
        instruction: Optional[str] = None,
        proprio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict full action chunk (without buffering).
        
        Args:
            rgb: RGB image (H, W, 3), uint8 or float [0,1]
            instruction: Language instruction (optional)
            proprio: Proprioceptive state (optional)
            
        Returns:
            Action chunk of shape (action_horizon, action_dim)
        """
        # Preprocess image
        rgb_tensor = self._preprocess_image(rgb)
        
        # Encode observation → state
        state = self.dual_encoder(rgb_tensor)
        
        # Get goal embedding from world model
        goal = self.world_model.get_goal_embedding(state)
        
        # Generate action chunk
        action_chunk = self.policy.sample(
            state=state,
            goal=goal,
            num_steps=self.num_inference_steps,
        )
        
        return action_chunk[0].cpu().numpy()
    
    def _preprocess_image(self, rgb: np.ndarray) -> torch.Tensor:
        """Convert RGB numpy array to model input tensor."""
        # Handle uint8 images
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        
        # Ensure float32
        rgb = rgb.astype(np.float32)
        
        # Ensure (H, W, 3) format
        if rgb.ndim == 4:  # (1, H, W, 3) or (1, 3, H, W)
            rgb = rgb[0]
        if rgb.shape[0] == 3 and rgb.ndim == 3:  # (3, H, W)
            rgb = rgb.transpose(1, 2, 0)
        
        # Clip to valid range
        rgb = np.clip(rgb, 0.0, 1.0)
        
        # Resize to expected input size (224x224)
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        img = img.resize((224, 224), Image.BILINEAR)
        rgb = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor: (1, 3, H, W)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return rgb_tensor.to(self.device)
    
    def reset(self) -> None:
        """Reset action buffer (call at start of each episode)."""
        self.action_buffer = None
        self.buffer_idx = 0
    
    @property
    def action_dim(self) -> int:
        """Return action dimension."""
        return self.cfg.data.get("action_dim", 7)
    
    def to(self, device: str) -> "GeoFlowVLAPolicy":
        """Move all models to device."""
        self.device = torch.device(device)
        self.dual_encoder = self.dual_encoder.to(self.device)
        self.world_model = self.world_model.to(self.device)
        self.policy = self.policy.to(self.device)
        return self
