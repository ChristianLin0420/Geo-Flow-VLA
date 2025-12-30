"""
Phase 2: Policy Training with Flow Matching and CPR for Geo-Flow VLA.

Trains the DiT policy to generate action trajectories using:
- Conditional Flow Matching (CFM) for behavior cloning
- CPR (Conditional Policy Regularization) for geometric consistency

Training Process:
    1. Load frozen world model and dual encoder
    2. Discriminator Step: Train D to distinguish real vs policy goals
    3. Policy Step:
       - CFM loss (behavior cloning)
       - CPR guidance (adversarial regularization)
    4. Save policy checkpoints

WandB Logging:
    - Flow matching loss
    - CPR discriminator metrics
    - Action prediction quality
    - Generated trajectory visualizations

Usage:
    python -m geo_flow_vla.train.phase2_policy
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import numpy as np
from tqdm import tqdm

from ..models.dual_encoder import DualEncoder
from ..models.world_model import FBWorldModel
from ..models.diffusion_policy import DiffusionPolicy
from ..models.discriminator import CPRDiscriminator
from ..losses.flow_matching_loss import FlowMatchingLoss
from ..losses.cpr_regularizer import CPRRegularizer
from ..data.libero_dataset import LIBERODataset, create_libero_dataloaders
from ..data.normalizer import StateActionNormalizer
from ..data.transforms import create_augmentation
from ..utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    get_local_rank,
    wrap_model_ddp,
    create_distributed_dataloader,
    set_epoch_sampler,
    reduce_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation metric and stops training if no improvement
    for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss (lower is better), "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, value: float, epoch: int = 0) -> bool:
        """
        Check if should stop.
        
        Args:
            value: Current validation metric
            epoch: Current epoch number
            
        Returns:
            True if should stop training
        """
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            logger.info(f"Early stopping: new best value {value:.4f} at epoch {epoch}")
        else:
            self.counter += 1
            logger.info(f"Early stopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered! Best: {self.best_value:.4f} at epoch {self.best_epoch}")
        
        return self.should_stop
    
    def state_dict(self) -> Dict:
        """Save early stopping state."""
        return {
            "best_value": self.best_value,
            "counter": self.counter,
            "best_epoch": self.best_epoch,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load early stopping state."""
        self.best_value = state["best_value"]
        self.counter = state["counter"]
        self.best_epoch = state["best_epoch"]


class PolicyTrainer:
    """
    Trainer for Phase 2: Diffusion Policy with CPR.
    
    Implements adversarial training:
    - Discriminator distinguishes real vs policy-generated goals
    - Policy is regularized to produce geometrically plausible trajectories
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device = torch.device("cuda"),
        is_distributed: bool = False,
    ) -> None:
        """
        Args:
            cfg: Hydra configuration
            device: Training device
            is_distributed: Whether using distributed training
        """
        self.cfg = cfg
        self.device = device
        self.is_distributed = is_distributed
        self.is_main = is_main_process()
        self.world_size = get_world_size()
        
        # Build models
        self._build_models()
        
        # Build optimizers
        self._build_optimizers()
        
        # Build data loaders
        self._build_dataloaders()
        
        # Build losses
        self._build_losses()
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if cfg.hardware.precision != "fp32" else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Checkpoint directory (only create on main process)
        self.ckpt_dir = Path(cfg.checkpoint.dir) / "phase2"
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Image augmentation for reducing overfitting
        self.augmentation = create_augmentation(cfg)
        if self.augmentation is not None:
            self.augmentation = self.augmentation.to(self.device)
            logger.info(f"✓ Image augmentation enabled")
        
        # Early stopping
        phase2_cfg = cfg.training.phase2
        if phase2_cfg.get("early_stopping", False):
            self.early_stopping = EarlyStopping(
                patience=phase2_cfg.get("early_stopping_patience", 10),
                min_delta=phase2_cfg.get("early_stopping_min_delta", 0.001),
                mode="min",  # Stop when val loss stops decreasing
            )
            logger.info(f"✓ Early stopping enabled (patience={self.early_stopping.patience})")
        else:
            self.early_stopping = None

    def _build_models(self) -> None:
        """Initialize all models."""
        cfg = self.cfg
        
        # Dual encoder (frozen)
        self.dual_encoder = DualEncoder(
            semantic_dim=cfg.model.semantic_dim,
            geometric_dim=cfg.model.geometric_dim,
            output_dim=cfg.model.state_dim,
            use_mock_moge=cfg.encoders.moge.use_mock,
        ).to(self.device)
        
        for param in self.dual_encoder.parameters():
            param.requires_grad = False
        self.dual_encoder.eval()
        
        # World model (frozen, loaded from Phase 1)
        self.world_model = FBWorldModel(
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            action_horizon=cfg.model.action_horizon,
            latent_dim=cfg.model.fb.latent_dim,
            hidden_dim=cfg.model.fb.hidden_dim,
            num_residual_blocks=cfg.model.fb.num_residual_blocks,
        ).to(self.device)
        
        # Load pretrained world model
        # Check if custom path is specified, otherwise auto-detect
        custom_wm_path = cfg.training.phase2.get("world_model_path", None)
        if custom_wm_path is not None:
            world_model_path = Path(custom_wm_path)
        else:
            world_model_path = Path(cfg.checkpoint.dir) / "phase1" / "world_model.pth"
        
        if world_model_path.exists():
            checkpoint = torch.load(world_model_path, map_location=self.device, weights_only=False)
            # Support both full checkpoint (epoch_*.pt) and state_dict only (world_model.pth)
            if isinstance(checkpoint, dict) and "world_model_state_dict" in checkpoint:
                state_dict = checkpoint["world_model_state_dict"]
                logger.info(f"Loaded world model from full checkpoint {world_model_path} (epoch {checkpoint.get('epoch', '?')})")
            else:
                state_dict = checkpoint
                logger.info(f"Loaded world model from {world_model_path}")
            self.world_model.load_state_dict(state_dict)
        else:
            logger.warning(f"World model not found at {world_model_path}, using random init")
        
        for param in self.world_model.parameters():
            param.requires_grad = False
        self.world_model.eval()
        
        # Diffusion policy (trainable)
        self.policy = DiffusionPolicy(
            action_dim=cfg.model.action_dim,
            action_horizon=cfg.model.action_horizon,
            state_dim=cfg.model.state_dim,
            latent_dim=cfg.model.fb.latent_dim,
            hidden_dim=cfg.model.dit.hidden_dim,
            num_layers=cfg.model.dit.num_layers,
            num_heads=cfg.model.dit.num_heads,
            mlp_ratio=cfg.model.dit.mlp_ratio,
            dropout=cfg.model.dit.dropout,
        ).to(self.device)
        
        # CPR discriminator (trainable)
        self.discriminator = CPRDiscriminator(
            state_dim=cfg.model.state_dim,
            latent_dim=cfg.model.fb.latent_dim,
            hidden_dim=cfg.model.discriminator.hidden_dim,
            num_layers=cfg.model.discriminator.num_layers,
            gradient_penalty_weight=cfg.model.discriminator.gradient_penalty,
        ).to(self.device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.policy = wrap_model_ddp(
                self.policy,
                self.device,
                find_unused_parameters=cfg.hardware.get("find_unused_params", False),
            )
            self.discriminator = wrap_model_ddp(
                self.discriminator,
                self.device,
                find_unused_parameters=cfg.hardware.get("find_unused_params", False),
            )
        
        logger.info(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")

    @property
    def policy_module(self):
        """Get the underlying policy module (unwrapped from DDP if needed)."""
        if self.is_distributed:
            return self.policy.module
        return self.policy

    @property
    def discriminator_module(self):
        """Get the underlying discriminator module (unwrapped from DDP if needed)."""
        if self.is_distributed:
            return self.discriminator.module
        return self.discriminator

    def _build_optimizers(self) -> None:
        """Initialize optimizers and schedulers."""
        cfg = self.cfg.training.phase2
        
        # Policy optimizer
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.lr_policy,
            weight_decay=cfg.weight_decay,
        )
        
        # Discriminator optimizer
        self.disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=cfg.lr_discriminator,
            weight_decay=cfg.weight_decay,
        )
        
        # Learning rate schedulers
        self.total_steps = len(self.train_loader) * cfg.epochs if hasattr(self, 'train_loader') else 100000
        
        def policy_lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / cfg.warmup_steps
            progress = (step - cfg.warmup_steps) / max(1, self.total_steps - cfg.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.policy_optimizer, policy_lr_lambda
        )
        
        self.disc_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.disc_optimizer, lambda step: 1.0  # Constant LR for discriminator
        )

    def _build_dataloaders(self) -> None:
        """Initialize data loaders with distributed sampling if enabled."""
        cfg = self.cfg
        
        # Create dataset based on config
        from torch.utils.data import random_split
        
        dataset_type = cfg.data.get("dataset", "libero")
        
        if dataset_type == "calvin":
            from ..data.calvin_dataset import CALVINDataset
            full_dataset = CALVINDataset(
                data_root=cfg.data.data_root,
                env=cfg.data.get("calvin_env", "D"),
                action_horizon=cfg.model.action_horizon,
                image_size=cfg.data.image_size,
            )
        elif dataset_type == "rlbench":
            from ..data.rlbench_dataset import RLBenchDataset
            # Parse tasks - can be string category or list of task names
            rlbench_tasks = cfg.data.get("rlbench_tasks", "all")
            if isinstance(rlbench_tasks, str):
                tasks = rlbench_tasks  # Let RLBenchDataset handle category names
            else:
                tasks = list(rlbench_tasks)
            full_dataset = RLBenchDataset(
                data_root=cfg.data.data_root,
                tasks=tasks,
                split="train",
                action_horizon=cfg.model.action_horizon,
                image_size=cfg.data.image_size,
                camera=cfg.data.get("rlbench_camera", "front_rgb"),
            )
        else:  # Default to LIBERO
            full_dataset = LIBERODataset(
                data_root=cfg.data.data_root,
                suite=cfg.data.get("libero_suite", "full"),
                action_horizon=cfg.model.action_horizon,
                image_size=cfg.data.image_size,
            )
        
        # Split train/val
        val_size = int(len(full_dataset) * 0.1)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        
        # Create dataloaders with distributed sampling if enabled
        if self.is_distributed:
            self.train_loader = create_distributed_dataloader(
                train_dataset,
                batch_size=cfg.training.phase2.batch_size,
                num_workers=cfg.data.num_workers,
                shuffle=True,
                pin_memory=cfg.data.pin_memory,
            )
            self.val_loader = create_distributed_dataloader(
                val_dataset,
                batch_size=cfg.training.phase2.batch_size,
                num_workers=cfg.data.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=cfg.data.pin_memory,
            )
        else:
            from torch.utils.data import DataLoader
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.training.phase2.batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                drop_last=True,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.training.phase2.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
            )
        
        self.total_steps = len(self.train_loader) * cfg.training.phase2.epochs
        
        # Rebuild optimizers with correct total_steps
        self._build_optimizers()
        
        logger.info(f"Train samples: {len(train_dataset)} (per GPU batch: {cfg.training.phase2.batch_size})")
        logger.info(f"Total steps: {self.total_steps}")

    def _build_losses(self) -> None:
        """Initialize loss functions."""
        cfg = self.cfg.training.phase2
        
        self.cfm_loss = FlowMatchingLoss(
            schedule=cfg.schedule,
        )
        
        self.cpr = CPRRegularizer(
            lambda_start=cfg.cpr_lambda_start,
            lambda_end=cfg.cpr_lambda_end,
            warmup_steps=cfg.cpr_warmup_epochs * len(self.train_loader),
        )

    @torch.no_grad()
    def encode_batch(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode images to state embeddings."""
        rgb = rgb.to(self.device)
        return self.dual_encoder(rgb)

    @torch.no_grad()
    def get_goal_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Get goal embedding from world model."""
        return self.world_model.encode_goal(state)

    def train_discriminator_step(
        self,
        real_state: torch.Tensor,
        real_future_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """One discriminator training step."""
        # Get real goals (from future states in dataset)
        real_goal = self.get_goal_embedding(real_future_state)
        
        # Get fake goals (from current states - self-prediction)
        fake_goal = self.get_goal_embedding(real_state)
        
        self.disc_optimizer.zero_grad()
        
        # Compute discriminator loss
        disc_result = self.cpr.discriminator_loss(
            self.discriminator,
            real_state, real_goal,
            real_state, fake_goal,
        )
        
        if self.scaler is not None:
            self.scaler.scale(disc_result["discriminator_loss"]).backward()
            self.scaler.unscale_(self.disc_optimizer)
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.cfg.training.phase2.gradient_clip,
            )
            self.scaler.step(self.disc_optimizer)
        else:
            disc_result["discriminator_loss"].backward()
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.cfg.training.phase2.gradient_clip,
            )
            self.disc_optimizer.step()
        
        disc_result["grad_norm"] = disc_grad_norm
        
        return disc_result

    def train_policy_step(
        self,
        state: torch.Tensor,
        future_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """One policy training step."""
        # Get goal from future state (for conditioning)
        goal = self.get_goal_embedding(future_state)
        
        # Get fake goal (for CPR)
        fake_goal = self.get_goal_embedding(state)
        
        self.policy_optimizer.zero_grad()
        
        if self.scaler is not None:
            with autocast('cuda'):
                # CFM loss
                cfm_result = self.cfm_loss(
                    self.policy, actions, state, goal, return_metrics=True
                )
                cfm_loss = cfm_result["loss"]
                
                # CPR loss
                cpr_result = self.cpr.policy_loss(
                    self.discriminator, state, fake_goal
                )
                
                # Total loss
                total_loss = cfm_loss + cpr_result["cpr_loss"]
            
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.policy_optimizer)
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.cfg.training.phase2.gradient_clip,
            )
            self.scaler.step(self.policy_optimizer)
            self.scaler.update()
        else:
            # CFM loss
            cfm_result = self.cfm_loss(
                self.policy, actions, state, goal, return_metrics=True
            )
            cfm_loss = cfm_result["loss"]
            
            # CPR loss
            cpr_result = self.cpr.policy_loss(
                self.discriminator, state, fake_goal
            )
            
            # Total loss
            total_loss = cfm_loss + cpr_result["cpr_loss"]
            
            total_loss.backward()
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.cfg.training.phase2.gradient_clip,
            )
            self.policy_optimizer.step()
        
        self.policy_scheduler.step()
        
        # Update CPR lambda schedule
        self.cpr.step_scheduler()
        
        return {
            "total_loss": total_loss,
            "cfm_loss": cfm_loss,
            "grad_norm": policy_grad_norm,
            **cpr_result,
            **{k: v for k, v in cfm_result.items() if k != "loss"},
        }

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch with distributed support."""
        self.policy.train()
        self.discriminator.train()
        
        epoch_metrics = {}
        num_batches = 0
        
        # Only show progress bar on main process
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", disable=not self.is_main)
        
        for batch in pbar:
            rgb, depth, proprio, instruction, actions = batch
            
            # Move to device
            rgb = rgb.to(self.device)
            actions = actions.to(self.device)
            
            # Apply image augmentation for regularization
            if self.augmentation is not None:
                rgb = self.augmentation(rgb)
            
            # Encode states
            with torch.no_grad():
                if rgb.dim() == 5:
                    current_rgb = rgb[:, 0]
                    future_rgb = rgb[:, -1]
                else:
                    current_rgb = rgb
                    future_rgb = rgb
                
                current_state = self.encode_batch(current_rgb)
                future_state = self.encode_batch(future_rgb)
            
            # Train discriminator
            for _ in range(self.cfg.training.phase2.discriminator_steps):
                disc_metrics = self.train_discriminator_step(
                    current_state, future_state
                )
            
            # Train policy
            policy_metrics = self.train_policy_step(
                current_state, future_state, actions
            )
            
            # Accumulate metrics
            all_metrics = {
                **{f"disc_{k}": v for k, v in disc_metrics.items()},
                **{f"policy_{k}": v for k, v in policy_metrics.items()},
            }
            
            for k, v in all_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                if isinstance(v, torch.Tensor):
                    epoch_metrics[k] += v.item()
                else:
                    epoch_metrics[k] += v
            num_batches += 1
            
            # Log to wandb (only on main process)
            if self.is_main and self.global_step % self.cfg.logging.log_every == 0:
                log_dict = {
                    "phase2/loss_cfm": policy_metrics["cfm_loss"].item(),
                    "phase2/loss_total": policy_metrics["total_loss"].item(),
                    "phase2/discriminator_loss": disc_metrics["discriminator_loss"].item(),
                    "phase2/discriminator_acc_real": disc_metrics["real_accuracy"].item(),
                    "phase2/discriminator_acc_fake": disc_metrics["fake_accuracy"].item(),
                    "phase2/cpr_lambda": policy_metrics["cpr_lambda"].item(),
                    "phase2/grad_norm_dit": policy_metrics["grad_norm"].item() if isinstance(policy_metrics["grad_norm"], torch.Tensor) else policy_metrics["grad_norm"],
                    "phase2/grad_norm_disc": disc_metrics["grad_norm"].item() if isinstance(disc_metrics["grad_norm"], torch.Tensor) else disc_metrics["grad_norm"],
                    "phase2/lr_policy": self.policy_scheduler.get_last_lr()[0],
                    "phase2/step": self.global_step,
                }
                
                if "discriminator_prob" in policy_metrics:
                    log_dict["phase2/cpr_reward"] = policy_metrics["discriminator_prob"].item()
                
                wandb.log(log_dict, step=self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "cfm": f"{policy_metrics['cfm_loss'].item():.4f}",
                "cpr": f"{policy_metrics['cpr_loss'].item():.4f}",
                "d_acc": f"{(disc_metrics['real_accuracy'].item() + disc_metrics['fake_accuracy'].item()) / 2:.2f}",
            })
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, num_batches)
        
        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.policy.eval()
        
        val_metrics = {
            "cfm_loss": 0.0,
            "action_mse": 0.0,
            "action_mae": 0.0,
        }
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", disable=not self.is_main):
            rgb, depth, proprio, instruction, actions = batch
            
            rgb = rgb.to(self.device)
            actions = actions.to(self.device)
            
            if rgb.dim() == 5:
                current_rgb = rgb[:, 0]
                future_rgb = rgb[:, -1]
            else:
                current_rgb = rgb
                future_rgb = rgb
            
            current_state = self.encode_batch(current_rgb)
            future_state = self.encode_batch(future_rgb)
            goal = self.get_goal_embedding(future_state)
            
            # CFM loss
            cfm_result = self.cfm_loss(self.policy, actions, current_state, goal)
            val_metrics["cfm_loss"] += cfm_result["loss"].item()
            
            # Generate predictions for MSE
            pred_actions = self.policy_module.sample(
                current_state, goal,
                num_steps=self.cfg.training.phase2.num_inference_steps,
            )
            
            val_metrics["action_mse"] += F.mse_loss(pred_actions, actions).item()
            val_metrics["action_mae"] += F.l1_loss(pred_actions, actions).item()
            
            num_batches += 1
        
        for k in val_metrics:
            val_metrics[k] /= max(1, num_batches)
        
        return val_metrics

    def _log_trajectory_visualization(
        self,
        pred_actions: torch.Tensor,
        gt_actions: torch.Tensor,
    ) -> None:
        """Log trajectory comparison to wandb."""
        import matplotlib.pyplot as plt
        
        # Plot first sample, first 3 action dimensions
        pred = pred_actions[0, :, :3].cpu().numpy()
        gt = gt_actions[0, :, :3].cpu().numpy()
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        labels = ["X", "Y", "Z"]
        
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(gt[:, i], 'b-', label='Ground Truth', linewidth=2)
            ax.plot(pred[:, i], 'r--', label='Predicted', linewidth=2)
            ax.set_ylabel(f'{label} Position')
            ax.legend()
            ax.grid(True)
        
        axes[-1].set_xlabel('Timestep')
        fig.suptitle('Predicted vs Ground Truth Trajectory')
        plt.tight_layout()
        
        wandb.log({"phase2/action_trajectory": wandb.Image(fig)}, step=self.global_step)
        plt.close(fig)

    def _log_3d_point_cloud_with_trajectory(
        self,
        rgb: torch.Tensor,
        pred_actions: torch.Tensor,
    ) -> None:
        """Log 3D point cloud with predicted trajectory overlaid.
        
        Uses scale-to-scene approach and creates thick gradient-colored 
        trajectory by densely interpolating and duplicating points.
        """
        try:
            from scipy.interpolate import interp1d
            from ..utils.camera_transforms import scale_trajectory_to_scene
            
            # Get point cloud from MoGe-2
            if rgb.dim() == 5:
                rgb_frame = rgb[0, 0]
            else:
                rgb_frame = rgb[0]
            
            point_map = self.dual_encoder.get_point_map(rgb_frame.unsqueeze(0))
            
            if point_map is not None:
                H, W = point_map.shape[-2:]
                points = point_map[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
                
                rgb_np = rgb_frame.permute(1, 2, 0).cpu().numpy()
                rgb_np = np.clip(rgb_np, 0, 1)
                rgb_colors = (rgb_np * 255).reshape(-1, 3)
                
                valid_mask = np.isfinite(points).all(axis=1)
                points = points[valid_mask]
                rgb_colors = rgb_colors[valid_mask]
                
                if len(points) == 0:
                    logger.warning("No valid points after filtering NaN/Inf")
                    return
                
                # Downsample scene point cloud
                stride = 1
                points_ds = points[::stride].astype(np.float64)
                colors_ds = rgb_colors[::stride]
                
                # Get predicted trajectory
                traj_points = pred_actions[0, :, :3].cpu().numpy().astype(np.float64)
                traj_valid = np.isfinite(traj_points).all(axis=1)
                traj_points = traj_points[traj_valid]
                
                if len(traj_points) == 0:
                    logger.warning("No valid trajectory points")
                    return
                
                # === HANDLE CALVIN DELTA ACTIONS ===
                # CALVIN uses position deltas, not absolute positions
                # Accumulate deltas to get actual trajectory positions
                dataset_type = self.cfg.data.get("dataset", "libero")
                if dataset_type == "calvin":
                    traj_points = np.cumsum(traj_points, axis=0)
                
                # Scale trajectory to scene
                traj_points = scale_trajectory_to_scene(
                    traj_points, 
                    points_ds, 
                    scale_factor=0.4
                )
                
                num_traj = len(traj_points)
                
                # === DENSE INTERPOLATION for smooth line appearance ===
                if num_traj >= 2:
                    try:
                        t_orig = np.arange(num_traj)
                        # 20x interpolation for dense line
                        t_interp = np.linspace(0, num_traj - 1, num_traj * 20)
                        interp_func = interp1d(t_orig, traj_points, axis=0, kind='cubic')
                        traj_points = interp_func(t_interp)
                    except Exception:
                        pass
                
                # === CREATE THICK LINE by duplicating with offsets ===
                # Compute trajectory extent for scaling offsets
                traj_extent = np.abs(traj_points - traj_points.mean(axis=0)).max()
                thickness = traj_extent * 0.02  # 2% of trajectory extent
                
                # Create multiple offset copies for thickness
                all_traj_points = [traj_points]  # Center line
                offsets = [
                    [thickness, 0, 0], [-thickness, 0, 0],
                    [0, thickness, 0], [0, -thickness, 0],
                    [0, 0, thickness], [0, 0, -thickness],
                    [thickness, thickness, 0], [-thickness, -thickness, 0],
                    [thickness, 0, thickness], [-thickness, 0, -thickness],
                    [0, thickness, thickness], [0, -thickness, -thickness],
                ]
                for offset in offsets:
                    all_traj_points.append(traj_points + np.array(offset))
                
                traj_points_thick = np.vstack(all_traj_points)
                
                # === COLOR GRADIENT: Blue (start) -> Green -> Yellow -> Red (end) ===
                num_pts_per_line = len(traj_points)
                t = np.linspace(0, 1, num_pts_per_line)
                
                # Rainbow gradient: Blue -> Cyan -> Green -> Yellow -> Red
                colors_single = np.zeros((num_pts_per_line, 3))
                colors_single[:, 0] = np.clip(2 * t - 0.5, 0, 1) * 255      # R: ramps up in second half
                colors_single[:, 1] = np.clip(1 - np.abs(2 * t - 1), 0, 1) * 255  # G: peaks at middle
                colors_single[:, 2] = np.clip(1 - 2 * t, 0, 1) * 255       # B: ramps down in first half
                
                # Repeat colors for all thickness copies
                num_copies = len(all_traj_points)
                traj_colors = np.tile(colors_single, (num_copies, 1))
                
                # Add bright start marker (larger green sphere)
                start_pos = traj_points[0]
                start_offsets = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            start_offsets.append(start_pos + np.array([dx, dy, dz]) * thickness * 2)
                start_markers = np.array(start_offsets)
                start_colors = np.full((len(start_markers), 3), [0.0, 255.0, 0.0])  # Green
                
                # Add bright end marker (larger red sphere)
                end_pos = traj_points[-1]
                end_offsets = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            end_offsets.append(end_pos + np.array([dx, dy, dz]) * thickness * 2)
                end_markers = np.array(end_offsets)
                end_colors = np.full((len(end_markers), 3), [255.0, 0.0, 0.0])  # Red
                
                # Combine scene and trajectory
                all_points = np.vstack([points_ds, traj_points_thick, start_markers, end_markers])
                all_colors = np.vstack([colors_ds, traj_colors, start_colors, end_colors])
                
                # Create combined array
                N = len(all_points)
                point_cloud = np.zeros((N, 6), dtype=np.float64)
                point_cloud[:, :3] = all_points
                point_cloud[:, 3:] = all_colors
                
                wandb.log({
                    "phase2/3d_scene_with_trajectory": wandb.Object3D(point_cloud)
                }, step=self.global_step)
                
        except Exception as e:
            logger.warning(f"Failed to log 3D point cloud with trajectory: {e}")


    def _log_2d_trajectory_overlay(
        self,
        rgb: torch.Tensor,
        pred_actions: torch.Tensor,
        gt_actions: torch.Tensor = None,
    ) -> None:
        """Log 2D image with trajectory overlay for easier visualization."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        
        try:
            if rgb.dim() == 5:
                rgb_frame = rgb[0, 0]
            else:
                rgb_frame = rgb[0]
            
            img = rgb_frame.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            
            # Flip image vertically only for LIBERO (agentview camera orientation)
            # RLBench and CALVIN images are already in correct orientation
            dataset_type = self.cfg.data.get("dataset", "libero")
            if dataset_type == "libero":
                img = np.flipud(img)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left: RGB image
            axes[0].imshow(img)
            axes[0].set_title('RGB Observation')
            axes[0].axis('off')
            
            # Right: Trajectory in XY plane
            ax = axes[1]
            
            # Get predicted trajectory
            traj_pred = pred_actions[0, :, :3].cpu().numpy()
            traj_valid = np.isfinite(traj_pred).all(axis=1)
            traj_pred = traj_pred[traj_valid]
            
            if len(traj_pred) == 0:
                plt.close(fig)
                return
            
            # Get ground truth trajectory
            traj_gt = None
            if gt_actions is not None:
                traj_gt_raw = gt_actions[0, :, :3].cpu().numpy()
                gt_valid = np.isfinite(traj_gt_raw).all(axis=1)
                traj_gt = traj_gt_raw[gt_valid]
            
            # === HANDLE CALVIN DELTA ACTIONS ===
            # CALVIN uses position deltas, not absolute positions
            # Accumulate deltas to get actual trajectory positions
            dataset_type = self.cfg.data.get("dataset", "libero")
            if dataset_type == "calvin":
                traj_pred = np.cumsum(traj_pred, axis=0)
                if traj_gt is not None and len(traj_gt) > 0:
                    traj_gt = np.cumsum(traj_gt, axis=0)
            
            # === CENTER BOTH TRAJECTORIES ON SAME ORIGIN ===
            # Use ground truth center as reference, shift prediction to align starts
            if traj_gt is not None and len(traj_gt) > 0:
                # Align predicted trajectory's START to ground truth START
                offset = traj_gt[0, :2] - traj_pred[0, :2]
                traj_pred_aligned = traj_pred.copy()
                traj_pred_aligned[:, :2] += offset
            else:
                traj_pred_aligned = traj_pred
            
            # Plot predicted trajectory with color gradient
            num_points = len(traj_pred_aligned)
            colors = plt.cm.coolwarm(np.linspace(0, 1, num_points))
            
            # Draw trajectory line segments
            if num_points >= 2:
                segments = np.array([[traj_pred_aligned[i, :2], traj_pred_aligned[i+1, :2]] 
                                    for i in range(num_points-1)])
                lc = LineCollection(segments, colors=colors[:-1], linewidths=3, alpha=0.8)
                ax.add_collection(lc)
            
            # Scatter points
            ax.scatter(traj_pred_aligned[:, 0], traj_pred_aligned[:, 1], c=np.arange(num_points), 
                    cmap='coolwarm', s=50, zorder=3, edgecolors='white', linewidths=0.5)
            
            # Start/End markers for prediction
            ax.scatter(traj_pred_aligned[0, 0], traj_pred_aligned[0, 1], c='lime', s=200, marker='o', 
                    edgecolors='black', linewidths=2, label='Pred Start', zorder=5)
            ax.scatter(traj_pred_aligned[-1, 0], traj_pred_aligned[-1, 1], c='red', s=200, marker='*', 
                    edgecolors='black', linewidths=2, label='Pred End', zorder=5)
            
            # Direction arrows
            arrow_indices = np.linspace(0, num_points-2, min(5, num_points-1), dtype=int)
            for i in arrow_indices:
                dx = traj_pred_aligned[i+1, 0] - traj_pred_aligned[i, 0]
                dy = traj_pred_aligned[i+1, 1] - traj_pred_aligned[i, 1]
                scale = 0.3
                ax.annotate('', xy=(traj_pred_aligned[i, 0] + dx*scale, traj_pred_aligned[i, 1] + dy*scale),
                        xytext=(traj_pred_aligned[i, 0], traj_pred_aligned[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
            
            # Plot ground truth
            if traj_gt is not None and len(traj_gt) > 0:
                ax.plot(traj_gt[:, 0], traj_gt[:, 1], 'g--', linewidth=3, 
                    alpha=0.8, label='Ground Truth')
                # GT start/end markers
                ax.scatter(traj_gt[0, 0], traj_gt[0, 1], c='green', s=150, marker='o', 
                        edgecolors='black', linewidths=2, zorder=4)
                ax.scatter(traj_gt[-1, 0], traj_gt[-1, 1], c='darkgreen', s=150, marker='s', 
                        edgecolors='black', linewidths=2, label='GT End', zorder=4)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Trajectory (XY Plane) - Step {self.global_step}\n(Prediction aligned to GT start)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='datalim')
            ax.autoscale()
            
            plt.tight_layout()
            
            wandb.log({
                "phase2/2d_trajectory_overlay": wandb.Image(fig)
            }, step=self.global_step)
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to log 2D trajectory: {e}")


    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "policy_state_dict": self.policy_module.state_dict(),
            "discriminator_state_dict": self.discriminator_module.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
            "policy_scheduler_state_dict": self.policy_scheduler.state_dict(),
            "cpr_step": self.cpr.step.item(),
            "best_loss": self.best_loss,
            "config": OmegaConf.to_container(self.cfg),
            "wandb_run_id": wandb.run.id if wandb.run else None,
        }
        
        # Save early stopping state if enabled
        if self.early_stopping is not None:
            checkpoint["early_stopping_state"] = self.early_stopping.state_dict()
        
        torch.save(checkpoint, self.ckpt_dir / "latest.pt")
        
        if self.epoch % self.cfg.training.phase2.save_every == 0:
            torch.save(checkpoint, self.ckpt_dir / f"epoch_{self.epoch}.pt")
        
        if is_best:
            torch.save(checkpoint, self.ckpt_dir / "best.pt")
            torch.save(self.policy_module.state_dict(), self.ckpt_dir / "policy.pth")

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_module.load_state_dict(checkpoint["policy_state_dict"])
        self.discriminator_module.load_state_dict(checkpoint["discriminator_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])
        self.cpr.step.fill_(checkpoint.get("cpr_step", 0))
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        
        # Restore early stopping state if present
        if self.early_stopping is not None and "early_stopping_state" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping_state"])
            logger.info(f"Restored early stopping: best={self.early_stopping.best_value:.4f} at epoch {self.early_stopping.best_epoch}")
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self) -> None:
        """Main training loop with distributed support."""
        if self.is_main:
            logger.info("Starting Phase 2: Policy Training with CPR")
            logger.info(f"Distributed: {self.is_distributed}, World Size: {self.world_size}")
        
        for epoch in range(self.epoch, self.cfg.training.phase2.epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.is_distributed:
                set_epoch_sampler(self.train_loader, epoch)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate (on all processes, but only log on main)
            if epoch % self.cfg.training.phase2.eval_every == 0:
                val_metrics = self.validate()
                
                # Only log and save on main process
                if self.is_main:
                    wandb.log({
                        "phase2/val_loss_cfm": val_metrics["cfm_loss"],
                        "phase2/val_action_mse": val_metrics["action_mse"],
                        "phase2/val_action_mae": val_metrics["action_mae"],
                        "phase2/epoch": epoch,
                    }, step=self.global_step)
                    
                    # Generate trajectory visualization
                    if epoch % self.cfg.logging.vis_every == 0:
                        self._generate_sample_trajectories()
                    
                    is_best = val_metrics["cfm_loss"] < self.best_loss
                    if is_best:
                        self.best_loss = val_metrics["cfm_loss"]
                    
                    self.save_checkpoint(is_best=is_best)
                    
                    logger.info(
                        f"Epoch {epoch}: "
                        f"Train CFM={train_metrics['policy_cfm_loss']:.4f}, "
                        f"Val CFM={val_metrics['cfm_loss']:.4f}, "
                        f"MSE={val_metrics['action_mse']:.4f}"
                    )
                    
                    # Early stopping check (based on validation loss)
                    if self.early_stopping is not None:
                        if self.early_stopping(val_metrics["cfm_loss"], epoch):
                            logger.info(f"⚠ Early stopping at epoch {epoch}!")
                            logger.info(f"  Best val loss: {self.early_stopping.best_value:.4f} at epoch {self.early_stopping.best_epoch}")
                            break
        
        if self.is_main:
            logger.info("Phase 2 training complete!")
            logger.info(f"Best validation loss: {self.best_loss:.4f}")
            logger.info(f"Policy saved to: {self.ckpt_dir / 'policy.pth'}")

    @torch.no_grad()
    def _generate_sample_trajectories(self) -> None:
        """Generate and log sample trajectories."""
        self.policy.eval()
        
        # Get one validation batch
        batch = next(iter(self.val_loader))
        rgb, depth, proprio, instruction, actions = batch
        
        rgb = rgb[:4].to(self.device)  # First 4 samples
        actions = actions[:4].to(self.device)
        
        if rgb.dim() == 5:
            current_rgb = rgb[:, 0]
            future_rgb = rgb[:, -1]
        else:
            current_rgb = rgb
            future_rgb = rgb
        
        current_state = self.encode_batch(current_rgb)
        future_state = self.encode_batch(future_rgb)
        goal = self.get_goal_embedding(future_state)
        
        # Generate trajectories
        pred_actions = self.policy_module.sample(
            current_state, goal,
            num_steps=self.cfg.training.phase2.num_inference_steps,
        )
        
        # Log trajectory visualization
        self._log_trajectory_visualization(pred_actions, actions)
        
        # Log 3D point cloud with trajectory overlay (improved with color gradient)
        self._log_3d_point_cloud_with_trajectory(rgb, pred_actions)
        
        # Log 2D trajectory overlay for easier visualization
        self._log_2d_trajectory_overlay(rgb, pred_actions, actions)


def train_policy(cfg: DictConfig) -> None:
    """
    Main entry point for Phase 2 training.
    
    Supports both single-GPU and multi-GPU distributed training.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup distributed training
    rank, world_size, is_distributed = setup_distributed()
    
    # Set device based on distributed mode
    if is_distributed:
        local_rank = get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Distributed training: rank={rank}/{world_size}, local_rank={local_rank}, device={device}")
    else:
        device = torch.device(cfg.hardware.device)
    
    # Set random seed (different per rank)
    seed = cfg.hardware.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize wandb (only on main process)
    if is_main_process():

        # Check if resuming from checkpoint to get wandb run_id
        wandb_run_id = None
        wandb_resume = None
        if cfg.checkpoint.resume:
            resume_ckpt = torch.load(cfg.checkpoint.resume, map_location="cpu", weights_only=False)
            wandb_run_id = resume_ckpt.get("wandb_run_id")
            if wandb_run_id:
                wandb_resume = "must"
                logger.info(f"Resuming wandb run: {wandb_run_id}")

        # Get dataset-specific tag for wandb
        dataset_type = cfg.data.get("dataset", "libero")
        if dataset_type == "calvin":
            dataset_tag = cfg.data.get("calvin_env", "D")
        elif dataset_type == "rlbench":
            dataset_tag = cfg.data.get("rlbench_tasks", "all")
        else:
            dataset_tag = cfg.data.get("libero_suite", "full")
        
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.logging.wandb.name,  # Custom run name (None = auto-generated)
            config=OmegaConf.to_container(cfg),
            tags=["phase2", dataset_type, dataset_tag, f"gpus_{world_size}"],
            group=cfg.logging.wandb.group or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_code=cfg.logging.wandb.save_code,
            id=wandb_run_id,
            resume=wandb_resume,
        )
    
    try:
        trainer = PolicyTrainer(cfg, device, is_distributed=is_distributed)
        
        if cfg.checkpoint.resume:
            trainer.load_checkpoint(cfg.checkpoint.resume)
        
        trainer.train()
        
    finally:
        if is_main_process():
            wandb.finish()
        cleanup_distributed()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    train_policy(cfg)


if __name__ == "__main__":
    main()

