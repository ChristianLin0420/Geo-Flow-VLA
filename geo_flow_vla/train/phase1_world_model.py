"""
Phase 1: Unsupervised World Model Training for Geo-Flow VLA.

Trains the Forward-Backward (FB) world model to learn manipulation physics.
The world model learns:
- Backward Network: Maps future states to goal embeddings on unit sphere
- Forward Network: Predicts future state features given current state, action, goal

Training Process:
    1. Load LIBERO trajectories
    2. Extract states via frozen dual encoder
    3. Train F and B networks with FB objective
    4. Save world_model.pth

WandB Logging:
    - FB losses (forward, backward, total)
    - Embedding visualizations (t-SNE)
    - Gradient norms
    - Learning dynamics

Usage:
    python -m geo_flow_vla.train.phase1_world_model
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import numpy as np
from tqdm import tqdm

from ..models.dual_encoder import DualEncoder
from ..models.world_model import FBWorldModel
from ..losses.fb_objective import FBObjective
from ..data.libero_dataset import LIBERODataset, create_libero_dataloaders
from ..data.normalizer import StateActionNormalizer
from ..utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    wrap_model_ddp,
    create_distributed_dataloader,
    set_epoch_sampler,
    reduce_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorldModelTrainer:
    """
    Trainer for Phase 1: Forward-Backward World Model.
    
    Handles:
    - Data loading and preprocessing
    - Training loop with gradient accumulation
    - Checkpointing and logging
    - Visualization generation
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
        
        # Build optimizer and scheduler
        self._build_optimizer()
        
        # Build data loaders
        self._build_dataloaders()
        
        # Build loss function
        self.fb_objective = FBObjective(
            forward_weight=cfg.training.phase1.forward_weight,
            backward_weight=cfg.training.phase1.backward_weight,
        )
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if cfg.hardware.precision != "fp32" else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Checkpoint directory (only create on main process)
        self.ckpt_dir = Path(cfg.checkpoint.dir) / "phase1"
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @property
    def world_model_module(self):
        """Get the underlying world model module (unwrapped from DDP if needed)."""
        if self.is_distributed:
            return self.world_model.module
        return self.world_model

    def _build_models(self) -> None:
        """Initialize dual encoder and world model."""
        cfg = self.cfg
        
        # Dual encoder (frozen)
        self.dual_encoder = DualEncoder(
            semantic_dim=cfg.model.semantic_dim,
            geometric_dim=cfg.model.geometric_dim,
            output_dim=cfg.model.state_dim,
            use_mock_moge=cfg.encoders.moge.use_mock,
        ).to(self.device)
        
        # Freeze dual encoder
        for param in self.dual_encoder.parameters():
            param.requires_grad = False
        self.dual_encoder.eval()
        
        # World model (trainable)
        self.world_model = FBWorldModel(
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            action_horizon=cfg.model.action_horizon,
            latent_dim=cfg.model.fb.latent_dim,
            hidden_dim=cfg.model.fb.hidden_dim,
            num_residual_blocks=cfg.model.fb.num_residual_blocks,
            ema_tau=cfg.model.fb.ema_tau,
        ).to(self.device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.world_model = wrap_model_ddp(
                self.world_model, 
                self.device,
                find_unused_parameters=cfg.hardware.get("find_unused_params", False),
            )
        
        logger.info(f"World model parameters: {sum(p.numel() for p in self.world_model.parameters()):,}")

    def _build_optimizer(self) -> None:
        """Initialize optimizer and learning rate scheduler."""
        cfg = self.cfg.training.phase1
        
        self.optimizer = torch.optim.AdamW(
            self.world_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / cfg.warmup_steps
            progress = (step - cfg.warmup_steps) / max(1, self.total_steps - cfg.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _build_dataloaders(self) -> None:
        """Initialize data loaders with distributed sampling if enabled."""
        cfg = self.cfg
        
        # Create dataset
        from torch.utils.data import random_split
        
        full_dataset = LIBERODataset(
            data_root=cfg.data.data_root,
            suite=cfg.data.libero_suite,
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
                batch_size=cfg.training.phase1.batch_size,
                num_workers=cfg.data.num_workers,
                shuffle=True,
                pin_memory=cfg.data.pin_memory,
            )
            self.val_loader = create_distributed_dataloader(
                val_dataset,
                batch_size=cfg.training.phase1.batch_size,
                num_workers=cfg.data.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=cfg.data.pin_memory,
            )
        else:
            from torch.utils.data import DataLoader
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.training.phase1.batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                drop_last=True,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.training.phase1.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
            )
        
        self.total_steps = len(self.train_loader) * cfg.training.phase1.epochs
        
        logger.info(f"Train samples: {len(train_dataset)} (per GPU: {len(self.train_loader.dataset) // self.world_size})")
        logger.info(f"Val samples: {len(val_dataset)}")

    @torch.no_grad()
    def encode_batch(
        self,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Encode batch of images to state embeddings."""
        rgb = rgb.to(self.device)
        return self.dual_encoder(rgb)

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch with distributed support."""
        self.world_model.train()
        
        epoch_losses = {
            "loss": 0.0,
            "forward_loss": 0.0,
            "backward_loss": 0.0,
        }
        num_batches = 0
        
        # Only show progress bar on main process
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", disable=not self.is_main)
        
        for batch in pbar:
            rgb, depth, proprio, instruction, actions = batch
            
            # Move to device
            rgb = rgb.to(self.device)
            actions = actions.to(self.device)
            
            # Encode states (using first and last frames)
            # For trajectory, we need current and future state
            with torch.no_grad():
                # Current state from first frame
                if rgb.dim() == 5:  # (B, T, C, H, W)
                    current_rgb = rgb[:, 0]
                    future_rgb = rgb[:, -1]
                else:  # (B, C, H, W) - single frame
                    current_rgb = rgb
                    future_rgb = rgb  # Fallback
                
                current_state = self.encode_batch(current_rgb)
                future_state = self.encode_batch(future_rgb)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast('cuda'):
                    loss_dict = self.fb_objective(
                        self.world_model,
                        current_state,
                        future_state,
                        actions,
                    )
                
                self.scaler.scale(loss_dict["loss"]).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.world_model.parameters(),
                    self.cfg.training.phase1.gradient_clip,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.fb_objective(
                    self.world_model,
                    current_state,
                    future_state,
                    actions,
                )
                
                loss_dict["loss"].backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.world_model.parameters(),
                    self.cfg.training.phase1.gradient_clip,
                )
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update EMA target network (access underlying module for custom methods)
            self.world_model_module.update_target_network()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            num_batches += 1
            
            # Log to wandb (only on main process)
            if self.is_main and self.global_step % self.cfg.logging.log_every == 0:
                wandb.log({
                    "phase1/loss_total": loss_dict["loss"].item(),
                    "phase1/loss_forward": loss_dict["forward_loss"].item(),
                    "phase1/loss_backward": loss_dict["backward_loss"].item(),
                    "phase1/z_norm_mean": loss_dict["z_norm"].item(),
                    "phase1/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "phase1/lr": self.scheduler.get_last_lr()[0],
                    "phase1/step": self.global_step,
                }, step=self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['loss'].item():.4f}",
                "fwd": f"{loss_dict['forward_loss'].item():.4f}",
                "bwd": f"{loss_dict['backward_loss'].item():.4f}",
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(1, num_batches)
        
        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.world_model.eval()
        
        val_losses = {
            "loss": 0.0,
            "forward_loss": 0.0,
            "backward_loss": 0.0,
        }
        num_batches = 0
        
        all_z = []
        last_rgb = None  # Store for 3D visualization
        last_current_state = None
        last_future_state = None
        last_actions = None
        
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
            
            loss_dict = self.fb_objective(
                self.world_model,
                current_state,
                future_state,
                actions,
            )
            
            for key in val_losses:
                if key in loss_dict:
                    val_losses[key] += loss_dict[key].item()
            num_batches += 1
            
            # Collect embeddings for visualization
            z = self.world_model_module.encode_goal(future_state)
            all_z.append(z.cpu())
            
            # Store last batch for visualizations
            last_rgb = current_rgb
            last_current_state = current_state
            last_future_state = future_state
            last_actions = actions
        
        for key in val_losses:
            val_losses[key] /= max(1, num_batches)
        
        # Generate visualizations (only on main process)
        if self.is_main and self.epoch % self.cfg.logging.vis_every == 0:
            all_z = torch.cat(all_z, dim=0)[:1000]  # Limit for speed
            self._log_tsne(all_z)
            
            # Log 3D point cloud
            if last_rgb is not None:
                self._log_3d_point_cloud(last_rgb)
            
            # Log forward prediction error heatmap
            if last_current_state is not None:
                self._log_forward_prediction_error(
                    last_current_state, last_future_state, last_actions
                )
        
        return val_losses

    def _log_tsne(self, embeddings: torch.Tensor) -> None:
        """Generate and log t-SNE visualization."""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            # Ensure embeddings are on CPU and detached
            embeddings_np = embeddings.detach().cpu().numpy()
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            z_2d = tsne.fit_transform(embeddings_np)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=5)
            ax.set_title(f"FB Embeddings t-SNE (Epoch {self.epoch})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            
            wandb.log({"phase1/z_tsne": wandb.Image(fig)}, step=self.global_step)
            plt.close(fig)
            
            # Log histogram of z components
            wandb.log({
                "phase1/z_histogram": wandb.Histogram(embeddings_np.flatten())
            }, step=self.global_step)
            
            logger.info(f"Logged t-SNE and histogram for {len(embeddings_np)} embeddings")
            
        except ImportError:
            logger.warning("sklearn not available for t-SNE visualization")
        except Exception as e:
            logger.warning(f"Failed to generate t-SNE/histogram: {e}")

    def _log_3d_point_cloud(self, rgb: torch.Tensor) -> None:
        """Generate and log 3D point cloud visualization from MoGe-2."""
        try:
            # Get point cloud from MoGe-2
            # rgb shape: (B, C, H, W) or (B, T, C, H, W)
            if rgb.dim() == 5:
                rgb_frame = rgb[0, 0]  # Take first frame of first batch
            else:
                rgb_frame = rgb[0]  # Take first batch
            
            # Get point map from dual encoder's MoGe lifter
            point_map = self.dual_encoder.get_point_map(rgb_frame.unsqueeze(0))  # (1, 3, H, W)
            
            if point_map is not None:
                # Reshape to (N, 3)
                H, W = point_map.shape[-2:]
                points = point_map[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()  # (H*W, 3)
                
                # Get RGB colors - images are already in [0, 1] from dataset
                rgb_np = rgb_frame.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                
                # Debug: log color stats
                logger.info(f"RGB stats: min={rgb_np.min():.3f}, max={rgb_np.max():.3f}, mean={rgb_np.mean():.3f}")
                
                # Ensure [0, 1] range (should already be, but clip for safety)
                rgb_np = np.clip(rgb_np, 0, 1)
                
                # WandB Object3D expects colors as uint8 [0, 255]
                rgb_colors = (rgb_np * 255).astype(np.uint8).reshape(-1, 3)  # (H*W, 3)
                
                # Downsample for performance
                stride = 1
                points_ds = points[::stride].astype(np.float64)  # WandB prefers float64 for coords
                colors_ds = rgb_colors[::stride]
                
                # Combine: WandB Object3D expects (N, 6) with XYZ as float and RGB as [0,255] uint8
                # But numpy concat requires same dtype, so we need to use structured approach
                # Create the array in the format WandB expects
                N = len(points_ds)
                point_cloud = np.zeros((N, 6), dtype=np.float64)
                point_cloud[:, :3] = points_ds  # XYZ
                point_cloud[:, 3:] = colors_ds.astype(np.float64)  # RGB as float [0, 255]
                
                # Log to WandB
                wandb.log({
                    "phase1/3d_point_cloud": wandb.Object3D(point_cloud)
                }, step=self.global_step)
                
                logger.info(f"Logged 3D point cloud with {N} points, colors [0-255]")
                
        except Exception as e:
            logger.warning(f"Failed to log 3D point cloud: {e}")

    def _log_forward_prediction_error(
        self,
        current_states: torch.Tensor,
        future_states: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """Log forward prediction error heatmap."""
        try:
            import matplotlib.pyplot as plt
            
            with torch.no_grad():
                # Get goal embedding from future states
                z = self.world_model_module.encode_goal(future_states)
                
                # Predict future states
                pred_states = self.world_model_module.predict_future(
                    current_states, actions, z
                )
                
                # Compute per-dimension prediction error
                errors = (pred_states - future_states).abs().cpu().numpy()  # (B, state_dim)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 4))
                im = ax.imshow(errors[:16].T, aspect='auto', cmap='viridis')
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("State Dimension")
                ax.set_title(f"Forward Prediction Error (Epoch {self.epoch})")
                plt.colorbar(im, ax=ax, label="Absolute Error")
                plt.tight_layout()
                
                wandb.log({
                    "phase1/pred_error_heatmap": wandb.Image(fig)
                }, step=self.global_step)
                plt.close(fig)
                
        except Exception as e:
            logger.warning(f"Failed to log prediction error heatmap: {e}")

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch + 1,
            "global_step": self.global_step,
            "world_model_state_dict": self.world_model_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": OmegaConf.to_container(self.cfg),
            "wandb_run_id": wandb.run.id if wandb.run else None,
        }
        
        # Save latest
        torch.save(checkpoint, self.ckpt_dir / "latest.pt")
        
        # Save periodic
        if self.epoch % self.cfg.training.phase1.save_every == 0:
            torch.save(checkpoint, self.ckpt_dir / f"epoch_{self.epoch}.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.ckpt_dir / "best.pt")
            # Also save just the world model for easy loading
            torch.save(
                self.world_model_module.state_dict(),
                self.ckpt_dir / "world_model.pth",
            )

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.world_model_module.load_state_dict(checkpoint["world_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self) -> None:
        """Main training loop with distributed training support."""
        if self.is_main:
            logger.info("Starting Phase 1: World Model Training")
            logger.info(f"Distributed: {self.is_distributed}, World Size: {self.world_size}")
        
        for epoch in range(self.epoch, self.cfg.training.phase1.epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler (ensures proper shuffling)
            if self.is_distributed:
                set_epoch_sampler(self.train_loader, epoch)
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate (on all processes, but only log on main)
            if epoch % self.cfg.training.phase1.eval_every == 0:
                val_losses = self.validate()
                
                # Only log and save on main process
                if self.is_main:
                    # Log validation metrics
                    wandb.log({
                        "phase1/val_loss_total": val_losses["loss"],
                        "phase1/val_loss_forward": val_losses["forward_loss"],
                        "phase1/val_loss_backward": val_losses["backward_loss"],
                        "phase1/epoch": epoch,
                    }, step=self.global_step)
                    
                    # Check for best model
                    is_best = val_losses["loss"] < self.best_loss
                    if is_best:
                        self.best_loss = val_losses["loss"]
                    
                    self.save_checkpoint(is_best=is_best)
                    
                    logger.info(
                        f"Epoch {epoch}: "
                        f"Train Loss={train_losses['loss']:.4f}, "
                        f"Val Loss={val_losses['loss']:.4f}"
                    )
        
        if self.is_main:
            logger.info("Phase 1 training complete!")
            logger.info(f"Best validation loss: {self.best_loss:.4f}")
            logger.info(f"Model saved to: {self.ckpt_dir / 'world_model.pth'}")


def train_world_model(cfg: DictConfig) -> None:
    """
    Main entry point for Phase 1 training.
    
    Supports both single-GPU and multi-GPU distributed training.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup distributed training if enabled
    rank, world_size, is_distributed = setup_distributed()
    
    # Set device based on distributed mode
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Distributed training: rank={rank}/{world_size}, device={device}")
    else:
        device = torch.device(cfg.hardware.device)
    
    # Set random seed (different per rank for data augmentation diversity)
    seed = cfg.hardware.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize wandb (only on main process)
    if is_main_process():
        # Check if resuming from checkpoint to get wandb run_id
        wandb_run_id = None
        wandb_resume = None
        if cfg.checkpoint.resume:
            resume_ckpt = torch.load(cfg.checkpoint.resume, map_location="cpu")
            wandb_run_id = resume_ckpt.get("wandb_run_id")
            if wandb_run_id:
                wandb_resume = "must"
                logger.info(f"Resuming wandb run: {wandb_run_id}")
        
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.logging.wandb.name,  # Custom run name (None = auto-generated)
            config=OmegaConf.to_container(cfg),
            tags=["phase1", cfg.data.libero_suite, f"gpus_{world_size}"],
            group=cfg.logging.wandb.group or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_code=cfg.logging.wandb.save_code,
            id=wandb_run_id,
            resume=wandb_resume,
        )
    
    try:
        # Create trainer and run
        trainer = WorldModelTrainer(cfg, device, is_distributed=is_distributed)
        
        # Resume if checkpoint exists
        if cfg.checkpoint.resume:
            trainer.load_checkpoint(cfg.checkpoint.resume)
        
        trainer.train()
        
    finally:
        # Cleanup
        if is_main_process():
            wandb.finish()
        cleanup_distributed()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    train_world_model(cfg)


if __name__ == "__main__":
    main()

