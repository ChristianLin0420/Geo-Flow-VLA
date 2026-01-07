"""
Alignment Visualization Utilities for Geo-Flow VLA.

Provides comprehensive visualizations for observing language-vision-action alignments
during training. All functions return wandb-loggable figures.

Usage:
    from geo_flow_vla.utils.visualizations import AlignmentVisualizer
    
    visualizer = AlignmentVisualizer(cfg.logging)
    visualizer.log_all(epoch=current_epoch, step=global_step, ...)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Lazy imports for visualization libraries
def _get_plt():
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    return plt

def _get_sns():
    import seaborn as sns
    return sns


class AlignmentVisualizer:
    """
    Centralized visualization class for alignment monitoring.
    
    Integrates with wandb for logging during training.
    Uses existing config: logging.vis_every (epochs), logging.num_vis_samples
    """
    
    def __init__(self, logging_cfg: DictConfig):
        """
        Args:
            logging_cfg: The logging section of your config (cfg.logging)
        """
        self.vis_every = logging_cfg.get("vis_every", 1)  # Epochs
        self.num_vis_samples = logging_cfg.get("num_vis_samples", 1)
        self.enabled = logging_cfg.wandb.get("enabled", True)
        self._cache = {}  # For accumulating embeddings across batches
        
    def should_log(self, epoch: int) -> bool:
        """Check if we should log at this epoch."""
        return self.enabled and (epoch % self.vis_every == 0)
    
    def clear_cache(self):
        """Clear accumulated embeddings."""
        self._cache = {}
    
    # =========================================================================
    # 1. Language-Goal Embedding Alignment
    # =========================================================================
    
    def visualize_language_goal_alignment(
        self,
        language_embeddings: Tensor,
        goal_embeddings: Tensor,
        task_labels: Optional[List[str]] = None,
        projection_layer: Optional[torch.nn.Module] = None,
    ) -> Dict[str, any]:
        """
        Joint visualization showing if language and goal embeddings align.
        
        Args:
            language_embeddings: (B, lang_dim) from CLIP
            goal_embeddings: (B, latent_dim) from Backward network
            task_labels: Optional task names for coloring
            projection_layer: Optional projection from lang_dim to latent_dim
            
        Returns:
            Dict of wandb-loggable objects
        """
        import wandb
        plt = _get_plt()
        
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning("sklearn not available for t-SNE")
            return {}
        
        # Use configured number of samples - ensure we use minimum of both embedding lengths
        min_embeddings = min(len(language_embeddings), len(goal_embeddings))
        n_samples = min(min_embeddings, self.num_vis_samples * 64)
        if n_samples < min_embeddings:
            idx = torch.randperm(n_samples)
            language_embeddings = language_embeddings[:n_samples][idx]
            goal_embeddings = goal_embeddings[:n_samples][idx]
            if task_labels is not None and len(task_labels) > 0:
                # Safe indexing: wrap indices if task_labels is shorter
                task_labels = [task_labels[i.item() % len(task_labels)] for i in idx]
        else:
            # Truncate both to the same size without shuffling
            language_embeddings = language_embeddings[:n_samples]
            goal_embeddings = goal_embeddings[:n_samples]
            if task_labels is not None and len(task_labels) > n_samples:
                task_labels = task_labels[:n_samples]
        
        lang_np = language_embeddings.detach().cpu().numpy()
        goal_np = goal_embeddings.detach().cpu().numpy()
        
        logger.info(f"[DEBUG] Language-goal viz: lang_np shape={lang_np.shape}, goal_np shape={goal_np.shape}")
        
        # Project language to goal dimension if needed
        if projection_layer is not None:
            with torch.no_grad():
                lang_proj = projection_layer(language_embeddings).cpu().numpy()
        else:
            if lang_np.shape[1] != goal_np.shape[1]:
                from sklearn.decomposition import PCA
                n_samples_pca = lang_np.shape[0]
                # PCA n_components must be <= min(n_samples, n_features)
                target_dim = min(goal_np.shape[1], n_samples_pca, lang_np.shape[1])
                logger.info(f"[DEBUG] PCA projection: target_dim={target_dim} (from goal_dim={goal_np.shape[1]}, n_samples={n_samples_pca}, lang_dim={lang_np.shape[1]})")
                
                if target_dim < 2:
                    logger.warning(f"Cannot perform PCA with target_dim={target_dim}, skipping language-goal viz")
                    return {}
                
                pca_lang = PCA(n_components=target_dim)
                lang_proj = pca_lang.fit_transform(lang_np)
                # Also reduce goal_np to match dimensions for fair comparison
                if target_dim < goal_np.shape[1]:
                    pca_goal = PCA(n_components=target_dim)
                    goal_np = pca_goal.fit_transform(goal_np)
                    logger.info(f"[DEBUG] Reduced goal_np to {goal_np.shape} for matching dimensions")
            else:
                lang_proj = lang_np
        
        n = len(lang_proj)
        combined = np.concatenate([lang_proj, goal_np], axis=0)
        
        perplexity = min(30, len(combined) // 4)
        if perplexity < 5:
            logger.warning("Too few samples for t-SNE visualization")
            return {}
            
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = tsne.fit_transform(combined)
        
        if task_labels is not None:
            unique_tasks = list(set(task_labels))
            color_map = {task: i for i, task in enumerate(unique_tasks)}
            colors = [color_map[t] for t in task_labels]
        else:
            colors = list(range(n))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        ax = axes[0]
        ax.scatter(coords[:n, 0], coords[:n, 1], c=colors, cmap='tab20', marker='o', 
                  s=80, alpha=0.7, label='Language (CLIP)', edgecolors='black', linewidths=0.5)
        ax.scatter(coords[n:, 0], coords[n:, 1], c=colors, cmap='tab20', marker='^', 
                  s=80, alpha=0.7, label='Goal (Backward)', edgecolors='black', linewidths=0.5)
        
        for i in range(n):
            ax.plot([coords[i, 0], coords[n+i, 0]], [coords[i, 1], coords[n+i, 1]], 
                   'k-', alpha=0.15, linewidth=0.8)
        
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title("Language <-> Goal Embedding Alignment (t-SNE)", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        
        ax2 = axes[1]
        lang_norm = lang_proj / (np.linalg.norm(lang_proj, axis=1, keepdims=True) + 1e-8)
        goal_norm = goal_np / (np.linalg.norm(goal_np, axis=1, keepdims=True) + 1e-8)
        paired_sim = np.sum(lang_norm * goal_norm, axis=1)
        cross_sim = np.sum(lang_norm * np.roll(goal_norm, 1, axis=0), axis=1)
        
        ax2.hist(paired_sim, bins=30, alpha=0.7, label='Paired (same sample)', color='green', edgecolor='black')
        ax2.hist(cross_sim, bins=30, alpha=0.7, label='Unpaired (shifted)', color='red', edgecolor='black')
        ax2.axvline(paired_sim.mean(), color='green', linestyle='--', linewidth=2)
        ax2.axvline(cross_sim.mean(), color='red', linestyle='--', linewidth=2)
        ax2.legend(fontsize=9)
        ax2.set_xlabel("Cosine Similarity")
        ax2.set_ylabel("Count")
        ax2.set_title("Language-Goal Similarity Distribution")
        
        plt.tight_layout()
        
        result = {
            "alignment/language_goal_tsne": wandb.Image(fig),
            "alignment/lang_goal_paired_sim_mean": paired_sim.mean(),
            "alignment/lang_goal_paired_sim_std": paired_sim.std(),
            "alignment/lang_goal_unpaired_sim_mean": cross_sim.mean(),
        }
        
        plt.close(fig)
        return result

    # =========================================================================
    # 2. Cross-Attention Heatmap
    # =========================================================================
    
    def visualize_cross_attention(
        self,
        rgb: Tensor,
        attention_weights: Tensor,
        positions: Optional[Tensor] = None,
        sample_idx: int = 0,
    ) -> Dict[str, any]:
        """Visualize which geometric regions semantic tokens attend to."""
        import wandb
        plt = _get_plt()
        
        rgb_np = rgb[sample_idx].permute(1, 2, 0).detach().cpu().numpy()
        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        
        attn = attention_weights[sample_idx].detach().cpu().numpy()
        attn_avg = attn.mean(axis=(0, 1))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].imshow(rgb_np)
        axes[0].set_title("RGB Input")
        axes[0].axis('off')
        
        ax = axes[1]
        H, W = rgb_np.shape[:2]
        n_g = len(attn_avg)
        spatial_size = int(np.sqrt(n_g))
        
        if spatial_size * spatial_size == n_g:
            attn_map = attn_avg.reshape(spatial_size, spatial_size)
            from scipy.ndimage import zoom
            scale = H / spatial_size
            attn_upsampled = zoom(attn_map, scale, order=1)
            ax.imshow(rgb_np)
            im = ax.imshow(attn_upsampled, cmap='hot', alpha=0.6)
            plt.colorbar(im, ax=ax, label='Attention')
            ax.set_title("Attention Overlay")
            ax.axis('off')
        else:
            ax.bar(range(len(attn_avg)), attn_avg)
            ax.set_xlabel("Geometric Token Index")
            ax.set_ylabel("Attention Weight")
            ax.set_title("Attention Distribution")
        
        ax3 = axes[2]
        num_heads = attn.shape[0]
        head_attn = attn.mean(axis=1)
        im = ax3.imshow(head_attn, aspect='auto', cmap='viridis')
        ax3.set_xlabel("Geometric Token")
        ax3.set_ylabel("Attention Head")
        ax3.set_title("Per-Head Attention Pattern")
        plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        
        result = {
            "attention/cross_attention_vis": wandb.Image(fig),
            "attention/max_attention": float(attn_avg.max()),
            "attention/attention_entropy": float(-np.sum(attn_avg * np.log(attn_avg + 1e-10))),
        }
        
        plt.close(fig)
        return result

    # =========================================================================
    # 3. Forward Prediction Trajectory
    # =========================================================================
    
    def visualize_forward_rollout(
        self,
        forward_net: torch.nn.Module,
        current_state: Tensor,
        actions: Tensor,
        goal: Tensor,
        target_states: Optional[Tensor] = None,
        num_steps: int = 5,
    ) -> Dict[str, any]:
        """Visualize multi-step forward prediction in state space."""
        import wandb
        plt = _get_plt()
        
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            logger.warning("sklearn not available for PCA")
            return {}
        
        device = current_state.device
        
        # Ensure consistent batch sizes - use smallest common batch
        batch_size = min(current_state.shape[0], actions.shape[0], goal.shape[0])
        if batch_size == 0:
            logger.warning("Empty batch for forward rollout visualization")
            return {}
        
        current_state = current_state[:batch_size]
        actions = actions[:batch_size]
        goal = goal[:batch_size]
        
        # Validate state dimension matches forward_net expectations
        # Handle wrapped modules (e.g., DistributedDataParallel)
        net_to_check = forward_net
        if hasattr(forward_net, 'module'):
            net_to_check = forward_net.module
        expected_state_dim = getattr(net_to_check, 'state_dim', None)
        
        logger.info(f"[DEBUG] Forward rollout: current_state shape={current_state.shape}, "
                   f"actions shape={actions.shape}, goal shape={goal.shape}, "
                   f"expected_state_dim={expected_state_dim}, forward_net type={type(forward_net).__name__}")
        
        if expected_state_dim is not None and current_state.shape[-1] != expected_state_dim:
            logger.warning(f"State dim mismatch: got {current_state.shape[-1]}, expected {expected_state_dim}. Skipping forward rollout viz.")
            return {}
        
        # If we couldn't get expected_state_dim, check if dimensions are reasonable
        if expected_state_dim is None:
            logger.warning(f"Could not determine expected_state_dim from forward_net. "
                          f"current_state dim={current_state.shape[-1]}. Proceeding with caution.")
            # Skip if state dim seems too small (likely raw observations, not encoded state)
            if current_state.shape[-1] < 64:
                logger.warning(f"State dim {current_state.shape[-1]} too small, likely not encoded. Skipping forward rollout viz.")
                return {}
        
        # Handle actions shape: expect (B, horizon, action_dim)
        if actions.dim() == 2:
            # Actions might be flattened, try to infer action_dim
            # Common action dims: 7 (robot arm), 8, 4, etc.
            for action_dim in [7, 8, 4, 6]:
                if actions.shape[1] % action_dim == 0:
                    action_horizon = actions.shape[1] // action_dim
                    actions = actions.view(batch_size, action_horizon, action_dim)
                    break
            else:
                logger.warning(f"Cannot reshape actions {actions.shape} to 3D tensor")
                return {}
        
        action_horizon = actions.shape[1]
        action_dim = actions.shape[2]
        
        # Get expected action horizon from forward_net if available
        expected_action_horizon = getattr(net_to_check, 'action_horizon', None)
        logger.info(f"[DEBUG] Forward rollout: action_horizon={action_horizon}, action_dim={action_dim}, "
                   f"expected_action_horizon={expected_action_horizon}")
        
        # ForwardNetwork expects full action sequence (B, action_horizon, action_dim)
        # It outputs a single predicted future state, not a trajectory
        # We'll visualize multiple samples' predictions to show the model's behavior
        
        num_vis_samples = min(num_steps, batch_size)  # Reuse num_steps as number of samples to visualize
        
        # Ensure all inputs are on the same device as the forward_net
        try:
            # Get device from forward_net parameters
            net_device = next(forward_net.parameters()).device
        except StopIteration:
            net_device = device  # Fallback to input device
        
        logger.info(f"[DEBUG] Forward rollout devices: current_state={current_state.device}, "
                   f"actions={actions.device}, goal={goal.device}, forward_net={net_device}")
        
        # Move inputs to forward_net's device
        current_state = current_state.to(net_device)
        actions = actions.to(net_device)
        goal = goal.to(net_device)
        
        with torch.no_grad():
            try:
                # Call forward_net with full action sequence for multiple samples
                pred_future_states = forward_net(
                    current_state[:num_vis_samples], 
                    actions[:num_vis_samples], 
                    goal[:num_vis_samples]
                )
                logger.info(f"[DEBUG] Forward net prediction successful: output shape={pred_future_states.shape}")
            except Exception as e:
                logger.warning(f"Forward net failed: {e}")
                return {}
        
        # Collect start states and predicted end states
        start_states = current_state[:num_vis_samples].detach().cpu().numpy()
        pred_states = pred_future_states.detach().cpu().numpy()
        
        # Stack for PCA: [start_1, pred_1, start_2, pred_2, ...]
        all_states = []
        for i in range(num_vis_samples):
            all_states.append(start_states[i])
            all_states.append(pred_states[i])
        all_states = np.array(all_states)
        
        # Need at least 2 samples for meaningful PCA
        n_pca_components = min(2, len(all_states), all_states.shape[1])
        if n_pca_components < 2:
            logger.warning(f"Not enough dimensions for PCA visualization")
            return {}
        
        pca = PCA(n_components=n_pca_components)
        all_2d = pca.fit_transform(all_states)
        
        # Separate back into start and pred pairs
        start_2d = all_2d[0::2]  # Even indices: start states
        pred_2d = all_2d[1::2]   # Odd indices: predicted states
        
        # Handle target states if provided
        target_2d = None
        if target_states is not None and len(target_states) >= num_vis_samples:
            target_np = target_states[:num_vis_samples].detach().cpu().numpy()
            # Project target states using the same PCA
            target_2d = pca.transform(target_np)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        colors = plt.cm.viridis(np.linspace(0, 1, num_vis_samples))
        
        # Draw arrows from start to predicted end for each sample
        for i in range(num_vis_samples):
            ax.annotate('', xy=pred_2d[i], xytext=start_2d[i],
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=2, alpha=0.7))
        
        # Plot start states
        ax.scatter(start_2d[:, 0], start_2d[:, 1], c='green', s=150, marker='o', 
                  zorder=5, edgecolors='black', label='Start States')
        # Plot predicted states
        ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c='red', s=150, marker='s', 
                  zorder=5, edgecolors='black', label='Predicted Future')
        
        if target_2d is not None:
            ax.scatter(target_2d[:, 0], target_2d[:, 1], c='blue', s=150, marker='D', 
                      zorder=5, edgecolors='black', alpha=0.7, label='Ground Truth')
        
        ax.legend(fontsize=10)
        ax.set_title(f"Forward Model: Start → Predicted ({num_vis_samples} samples)", fontsize=12)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        # Compare norms: start states vs predicted states
        start_norms = np.linalg.norm(start_states, axis=1)
        pred_norms = np.linalg.norm(pred_states, axis=1)
        
        x = np.arange(num_vis_samples)
        width = 0.35
        ax2.bar(x - width/2, start_norms, width, label='Start State', color='green', alpha=0.7)
        ax2.bar(x + width/2, pred_norms, width, label='Predicted Future', color='red', alpha=0.7)
        
        if target_states is not None and len(target_states) >= num_vis_samples:
            target_norms = np.linalg.norm(target_states[:num_vis_samples].detach().cpu().numpy(), axis=1)
            ax2.scatter(x, target_norms, c='blue', s=100, marker='D', zorder=5, label='Ground Truth')
        
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("State Norm")
        ax2.set_title("State Magnitude: Start vs Predicted")
        ax2.legend()
        ax2.set_xticks(x)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        metrics = {"world_model/forward_rollout": wandb.Image(fig)}
        
        # Compute prediction error if ground truth available
        if target_states is not None and len(target_states) >= num_vis_samples:
            target_np = target_states[:num_vis_samples].detach().cpu().numpy()
            pred_errors = np.linalg.norm(pred_states - target_np, axis=1)
            metrics["world_model/mean_pred_error"] = float(pred_errors.mean())
            metrics["world_model/std_pred_error"] = float(pred_errors.std())
        
        plt.close(fig)
        return metrics

    # =========================================================================
    # 4. InfoNCE Contrastive Similarity Matrix
    # =========================================================================
    
    def visualize_contrastive_matrix(
        self,
        z_online: Tensor,
        z_target: Tensor,
        task_labels: Optional[List[str]] = None,
        temperature: float = 0.1,
    ) -> Dict[str, any]:
        """Visualize the InfoNCE similarity matrix."""
        import wandb
        plt = _get_plt()
        sns = _get_sns()
        
        max_samples = min(64, len(z_online))
        if len(z_online) > max_samples:
            idx = torch.randperm(len(z_online))[:max_samples]
            z_online = z_online[idx]
            z_target = z_target[idx]
            if task_labels is not None:
                task_labels = [task_labels[i] for i in idx.tolist()]
        
        z_on = z_online.detach().cpu()
        z_tg = z_target.detach().cpu()
        
        sim_matrix = torch.mm(z_on, z_tg.t()) / temperature
        sim_softmax = torch.softmax(sim_matrix, dim=-1).numpy()
        sim_raw = sim_matrix.numpy()
        
        B = len(sim_raw)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        ax = axes[0]
        labels = [t[:15] for t in task_labels] if task_labels and len(task_labels) <= 20 else False
        sns.heatmap(sim_raw, ax=ax, cmap='RdBu_r', center=0, xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Similarity / t'})
        ax.set_title(f"Raw Similarity Matrix (t={temperature})")
        ax.set_xlabel("Target (z_target)")
        ax.set_ylabel("Online (z_online)")
        
        ax2 = axes[1]
        sns.heatmap(sim_softmax, ax=ax2, cmap='viridis', xticklabels=labels, yticklabels=labels,
                   vmin=0, vmax=1, cbar_kws={'label': 'Softmax Probability'})
        ax2.set_title("Softmax Similarity (InfoNCE Attention)")
        
        ax3 = axes[2]
        diag = np.diag(sim_softmax)
        off_diag_mask = ~np.eye(B, dtype=bool)
        off_diag = sim_softmax[off_diag_mask]
        
        ax3.hist(diag, bins=min(20, B), alpha=0.7, label='Positive (diagonal)', 
                color='green', edgecolor='black', density=True)
        ax3.hist(off_diag, bins=30, alpha=0.7, label='Negative (off-diag)', 
                color='red', edgecolor='black', density=True)
        ax3.axvline(diag.mean(), color='green', linestyle='--', linewidth=2)
        ax3.axvline(off_diag.mean(), color='red', linestyle='--', linewidth=2)
        ax3.legend(fontsize=10)
        ax3.set_xlabel("Softmax Probability")
        ax3.set_ylabel("Density")
        ax3.set_title("Positive vs Negative Distribution")
        
        plt.tight_layout()
        
        accuracy = (np.argmax(sim_softmax, axis=1) == np.arange(B)).mean()
        
        result = {
            "contrastive/similarity_matrix": wandb.Image(fig),
            "contrastive/infonce_accuracy": float(accuracy),
            "contrastive/positive_mean": float(diag.mean()),
            "contrastive/negative_mean": float(off_diag.mean()),
            "contrastive/pos_neg_gap": float(diag.mean() - off_diag.mean()),
        }
        
        plt.close(fig)
        return result

    # =========================================================================
    # 5. Flow Matching Denoising Trajectory
    # =========================================================================
    
    def visualize_flow_matching(
        self,
        policy: torch.nn.Module,
        state: Tensor,
        goal: Tensor,
        ground_truth: Tensor,
        num_vis_steps: int = 10,
    ) -> Dict[str, any]:
        """Visualize the flow matching denoising trajectory."""
        import wandb
        plt = _get_plt()
        
        device = state.device
        action_horizon = ground_truth.shape[1]
        action_dim = ground_truth.shape[2]
        
        x = torch.randn(1, action_horizon, action_dim, device=device)
        trajectories = [x.clone().cpu()]
        times = torch.linspace(0, 1, num_vis_steps + 1, device=device)
        
        with torch.no_grad():
            for i in range(num_vis_steps):
                t = times[i]
                dt = times[i + 1] - times[i]
                t_batch = torch.full((1,), t.item(), device=device)
                v = policy(x, t_batch, state[:1], goal[:1])
                x = x + v * dt
                trajectories.append(x.clone().cpu())
        
        trajectories = [t[0].numpy() for t in trajectories]
        gt = ground_truth[0].cpu().numpy()
        
        num_dims = min(3, action_dim)
        fig, axes = plt.subplots(num_dims, 1, figsize=(14, 3 * num_dims))
        if num_dims == 1:
            axes = [axes]
        
        cmap = plt.cm.plasma
        
        for dim_idx in range(num_dims):
            ax = axes[dim_idx]
            for step, traj in enumerate(trajectories):
                alpha = 0.2 + 0.8 * (step / num_vis_steps)
                color = cmap(step / num_vis_steps)
                ax.plot(traj[:, dim_idx], color=color, alpha=alpha, linewidth=1)
            
            ax.plot(trajectories[-1][:, dim_idx], color=cmap(1.0), linewidth=2.5, label='Final Prediction')
            ax.plot(gt[:, dim_idx], 'k--', linewidth=2.5, label='Ground Truth')
            ax.set_ylabel(f"Action Dim {dim_idx}")
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[0].set_title("Flow Matching: Noise -> Action Trajectory", fontsize=12)
        axes[-1].set_xlabel("Timestep")
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes, label='Flow Time t', shrink=0.6)
        
        plt.tight_layout()
        
        final_pred = trajectories[-1]
        mse = np.mean((final_pred - gt) ** 2)
        
        result = {
            "flow_matching/denoising_trajectory": wandb.Image(fig),
            "flow_matching/final_mse": float(mse),
            "flow_matching/final_mae": float(np.mean(np.abs(final_pred - gt))),
        }
        
        plt.close(fig)
        return result

    # =========================================================================
    # 6. Language-State Clustering
    # =========================================================================
    
    def visualize_language_state_clustering(
        self,
        states: Tensor,
        language_embeddings: Tensor,
        task_labels: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """Visualize state space colored by language instruction similarity."""
        import wandb
        plt = _get_plt()
        
        try:
            from sklearn.manifold import TSNE
            from sklearn.cluster import KMeans
            from scipy.spatial.distance import pdist, squareform
        except ImportError:
            logger.warning("sklearn/scipy not available")
            return {}
        
        states_np = states.detach().cpu().numpy()
        lang_np = language_embeddings.detach().cpu().numpy()
        
        # Check for NaN/Inf in embeddings
        if np.any(~np.isfinite(states_np)) or np.any(~np.isfinite(lang_np)):
            logger.warning("NaN or Inf detected in embeddings, skipping clustering viz")
            return {}
        
        # Check for zero-norm vectors (would cause cosine distance issues)
        state_norms = np.linalg.norm(states_np, axis=1)
        lang_norms = np.linalg.norm(lang_np, axis=1)
        if np.any(state_norms < 1e-8) or np.any(lang_norms < 1e-8):
            logger.warning("Zero-norm vectors detected, skipping clustering viz")
            return {}
        
        n_samples = len(states_np)
        if n_samples < 10:
            return {}
        
        # Check for sufficient variance in language embeddings
        lang_std = np.std(lang_np)
        if lang_std < 1e-6:
            logger.warning(f"Language embeddings have near-zero variance ({lang_std:.2e}), skipping clustering viz")
            return {}
        
        perplexity = min(30, n_samples // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        states_2d = tsne.fit_transform(states_np)
        
        n_clusters = max(2, min(10, n_samples // 5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        lang_clusters = kmeans.fit_predict(lang_np)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        ax = axes[0]
        ax.scatter(states_2d[:, 0], states_2d[:, 1], c=lang_clusters, cmap='tab10', 
                  s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.set_title("State Space Colored by Language Cluster", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        
        ax2 = axes[1]
        if n_samples > 200:
            sample_idx = np.random.choice(n_samples, 200, replace=False)
            lang_sample, state_sample = lang_np[sample_idx], states_np[sample_idx]
        else:
            lang_sample, state_sample = lang_np, states_np
        
        # Use euclidean distance as fallback if cosine fails
        try:
            lang_dist = squareform(pdist(lang_sample, 'cosine'))
            state_dist = squareform(pdist(state_sample, 'cosine'))
        except Exception as e:
            logger.warning(f"Cosine distance failed: {e}, trying euclidean")
            lang_dist = squareform(pdist(lang_sample, 'euclidean'))
            state_dist = squareform(pdist(state_sample, 'euclidean'))
        
        # Check for NaN in distances
        if np.any(~np.isfinite(lang_dist)) or np.any(~np.isfinite(state_dist)):
            logger.warning("NaN in distance matrix, skipping correlation plot")
            plt.close(fig)
            return {}
        
        triu_idx = np.triu_indices(len(lang_dist), k=1)
        lang_flat, state_flat = lang_dist[triu_idx], state_dist[triu_idx]
        
        ax2.scatter(lang_flat, state_flat, alpha=0.2, s=10, c='blue')
        
        # Safe polyfit with error handling
        corr = 0.0
        try:
            z = np.polyfit(lang_flat, state_flat, 1)
            p = np.poly1d(z)
            x_line = np.linspace(lang_flat.min(), lang_flat.max(), 100)
            ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend')
        except Exception as e:
            logger.warning(f"Polyfit failed: {e}, skipping trend line")
        
        # Safe correlation computation
        try:
            corr = np.corrcoef(lang_flat, state_flat)[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
        except Exception:
            corr = 0.0
        
        ax2.set_xlabel("Language Distance (cosine)")
        ax2.set_ylabel("State Distance (cosine)")
        ax2.set_title(f"Language vs State Similarity\nCorrelation: {corr:.3f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        result = {
            "clustering/language_state_clustering": wandb.Image(fig),
            "clustering/lang_state_correlation": float(corr),
            "clustering/num_language_clusters": n_clusters,
        }
        
        plt.close(fig)
        return result

    # =========================================================================
    # 7. Goal Embedding Sphere Visualization
    # =========================================================================
    
    def visualize_goal_sphere(
        self,
        goal_embeddings: Tensor,
        task_labels: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """Visualize goal embeddings on the unit sphere."""
        import wandb
        plt = _get_plt()
        
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            return {}
        
        z_np = goal_embeddings.detach().cpu().numpy()
        n_samples = len(z_np)
        
        logger.info(f"[DEBUG] Goal sphere viz: n_samples={n_samples}, embedding_dim={z_np.shape[1]}")
        
        # Need at least 4 samples for meaningful visualization
        if n_samples < 4:
            logger.warning(f"Too few samples ({n_samples}) for goal sphere visualization")
            return {}
        
        # PCA n_components must be <= min(n_samples, n_features)
        n_pca_components = min(3, n_samples, z_np.shape[1])
        pca = PCA(n_components=n_pca_components)
        z_3d = pca.fit_transform(z_np)
        z_3d_norm = z_3d / (np.linalg.norm(z_3d, axis=1, keepdims=True) + 1e-8)
        
        fig = plt.figure(figsize=(16, 6))
        
        ax1 = fig.add_subplot(131, projection='3d')
        if task_labels is not None:
            unique_tasks = list(set(task_labels))
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_tasks)))
            color_map = {task: colors[i] for i, task in enumerate(unique_tasks)}
            for task in unique_tasks[:10]:
                mask = np.array([l == task for l in task_labels])
                ax1.scatter(z_3d_norm[mask, 0], z_3d_norm[mask, 1], z_3d_norm[mask, 2],
                           c=[color_map[task]], label=task[:15], s=30, alpha=0.7)
            if len(unique_tasks) <= 10:
                ax1.legend(loc='upper left', fontsize=7)
        else:
            ax1.scatter(z_3d_norm[:, 0], z_3d_norm[:, 1], z_3d_norm[:, 2], 
                       c=range(n_samples), cmap='viridis', s=30, alpha=0.7)
        
        u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
        ax1.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), 
                          color='gray', alpha=0.1, linewidth=0.5)
        ax1.set_title(f"Goal Embeddings on Unit Sphere\n(PCA: {pca.explained_variance_ratio_.sum():.1%} var)")
        
        ax2 = fig.add_subplot(132)
        cos_sim = np.clip(np.dot(z_np, z_np.T), -1, 1)
        angles = np.arccos(cos_sim) * 180 / np.pi
        angles_flat = angles[np.triu_indices(n_samples, k=1)]
        
        logger.info(f"[DEBUG] Angles histogram: angles_flat len={len(angles_flat)}, "
                   f"range=[{angles_flat.min():.2f}, {angles_flat.max():.2f}] deg, "
                   f"std={angles_flat.std():.4f}" if len(angles_flat) > 0 else "[DEBUG] No angles to plot")
        
        # Adaptive binning for angles histogram to handle small sample sizes
        if len(angles_flat) > 0:
            n_angle_bins = max(3, min(50, len(angles_flat) // 5))
            try:
                ax2.hist(angles_flat, bins=n_angle_bins, alpha=0.7, edgecolor='black', color='steelblue')
            except ValueError as e:
                logger.warning(f"[DEBUG] Angles histogram failed with {n_angle_bins} bins: {e}")
                try:
                    ax2.hist(angles_flat, bins='auto', alpha=0.7, edgecolor='black', color='steelblue')
                except ValueError as e2:
                    logger.warning(f"[DEBUG] Angles histogram 'auto' also failed: {e2}")
                    try:
                        ax2.hist(angles_flat, bins=3, alpha=0.7, edgecolor='black', color='steelblue')
                    except ValueError:
                        # Ultimate fallback: bar plot
                        ax2.bar(range(min(50, len(angles_flat))), sorted(angles_flat)[:50], 
                               alpha=0.7, color='steelblue', width=1.0)
            ax2.axvline(90, color='red', linestyle='--', linewidth=2, label='90deg (orthogonal)')
            ax2.axvline(angles_flat.mean(), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean: {angles_flat.mean():.1f}deg')
            ax2.legend(fontsize=9)
        ax2.set_xlabel("Angular Distance (degrees)")
        ax2.set_title("Distribution of Goal Embedding Angles")
        
        ax3 = fig.add_subplot(133)
        norms = np.linalg.norm(z_np, axis=1)
        logger.info(f"[DEBUG] Goal sphere: n_samples={n_samples}, norms range=[{norms.min():.4f}, {norms.max():.4f}], "
                   f"norms std={norms.std():.6f}")
        
        # Robust histogram binning with multiple fallback strategies
        histogram_success = False
        try:
            n_unique = len(np.unique(np.round(norms, decimals=4)))
            # More conservative binning: ensure we have enough data per bin
            n_bins = max(3, min(20, n_unique // 3, n_samples // 5))
            logger.info(f"[DEBUG] Trying histogram with n_bins={n_bins}, n_unique={n_unique}")
            ax3.hist(norms, bins=n_bins, alpha=0.7, edgecolor='black', color='coral')
            histogram_success = True
        except ValueError as e:
            logger.warning(f"[DEBUG] Histogram failed with n_bins={n_bins}: {e}")
            try:
                ax3.hist(norms, bins='auto', alpha=0.7, edgecolor='black', color='coral')
                histogram_success = True
            except ValueError as e2:
                logger.warning(f"[DEBUG] Histogram 'auto' also failed: {e2}")
                try:
                    # Last resort: use a fixed small number of bins
                    ax3.hist(norms, bins=3, alpha=0.7, edgecolor='black', color='coral')
                    histogram_success = True
                except ValueError as e3:
                    logger.warning(f"[DEBUG] Even 3 bins failed: {e3}. Using bar plot instead.")
                    # Ultimate fallback: just plot sorted values
                    ax3.bar(range(len(norms)), sorted(norms), alpha=0.7, color='coral', width=1.0)
                    histogram_success = True
        
        ax3.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Unit Norm')
        ax3.axvline(norms.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {norms.mean():.3f}')
        ax3.legend(fontsize=9)
        ax3.set_xlabel("Embedding Norm" if histogram_success else "Sample Index (sorted)")
        ax3.set_title("Goal Embedding Norm Distribution")
        
        plt.tight_layout()
        
        result = {
            "goal_sphere/visualization": wandb.Image(fig),
            "goal_sphere/mean_angular_distance": float(angles_flat.mean()),
            "goal_sphere/std_angular_distance": float(angles_flat.std()),
            "goal_sphere/mean_norm": float(norms.mean()),
            "goal_sphere/std_norm": float(norms.std()),
            "goal_sphere/collapse_risk": float(1.0 - min(angles_flat.std() / 30, 1.0)),
        }
        
        plt.close(fig)
        return result

    # =========================================================================
    # Combined Logging Helper
    # =========================================================================
    
    def log_all_visualizations(
        self,
        epoch: int,
        step: int,
        z_online: Optional[Tensor] = None,
        z_target: Optional[Tensor] = None,
        language_embeddings: Optional[Tensor] = None,
        goal_embeddings: Optional[Tensor] = None,
        states: Optional[Tensor] = None,
        forward_net: Optional[torch.nn.Module] = None,
        current_state: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        goal: Optional[Tensor] = None,
        policy: Optional[torch.nn.Module] = None,
        ground_truth_actions: Optional[Tensor] = None,
        attention_weights: Optional[Tensor] = None,
        rgb: Optional[Tensor] = None,
        task_labels: Optional[List[str]] = None,
        temperature: float = 0.1,
    ) -> Dict[str, any]:
        """Log all available visualizations to wandb."""
        import wandb
        
        if not self.should_log(epoch):
            return {}
        
        all_metrics = {}
        
        if language_embeddings is not None and goal_embeddings is not None:
            try:
                metrics = self.visualize_language_goal_alignment(language_embeddings, goal_embeddings, task_labels)
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(f"Failed language-goal viz: {e}")
        
        if attention_weights is not None and rgb is not None:
            try:
                metrics = self.visualize_cross_attention(rgb, attention_weights)
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(f"Failed cross-attention viz: {e}")
        
        if forward_net is not None and current_state is not None and actions is not None and goal is not None:
            try:
                metrics = self.visualize_forward_rollout(forward_net, current_state, actions, goal)
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(f"Failed forward rollout viz: {e}")
        
        if z_online is not None and z_target is not None:
            try:
                metrics = self.visualize_contrastive_matrix(z_online, z_target, task_labels, temperature)
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(f"Failed contrastive viz: {e}")
        
        if policy is not None and states is not None and goal is not None and ground_truth_actions is not None:
            try:
                metrics = self.visualize_flow_matching(policy, states, goal, ground_truth_actions)
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(f"Failed flow matching viz: {e}")
        
        if states is not None and language_embeddings is not None:
            try:
                metrics = self.visualize_language_state_clustering(states, language_embeddings, task_labels)
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(f"Failed clustering viz: {e}")
        
        if goal_embeddings is not None:
            try:
                metrics = self.visualize_goal_sphere(goal_embeddings, task_labels)
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(f"Failed goal sphere viz: {e}")
        
        if all_metrics:
            wandb.log(all_metrics, step=step)
        
        return all_metrics
