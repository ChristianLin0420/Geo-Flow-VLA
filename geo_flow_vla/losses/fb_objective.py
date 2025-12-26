"""
Forward-Backward (FB) Objective for Geo-Flow VLA.

Implements the unsupervised representation learning objective for the world model.
Combines forward prediction loss with contrastive backward loss.

Losses:
    L_forward = ||F(s_t, a_t, B_target(s_{t+k})) - sg(s_{t+k})||²
    L_backward = InfoNCE(B(s), B_target(s))
    L_FB = L_forward + β * L_backward

References:
    - Touati & Ollivier, "Learning One Representation to Optimize All Rewards" (2021)
    - Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL" (2022)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_fb_loss(
    world_model: nn.Module,
    current_state: Tensor,
    future_state: Tensor,
    actions: Tensor,
    forward_weight: float = 1.0,
    backward_weight: float = 0.5,
) -> Dict[str, Tensor]:
    """
    Compute Forward-Backward loss.
    
    Args:
        world_model: FBWorldModel instance
        current_state: State at time t (B, state_dim)
        future_state: State at time t+k (B, state_dim)
        actions: Action sequence from t to t+k (B, T, action_dim)
        forward_weight: Weight for forward prediction loss
        backward_weight: Weight for backward contrastive loss
        
    Returns:
        Dictionary with losses
    """
    return world_model.compute_loss(
        current_state,
        future_state,
        actions,
        forward_weight=forward_weight,
        backward_weight=backward_weight,
    )


class FBObjective(nn.Module):
    """
    Forward-Backward objective for unsupervised world model training.
    
    Provides:
    - Forward prediction loss (MSE)
    - Backward contrastive loss (InfoNCE)
    - Optional auxiliary losses (diversity, smoothness)
    """

    def __init__(
        self,
        forward_weight: float = 1.0,
        backward_weight: float = 0.5,
        temperature: float = 0.1,
        diversity_weight: float = 0.0,
        smoothness_weight: float = 0.0,
    ) -> None:
        """
        Args:
            forward_weight: Weight for forward prediction loss
            backward_weight: Weight for backward contrastive loss
            temperature: Temperature for InfoNCE
            diversity_weight: Weight for embedding diversity loss
            smoothness_weight: Weight for temporal smoothness loss
        """
        super().__init__()
        
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        self.smoothness_weight = smoothness_weight

    def forward_loss(
        self,
        predicted_state: Tensor,
        target_state: Tensor,
    ) -> Tensor:
        """
        Compute forward prediction loss.
        
        Uses MSE with stop-gradient on target.
        
        Args:
            predicted_state: Model prediction (B, state_dim)
            target_state: Actual future state (B, state_dim)
            
        Returns:
            Forward loss scalar
        """
        target = target_state.detach()
        return F.mse_loss(predicted_state, target)

    def backward_loss(
        self,
        z_online: Tensor,
        z_target: Tensor,
    ) -> Tensor:
        """
        Compute backward contrastive loss (InfoNCE).
        
        Positive pairs: (z_online[i], z_target[i])
        Negative pairs: (z_online[i], z_target[j]) for i != j
        
        Args:
            z_online: Online network embeddings (B, latent_dim)
            z_target: Target network embeddings (B, latent_dim)
            
        Returns:
            Contrastive loss scalar
        """
        B = z_online.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z_online, z_target.t()) / self.temperature
        
        # Labels: diagonal is positive
        labels = torch.arange(B, device=z_online.device)
        
        # InfoNCE loss (cross-entropy with positives on diagonal)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def diversity_loss(self, z: Tensor) -> Tensor:
        """
        Encourage diverse embeddings (prevent collapse).
        
        Maximizes variance of embeddings across batch.
        
        Args:
            z: Embeddings (B, latent_dim)
            
        Returns:
            Diversity loss (negative, to be minimized)
        """
        # Compute covariance
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = torch.mm(z_centered.t(), z_centered) / (z.shape[0] - 1)
        
        # Off-diagonal elements should be small
        off_diag = cov - torch.diag(torch.diag(cov))
        loss = off_diag.pow(2).sum() / (z.shape[1] ** 2)
        
        # Diagonal elements should be large (high variance)
        var_loss = -torch.diag(cov).mean()
        
        return loss + 0.1 * var_loss

    def smoothness_loss(
        self,
        z_t: Tensor,
        z_t_plus_1: Tensor,
    ) -> Tensor:
        """
        Encourage temporally smooth embeddings.
        
        Adjacent timesteps should have similar embeddings.
        
        Args:
            z_t: Embeddings at time t (B, latent_dim)
            z_t_plus_1: Embeddings at time t+1 (B, latent_dim)
            
        Returns:
            Smoothness loss
        """
        return F.mse_loss(z_t, z_t_plus_1)

    def forward(
        self,
        world_model: nn.Module,
        current_state: Tensor,
        future_state: Tensor,
        actions: Tensor,
        adjacent_state: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute complete FB objective.
        
        Args:
            world_model: FBWorldModel instance
            current_state: State at time t (B, state_dim)
            future_state: State at time t+k (B, state_dim)
            actions: Actions from t to t+k (B, T, action_dim)
            adjacent_state: Optional state at t+1 for smoothness loss
            
        Returns:
            Dictionary with all losses
        """
        # Forward pass through world model
        outputs = world_model(current_state, future_state, actions)
        
        # Core losses
        l_forward = self.forward_loss(
            outputs["predicted_state"],
            outputs["target_state"],
        )
        
        l_backward = self.backward_loss(
            outputs["z_online"],
            outputs["z_target"],
        )
        
        # Total loss
        total_loss = (
            self.forward_weight * l_forward +
            self.backward_weight * l_backward
        )
        
        result = {
            "loss": total_loss,
            "forward_loss": l_forward,
            "backward_loss": l_backward,
            "z_norm": outputs["z_online"].norm(dim=-1).mean(),
        }
        
        # Optional diversity loss
        if self.diversity_weight > 0:
            l_diversity = self.diversity_loss(outputs["z_online"])
            total_loss = total_loss + self.diversity_weight * l_diversity
            result["diversity_loss"] = l_diversity
            result["loss"] = total_loss
        
        # Optional smoothness loss
        if self.smoothness_weight > 0 and adjacent_state is not None:
            z_t = world_model.encode_goal(current_state)
            z_t1 = world_model.encode_goal(adjacent_state)
            l_smooth = self.smoothness_loss(z_t, z_t1)
            total_loss = total_loss + self.smoothness_weight * l_smooth
            result["smoothness_loss"] = l_smooth
            result["loss"] = total_loss
        
        return result

    def compute_metrics(
        self,
        z_online: Tensor,
        z_target: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute additional metrics for logging.
        
        Args:
            z_online: Online embeddings
            z_target: Target embeddings
            
        Returns:
            Dictionary with metrics
        """
        with torch.no_grad():
            # Cosine similarity between online and target
            cos_sim = F.cosine_similarity(z_online, z_target, dim=-1).mean()
            
            # Embedding statistics
            z_mean = z_online.mean()
            z_std = z_online.std()
            
            # Uniformity (how spread out embeddings are)
            pairwise_dist = torch.cdist(z_online, z_online)
            uniformity = pairwise_dist.mean()
            
            return {
                "cosine_similarity": cos_sim,
                "z_mean": z_mean,
                "z_std": z_std,
                "uniformity": uniformity,
            }

