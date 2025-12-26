"""
Forward-Backward (FB) World Model for Geo-Flow VLA.

Implements unsupervised representation learning for manipulation:
- Backward Network (B): Maps states to goal embeddings on unit sphere
- Forward Network (F): Predicts future state features given current state, action, and goal

The FB objective learns the "physics of manipulation" without reward labels.

Architecture:
    Backward: s_t → z ∈ S^{d-1} (unit sphere)
    Forward: (s_t, a_t, z) → predicted s_{t+k}

References:
    - Touati & Ollivier, "Learning One Representation to Optimize All Rewards" (2021)
    - Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL" (2022)
"""

from typing import Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock(nn.Module):
    """
    Residual MLP block for stable gradient flow.
    
    Architecture: LayerNorm → Linear → GELU → Linear → Add
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        hidden_dim = int(dim * hidden_mult)
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class BackwardNetwork(nn.Module):
    """
    Backward Network (B): Maps future states to goal embeddings.
    
    Output lives on the unit sphere S^{d-1} via L2 normalization.
    This prevents representation collapse and enables contrastive learning.
    
    Architecture:
        s_{t+k} → MLP → L2 Normalize → z ∈ R^d, ||z|| = 1
    """

    def __init__(
        self,
        state_dim: int = 512,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            state_dim: Input state dimension (from dual encoder)
            latent_dim: Output embedding dimension (on unit sphere)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Build MLP
        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.encoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, state: Tensor, normalize: bool = True) -> Tensor:
        """
        Map state to goal embedding on unit sphere.
        
        Args:
            state: State embedding (B, state_dim)
            normalize: Whether to L2-normalize output
            
        Returns:
            Goal embedding z (B, latent_dim), ||z|| = 1 if normalized
        """
        z = self.encoder(state)
        
        if normalize:
            z = F.normalize(z, dim=-1, eps=1e-8)
        
        return z


class ForwardNetwork(nn.Module):
    """
    Forward Network (F): Predicts future state features.
    
    Given current state s_t, action sequence a_t, and goal embedding z,
    predicts the state features at time t+k.
    
    Architecture:
        [s_t, flatten(a_t), z] → MLP with Residual Blocks → predicted s_{t+k}
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 7,
        action_horizon: int = 16,
        latent_dim: int = 256,
        hidden_dim: int = 1024,
        num_residual_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            state_dim: State embedding dimension
            action_dim: Single action dimension
            action_horizon: Number of action steps
            latent_dim: Goal embedding dimension
            hidden_dim: Hidden dimension for residual blocks
            num_residual_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim
        
        # Separate encoders for each input modality
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * action_horizon, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )
        
        self.goal_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )
        
        # Fusion layer
        concat_dim = (hidden_dim // 4) * 3
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Residual trunk
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) 
              for _ in range(num_residual_blocks)]
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state: Tensor,
        actions: Tensor,
        goal: Tensor,
    ) -> Tensor:
        """
        Predict future state features.
        
        Args:
            state: Current state embedding (B, state_dim)
            actions: Action sequence (B, action_horizon, action_dim)
            goal: Goal embedding from Backward network (B, latent_dim)
            
        Returns:
            Predicted future state features (B, state_dim)
        """
        B = state.shape[0]
        
        # Encode each modality
        state_enc = self.state_encoder(state)  # (B, hidden//4)
        
        actions_flat = actions.reshape(B, -1)  # (B, action_horizon * action_dim)
        action_enc = self.action_encoder(actions_flat)  # (B, hidden//4)
        
        goal_enc = self.goal_encoder(goal)  # (B, hidden//4)
        
        # Concatenate and fuse
        concat = torch.cat([state_enc, action_enc, goal_enc], dim=-1)
        fused = self.fusion(concat)  # (B, hidden)
        
        # Residual processing
        features = self.residual_blocks(fused)  # (B, hidden)
        
        # Project to state dimension
        predicted_state = self.output_proj(features)  # (B, state_dim)
        
        return predicted_state


class FBWorldModel(nn.Module):
    """
    Complete Forward-Backward World Model.
    
    Combines BackwardNetwork and ForwardNetwork with:
    - EMA target network for stable training
    - Contrastive loss for backward network
    - TD-style prediction loss for forward network
    
    Training Objective:
        L_FB = L_forward + β * L_backward
        
        L_forward = ||F(s_t, a_t, B_target(s_{t+k})) - sg(s_{t+k})||²
        L_backward = InfoNCE contrastive loss on z embeddings
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 7,
        action_horizon: int = 16,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        forward_hidden_dim: int = 1024,
        num_residual_blocks: int = 2,
        ema_tau: float = 0.005,
        temperature: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            state_dim: State embedding dimension
            action_dim: Single action dimension
            action_horizon: Number of action steps
            latent_dim: Goal embedding dimension
            hidden_dim: Hidden dim for backward network
            forward_hidden_dim: Hidden dim for forward network
            num_residual_blocks: Residual blocks in forward network
            ema_tau: EMA update rate for target network
            temperature: Temperature for contrastive loss
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.ema_tau = ema_tau
        self.temperature = temperature
        
        # Backward network (online)
        self.backward_net = BackwardNetwork(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # Backward network (target - EMA updated)
        self.backward_net_target = copy.deepcopy(self.backward_net)
        for param in self.backward_net_target.parameters():
            param.requires_grad = False
        
        # Forward network
        self.forward_net = ForwardNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            latent_dim=latent_dim,
            hidden_dim=forward_hidden_dim,
            num_residual_blocks=num_residual_blocks,
            dropout=dropout,
        )

    @torch.no_grad()
    def update_target_network(self) -> None:
        """Update target network with exponential moving average."""
        for param, target_param in zip(
            self.backward_net.parameters(),
            self.backward_net_target.parameters(),
        ):
            target_param.data.mul_(1 - self.ema_tau)
            target_param.data.add_(self.ema_tau * param.data)

    def encode_goal(self, future_state: Tensor, use_target: bool = False) -> Tensor:
        """
        Encode future state to goal embedding.
        
        Args:
            future_state: State at time t+k (B, state_dim)
            use_target: Use target network (for training stability)
            
        Returns:
            Goal embedding z (B, latent_dim)
        """
        if use_target:
            with torch.no_grad():
                return self.backward_net_target(future_state)
        return self.backward_net(future_state)

    def predict_future(
        self,
        state: Tensor,
        actions: Tensor,
        goal: Tensor,
    ) -> Tensor:
        """
        Predict future state features.
        
        Args:
            state: Current state (B, state_dim)
            actions: Action sequence (B, T, action_dim)
            goal: Goal embedding (B, latent_dim)
            
        Returns:
            Predicted future state (B, state_dim)
        """
        return self.forward_net(state, actions, goal)

    def forward(
        self,
        current_state: Tensor,
        future_state: Tensor,
        actions: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Full forward pass for training.
        
        Args:
            current_state: State at time t (B, state_dim)
            future_state: State at time t+k (B, state_dim)
            actions: Action sequence from t to t+k (B, T, action_dim)
            
        Returns:
            Dictionary with predicted state and goal embeddings
        """
        # Get goal embedding from future state (using target network)
        z_target = self.encode_goal(future_state, use_target=True)
        
        # Get goal embedding from online network (for contrastive loss)
        z_online = self.encode_goal(future_state, use_target=False)
        
        # Predict future state
        predicted_state = self.predict_future(current_state, actions, z_target)
        
        return {
            "predicted_state": predicted_state,
            "z_target": z_target,
            "z_online": z_online,
            "target_state": future_state,
        }

    def compute_forward_loss(
        self,
        predicted_state: Tensor,
        target_state: Tensor,
    ) -> Tensor:
        """
        Compute forward prediction loss.
        
        Uses MSE between predicted and actual future state features.
        Stop gradient on target to learn dynamics, not reconstruction.
        
        Args:
            predicted_state: Model prediction (B, state_dim)
            target_state: Actual future state (B, state_dim)
            
        Returns:
            Forward loss scalar
        """
        # Stop gradient on target
        target = target_state.detach()
        
        loss = F.mse_loss(predicted_state, target)
        
        return loss

    def compute_backward_loss(
        self,
        z_online: Tensor,
        z_target: Tensor,
    ) -> Tensor:
        """
        Compute contrastive loss for backward network.
        
        Uses InfoNCE loss to ensure embeddings capture reachability.
        Positive pairs: (z_online[i], z_target[i])
        Negative pairs: (z_online[i], z_target[j]) for i != j
        
        Args:
            z_online: Online network embeddings (B, latent_dim)
            z_target: Target network embeddings (B, latent_dim)
            
        Returns:
            Backward (contrastive) loss scalar
        """
        B = z_online.shape[0]
        
        # Compute similarity matrix
        # z_online: (B, d), z_target: (B, d)
        sim_matrix = torch.mm(z_online, z_target.t()) / self.temperature  # (B, B)
        
        # Labels: diagonal is positive
        labels = torch.arange(B, device=z_online.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def compute_loss(
        self,
        current_state: Tensor,
        future_state: Tensor,
        actions: Tensor,
        forward_weight: float = 1.0,
        backward_weight: float = 0.5,
    ) -> Dict[str, Tensor]:
        """
        Compute total FB loss.
        
        Args:
            current_state: State at time t
            future_state: State at time t+k
            actions: Action sequence
            forward_weight: Weight for forward loss
            backward_weight: Weight for backward (contrastive) loss
            
        Returns:
            Dictionary with individual and total losses
        """
        # Forward pass
        outputs = self.forward(current_state, future_state, actions)
        
        # Compute losses
        forward_loss = self.compute_forward_loss(
            outputs["predicted_state"],
            outputs["target_state"],
        )
        
        backward_loss = self.compute_backward_loss(
            outputs["z_online"],
            outputs["z_target"],
        )
        
        total_loss = forward_weight * forward_loss + backward_weight * backward_loss
        
        return {
            "loss": total_loss,
            "forward_loss": forward_loss,
            "backward_loss": backward_loss,
            "z_norm": outputs["z_online"].norm(dim=-1).mean(),
        }

    def get_goal_embedding(self, state: Tensor) -> Tensor:
        """
        Get goal embedding for policy conditioning.
        
        Used during policy training to condition DiT on learned goals.
        
        Args:
            state: State embedding (B, state_dim)
            
        Returns:
            Goal embedding (B, latent_dim)
        """
        with torch.no_grad():
            return self.backward_net(state)

