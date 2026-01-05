"""
Language-Conditioned Forward-Backward (FB) World Model for Geo-Flow VLA.

Key improvement over original:
- BackwardNetwork takes (state, instruction) -> z
- Goal embeddings are conditioned on language instructions
- Proper contrastive learning across trajectories

Architecture:
    Backward: B(state, instruction) → z ∈ S^{d-1} (unit sphere)
    Forward: F(state, actions, z) → predicted s_{t+k}

At inference:
    z = B(current_state, instruction) gives "where should I go given my task"
"""

from typing import Dict, List, Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .language_encoder import CLIPLanguageEncoder, create_language_encoder


class ResidualBlock(nn.Module):
    """Residual MLP block for stable gradient flow."""

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


class LanguageConditionedBackwardNetwork(nn.Module):
    """
    Backward Network: B(state, instruction) -> z
    
    Maps state + language instruction to goal embedding on unit sphere.
    This answers: "Given where I am and what I'm told to do, what goal should I aim for?"
    
    Architecture:
        state → state_proj → h_s
        instruction → lang_proj → h_l
        CrossAttention(h_s queries h_l) → fused
        MLP(fused) → z ∈ S^{d-1}
    """

    def __init__(
        self,
        state_dim: int = 512,
        language_dim: int = 768,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            state_dim: Input state dimension (from dual encoder)
            language_dim: Language embedding dimension (from CLIP)
            latent_dim: Output goal embedding dimension (on unit sphere)
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers after attention
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Project state and language to same dimension
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.lang_proj = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Cross-attention: state attends to language
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # MLP layers after attention
        mlp_layers = []
        for _ in range(num_layers):
            mlp_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        
        # Final projection to latent space
        self.goal_head = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize final layer with small weights for stable training
        nn.init.xavier_uniform_(self.goal_head.weight, gain=0.01)
        nn.init.zeros_(self.goal_head.bias)

    def forward(
        self,
        state: Tensor,
        lang_embedding: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        """
        Map state + instruction to goal embedding.
        
        Args:
            state: State embedding (B, state_dim)
            lang_embedding: Language embedding from CLIP (B, language_dim)
            normalize: L2 normalize output to unit sphere
            
        Returns:
            Goal embedding z (B, latent_dim), ||z|| = 1 if normalized
        """
        # Project to hidden dimension
        state_h = self.state_proj(state)  # (B, hidden_dim)
        lang_h = self.lang_proj(lang_embedding)  # (B, hidden_dim)
        
        # Reshape for attention: (B, 1, hidden_dim)
        state_tokens = state_h.unsqueeze(1)
        lang_tokens = lang_h.unsqueeze(1)
        
        # Cross-attention: state queries, language is key/value
        attended, _ = self.cross_attention(
            state_tokens, lang_tokens, lang_tokens
        )
        
        # Residual connection + layer norm
        fused = self.cross_norm(state_tokens + attended).squeeze(1)  # (B, hidden_dim)
        
        # MLP processing
        features = self.mlp(fused)
        
        # Project to goal latent
        z = self.goal_head(features)
        
        if normalize:
            z = F.normalize(z, dim=-1, eps=1e-8)
        
        return z


class ForwardNetwork(nn.Module):
    """
    Forward Network: F(state, actions, goal) -> predicted_future_state
    
    Predicts future state given current state, action sequence, and goal.
    This learns the dynamics: "If I'm here, do these actions aiming for goal z,
    what state will I reach?"
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 8,
        action_horizon: int = 16,
        latent_dim: int = 256,
        hidden_dim: int = 1024,
        num_residual_blocks: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            state_dim: State embedding dimension
            action_dim: Single action dimension
            action_horizon: Number of action steps
            latent_dim: Goal embedding dimension
            hidden_dim: Hidden dimension for processing
            num_residual_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim
        
        # Separate encoders for each input modality
        quarter_dim = hidden_dim // 4
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, quarter_dim),
            nn.LayerNorm(quarter_dim),
            nn.GELU(),
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * action_horizon, quarter_dim),
            nn.LayerNorm(quarter_dim),
            nn.GELU(),
        )
        
        self.goal_encoder = nn.Sequential(
            nn.Linear(latent_dim, quarter_dim),
            nn.LayerNorm(quarter_dim),
            nn.GELU(),
        )
        
        # Fusion layer
        concat_dim = quarter_dim * 3
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


class LanguageConditionedFBWorldModel(nn.Module):
    """
    Complete Forward-Backward World Model with Language Conditioning.
    
    Key components:
    - Language encoder (CLIP, frozen)
    - Backward network: B(state, instruction) -> z (on unit sphere)
    - Forward network: F(state, actions, z) -> predicted_state
    - EMA target network for stable contrastive training
    
    Training Objective:
        L_FB = L_forward + β * L_backward
        
        L_forward = ||F(s_t, a_t, z) - sg(s_{t+k})||²
        L_backward = InfoNCE contrastive loss on z embeddings
        
    Where z = B(s_{t+k}, instruction) during training.
    At inference: z = B(s_current, instruction) - "where should I go for this task"
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 8,
        action_horizon: int = 16,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        forward_hidden_dim: int = 1024,
        num_residual_blocks: int = 2,
        ema_tau: float = 0.005,
        temperature: float = 0.1,
        dropout: float = 0.1,
        language_model: str = "openai/clip-vit-large-patch14",
        use_mock_language: bool = False,
    ):
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
            language_model: CLIP model name
            use_mock_language: Use mock language encoder (for testing)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim
        self.ema_tau = ema_tau
        self.temperature = temperature
        
        # Language encoder (frozen)
        self.language_encoder = create_language_encoder(
            model_name=language_model,
            use_mock=use_mock_language,
            freeze=True,
        )
        language_dim = self.language_encoder.output_dim
        self.language_dim = language_dim
        
        # Backward network (online) - language conditioned
        self.backward_net = LanguageConditionedBackwardNetwork(
            state_dim=state_dim,
            language_dim=language_dim,
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

    def to(self, device: torch.device) -> "LanguageConditionedFBWorldModel":
        """
        Ensure all components are moved to device properly.
        
        Override to handle deepcopy'd target network and lazy-loaded language encoder.
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        # Move language encoder (updates its internal device tracking)
        self.language_encoder = self.language_encoder.to(device)
        
        # Explicitly move target network (deepcopy can cause device issues)
        self.backward_net_target = self.backward_net_target.to(device)
        
        return super().to(device)

    def encode_instruction(
        self,
        instruction: Union[str, List[str]],
    ) -> Tensor:
        """
        Encode text instruction to embedding.
        
        Args:
            instruction: Single string or list of strings
            
        Returns:
            Language embedding (B, language_dim)
        """
        return self.language_encoder(instruction)

    def encode_goal(
        self,
        state: Tensor,
        lang_embedding: Tensor,
        use_target: bool = False,
    ) -> Tensor:
        """
        Encode state + instruction to goal embedding.
        
        Args:
            state: State embedding (B, state_dim)
            lang_embedding: Language embedding (B, language_dim)
            use_target: Use EMA target network (for training stability)
            
        Returns:
            Goal embedding z (B, latent_dim) on unit sphere
        """
        if use_target:
            with torch.no_grad():
                return self.backward_net_target(state, lang_embedding)
        return self.backward_net(state, lang_embedding)

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
        lang_embedding: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Full forward pass for training.
        
        Args:
            current_state: State at time t (B, state_dim)
            future_state: State at time t+k (B, state_dim)
            actions: Action sequence from t to t+k (B, T, action_dim)
            lang_embedding: Pre-computed language embedding (B, language_dim)
            
        Returns:
            Dictionary with:
                - predicted_state: Forward network output
                - z_target: Goal from target network (for forward conditioning)
                - z_online: Goal from online network (for contrastive loss)
                - target_state: Ground truth future state
        """
        # Get goal embedding from FUTURE state + instruction
        # Target network for forward conditioning
        z_target = self.encode_goal(future_state, lang_embedding, use_target=True)
        
        # Online network for contrastive learning
        z_online = self.encode_goal(future_state, lang_embedding, use_target=False)
        
        # Predict future state using target goal (no gradient through z_target)
        predicted_state = self.predict_future(current_state, actions, z_target.detach())
        
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
        
        Uses MSE between predicted and actual future state.
        Stop gradient on target to learn dynamics, not reconstruction.
        """
        target = target_state.detach()
        return F.mse_loss(predicted_state, target)

    def compute_backward_loss(
        self,
        z_online: Tensor,
        z_target: Tensor,
    ) -> Tensor:
        """
        Compute contrastive loss for backward network (InfoNCE).
        
        Positive pairs: (z_online[i], z_target[i]) from same trajectory
        Negative pairs: (z_online[i], z_target[j]) for i != j (cross-batch)
        
        This ensures embeddings from same trajectory are close,
        embeddings from different trajectories are far.
        """
        B = z_online.shape[0]
        
        # Compute similarity matrix
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
        lang_embedding: Tensor,
        forward_weight: float = 1.0,
        backward_weight: float = 0.5,
    ) -> Dict[str, Tensor]:
        """
        Compute total FB loss.
        
        Args:
            current_state: State at time t
            future_state: State at time t+k
            actions: Action sequence
            lang_embedding: Language embedding
            forward_weight: Weight for forward loss
            backward_weight: Weight for backward (contrastive) loss
            
        Returns:
            Dictionary with individual and total losses
        """
        # Forward pass
        outputs = self.forward(current_state, future_state, actions, lang_embedding)
        
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

    def get_goal_embedding(
        self,
        state: Tensor,
        instruction: Union[str, List[str]],
    ) -> Tensor:
        """
        Get goal embedding for inference.
        
        At inference time: B(current_state, instruction) -> z
        This gives "where should I go given where I am and what I need to do"
        
        Args:
            state: Current state embedding (B, state_dim)
            instruction: Task instruction string or list
            
        Returns:
            Goal embedding (B, latent_dim)
        """
        lang_embedding = self.encode_instruction(instruction)
        # Use online network at inference
        return self.encode_goal(state, lang_embedding, use_target=False)

