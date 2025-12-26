"""
Diffusion Transformer (DiT) Policy with Conditional Flow Matching.

Implements a DiT backbone for generating continuous robot action trajectories.
Uses Adaptive Layer Normalization (AdaLN-Zero) for conditioning on:
- Timestep t (flow matching noise level)
- State embedding s_t (from dual encoder)
- Goal embedding z (from backward network)

Architecture:
    Input: Noisy action x_t (B, T, action_dim), timestep t, condition C
    Output: Velocity field v_t (B, T, action_dim)

References:
    - Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, 2023)
    - Chi et al., "Diffusion Policy" (2023)
    - Lipman et al., "Flow Matching for Generative Modeling" (2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaptive modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Embed timesteps.
        
        Args:
            t: Timesteps (B,) in [0, 1]
            
        Returns:
            Embeddings (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        
        return embedding


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN-Zero conditioning.
    
    Architecture:
        1. AdaLN-modulated self-attention
        2. AdaLN-modulated MLP
        
    Conditioning is injected via adaptive layer norm parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            hidden_dim: Transformer hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Layer norms (will be modulated)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Self-attention
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        
        # MLP
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # AdaLN modulation: 6 parameters (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        
        # Zero-initialize the modulation output
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Forward pass with AdaLN conditioning.
        
        Args:
            x: Input tokens (B, T, hidden_dim)
            c: Conditioning embedding (B, hidden_dim)
            
        Returns:
            Output tokens (B, T, hidden_dim)
        """
        # Get modulation parameters from conditioning
        mod = self.adaLN_modulation(c)  # (B, 6 * hidden_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        h = self.norm1(x)
        h = modulate(h, shift1, scale1)
        h = self._attention(h)
        x = x + gate1.unsqueeze(1) * h
        
        # MLP with AdaLN
        h = self.norm2(x)
        h = modulate(h, shift2, scale2)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h
        
        return x

    def _attention(self, x: Tensor) -> Tensor:
        """Multi-head self-attention."""
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.attn_proj(out)
        
        return out


class FinalLayer(nn.Module):
    """Final layer with AdaLN-Zero for output projection."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(hidden_dim, output_dim)
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        
        # Zero-initialize
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Apply final layer with conditioning."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.proj(x)
        return x


class DiffusionPolicy(nn.Module):
    """
    Diffusion Transformer Policy with Conditional Flow Matching.
    
    Predicts velocity field v_t for transforming noise into action trajectories.
    
    Architecture:
        1. Action embedding: noisy_action → tokens
        2. Timestep embedding: t → timestep_embed
        3. Condition embedding: concat(state, goal) → condition
        4. DiT blocks with AdaLN conditioning
        5. Final layer → velocity field
    
    Training:
        Loss = ||v_θ(x_t, t, c) - (x_1 - x_0)||²
        where x_t = (1-t)x_0 + t*x_1, x_0 ~ N(0,I), x_1 = ground_truth
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        state_dim: int = 512,
        latent_dim: int = 256,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            action_dim: Dimension of single action
            action_horizon: Number of action steps to predict
            state_dim: State embedding dimension (from dual encoder)
            latent_dim: Goal embedding dimension (from backward network)
            hidden_dim: Transformer hidden dimension
            num_layers: Number of DiT blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Action embedding: per-step actions to tokens
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        
        # Positional embedding for action sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Timestep embedding
        self.timestep_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition embedding: state + goal → hidden_dim
        self.condition_embed = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize transformer weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)

    def forward(
        self,
        noisy_action: Tensor,
        timestep: Tensor,
        state: Tensor,
        goal: Tensor,
    ) -> Tensor:
        """
        Predict velocity field for flow matching.
        
        Args:
            noisy_action: Noisy action trajectory (B, T, action_dim)
            timestep: Diffusion timestep in [0, 1] (B,)
            state: State embedding from dual encoder (B, state_dim)
            goal: Goal embedding from backward network (B, latent_dim)
            
        Returns:
            Predicted velocity field (B, T, action_dim)
        """
        B, T, D = noisy_action.shape
        
        assert T == self.action_horizon, \
            f"Expected action_horizon {self.action_horizon}, got {T}"
        assert D == self.action_dim, \
            f"Expected action_dim {self.action_dim}, got {D}"
        
        # Embed actions
        x = self.action_embed(noisy_action)  # (B, T, hidden_dim)
        x = x + self.pos_embed
        
        # Embed timestep
        t_emb = self.timestep_embed(timestep)  # (B, hidden_dim)
        
        # Embed condition (state + goal)
        condition = torch.cat([state, goal], dim=-1)
        c_emb = self.condition_embed(condition)  # (B, hidden_dim)
        
        # Combine timestep and condition
        c = t_emb + c_emb  # (B, hidden_dim)
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final projection to action space
        v = self.final_layer(x, c)  # (B, T, action_dim)
        
        return v

    def sample(
        self,
        state: Tensor,
        goal: Tensor,
        num_steps: int = 50,
        schedule: str = "linear",
    ) -> Tensor:
        """
        Generate action trajectory using flow matching ODE.
        
        Uses Euler integration: x_{t+dt} = x_t + v(x_t, t) * dt
        
        Args:
            state: State embedding (B, state_dim)
            goal: Goal embedding (B, latent_dim)
            num_steps: Number of ODE integration steps
            schedule: Time schedule ("linear" or "cosine")
            
        Returns:
            Generated action trajectory (B, T, action_dim)
        """
        B = state.shape[0]
        device = state.device
        
        # Start from noise
        x = torch.randn(
            B, self.action_horizon, self.action_dim,
            device=device, dtype=state.dtype,
        )
        
        # Time steps from 0 to 1
        if schedule == "linear":
            times = torch.linspace(0, 1, num_steps + 1, device=device)
        else:
            # Cosine schedule (slower near boundaries)
            t = torch.linspace(0, 1, num_steps + 1, device=device)
            times = 0.5 * (1 - torch.cos(t * math.pi))
        
        # Euler integration
        for i in range(num_steps):
            t = times[i]
            dt = times[i + 1] - times[i]
            
            # Broadcast timestep
            t_batch = torch.full((B,), t.item(), device=device)
            
            # Predict velocity
            v = self.forward(x, t_batch, state, goal)
            
            # Euler step
            x = x + v * dt
        
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        state: Tensor,
        goal: Tensor,
        num_steps: int = 20,
    ) -> Tensor:
        """
        DDIM-style sampling (deterministic, faster).
        
        Args:
            state: State embedding (B, state_dim)
            goal: Goal embedding (B, latent_dim)
            num_steps: Number of sampling steps
            
        Returns:
            Generated action trajectory (B, T, action_dim)
        """
        return self.sample(state, goal, num_steps=num_steps, schedule="linear")

    def get_action(
        self,
        state: Tensor,
        goal: Tensor,
        num_steps: int = 50,
    ) -> Tensor:
        """
        Get first action from generated trajectory.
        
        For real-time control, typically only first action is executed.
        
        Args:
            state: State embedding (B, state_dim)
            goal: Goal embedding (B, latent_dim)
            num_steps: ODE integration steps
            
        Returns:
            First action (B, action_dim)
        """
        trajectory = self.sample(state, goal, num_steps)
        return trajectory[:, 0]

    def compute_loss(
        self,
        ground_truth_action: Tensor,
        state: Tensor,
        goal: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute flow matching loss.
        
        L = E_{t,x_0,x_1}[||v_θ(x_t, t, c) - (x_1 - x_0)||²]
        
        Args:
            ground_truth_action: Target action trajectory (B, T, action_dim)
            state: State embedding (B, state_dim)
            goal: Goal embedding (B, latent_dim)
            noise: Optional noise (B, T, action_dim), sampled if None
            
        Returns:
            Flow matching loss scalar
        """
        B = ground_truth_action.shape[0]
        device = ground_truth_action.device
        
        # Sample noise (x_0)
        if noise is None:
            noise = torch.randn_like(ground_truth_action)
        
        # Sample random timesteps
        t = torch.rand(B, device=device)
        
        # Interpolate: x_t = (1-t)*x_0 + t*x_1
        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * ground_truth_action
        
        # Target velocity: x_1 - x_0
        target_v = ground_truth_action - noise
        
        # Predict velocity
        pred_v = self.forward(x_t, t, state, goal)
        
        # MSE loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss

