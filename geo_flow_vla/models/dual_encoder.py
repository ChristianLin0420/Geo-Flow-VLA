"""
Dual Encoder for Geo-Flow VLA.

Fuses DINOv2-G (semantic features) with MoGe-2 (geometric features)
using cross-attention to create a unified state representation.

Architecture:
    RGB Image → DINOv2-G → Semantic Tokens (B, N, 1536)
    RGB Image → MoGe-2 → Point Map (B, 3, H, W) → Geometric Features (B, N, 256)
    Cross-Attention Fusion → State Embedding (B, 512)

Both encoders are frozen during policy training.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data.moge_lifting import get_moge_lifter, BaseMoGe2Lifter


class DINOv2Encoder(nn.Module):
    """
    DINOv2-G/14 encoder for semantic features.
    
    Extracts patch-level semantic tokens from RGB images.
    Uses the frozen DINOv2 giant model (1.1B parameters).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitg14",
        pretrained: bool = True,
        frozen: bool = True,
        output_dim: int = 1536,  # DINOv2-G output dimension
    ) -> None:
        """
        Args:
            model_name: DINOv2 model variant
            pretrained: Load pretrained weights
            frozen: Freeze encoder weights
            output_dim: Expected output dimension (for verification)
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.frozen = frozen
        
        # Load DINOv2 from torch hub
        self.encoder = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=pretrained,
        )
        
        # Verify output dimension
        actual_dim = self.encoder.embed_dim
        assert actual_dim == output_dim, \
            f"DINOv2 output dim {actual_dim} != expected {output_dim}"
        
        if frozen:
            self._freeze()

    def _freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Extract semantic features from RGB images.
        
        Args:
            x: RGB images, shape (B, 3, H, W), normalized to ImageNet stats
            
        Returns:
            Patch tokens, shape (B, N, output_dim) where N = (H/14) * (W/14)
        """
        if self.frozen:
            with torch.no_grad():
                features = self.encoder.forward_features(x)
        else:
            features = self.encoder.forward_features(x)
        
        # DINOv2 returns dict with 'x_norm_patchtokens'
        if isinstance(features, dict):
            tokens = features.get('x_norm_patchtokens', features.get('x_prenorm'))
        else:
            # Remove CLS token if present
            tokens = features[:, 1:] if features.shape[1] > (x.shape[2] // 14) ** 2 else features
        
        return tokens

    def get_cls_token(self, x: Tensor) -> Tensor:
        """Get CLS token for global representation."""
        if self.frozen:
            with torch.no_grad():
                features = self.encoder.forward_features(x)
        else:
            features = self.encoder.forward_features(x)
        
        if isinstance(features, dict):
            return features.get('x_norm_clstoken', features['x_prenorm'][:, 0])
        return features[:, 0]


class GeometricEncoder(nn.Module):
    """
    Geometric encoder using MoGe-2 point maps.
    
    Projects per-pixel 3D coordinates (and optionally normals)
    to a feature space compatible with semantic fusion.
    """

    def __init__(
        self,
        output_dim: int = 256,
        use_normals: bool = True,
        use_mock: bool = False,
        moge_model: str = "Ruicheng/moge-2-vitl-normal",
        num_sample_points: int = 256,
    ) -> None:
        """
        Args:
            output_dim: Output feature dimension
            use_normals: Include surface normals in features
            use_mock: Use mock MoGe-2 (for testing)
            moge_model: MoGe-2 model name
            num_sample_points: Number of points to sample for token representation
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.use_normals = use_normals
        self.num_sample_points = num_sample_points
        
        # MoGe-2 lifter (frozen)
        self.lifter = get_moge_lifter(
            use_mock=use_mock,
            model_name=moge_model,
        )
        
        # Input dimension: xyz (3) + optional normals (3)
        input_dim = 6 if use_normals else 3
        
        # Per-point feature extraction
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )
        
        # Positional encoding for 3D coordinates
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
        )

    @torch.no_grad()
    def get_point_map(self, rgb: Tensor) -> Dict[str, Tensor]:
        """
        Get 3D point map from RGB using MoGe-2.
        
        Args:
            rgb: RGB images (B, 3, H, W), values in [0, 1]
            
        Returns:
            Dictionary with points, normals, mask, etc.
        """
        return self.lifter.lift_batch(rgb)

    def forward(self, rgb: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract geometric features from RGB images.
        
        Args:
            rgb: RGB images (B, 3, H, W), values in [0, 1]
            
        Returns:
            Tuple of:
                - Geometric tokens (B, N, output_dim)
                - 3D positions for each token (B, N, 3)
        """
        B, C, H, W = rgb.shape
        
        # Get 3D geometry from MoGe-2
        moge_output = self.get_point_map(rgb)
        points = moge_output["points"]  # (B, H, W, 3)
        mask = moge_output["mask"]  # (B, H, W)
        
        # Flatten spatial dimensions
        points_flat = points.reshape(B, -1, 3)  # (B, H*W, 3)
        mask_flat = mask.reshape(B, -1)  # (B, H*W)
        
        # Build input features
        if self.use_normals and moge_output["normal"] is not None:
            normals = moge_output["normal"]  # (B, H, W, 3)
            normals_flat = normals.reshape(B, -1, 3)
            point_features = torch.cat([points_flat, normals_flat], dim=-1)  # (B, H*W, 6)
        else:
            point_features = points_flat  # (B, H*W, 3)
        
        # Sample points for efficiency (full resolution is expensive)
        if self.num_sample_points < H * W:
            # Sample valid points based on mask
            sampled_features = []
            sampled_positions = []
            
            for b in range(B):
                valid_idx = mask_flat[b].nonzero(as_tuple=True)[0]
                
                if len(valid_idx) >= self.num_sample_points:
                    # Random sample from valid points
                    perm = torch.randperm(len(valid_idx), device=rgb.device)[:self.num_sample_points]
                    sample_idx = valid_idx[perm]
                else:
                    # Use all valid + pad with zeros
                    sample_idx = valid_idx
                    pad_size = self.num_sample_points - len(valid_idx)
                    sample_idx = torch.cat([
                        sample_idx,
                        torch.zeros(pad_size, dtype=torch.long, device=rgb.device)
                    ])
                
                sampled_features.append(point_features[b, sample_idx])
                sampled_positions.append(points_flat[b, sample_idx])
            
            point_features = torch.stack(sampled_features)  # (B, N, 3 or 6)
            positions = torch.stack(sampled_positions)  # (B, N, 3)
        else:
            positions = points_flat
        
        # Encode point features
        tokens = self.point_encoder(point_features)  # (B, N, output_dim)
        
        # Add positional encoding
        pos_enc = self.pos_encoder(positions)
        tokens = tokens + pos_enc
        
        return tokens, positions


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module for fusing semantic and geometric features.
    
    Semantic tokens attend to geometric tokens to incorporate 3D structure.
    """

    def __init__(
        self,
        semantic_dim: int = 1536,
        geometric_dim: int = 256,
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            semantic_dim: Dimension of semantic tokens (DINOv2-G)
            geometric_dim: Dimension of geometric tokens
            output_dim: Output state dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.geometric_dim = geometric_dim
        self.output_dim = output_dim
        
        # Project semantic tokens to attention dimension
        self.semantic_proj = nn.Linear(semantic_dim, output_dim)
        
        # Project geometric tokens for keys/values
        self.geometric_key_proj = nn.Linear(geometric_dim, output_dim)
        self.geometric_value_proj = nn.Linear(geometric_dim, output_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm and FFN
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout),
        )
        
        # Global pooling to get single state vector
        self.global_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        semantic_tokens: Tensor,
        geometric_tokens: Tensor,
    ) -> Tensor:
        """
        Fuse semantic and geometric features.
        
        Args:
            semantic_tokens: DINOv2 patch tokens (B, N_s, semantic_dim)
            geometric_tokens: Geometric point features (B, N_g, geometric_dim)
            
        Returns:
            Fused state embedding (B, output_dim)
        """
        # Project to common dimension
        queries = self.semantic_proj(semantic_tokens)  # (B, N_s, output_dim)
        keys = self.geometric_key_proj(geometric_tokens)  # (B, N_g, output_dim)
        values = self.geometric_value_proj(geometric_tokens)  # (B, N_g, output_dim)
        
        # Cross-attention: semantic queries attend to geometric keys/values
        attended, _ = self.cross_attention(queries, keys, values)
        
        # Residual connection and normalization
        fused = self.norm1(queries + attended)
        
        # FFN
        fused = self.norm2(fused + self.ffn(fused))
        
        # Global average pooling over tokens
        pooled = fused.mean(dim=1)  # (B, output_dim)
        
        # Final projection
        state = self.global_pool(pooled)
        
        return state


class DualEncoder(nn.Module):
    """
    Complete Dual Encoder combining DINOv2-G and MoGe-2.
    
    Produces a unified state representation by fusing:
    - Semantic features from DINOv2-G/14
    - Geometric features from MoGe-2 3D point maps
    
    Both backbone encoders are frozen; only fusion layers are trainable.
    """

    def __init__(
        self,
        semantic_dim: int = 1536,
        geometric_dim: int = 256,
        output_dim: int = 512,
        dinov2_model: str = "dinov2_vitg14",
        moge_model: str = "Ruicheng/moge-2-vitl-normal",
        use_mock_moge: bool = False,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_backbones: bool = True,
    ) -> None:
        """
        Args:
            semantic_dim: DINOv2-G output dimension (1536)
            geometric_dim: Geometric feature dimension
            output_dim: Final state embedding dimension
            dinov2_model: DINOv2 model name
            moge_model: MoGe-2 model name
            use_mock_moge: Use mock MoGe-2 (for testing)
            num_heads: Cross-attention heads
            dropout: Dropout rate
            freeze_backbones: Freeze DINOv2 and MoGe-2
        """
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.geometric_dim = geometric_dim
        self.output_dim = output_dim
        
        # Semantic encoder (DINOv2-G)
        self.semantic_encoder = DINOv2Encoder(
            model_name=dinov2_model,
            pretrained=True,
            frozen=freeze_backbones,
            output_dim=semantic_dim,
        )
        
        # Geometric encoder (MoGe-2)
        self.geometric_encoder = GeometricEncoder(
            output_dim=geometric_dim,
            use_normals=True,
            use_mock=use_mock_moge,
            moge_model=moge_model,
        )
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            semantic_dim=semantic_dim,
            geometric_dim=geometric_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # ImageNet normalization (DINOv2 expects this)
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def normalize_image(self, x: Tensor) -> Tensor:
        """Normalize image to ImageNet statistics."""
        return (x - self.img_mean) / self.img_std

    def forward(
        self,
        rgb: Tensor,
        return_tokens: bool = False,
    ) -> Tensor:
        """
        Encode RGB image to state representation.
        
        Args:
            rgb: RGB images (B, 3, H, W), values in [0, 1]
            return_tokens: Also return intermediate tokens
            
        Returns:
            State embedding (B, output_dim)
            If return_tokens: also returns (semantic_tokens, geometric_tokens, positions)
        """
        # Normalize for DINOv2
        rgb_normalized = self.normalize_image(rgb)
        
        # Extract semantic features
        semantic_tokens = self.semantic_encoder(rgb_normalized)  # (B, N_s, 1536)
        
        # Extract geometric features (uses unnormalized RGB for MoGe)
        geometric_tokens, positions = self.geometric_encoder(rgb)  # (B, N_g, 256)
        
        # Fuse features
        state = self.fusion(semantic_tokens, geometric_tokens)  # (B, 512)
        
        if return_tokens:
            return state, semantic_tokens, geometric_tokens, positions
        
        return state

    def encode_semantic(self, rgb: Tensor) -> Tensor:
        """Get only semantic features."""
        rgb_normalized = self.normalize_image(rgb)
        return self.semantic_encoder(rgb_normalized)

    def encode_geometric(self, rgb: Tensor) -> Tuple[Tensor, Tensor]:
        """Get only geometric features."""
        return self.geometric_encoder(rgb)

    @torch.no_grad()
    def get_point_map(self, rgb: Tensor) -> Optional[Tensor]:
        """
        Get raw 3D point map from MoGe-2.
        
        Args:
            rgb: RGB images (B, 3, H, W), values in [0, 1]
            
        Returns:
            Point map (B, 3, H, W) or None if not available
        """
        try:
            moge_output = self.geometric_encoder.get_point_map(rgb)
            if moge_output is not None and "points" in moge_output:
                # points is (B, H, W, 3), convert to (B, 3, H, W)
                points = moge_output["points"].permute(0, 3, 1, 2)
                return points
        except Exception:
            pass
        return None

    @property
    def trainable_parameters(self) -> int:
        """Count trainable parameters (fusion only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

