"""
MoGe-2 3D Lifting Module for Geo-Flow VLA.

Provides batch inference utilities for MoGe-2 (Monocular Geometry Estimation)
to lift RGB images to metric 3D point maps.

MoGe-2 Output Structure:
    - points: (H, W, 3) - metric 3D point map in OpenCV coords (x right, y down, z forward)
    - depth: (H, W) - depth map
    - normal: (H, W, 3) - surface normals (for -normal variant)
    - mask: (H, W) - valid pixel binary mask
    - intrinsics: (3, 3) - normalized camera intrinsics

References:
    - Wang et al., "MoGe: Unlocking Accurate Monocular Geometry Estimation" CVPR 2025
    - GitHub: https://github.com/microsoft/MoGe
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class BaseMoGe2Lifter(ABC, nn.Module):
    """Abstract base class for MoGe-2 3D lifting."""

    @abstractmethod
    def lift_single(self, rgb: Tensor) -> Dict[str, Tensor]:
        """
        Lift a single RGB image to 3D.
        
        Args:
            rgb: RGB image, shape (3, H, W), values in [0, 1]
            
        Returns:
            Dictionary with keys: points, depth, normal, mask, intrinsics
        """
        pass

    @abstractmethod
    def lift_batch(self, rgb: Tensor) -> Dict[str, Tensor]:
        """
        Lift a batch of RGB images to 3D.
        
        Args:
            rgb: RGB batch, shape (B, 3, H, W), values in [0, 1]
            
        Returns:
            Dictionary with batched outputs
        """
        pass

    def get_point_map(self, rgb: Tensor) -> Tensor:
        """
        Get point map in image format.
        
        Args:
            rgb: RGB batch, shape (B, 3, H, W)
            
        Returns:
            Point map, shape (B, 3, H, W) - xyz coordinates per pixel
        """
        output = self.lift_batch(rgb)
        # Reshape from (B, H, W, 3) to (B, 3, H, W) for downstream processing
        points = output["points"].permute(0, 3, 1, 2)
        return points


class MoGe2Lifter(BaseMoGe2Lifter):
    """
    MoGe-2 3D Lifter using official model from HuggingFace.
    
    Loads the MoGe-2 model and provides batched inference utilities.
    """

    def __init__(
        self,
        model_name: str = "Ruicheng/moge-2-vitl-normal",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        resolution_level: int = 9,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model name or local path
            device: Target device (defaults to CUDA if available)
            dtype: Model precision
            resolution_level: MoGe resolution level (0-9, higher = more detail)
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.resolution_level = resolution_level
        
        self._model = None
        self._loaded = False

    def _load_model(self) -> None:
        """Lazy load MoGe-2 model."""
        if self._loaded:
            return
        
        try:
            # Import MoGe model - requires moge package installed
            from moge.model.v2 import MoGeModel
            
            logger.info(f"Loading MoGe-2 model: {self.model_name}")
            self._model = MoGeModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            
            # Freeze model
            for param in self._model.parameters():
                param.requires_grad = False
            
            self._loaded = True
            logger.info("MoGe-2 model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                "MoGe package not found. Install with: pip install git+https://github.com/microsoft/MoGe.git"
            ) from e

    @torch.no_grad()
    def lift_single(self, rgb: Tensor) -> Dict[str, Tensor]:
        """
        Lift a single RGB image to 3D using MoGe-2.
        
        Args:
            rgb: RGB image, shape (3, H, W), values in [0, 1]
            
        Returns:
            Dictionary containing:
                - points: (H, W, 3) metric 3D point map
                - depth: (H, W) depth map
                - normal: (H, W, 3) surface normals
                - mask: (H, W) valid pixel mask
                - intrinsics: (3, 3) camera intrinsics
        """
        self._load_model()
        
        assert rgb.dim() == 3 and rgb.shape[0] == 3, \
            f"Expected (3, H, W), got {rgb.shape}"
        
        # MoGe expects (3, H, W) with values in [0, 1]
        rgb = rgb.to(self.device, self.dtype)
        
        # Run inference
        output = self._model.infer(
            rgb,
            resolution_level=self.resolution_level,
        )
        
        return {
            "points": output["points"],      # (H, W, 3)
            "depth": output["depth"],        # (H, W)
            "normal": output.get("normal"),  # (H, W, 3) or None
            "mask": output["mask"],          # (H, W)
            "intrinsics": output["intrinsics"],  # (3, 3)
        }

    @torch.no_grad()
    def lift_batch(self, rgb: Tensor) -> Dict[str, Tensor]:
        """
        Lift a batch of RGB images to 3D.
        
        Args:
            rgb: RGB batch, shape (B, 3, H, W), values in [0, 1]
            
        Returns:
            Dictionary with batched outputs:
                - points: (B, H, W, 3)
                - depth: (B, H, W)
                - normal: (B, H, W, 3) or None
                - mask: (B, H, W)
                - intrinsics: (B, 3, 3)
        """
        self._load_model()
        
        assert rgb.dim() == 4 and rgb.shape[1] == 3, \
            f"Expected (B, 3, H, W), got {rgb.shape}"
        
        B = rgb.shape[0]
        rgb = rgb.to(self.device, self.dtype)
        
        # Process each image (MoGe-2 doesn't support batch inference natively)
        results = [self.lift_single(rgb[i]) for i in range(B)]
        
        # Stack results
        points = torch.stack([r["points"] for r in results])
        depth = torch.stack([r["depth"] for r in results])
        mask = torch.stack([r["mask"] for r in results])
        intrinsics = torch.stack([r["intrinsics"] for r in results])
        
        # Handle optional normal
        if results[0]["normal"] is not None:
            normal = torch.stack([r["normal"] for r in results])
        else:
            normal = None
        
        return {
            "points": points,
            "depth": depth,
            "normal": normal,
            "mask": mask,
            "intrinsics": intrinsics,
        }


class MockMoGe2Lifter(BaseMoGe2Lifter):
    """
    Mock MoGe-2 Lifter for testing and development.
    
    Generates synthetic 3D point maps when MoGe-2 weights are unavailable.
    Useful for:
    - Testing data pipeline without downloading large models
    - Development on machines without GPU
    - Unit testing
    """

    def __init__(
        self,
        default_depth: float = 1.0,
        focal_length: float = 500.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            default_depth: Default depth value for synthetic points
            focal_length: Focal length for camera projection
            device: Target device
        """
        super().__init__()
        
        self.default_depth = default_depth
        self.focal_length = focal_length
        self.device = device or torch.device("cpu")

    def lift_single(self, rgb: Tensor) -> Dict[str, Tensor]:
        """Generate synthetic 3D point map from RGB image."""
        assert rgb.dim() == 3 and rgb.shape[0] == 3
        
        C, H, W = rgb.shape
        device = rgb.device
        
        # Create pixel grid
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        
        # Synthetic depth: slight variation based on image brightness
        brightness = rgb.mean(dim=0)  # (H, W)
        depth = self.default_depth + 0.1 * (brightness - 0.5)
        
        # Camera parameters
        cx, cy = W / 2, H / 2
        fx = fy = self.focal_length
        
        # Back-project to 3D (OpenCV convention: x right, y down, z forward)
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        points = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
        
        # Synthetic normals (pointing towards camera)
        normal = torch.zeros(H, W, 3, device=device)
        normal[:, :, 2] = -1  # All normals point towards camera
        
        # All pixels valid
        mask = torch.ones(H, W, device=device, dtype=torch.bool)
        
        # Intrinsics matrix
        intrinsics = torch.tensor([
            [fx / W, 0, cx / W],
            [0, fy / H, cy / H],
            [0, 0, 1],
        ], device=device, dtype=torch.float32)
        
        return {
            "points": points,
            "depth": depth,
            "normal": normal,
            "mask": mask,
            "intrinsics": intrinsics,
        }

    def lift_batch(self, rgb: Tensor) -> Dict[str, Tensor]:
        """Generate synthetic 3D point maps for a batch."""
        assert rgb.dim() == 4 and rgb.shape[1] == 3
        
        B = rgb.shape[0]
        results = [self.lift_single(rgb[i]) for i in range(B)]
        
        return {
            "points": torch.stack([r["points"] for r in results]),
            "depth": torch.stack([r["depth"] for r in results]),
            "normal": torch.stack([r["normal"] for r in results]),
            "mask": torch.stack([r["mask"] for r in results]),
            "intrinsics": torch.stack([r["intrinsics"] for r in results]),
        }


def get_moge_lifter(
    use_mock: bool = False,
    model_name: str = "Ruicheng/moge-2-vitl-normal",
    device: Optional[torch.device] = None,
    **kwargs,
) -> BaseMoGe2Lifter:
    """
    Factory function to get MoGe-2 lifter.
    
    Args:
        use_mock: If True, return mock lifter
        model_name: HuggingFace model name for real lifter
        device: Target device
        **kwargs: Additional arguments for lifter
        
    Returns:
        MoGe2 lifter instance
    """
    if use_mock:
        return MockMoGe2Lifter(device=device, **kwargs)
    else:
        return MoGe2Lifter(model_name=model_name, device=device, **kwargs)


class MoGe2FeatureExtractor(nn.Module):
    """
    Wrapper that extracts features from MoGe-2 point maps.
    
    Projects per-pixel 3D coordinates to a learned feature space
    for downstream fusion with semantic features.
    """

    def __init__(
        self,
        output_dim: int = 256,
        use_mock: bool = False,
        model_name: str = "Ruicheng/moge-2-vitl-normal",
        include_normals: bool = True,
    ) -> None:
        """
        Args:
            output_dim: Output feature dimension
            use_mock: Use mock lifter
            model_name: MoGe model name
            include_normals: Include surface normals in features
        """
        super().__init__()
        
        self.lifter = get_moge_lifter(use_mock=use_mock, model_name=model_name)
        self.include_normals = include_normals
        
        # Input: xyz (3) + optional normals (3)
        input_dim = 6 if include_normals else 3
        
        # Learned projection
        self.projection = nn.Sequential(
            nn.Conv2d(input_dim, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, output_dim, 1),
        )

    def forward(self, rgb: Tensor) -> Tensor:
        """
        Extract geometric features from RGB images.
        
        Args:
            rgb: RGB images, shape (B, 3, H, W), values in [0, 1]
            
        Returns:
            Geometric features, shape (B, output_dim, H, W)
        """
        # Get 3D geometry
        with torch.no_grad():
            moge_output = self.lifter.lift_batch(rgb)
        
        # Combine point coordinates and normals
        points = moge_output["points"].permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        if self.include_normals and moge_output["normal"] is not None:
            normals = moge_output["normal"].permute(0, 3, 1, 2)  # (B, 3, H, W)
            features = torch.cat([points, normals], dim=1)  # (B, 6, H, W)
        else:
            features = points
        
        # Project to feature space
        return self.projection(features)

