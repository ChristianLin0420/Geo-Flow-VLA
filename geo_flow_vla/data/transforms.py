"""
Image augmentation transforms for training.

Applies color jitter, random crop, and other augmentations
to reduce overfitting in policy training.
"""

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, Union
import random
import logging

logger = logging.getLogger(__name__)


class ImageAugmentation(nn.Module):
    """
    Image augmentation for training.
    
    Applies:
    - Color jitter (brightness, contrast, saturation, hue)
    - Random crop and resize
    
    Designed to reduce overfitting by increasing data diversity.
    """
    
    def __init__(
        self,
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        random_crop: bool = True,
        crop_scale_min: float = 0.85,
        crop_scale_max: float = 1.0,
        image_size: int = 224,
        p_color_jitter: float = 0.8,  # Probability of applying color jitter
        p_crop: float = 0.5,  # Probability of applying random crop
    ):
        """
        Args:
            color_jitter: Enable color jitter augmentation
            brightness: Max brightness change factor
            contrast: Max contrast change factor
            saturation: Max saturation change factor
            hue: Max hue change factor
            random_crop: Enable random crop augmentation
            crop_scale_min: Minimum crop scale
            crop_scale_max: Maximum crop scale
            image_size: Target image size after crop
            p_color_jitter: Probability of applying color jitter
            p_crop: Probability of applying random crop
        """
        super().__init__()
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_crop = random_crop
        self.crop_scale_min = crop_scale_min
        self.crop_scale_max = crop_scale_max
        self.image_size = image_size
        self.p_color_jitter = p_color_jitter
        self.p_crop = p_crop
    
    def forward(self, img: Tensor) -> Tensor:
        """
        Apply augmentations.
        
        Args:
            img: Image tensor (C, H, W) or (B, C, H, W) or (B, T, C, H, W) in [0, 1]
            
        Returns:
            Augmented image, same shape
        """
        original_shape = img.shape
        
        # Handle different input shapes
        if img.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = img.shape
            img = img.view(B * T, C, H, W)
            img = self._augment_batch(img)
            img = img.view(B, T, C, H, W)
        elif img.dim() == 4:  # (B, C, H, W)
            img = self._augment_batch(img)
        elif img.dim() == 3:  # (C, H, W)
            img = self._augment_single(img)
        else:
            raise ValueError(f"Unexpected image dimension: {img.dim()}")
        
        return img
    
    def _augment_batch(self, img: Tensor) -> Tensor:
        """Augment batch of images."""
        # Apply same augmentation to all images in batch for consistency
        # (important for temporal sequences)
        return torch.stack([self._augment_single(img[i]) for i in range(img.shape[0])])
    
    def _augment_single(self, img: Tensor) -> Tensor:
        """Augment single image."""
        # Color jitter (with probability)
        if self.color_jitter and random.random() < self.p_color_jitter:
            img = self._apply_color_jitter(img)
        
        # Random crop (with probability)
        if self.random_crop and random.random() < self.p_crop:
            img = self._apply_random_crop(img)
        
        return img
    
    def _apply_color_jitter(self, img: Tensor) -> Tensor:
        """Apply random color jitter."""
        # Random order for color transforms (as in torchvision ColorJitter)
        # Note: Use f=factor default arg to capture value at lambda creation time
        transforms = []
        
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            transforms.append(lambda x, f=factor: TF.adjust_brightness(x, f))
        
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            transforms.append(lambda x, f=factor: TF.adjust_contrast(x, f))
        
        if self.saturation > 0:
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            transforms.append(lambda x, f=factor: TF.adjust_saturation(x, f))
        
        if self.hue > 0:
            factor = random.uniform(-self.hue, self.hue)
            transforms.append(lambda x, f=factor: TF.adjust_hue(x, f))
        
        # Shuffle and apply
        random.shuffle(transforms)
        for t in transforms:
            img = t(img)
        
        return torch.clamp(img, 0, 1)
    
    def _apply_random_crop(self, img: Tensor) -> Tensor:
        """Apply random crop and resize back."""
        C, H, W = img.shape
        
        # Random scale
        scale = random.uniform(self.crop_scale_min, self.crop_scale_max)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        
        # Random position
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        
        # Crop
        img = TF.crop(img, top, left, crop_h, crop_w)
        
        # Resize back to original size
        img = TF.resize(img, [self.image_size, self.image_size], antialias=True)
        
        return img
    
    def __repr__(self) -> str:
        return (
            f"ImageAugmentation("
            f"color_jitter={self.color_jitter}, "
            f"random_crop={self.random_crop}, "
            f"image_size={self.image_size})"
        )


def create_augmentation(cfg) -> Optional[ImageAugmentation]:
    """
    Create augmentation from config.
    
    Args:
        cfg: OmegaConf config with augmentation settings
        
    Returns:
        ImageAugmentation instance or None if disabled
    """
    aug_cfg = cfg.get("augmentation", {})
    
    if not aug_cfg.get("enabled", False):
        logger.info("Image augmentation disabled")
        return None
    
    # Check if any augmentation is actually enabled
    color_jitter = aug_cfg.get("color_jitter", False)
    random_crop = aug_cfg.get("random_crop", False)
    
    if not color_jitter and not random_crop:
        logger.info("No augmentation transforms enabled")
        return None
    
    augmentation = ImageAugmentation(
        color_jitter=color_jitter,
        brightness=aug_cfg.get("color_jitter_brightness", 0.2),
        contrast=aug_cfg.get("color_jitter_contrast", 0.2),
        saturation=aug_cfg.get("color_jitter_saturation", 0.2),
        hue=aug_cfg.get("color_jitter_hue", 0.05),
        random_crop=random_crop,
        crop_scale_min=aug_cfg.get("crop_scale_min", 0.85),
        crop_scale_max=aug_cfg.get("crop_scale_max", 1.0),
        image_size=cfg.data.get("image_size", 224),
    )
    
    logger.info(f"Image augmentation enabled: {augmentation}")
    return augmentation


class GaussianNoise(nn.Module):
    """Add Gaussian noise to images."""
    
    def __init__(self, std: float = 0.05, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p
    
    def forward(self, img: Tensor) -> Tensor:
        if self.training and random.random() < self.p:
            noise = torch.randn_like(img) * self.std
            return torch.clamp(img + noise, 0, 1)
        return img


class RandomErase(nn.Module):
    """Random erasing augmentation (cutout)."""
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.1),
        ratio: Tuple[float, float] = (0.3, 3.3),
    ):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def forward(self, img: Tensor) -> Tensor:
        if self.training and random.random() < self.p:
            C, H, W = img.shape
            
            # Random area
            area = H * W
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            
            # Random aspect ratio
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))
            
            if h < H and w < W:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                
                # Fill with random values
                img = img.clone()
                img[:, top:top+h, left:left+w] = torch.rand(C, h, w, device=img.device)
        
        return img

