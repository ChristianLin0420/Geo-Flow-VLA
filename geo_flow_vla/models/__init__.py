"""
Model components for Geo-Flow VLA.

Provides:
- DualEncoder: Fused DINOv2-G (semantic) + MoGe-2 (geometric) backbone
- FBWorldModel: Forward-Backward representation learning
- DiffusionPolicy: DiT backbone with Flow Matching
- CPRDiscriminator: Discriminator for policy regularization
"""

from .dual_encoder import DualEncoder, DINOv2Encoder, GeometricEncoder
from .world_model import FBWorldModel, BackwardNetwork, ForwardNetwork
from .diffusion_policy import DiffusionPolicy, DiTBlock
from .discriminator import CPRDiscriminator

__all__ = [
    "DualEncoder",
    "DINOv2Encoder",
    "GeometricEncoder",
    "FBWorldModel",
    "BackwardNetwork",
    "ForwardNetwork",
    "DiffusionPolicy",
    "DiTBlock",
    "CPRDiscriminator",
]

