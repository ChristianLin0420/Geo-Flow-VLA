"""
Model components for Geo-Flow VLA.

Provides:
- DualEncoder: Fused DINOv2-G (semantic) + MoGe-2 (geometric) backbone
- FBWorldModel: Forward-Backward representation learning (legacy)
- LanguageConditionedFBWorldModel: FB world model with language conditioning
- DiffusionPolicy: DiT backbone with Flow Matching
- CPRDiscriminator: Discriminator for policy regularization
- CLIPLanguageEncoder: CLIP-based text encoder for instructions
"""

from .dual_encoder import DualEncoder, DINOv2Encoder, GeometricEncoder
from .world_model import FBWorldModel, BackwardNetwork, ForwardNetwork
from .language_conditioned_world_model import (
    LanguageConditionedFBWorldModel,
    LanguageConditionedBackwardNetwork,
)
from .language_encoder import CLIPLanguageEncoder, create_language_encoder
from .diffusion_policy import DiffusionPolicy, DiTBlock
from .discriminator import CPRDiscriminator

__all__ = [
    "DualEncoder",
    "DINOv2Encoder",
    "GeometricEncoder",
    "FBWorldModel",
    "BackwardNetwork",
    "ForwardNetwork",
    "LanguageConditionedFBWorldModel",
    "LanguageConditionedBackwardNetwork",
    "CLIPLanguageEncoder",
    "create_language_encoder",
    "DiffusionPolicy",
    "DiTBlock",
    "CPRDiscriminator",
]

