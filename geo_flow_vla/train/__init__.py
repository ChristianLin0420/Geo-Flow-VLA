"""
Training scripts for Geo-Flow VLA.

Provides:
- Phase 1: Unsupervised world model (FB) training
- Phase 2: Policy training with Flow Matching and CPR
"""

from .phase1_world_model import train_world_model, WorldModelTrainer
from .phase2_policy import train_policy, PolicyTrainer

__all__ = [
    "train_world_model",
    "WorldModelTrainer",
    "train_policy",
    "PolicyTrainer",
]

