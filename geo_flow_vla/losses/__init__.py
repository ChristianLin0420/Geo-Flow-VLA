"""
Loss functions for Geo-Flow VLA training.

Provides:
- FlowMatchingLoss: Conditional Flow Matching (CFM) for policy training
- FBObjective: Forward-Backward representation learning objective
- CPRRegularizer: Conditional Policy Regularization via adversarial training
"""

from .flow_matching_loss import FlowMatchingLoss, compute_cfm_loss
from .fb_objective import FBObjective, compute_fb_loss
from .cpr_regularizer import CPRRegularizer, compute_cpr_loss

__all__ = [
    "FlowMatchingLoss",
    "compute_cfm_loss",
    "FBObjective", 
    "compute_fb_loss",
    "CPRRegularizer",
    "compute_cpr_loss",
]

