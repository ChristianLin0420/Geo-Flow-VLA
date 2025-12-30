"""
Geo-Flow-VLA Evaluation Module.

Provides evaluation pipelines for:
- LIBERO: MuJoCo-based manipulation benchmark
- RLBench: CoppeliaSim-based 18-task benchmark
- CALVIN: Long-horizon language-conditioned benchmark
"""

from .policy_wrapper import GeoFlowVLAPolicy
from .base_evaluator import BaseEvaluator

__all__ = [
    "GeoFlowVLAPolicy",
    "BaseEvaluator",
]
