"""
Data loading and preprocessing modules for Geo-Flow VLA.

Provides:
- Base dataset interface for manipulation benchmarks
- LIBERO dataset implementation
- Placeholder implementations for RLBench and CALVIN
- MoGe-2 3D lifting utilities
- Action/state normalization
"""

from .base_dataset import BaseManipulationDataset, TrajectoryData
from .libero_dataset import LIBERODataset
from .rlbench_dataset import RLBenchDataset
from .calvin_dataset import CALVINDataset
from .moge_lifting import MoGe2Lifter, MockMoGe2Lifter
from .normalizer import StateActionNormalizer

__all__ = [
    "BaseManipulationDataset",
    "TrajectoryData",
    "LIBERODataset",
    "RLBenchDataset",
    "CALVINDataset",
    "MoGe2Lifter",
    "MockMoGe2Lifter",
    "StateActionNormalizer",
]

