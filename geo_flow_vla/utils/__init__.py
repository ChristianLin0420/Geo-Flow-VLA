"""
Utility modules for Geo-Flow VLA.
"""

from .schedule import (
    LinearSchedule,
    CosineSchedule,
    OptimalTransportSchedule,
    get_schedule,
)
from .rotation_equivariance import (
    random_so3_rotation,
    rotate_point_cloud,
    rotate_actions,
    SE3Augmentation,
)
from .camera_transforms import (
    get_default_camera_extrinsics,
    transform_points_robot_to_camera,
    transform_trajectory_to_camera_frame,
    scale_trajectory_to_scene,
    get_dataset_camera,
    DATASET_CAMERA_CONFIG,
)
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    get_local_rank,
    wrap_model_ddp,
    create_distributed_dataloader,
    set_epoch_sampler,
    reduce_dict,
    DistributedTrainingContext,
)

__all__ = [
    # Schedules
    "LinearSchedule",
    "CosineSchedule",
    "OptimalTransportSchedule",
    "get_schedule",
    # Rotation equivariance
    "random_so3_rotation",
    "rotate_point_cloud",
    "rotate_actions",
    "SE3Augmentation",
    # Camera transforms
    "get_default_camera_extrinsics",
    "transform_points_robot_to_camera",
    "transform_trajectory_to_camera_frame",
    "scale_trajectory_to_scene",
    "get_dataset_camera",
    "DATASET_CAMERA_CONFIG",
    # Distributed training
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "wrap_model_ddp",
    "create_distributed_dataloader",
    "set_epoch_sampler",
    "reduce_dict",
    "DistributedTrainingContext",
]


# Alignment visualizations
from .visualizations import AlignmentVisualizer

__all__ += [
    "AlignmentVisualizer",
]
