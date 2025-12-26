"""
Distributed Training Utilities for Geo-Flow VLA.

Provides helpers for multi-GPU training with PyTorch DistributedDataParallel.
"""

import os
import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


def setup_distributed() -> Tuple[int, int, bool]:
    """
    Initialize distributed training environment.
    
    Returns:
        Tuple of (rank, world_size, is_distributed)
    """
    # Check if running with torchrun/torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, True
    else:
        # Single GPU or CPU training
        return 0, 1, False


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_value(value: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all processes.
    
    Args:
        value: Tensor to reduce
        average: If True, return average; otherwise return sum
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return value
    
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    
    if average:
        value /= get_world_size()
    
    return value


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce all values in a dictionary across all processes.
    
    Args:
        input_dict: Dictionary with tensor values
        average: If True, return averages; otherwise return sums
        
    Returns:
        Dictionary with reduced values
    """
    if not dist.is_initialized():
        return input_dict
    
    names = list(input_dict.keys())
    values = torch.stack([input_dict[k] for k in names])
    
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    
    if average:
        values /= get_world_size()
    
    return {k: v.item() for k, v in zip(names, values)}


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """
    Wrap a model with DistributedDataParallel if distributed training is enabled.
    
    Args:
        model: Model to wrap
        device: Device to place model on
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        DDP-wrapped model or original model
    """
    model = model.to(device)
    
    if dist.is_initialized():
        local_rank = get_local_rank()
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )
        logger.info(f"Wrapped model with DDP on device {local_rank}")
    
    return model


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with distributed sampling.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DataLoader with DistributedSampler if distributed
    """
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
    )


def set_epoch_sampler(dataloader: DataLoader, epoch: int):
    """
    Set the epoch for DistributedSampler to ensure proper shuffling.
    
    Args:
        dataloader: DataLoader with DistributedSampler
        epoch: Current epoch number
    """
    if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)


class DistributedTrainingContext:
    """Context manager for distributed training setup/cleanup."""
    
    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.is_distributed = False
    
    def __enter__(self):
        self.rank, self.world_size, self.is_distributed = setup_distributed()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()
        return False
    
    @property
    def is_main(self) -> bool:
        return self.rank == 0
    
    @property
    def device(self) -> torch.device:
        if self.is_distributed:
            return torch.device(f"cuda:{get_local_rank()}")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

