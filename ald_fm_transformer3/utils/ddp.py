from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """
    Initialise NCCL process group from torchrun env vars.
    Returns (rank, local_rank, world_size).
    Falls back to single-GPU (0, 0, 1) when not launched with torchrun.
    """
    if 'RANK' not in os.environ:
        return 0, 0, 1

    # 2-hour timeout: rank 0 may compute norm stats over hundreds of CSV files
    # before the first collective, which easily exceeds the default 10-min limit.
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
