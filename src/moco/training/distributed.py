"""torchrun DDP setup and manual gradient averaging."""
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    """Process layout of the current run; a single process looks like DDP with world_size 1."""

    is_ddp: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(timeout_minutes: int) -> DistInfo:
    """DDP when launched by torchrun (LOCAL_RANK set), otherwise a single process.

    Args:
        timeout_minutes: NCCL timeout; long, so other ranks can wait at the barrier while rank 0 validates.

    Returns:
        DistInfo for this process.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistInfo(False, 0, 1, 0, device)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    torch.cuda.set_device(local_rank)
    return DistInfo(True, dist.get_rank(), dist.get_world_size(), local_rank, torch.device(f"cuda:{local_rank}"))


def average_gradients(parameters: Iterable[torch.nn.Parameter], world_size: int) -> None:
    """All-reduce mean of .grad, for modules whose backward bypasses DDP's hooks."""
    for p in parameters:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size
