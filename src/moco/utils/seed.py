"""Seeding for python, numpy, torch, CUDA and DataLoader workers."""
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    """deterministic=False keeps cudnn.benchmark on for speed (results then vary slightly run to run)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """Forked workers inherit one RNG state; reseed python/numpy from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
