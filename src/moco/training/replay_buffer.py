"""History of generated fakes for discriminator updates (CycleGAN image pool)."""
import random

import torch
from torch import Tensor


class ReplayBuffer:
    """Each sample is either returned as-is or, with p=0.5 once full, swapped for a stored older fake. Kept on CPU."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data: list[Tensor] = []

    def push_and_pop(self, x: Tensor) -> Tensor:
        """Store the new fakes and return a batch mixing new and older ones.

        Args:
            x: (B, ...) fakes from the current generator step.

        Returns:
            (B, ...) on CPU; feeding D older fakes too stops it from chasing only the latest generator.
        """
        out = []
        for i in range(x.size(0)):
            item = x[i].unsqueeze(0).detach().cpu()
            if len(self.data) < self.max_size:
                self.data.append(item.clone())
                out.append(item)
            elif random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                out.append(self.data[idx].clone())
                self.data[idx] = item.clone()
            else:
                out.append(item)
        return torch.cat(out, dim=0)
