"""Learning-rate schedule and loss-weight warmup."""
from dataclasses import replace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

from moco.losses.gan import LossWeights


def build_scheduler(optimiser: Optimizer, n_epochs: int, warmup: int, warmup_start_factor: float,
                    final_factor: float) -> SequentialLR:
    """Per-epoch LR schedule: linear warmup, constant until the half-way point, then linear decay.

    Args:
        optimiser: Optimiser to schedule; call step() once per epoch.
        n_epochs: Total training epochs.
        warmup: Warmup epochs.
        warmup_start_factor: LR multiplier at epoch 0.
        final_factor: LR multiplier reached at the last epoch.

    Returns:
        The chained scheduler.
    """
    half = n_epochs // 2
    return SequentialLR(
        optimiser,
        schedulers=[
            LinearLR(optimiser, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup),
            ConstantLR(optimiser, factor=1.0, total_iters=half),
            LinearLR(optimiser, start_factor=1.0, end_factor=final_factor,
                     total_iters=max(n_epochs - half - warmup, 1)),
        ],
        milestones=[warmup, warmup + half],
    )


def epoch_loss_weights(base: LossWeights, epoch: int, warmup_epochs: int) -> LossWeights:
    """Ramp cycle and identity weights linearly from 1 to their full value over warmup_epochs."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return base
    t = epoch / warmup_epochs
    return replace(base, cyc=1.0 + (base.cyc - 1.0) * t, idt=1.0 + (base.idt - 1.0) * t)
