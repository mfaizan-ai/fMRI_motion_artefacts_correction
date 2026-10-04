"""CycleGAN losses: LSGAN adversarial (multi-scale), cycle and identity L1."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from moco.models.outputs import ModelOutputs


@dataclass
class LossWeights:
    """Weights of the generator loss terms; temporal and fc are used in sequence mode only."""

    adv: float = 1.0
    cyc: float = 10.0
    idt: float = 5.0
    temporal: float = 0.0
    fc: float = 0.0


def lsgan_generator_loss(scores_fake: list[Tensor]) -> Tensor:
    """Generator wants every fake patch scored 1; averaged over scales."""
    return sum(0.5 * F.mse_loss(s, torch.ones_like(s)) for s in scores_fake) / len(scores_fake)


def lsgan_discriminator_loss(scores_real: list[Tensor], scores_fake: list[Tensor],
                             real_target: float = 1.0, fake_target: float = 0.0) -> Tensor:
    """LSGAN with (optionally smoothed) targets, averaged over scales."""
    return sum(
        0.5 * (F.mse_loss(sr, torch.full_like(sr, real_target)) + F.mse_loss(sf, torch.full_like(sf, fake_target)))
        for sr, sf in zip(scores_real, scores_fake, strict=True)
    ) / len(scores_real)


def generator_loss(out: ModelOutputs, x_a: Tensor, x_b: Tensor, weights: LossWeights) -> dict[str, Tensor]:
    """Generator loss for one batch.

    Args:
        out: Forward pass of the CycleGAN on (x_a, x_b).
        x_a: (B, T, H, W, D) corrupted input.
        x_b: (B, T, H, W, D) clean input.
        weights: Loss weights for this epoch.

    Returns:
        Unweighted "adv", "cyc" (A->B->A, B->A->B), "idt" (A->A, B->B) and the weighted "total".
    """
    adv = lsgan_generator_loss(out.score_fake_b) + lsgan_generator_loss(out.score_fake_a)
    cyc = F.l1_loss(out.x_cycle_a, x_a) + F.l1_loss(out.x_cycle_b, x_b)
    idt = F.l1_loss(out.x_self_a, x_a) + F.l1_loss(out.x_self_b, x_b)
    total = weights.adv * adv + weights.cyc * cyc + weights.idt * idt
    return {"adv": adv, "cyc": cyc, "idt": idt, "total": total}


def r1_gradient_penalty(discriminator: torch.nn.Module, real: Tensor) -> Tensor:
    """R1 penalty: mean over the batch of ||grad_x D(x)||^2 on real samples.

    Args:
        discriminator: Multi-scale discriminator returning a list of score maps.
        real: (B, T, H, W, D) real samples.

    Returns:
        Scalar penalty; scores are summed over scales before taking the gradient.
    """
    real = real.detach().requires_grad_(True)
    output = sum(s.sum() for s in discriminator(real))
    (grad,) = torch.autograd.grad(outputs=output, inputs=real, create_graph=True)
    return grad.pow(2).reshape(grad.size(0), -1).sum(dim=1).mean()
