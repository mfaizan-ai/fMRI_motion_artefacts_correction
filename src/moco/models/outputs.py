"""Container for every tensor produced by one CycleGAN forward pass."""
from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutputs:
    """Chunks are (B, T, D, H, W). Feature maps are (B, C, d, h, w) for the disentangled model and
    (B, T, C, d, h, w) for the spatiotemporal one. Scores are lists of patch maps, finest scale first.
    A = motion-corrupted domain, B = motion-free domain."""

    c_a: Tensor
    c_b: Tensor
    a_global: Tensor
    a_spatial: Tensor
    a_global_b: Tensor
    a_spatial_b: Tensor
    x_hat_b: Tensor        # A -> B, the corrected output
    x_hat_a: Tensor        # B -> A, synthetic corruption
    x_self_a: Tensor
    x_self_b: Tensor
    c_hat_b: Tensor
    c_hat_a: Tensor
    a_hat_global: Tensor
    a_hat_spatial: Tensor
    x_cycle_a: Tensor      # A -> B -> A
    x_cycle_b: Tensor      # B -> A -> B
    score_real_b: list
    score_fake_b: list
    score_real_a: list
    score_fake_a: list
