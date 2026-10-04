"""Losses on ROI time series: LSGAN for the ROI discriminator and ROI-space cycle consistency."""
import torch
import torch.nn.functional as F
from torch import Tensor


def roi_discriminator_loss(real: dict[str, Tensor], fake: dict[str, Tensor], lambda_roi: float = 0.5) -> Tensor:
    """LSGAN on {"global": (B, 1), "roi": (B, n_rois)}: global term + lambda_roi * per-ROI term."""
    def lsgan(key: str) -> Tensor:
        return 0.5 * (F.mse_loss(real[key], torch.ones_like(real[key]))
                      + F.mse_loss(fake[key], torch.zeros_like(fake[key])))
    return lsgan("global") + lambda_roi * lsgan("roi")


def roi_generator_loss(fake: dict[str, Tensor], lambda_roi: float = 0.5) -> Tensor:
    """LSGAN generator side: corrected ROI time series should be scored real (1), global + per-ROI."""
    return (F.mse_loss(fake["global"], torch.ones_like(fake["global"]))
            + lambda_roi * F.mse_loss(fake["roi"], torch.ones_like(fake["roi"])))


def roi_cycle_loss(input_roi_ts: Tensor, cycle_roi_ts: Tensor) -> Tensor:
    """L1 between ROI time series (B, n_rois, T) of x_a and its A->B->A reconstruction.

    The input side is detached: it is a fixed target, not something the generator should move.
    """
    assert input_roi_ts.shape == cycle_roi_ts.shape
    return F.l1_loss(cycle_roi_ts, input_roi_ts.detach())
