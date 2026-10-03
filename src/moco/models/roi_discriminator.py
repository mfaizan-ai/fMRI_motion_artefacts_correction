"""Discriminator on ROI mean time series: one score per ROI plus a whole-brain score."""
import torch
import torch.nn as nn
from torch import Tensor


class MultiScaleROITemporalDiscriminator(nn.Module):
    """x: (B, n_rois, T) -> {"roi": (B, n_rois), "global": (B, 1)}. The temporal encoder is shared by all ROIs."""

    def __init__(self, n_rois: int = 400, temporal_features: int = 32):
        super().__init__()
        self.n_rois = n_rois
        self.temporal_features = temporal_features
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, temporal_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.roi_weight = nn.Parameter(torch.randn(n_rois, temporal_features) * 0.02)
        self.roi_bias = nn.Parameter(torch.zeros(n_rois))
        self.global_head = nn.Sequential(
            nn.Linear(n_rois * temporal_features, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        B, R, T = x.shape
        z = self.temporal_encoder(x.reshape(B * R, 1, T)).mean(dim=-1)
        z = z.reshape(B, R, self.temporal_features)
        roi_scores = torch.einsum("brf,rf->br", z, self.roi_weight) + self.roi_bias
        return {"roi": roi_scores, "global": self.global_head(z.flatten(start_dim=1))}
