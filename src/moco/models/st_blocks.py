"""Factorized R(3+1)D blocks: a 3D conv per volume, then mixing across the T timepoints.

Tensors are (B, T, C, D, H, W) inside these blocks.
"""
import torch
import torch.nn as nn
from torch import Tensor

from moco.models.blocks import AdaIN3D
from moco.models.convlstm import ConvLSTM3D


def per_volume(module: nn.Module, x: Tensor) -> Tensor:
    """Apply a 3D module to each timepoint: (B, T, C, D, H, W) -> (B, T, C', D', H', W')."""
    B, T = x.shape[:2]
    y = module(x.reshape(B * T, *x.shape[2:]))
    return y.reshape(B, T, *y.shape[1:])


class TemporalConv1D(nn.Module):
    """Residual Conv1d along T at every voxel (each voxel's time series mixed independently)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, D, H, W = x.shape
        xt = x.permute(0, 3, 4, 5, 2, 1).reshape(B * D * H * W, C, T)
        xt = self.act(self.norm(self.conv(xt)))
        return x + xt.reshape(B, D, H, W, C, T).permute(0, 5, 4, 1, 2, 3)


class ConvLSTMTemporal(nn.Module):
    """Residual bidirectional ConvLSTM mixer, a drop-in for TemporalConv1D.

    Forward and backward ConvLSTM3D (hidden C/4 each) -> 1x1x1 conv back to C -> x + scale * that,
    with scale learnable and initialised at 0.1 so the block starts close to identity.
    """

    def __init__(self, channels: int, spatial: tuple, kernel_size: int = 3):
        super().__init__()
        hidden = channels // 4
        d = (1, 1, channels, *spatial)
        self.fwd = ConvLSTM3D(hidden, kernel_size, "same", d)
        self.bwd = ConvLSTM3D(hidden, kernel_size, "same", d)
        self.proj = nn.Conv3d(2 * hidden, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        h = torch.cat([self.fwd(x), self.bwd(x.flip(1)).flip(1)], dim=2)
        return x + self.scale * per_volume(self.proj, h)


def make_temporal(channels: int, temporal_k: int, convlstm_spatial: tuple | None = None) -> nn.Module:
    """TemporalConv1D, or ConvLSTMTemporal when the block's spatial shape (D, H, W) is given."""
    if convlstm_spatial is None:
        return TemporalConv1D(channels, temporal_k)
    return ConvLSTMTemporal(channels, convlstm_spatial)


class FactorizedDownBlock(nn.Module):
    """Stride-2 conv per volume (InstanceNorm, or BatchNorm for the artefact encoder) + optional mixing."""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True, temporal_k: int = 3,
                 use_temporal: bool = False, convlstm_spatial: tuple | None = None):
        super().__init__()
        norm = nn.InstanceNorm3d(out_ch, affine=True) if use_norm else nn.BatchNorm3d(out_ch)
        self.spatial = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm),
            norm,
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_temporal = use_temporal
        self.temporal = make_temporal(out_ch, temporal_k, convlstm_spatial) if use_temporal else None

    def forward(self, x: Tensor) -> Tensor:
        x = per_volume(self.spatial, x)
        return self.temporal(x) if self.use_temporal else x


class FactorizedResBlock(nn.Module):
    """Spatial residual block per volume, then temporal mixing (always on)."""

    def __init__(self, channels: int, temporal_k: int = 3, convlstm_spatial: tuple | None = None):
        super().__init__()
        self.spatial_block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.temporal = make_temporal(channels, temporal_k, convlstm_spatial)

    def forward(self, x: Tensor) -> Tensor:
        x = per_volume(lambda v: self.act(v + self.spatial_block(v)), x)
        return self.temporal(x)


class FactorizedUpBlock(nn.Module):
    """Trilinear x2 upsample + conv per volume; temporal mixing off by default (too costly at high res)."""

    def __init__(self, in_ch: int, out_ch: int, temporal_k: int = 3, use_temporal: bool = False,
                 convlstm_spatial: tuple | None = None):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_temporal = use_temporal
        self.temporal = make_temporal(out_ch, temporal_k, convlstm_spatial) if use_temporal else None

    def forward(self, x: Tensor) -> Tensor:
        x = per_volume(self.spatial, x)
        return self.temporal(x) if self.use_temporal else x


class FactorizedAdaINResBlock(nn.Module):
    """AdaIN residual block per volume, each timepoint modulated by its own artefact code, then mixing.

    x: (B, T, C, D, H, W), a_global: (B, T, artefact_dim)."""

    def __init__(self, channels: int, artefact_dim: int = 64, temporal_k: int = 3,
                 convlstm_spatial: tuple | None = None):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.adain1 = AdaIN3D(channels, artefact_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.adain2 = AdaIN3D(channels, artefact_dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.temporal = make_temporal(channels, temporal_k, convlstm_spatial)

    def forward(self, x: Tensor, a_global: Tensor) -> Tensor:
        B, T, C, D, H, W = x.shape
        ag = a_global.reshape(B * T, -1)
        xs = x.reshape(B * T, C, D, H, W)
        h = self.act1(self.adain1(self.conv1(xs), ag))
        h = self.adain2(self.conv2(h), ag)
        xs = self.act2(h + xs)
        return self.temporal(xs.reshape(B, T, C, D, H, W))


class FactorizedDiscBlock(nn.Module):
    """Spectral-norm stride-2 conv per volume + optional temporal mixing."""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True, temporal_k: int = 3,
                 use_temporal: bool = False, convlstm_spatial: tuple | None = None):
        super().__init__()
        layers = [nn.utils.spectral_norm(nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_norm))]
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.spatial = nn.Sequential(*layers)
        self.use_temporal = use_temporal
        self.temporal = make_temporal(out_ch, temporal_k, convlstm_spatial) if use_temporal else None

    def forward(self, x: Tensor) -> Tensor:
        x = per_volume(self.spatial, x)
        return self.temporal(x) if self.use_temporal else x
