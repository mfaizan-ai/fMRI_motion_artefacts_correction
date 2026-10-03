"""3D convolutional building blocks shared by the channel-as-time (disentangled) model."""
import torch
import torch.nn as nn
from torch import Tensor


class ResBlock3D(nn.Module):
    """Residual block with InstanceNorm. x: (B, C, D, H, W) -> same shape."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class StridedConvBlock(nn.Module):
    """Stride-2 conv halving each spatial dim.

    InstanceNorm (content encoder) strips per-channel intensity statistics; BatchNorm
    (artefact encoder) keeps them, since intensity shifts are the motion signature.
    """

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        norm = nn.InstanceNorm3d(out_ch, affine=True) if use_norm else nn.BatchNorm3d(out_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm),
            norm,
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock3D(nn.Module):
    """Trilinear x2 upsample + conv (avoids transposed-conv checkerboarding)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class AdaIN3D(nn.Module):
    """AdaIN(x, a) = sigma(a) * IN(x) + mu(a). x: (B, C, D, H, W), a_global: (B, artefact_dim)."""

    def __init__(self, channels: int, artefact_dim: int = 64):
        super().__init__()
        self.norm = nn.InstanceNorm3d(channels, affine=False)
        self.mlp_mean = nn.Linear(artefact_dim, channels)
        self.mlp_std = nn.Linear(artefact_dim, channels)

    def forward(self, x: Tensor, a_global: Tensor) -> Tensor:
        mean = self.mlp_mean(a_global)[:, :, None, None, None]
        std = self.mlp_std(a_global)[:, :, None, None, None]
        return std * self.norm(x) + mean


class AdaINResBlock3D(nn.Module):
    """Residual block with both norms replaced by AdaIN on the same artefact code."""

    def __init__(self, channels: int, artefact_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.adain1 = AdaIN3D(channels, artefact_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.adain2 = AdaIN3D(channels, artefact_dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor, a_global: Tensor) -> Tensor:
        residual = x
        x = self.act1(self.adain1(self.conv1(x), a_global))
        x = self.adain2(self.conv2(x), a_global)
        return self.act2(x + residual)


class DiscConvBlock(nn.Module):
    """Spectral-norm stride-2 conv; no norm on the first block (pix2pix convention)."""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers = [nn.utils.spectral_norm(
            nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)
        )]
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


def append_temporal_diffs(x: Tensor) -> Tensor:
    """x: (B, T, D, H, W) -> (B, 2T-1, D, H, W): volumes followed by frame-to-frame differences."""
    return torch.cat([x, x[:, 1:] - x[:, :-1]], dim=1)
