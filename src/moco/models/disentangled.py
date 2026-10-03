"""Disentangled CycleGAN treating the T timepoints of a chunk as input channels (legacy model)."""
import torch
import torch.nn as nn
from torch import Tensor

from moco.models.base import CycleGANBase
from moco.models.blocks import (
    AdaINResBlock3D,
    DiscConvBlock,
    ResBlock3D,
    StridedConvBlock,
    UpBlock3D,
    append_temporal_diffs,
)


class ContentEncoder(nn.Module):
    """x: (B, T, D, H, W) -> content (B, 6*base_ch, D/8, H/8, W/8). InstanceNorm keeps it intensity-invariant."""

    def __init__(self, in_channels: int, base_channels: int = 64, n_res_blocks: int = 5):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 6
        self.down1 = StridedConvBlock(in_channels, c1)
        self.down2 = StridedConvBlock(c1, c2)
        self.down3 = StridedConvBlock(c2, c3)
        self.res_blocks = nn.Sequential(*[ResBlock3D(c3) for _ in range(n_res_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        return self.res_blocks(self.down3(self.down2(self.down1(x))))


class ArtefactEncoder(nn.Module):
    """x: (B, T, D, H, W) -> a_global (B, global_code_dim), a_spatial (B, spatial_code_ch, D/8, H/8, W/8)."""

    def __init__(self, in_channels: int, base_channels: int = 64, global_code_dim: int = 64,
                 spatial_code_ch: int = 32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.down1 = StridedConvBlock(in_channels, c1, use_norm=False)
        self.down2 = StridedConvBlock(c1, c2, use_norm=False)
        self.down3 = StridedConvBlock(c2, c3, use_norm=False)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 2, global_code_dim),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv3d(c3, spatial_code_ch, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.down3(self.down2(self.down1(x)))
        return self.global_mlp(self.global_pool(x)), self.spatial_branch(x)


def _upsampling_head(content_ch: int, out_channels: int) -> dict[str, nn.Module]:
    ch1, ch2, ch3 = content_ch // 2, content_ch // 4, content_ch // 8
    return dict(
        up1=UpBlock3D(content_ch, ch1),
        up2=UpBlock3D(ch1, ch2),
        up3=UpBlock3D(ch2, ch3),
        out_conv=nn.Sequential(nn.Conv3d(ch3, out_channels, 3, padding=1, bias=True), nn.Tanh()),
    )


class MotionFreeDecoder(nn.Module):
    """G_B: content (B, C, d, h, w) -> chunk (B, T, D, H, W). Sees no artefact information."""

    def __init__(self, content_ch: int, out_channels: int, n_res_blocks: int = 4):
        super().__init__()
        self.res_blocks = nn.Sequential(*[ResBlock3D(content_ch) for _ in range(n_res_blocks)])
        for name, module in _upsampling_head(content_ch, out_channels).items():
            setattr(self, name, module)

    def forward(self, content: Tensor) -> Tensor:
        x = self.up3(self.up2(self.up1(self.res_blocks(content))))
        return self.out_conv(x)


class MotionCorruptedDecoder(nn.Module):
    """G_A: content + artefact codes -> corrupted chunk (B, T, D, H, W).

    a_global modulates the bottleneck via AdaIN; a_spatial is concatenated and merged before upsampling."""

    def __init__(self, content_ch: int, artefact_dim: int, spatial_art_ch: int, out_channels: int,
                 n_adain_blocks: int = 4):
        super().__init__()
        self.adain_blocks = nn.ModuleList(
            [AdaINResBlock3D(content_ch, artefact_dim) for _ in range(n_adain_blocks)]
        )
        self.spatial_merge = nn.Sequential(
            nn.Conv3d(content_ch + spatial_art_ch, content_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(content_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        for name, module in _upsampling_head(content_ch, out_channels).items():
            setattr(self, name, module)

    def forward(self, content: Tensor, a_global: Tensor, a_spatial: Tensor) -> Tensor:
        x = content
        for block in self.adain_blocks:
            x = block(x, a_global)
        x = self.spatial_merge(torch.cat([x, a_spatial], dim=1))
        return self.out_conv(self.up3(self.up2(self.up1(x))))


class MultiScalePatchDiscriminator3D(nn.Module):
    """x: (B, T, D, H, W) -> list of patch score maps (B, 1, d, h, w), finest first."""

    def __init__(self, in_timepoints: int, base_ch: int = 64, num_scales: int = 2,
                 use_temporal_diffs: bool = False):
        super().__init__()
        self.use_temporal_diffs = use_temporal_diffs
        self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)
        in_ch = 2 * in_timepoints - 1 if use_temporal_diffs else in_timepoints
        c = base_ch
        self.cnns = nn.ModuleList([
            nn.Sequential(
                DiscConvBlock(in_ch, c, use_norm=False),
                DiscConvBlock(c, c * 2),
                DiscConvBlock(c * 2, c * 4),
                DiscConvBlock(c * 4, c * 8),
                nn.utils.spectral_norm(nn.Conv3d(c * 8, 1, kernel_size=3, padding=1, bias=True)),
            )
            for _ in range(num_scales)
        ])

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.use_temporal_diffs:
            x = append_temporal_diffs(x)
        outputs = []
        for cnn in self.cnns:
            outputs.append(cnn(x))
            x = self.downsample(x)
        return outputs


class DisentangledCycleGAN(CycleGANBase):
    """Channel-as-time model: chunks (B, T, D, H, W) with T fed as conv input channels."""

    def __init__(self, in_timepoints: int = 20, content_base_ch: int = 64, content_n_res: int = 5,
                 artefact_base_ch: int = 64, global_code_dim: int = 64, spatial_code_ch: int = 32,
                 disc_base_ch: int = 64, num_disc_scales: int = 2, disc_temporal_diffs: bool = False,
                 residual: bool = False):
        super().__init__()
        self.residual = residual
        content_ch = content_base_ch * 6
        self.E_c = ContentEncoder(in_timepoints, content_base_ch, content_n_res)
        self.E_a = ArtefactEncoder(in_timepoints, artefact_base_ch, global_code_dim, spatial_code_ch)
        self.G_B = MotionFreeDecoder(content_ch, in_timepoints)
        self.G_A = MotionCorruptedDecoder(content_ch, global_code_dim, spatial_code_ch, in_timepoints)
        self.D_B = MultiScalePatchDiscriminator3D(in_timepoints, disc_base_ch, num_disc_scales,
                                                  disc_temporal_diffs)
        self.D_A = MultiScalePatchDiscriminator3D(in_timepoints, disc_base_ch, num_disc_scales,
                                                  disc_temporal_diffs)
