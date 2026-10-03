"""Factorized R(3+1)D disentangled CycleGAN: each volume is a 1-channel 3D image, time mixed at the bottleneck."""
import torch
import torch.nn as nn
from torch import Tensor

from moco.models.base import CycleGANBase
from moco.models.st_blocks import (
    FactorizedAdaINResBlock,
    FactorizedDiscBlock,
    FactorizedDownBlock,
    FactorizedResBlock,
    FactorizedUpBlock,
    per_volume,
)


class STContentEncoder(nn.Module):
    """x: (B, T, D, H, W) -> content (B, T, 6*base_ch, D/8, H/8, W/8). Down blocks are spatial only."""

    def __init__(self, base_ch: int = 64, n_res: int = 5, temporal_k: int = 3,
                 convlstm_spatial: tuple | None = None):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 6
        self.down1 = FactorizedDownBlock(1, c1, temporal_k=temporal_k)
        self.down2 = FactorizedDownBlock(c1, c2, temporal_k=temporal_k)
        self.down3 = FactorizedDownBlock(c2, c3, temporal_k=temporal_k)
        self.res = nn.ModuleList(
            [FactorizedResBlock(c3, temporal_k, convlstm_spatial) for _ in range(n_res)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.down3(self.down2(self.down1(x.unsqueeze(2))))
        for block in self.res:
            x = block(x)
        return x


class STArtefactEncoder(nn.Module):
    """x: (B, T, D, H, W) -> a_global (B, T, global_code_dim), a_spatial (B, T, spatial_code_ch, D/8, H/8, W/8).

    BatchNorm keeps intensity statistics; no temporal mixing, so every TR gets its own artefact code."""

    def __init__(self, base_ch: int = 64, global_code_dim: int = 64, spatial_code_ch: int = 32,
                 temporal_k: int = 3):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.down1 = FactorizedDownBlock(1, c1, use_norm=False, temporal_k=temporal_k)
        self.down2 = FactorizedDownBlock(c1, c2, use_norm=False, temporal_k=temporal_k)
        self.down3 = FactorizedDownBlock(c2, c3, use_norm=False, temporal_k=temporal_k)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch * 2, global_code_dim),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv3d(c3, spatial_code_ch, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.down3(self.down2(self.down1(x.unsqueeze(2))))
        a_global = per_volume(lambda v: self.global_mlp(self.global_pool(v)), x)
        return a_global, per_volume(self.spatial_branch, x)


def _out_conv(ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(ch, 1, kernel_size=3, padding=1, bias=True), nn.Tanh())


class STMotionFreeDecoder(nn.Module):
    """G_B: content (B, T, C, d, h, w) -> chunk (B, T, D, H, W). Temporal mixing only in the bottleneck blocks."""

    def __init__(self, content_ch: int = 384, n_res: int = 4, temporal_k: int = 3,
                 convlstm_spatial: tuple | None = None):
        super().__init__()
        ch1, ch2, ch3 = content_ch // 2, content_ch // 4, content_ch // 8
        self.res = nn.ModuleList(
            [FactorizedResBlock(content_ch, temporal_k, convlstm_spatial) for _ in range(n_res)]
        )
        self.up1 = FactorizedUpBlock(content_ch, ch1, temporal_k=temporal_k)
        self.up2 = FactorizedUpBlock(ch1, ch2, temporal_k=temporal_k)
        self.up3 = FactorizedUpBlock(ch2, ch3, temporal_k=temporal_k)
        self.out_conv = _out_conv(ch3)

    def forward(self, content: Tensor) -> Tensor:
        x = content
        for block in self.res:
            x = block(x)
        x = self.up3(self.up2(self.up1(x)))
        return per_volume(self.out_conv, x).squeeze(2)


class STMotionCorruptedDecoder(nn.Module):
    """G_A: content (B, T, C, d, h, w) + a_global (B, T, A) + a_spatial (B, T, S, d, h, w) -> (B, T, D, H, W)."""

    def __init__(self, content_ch: int = 384, artefact_dim: int = 64, spatial_art_ch: int = 32,
                 n_adain: int = 4, temporal_k: int = 3, convlstm_spatial: tuple | None = None):
        super().__init__()
        ch1, ch2, ch3 = content_ch // 2, content_ch // 4, content_ch // 8
        self.adain_blocks = nn.ModuleList([
            FactorizedAdaINResBlock(content_ch, artefact_dim, temporal_k, convlstm_spatial)
            for _ in range(n_adain)
        ])
        self.spatial_merge = nn.Sequential(
            nn.Conv3d(content_ch + spatial_art_ch, content_ch, 1, bias=False),
            nn.InstanceNorm3d(content_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up1 = FactorizedUpBlock(content_ch, ch1, temporal_k=temporal_k)
        self.up2 = FactorizedUpBlock(ch1, ch2, temporal_k=temporal_k)
        self.up3 = FactorizedUpBlock(ch2, ch3, temporal_k=temporal_k)
        self.out_conv = _out_conv(ch3)

    def forward(self, content: Tensor, a_global: Tensor, a_spatial: Tensor) -> Tensor:
        x = content
        for block in self.adain_blocks:
            x = block(x, a_global)
        x = per_volume(self.spatial_merge, torch.cat([x, a_spatial], dim=2))
        x = self.up3(self.up2(self.up1(x)))
        return per_volume(self.out_conv, x).squeeze(2)


class STScaleCNN(nn.Module):
    """One PatchGAN scale: (B, T, 1, D, H, W) -> (B, 1, d, h, w), patch scores averaged over T.

    Only the last (coarsest, cheapest) block mixes time."""

    def __init__(self, base_ch: int, temporal_k: int, convlstm_spatial: tuple | None = None):
        super().__init__()
        c = base_ch
        self.blocks = nn.ModuleList([
            FactorizedDiscBlock(1, c, use_norm=False, temporal_k=temporal_k),
            FactorizedDiscBlock(c, c * 2, temporal_k=temporal_k),
            FactorizedDiscBlock(c * 2, c * 4, temporal_k=temporal_k),
            FactorizedDiscBlock(c * 4, c * 8, temporal_k=temporal_k, use_temporal=True,
                                convlstm_spatial=convlstm_spatial),
        ])
        self.out_conv = nn.utils.spectral_norm(nn.Conv3d(c * 8, 1, kernel_size=3, padding=1, bias=True))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return per_volume(self.out_conv, x).mean(dim=1)


class STMultiScaleDiscriminator(nn.Module):
    """x: (B, T, D, H, W) -> list of (B, 1, d, h, w) score maps, finest first."""

    def __init__(self, base_ch: int = 64, num_scales: int = 2, temporal_k: int = 3,
                 convlstm_input_spatial: tuple | None = None):
        super().__init__()
        self.num_scales = num_scales
        self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)
        # ConvLSTM peepholes need each scale's last-block grid: the pool gives ceil(n/2) between
        # scales and four k4/s2/p1 blocks give n // 16
        scale_spatial = [None] * num_scales
        if convlstm_input_spatial is not None:
            dims = tuple(convlstm_input_spatial)
            for i in range(num_scales):
                scale_spatial[i] = tuple(n // 16 for n in dims)
                dims = tuple((n + 1) // 2 for n in dims)
        self.cnns = nn.ModuleList([STScaleCNN(base_ch, temporal_k, s) for s in scale_spatial])

    def forward(self, x: Tensor) -> list[Tensor]:
        x = x.unsqueeze(2)
        outputs = []
        for i, cnn in enumerate(self.cnns):
            outputs.append(cnn(x))
            if i < self.num_scales - 1:
                x = per_volume(self.downsample, x)
        return outputs


class SpatioTemporalCycleGAN(CycleGANBase):
    """Factorized R(3+1)D CycleGAN. With use_convlstm every temporal mixer is a bidirectional ConvLSTM."""

    def __init__(self, spatial_dims: tuple = (64, 72, 56), content_base_ch: int = 64, content_n_res: int = 5,
                 artefact_base_ch: int = 64, global_code_dim: int = 64, spatial_code_ch: int = 32,
                 disc_base_ch: int = 64, num_disc_scales: int = 2, temporal_k: int = 3,
                 residual: bool = False, use_convlstm: bool = False):
        super().__init__()
        self.residual = residual
        content_ch = content_base_ch * 6
        # three k4/s2/p1 down blocks -> bottleneck (D//8, H//8, W//8)
        bottleneck = tuple(n // 8 for n in spatial_dims) if use_convlstm else None
        disc_spatial = tuple(spatial_dims) if use_convlstm else None

        self.E_c = STContentEncoder(content_base_ch, content_n_res, temporal_k, bottleneck)
        self.E_a = STArtefactEncoder(artefact_base_ch, global_code_dim, spatial_code_ch, temporal_k)
        self.G_B = STMotionFreeDecoder(content_ch, n_res=4, temporal_k=temporal_k, convlstm_spatial=bottleneck)
        self.G_A = STMotionCorruptedDecoder(content_ch, global_code_dim, spatial_code_ch, n_adain=4,
                                            temporal_k=temporal_k, convlstm_spatial=bottleneck)
        self.D_B = STMultiScaleDiscriminator(disc_base_ch, num_disc_scales, temporal_k, disc_spatial)
        self.D_A = STMultiScaleDiscriminator(disc_base_ch, num_disc_scales, temporal_k, disc_spatial)
