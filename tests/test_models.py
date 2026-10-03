import pytest
import torch

from moco.models.convlstm import ConvLSTM3D
from moco.models.disentangled import DisentangledCycleGAN
from moco.models.spatiotemporal import SpatioTemporalCycleGAN

SPATIAL = (64, 64, 64)  # coarsest discriminator scale needs >1 voxel for InstanceNorm
SMALL = dict(content_base_ch=8, content_n_res=1, artefact_base_ch=8, global_code_dim=8, spatial_code_ch=4,
             disc_base_ch=8, num_disc_scales=2, residual=True)


def _chunk(b=2, t=5):
    x = torch.randn(b, t, *SPATIAL)
    x[..., :4, :, :] = 0  # background
    return x


@pytest.mark.parametrize("use_convlstm", [False, True])
def test_spatiotemporal_shapes(use_convlstm):
    model = SpatioTemporalCycleGAN(spatial_dims=SPATIAL, use_convlstm=use_convlstm, **SMALL).eval()
    x_a, x_b = _chunk(), _chunk()
    with torch.no_grad():
        out = model(x_a, x_b)
    assert out.x_hat_b.shape == x_a.shape and out.x_cycle_a.shape == x_a.shape
    assert out.c_a.shape == (2, 5, 48, 8, 8, 8)
    assert out.a_global.shape == (2, 5, 8)
    assert [s.shape for s in out.score_fake_b] == [(2, 1, 4, 4, 4), (2, 1, 2, 2, 2)]


def test_disentangled_shapes():
    model = DisentangledCycleGAN(in_timepoints=5, **SMALL).eval()
    x_a = _chunk()
    with torch.no_grad():
        out = model(x_a, _chunk())
    assert out.x_hat_b.shape == x_a.shape
    assert out.c_a.shape == (2, 48, 8, 8, 8)


def test_residual_correction_keeps_background_zero():
    model = SpatioTemporalCycleGAN(spatial_dims=SPATIAL, **SMALL).eval()
    x = _chunk()
    with torch.no_grad():
        corrected = model.correct(x)
    assert torch.all(corrected[x == 0] == 0)


def test_convlstm_is_causal():
    lstm = ConvLSTM3D(filters=4, kernel_size=3, pad="same", d=(1, 5, 3, 4, 4, 4))
    x = torch.randn(1, 5, 3, 4, 4, 4)
    changed = x.clone()
    changed[:, -1] += 1
    with torch.no_grad():
        assert torch.allclose(lstm(x)[:, :-1], lstm(changed)[:, :-1])
