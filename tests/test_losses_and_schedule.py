import torch

from moco.losses.sequence import fc_strength_mask, pearson_corr_matrix
from moco.training.schedule import build_scheduler


def test_fc_mask_strategies():
    fc = torch.tensor([[1.0, 0.9, 0.1], [0.9, 1.0, -0.5], [0.1, -0.5, 1.0]])
    assert fc_strength_mask(fc, "threshold", threshold=0.4).sum() == 2  # |r| 0.9 and 0.5, upper triangle only
    top = fc_strength_mask(fc, "topk", top_k=1)
    assert top.sum() == 1 and top[0, 1]
    assert not torch.any(torch.diagonal(fc_strength_mask(fc, "percentile", percentile=0)))


def test_pearson_handles_constant_roi():
    x = torch.randn(50, 3)
    x[:, 2] = 7.0
    corr = pearson_corr_matrix(x)
    assert torch.allclose(torch.diagonal(corr), torch.ones(3), atol=1e-5)
    assert corr[2, 0] == 0 and corr[0, 2] == 0


def test_lr_schedule_phases():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1.0)
    sched = build_scheduler(opt, n_epochs=20, warmup=2, warmup_start_factor=0.01, final_factor=1e-6)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == 0.01 and lrs[2] == 1.0 and lrs[12] == 1.0  # warmup, then constant until warmup + half
    assert lrs[13] < 1.0 and lrs[19] < lrs[13]                 # then linear decay
