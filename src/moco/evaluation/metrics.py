"""fMRI quality metrics on chunks (B, T, X, Y, Z) in BOLD units, restricted to a ground-truth brain mask."""
import torch
from torch import Tensor


def _expand_mask(mask: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """mask: (X, Y, Z) or (B, X, Y, Z) -> (B, X, Y, Z) bool, plus per-sample voxel counts (B,)."""
    mask = torch.as_tensor(mask, device=x.device, dtype=torch.bool)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    if mask.shape != (x.shape[0], *x.shape[-3:]):
        raise ValueError(f"mask {tuple(mask.shape)} incompatible with {tuple(x.shape)}")
    voxel_count = mask.sum(dim=(-3, -2, -1))
    if torch.any(voxel_count == 0):
        raise ValueError("every brain mask needs at least one voxel")
    return mask, voxel_count


def dvars(x: Tensor, mask: Tensor | None = None) -> float:
    """Mean over frames of sqrt(mean_brain((x_t - x_{t-1})^2))."""
    diff_sq = (x[:, 1:] - x[:, :-1]).float().square()
    if mask is None:
        mean_sq = diff_sq.mean(dim=(-3, -2, -1))
    else:
        mask, voxel_count = _expand_mask(mask, x)
        mean_sq = (diff_sq * mask.to(diff_sq.dtype)[:, None]).sum(dim=(-3, -2, -1)) / voxel_count[:, None]
    return mean_sq.sqrt().mean().item()


def tsnr(x: Tensor, mask: Tensor | None = None) -> float:
    """Mean over brain voxels of |temporal mean| / temporal std; near-constant voxels excluded."""
    mean, std = x.mean(dim=1), x.std(dim=1)
    valid = std > 1e-3
    if mask is not None:
        valid = valid & _expand_mask(mask, x)[0]
    if valid.sum() == 0:
        return 0.0
    return (mean[valid].abs() / std[valid]).mean().item()


def global_signal_std(x: Tensor, mask: Tensor | None = None) -> float:
    """Temporal std of the brain-mean signal, averaged over the batch."""
    if mask is None:
        gs = x.mean(dim=(-3, -2, -1))
    else:
        mask, voxel_count = _expand_mask(mask, x)
        gs = (x * mask.to(x.dtype)[:, None]).sum(dim=(-3, -2, -1)) / voxel_count[:, None]
    return gs.std(dim=1).mean().item()


def spatial_smoothness(x: Tensor) -> float:
    """Mean absolute finite difference over the three spatial axes (lower = smoother)."""
    grads = [x.diff(dim=d).abs().mean() for d in (2, 3, 4)]
    return (sum(grads) / 3.0).item()


def fmri_metrics(x_input: Tensor, x_corrected: Tensor) -> dict[str, float]:
    """Input vs corrected metrics. The brain mask comes from the real input (x_input[:, 0] != 0), never
    from the output, whose background is not guaranteed to be 0."""
    xi, xc = x_input.detach().cpu(), x_corrected.detach().cpu()
    mask = xi[:, 0] != 0
    d_in, d_out = dvars(xi, mask), dvars(xc, mask)
    t_in, t_out = tsnr(xi, mask), tsnr(xc, mask)
    g_in, g_out = global_signal_std(xi, mask), global_signal_std(xc, mask)
    s_in, s_out = spatial_smoothness(xi), spatial_smoothness(xc)
    return {
        "dvars_input": d_in, "dvars_corrected": d_out, "dvars_improvement": d_in - d_out,
        "tsnr_input": t_in, "tsnr_corrected": t_out, "tsnr_improvement": t_out - t_in,
        "gs_std_input": g_in, "gs_std_corrected": g_out, "gs_std_improvement": g_in - g_out,
        "smoothness_input": s_in, "smoothness_corrected": s_out, "smoothness_ratio": s_out / (s_in + 1e-8),
    }


def val_score(metrics: dict[str, float], cfg) -> float:
    """Model-selection score: tSNR, DVARS and global-signal improvements, each scaled by its expected
    ideal improvement, minus a penalty when spatial smoothness rises above cfg.smoothness_tolerance.

    Note: nothing here penalises suppressed temporal variance, which this score can reward."""
    score = (metrics["tsnr_improvement"] / cfg.tsnr_scale
             + metrics["dvars_improvement"] / cfg.dvars_scale
             + metrics["gs_std_improvement"] / cfg.gs_std_scale)
    if metrics["smoothness_ratio"] > cfg.smoothness_tolerance:
        score -= (metrics["smoothness_ratio"] - cfg.smoothness_tolerance) * cfg.smoothness_penalty
    return score
