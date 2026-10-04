"""One training epoch and one validation pass."""
import contextlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from moco.atlas import SchaeferAtlas, age_group_for_subject, roi_timeseries_batch
from moco.data.grade import denormalize_chunk
from moco.data.psc import psc_denormalise
from moco.evaluation.metrics import fmri_metrics
from moco.losses.gan import LossWeights, generator_loss, lsgan_discriminator_loss, r1_gradient_penalty
from moco.losses.roi import roi_cycle_loss, roi_discriminator_loss, roi_generator_loss
from moco.losses.sequence import fc_loss, temporal_consistency_loss
from moco.training.distributed import average_gradients
from moco.training.replay_buffer import ReplayBuffer


@dataclass
class TrainContext:
    """Everything train_one_epoch needs besides the batch stream."""
    model: torch.nn.Module            # possibly DDP-wrapped
    opt_G: torch.optim.Optimizer
    opt_D: torch.optim.Optimizer
    device: torch.device
    replay_a: ReplayBuffer
    replay_b: ReplayBuffer
    is_ddp: bool = False
    world_size: int = 1
    atlases: dict[str, SchaeferAtlas] | None = None
    roi_disc: torch.nn.Module | None = None
    opt_D_roi: torch.optim.Optimizer | None = None


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The model itself, without the DDP wrapper."""
    return model.module if isinstance(model, DDP) else model


def _first_scale_mean(scores: list[Tensor]) -> float:
    """Mean discriminator score at full resolution, for logging."""
    return scores[0].mean().item()


def _set_requires_grad(params: Iterable[Tensor], flag: bool) -> None:
    for p in params:
        p.requires_grad_(flag)


def _batch_inputs(batch: dict, device: torch.device, sequence_mode: bool) -> tuple[Tensor, Tensor]:
    """Flat batches are (B, T, X, Y, Z); a sequence item (1, S, T, ...) is squeezed so S acts as the batch."""
    if sequence_mode:
        assert batch["A"].shape[0] == 1, "sequence mode needs batch_size=1"
        return batch["A"].squeeze(0).to(device), batch["B"].squeeze(0).to(device)
    return batch["A"].to(device), batch["B"].to(device)


def _path_keys(batch: dict) -> tuple[str, str]:
    # the grade dataset names paths A_paths/B_paths, the PSC datasets path_A/path_B
    return ("A_paths", "B_paths") if "A_paths" in batch else ("path_A", "path_B")


def _sequence_losses(x_a: Tensor, x_hat_b: Tensor, weights: LossWeights, atlas: SchaeferAtlas | None,
                     fc_cfg: DictConfig) -> tuple[Tensor, dict]:
    """Weighted temporal-consistency and FC losses on one sequence.

    Args:
        x_a: (S, T, X, Y, Z) S consecutive input chunks.
        x_hat_b: (S, T, X, Y, Z) their corrections.
        weights: Loss weights; a term with weight 0 is not computed.
        atlas: Age-matched atlas for FC; None skips the FC term.
        fc_cfg: `train.sequence.fc` config (mask strategy and its parameters).

    Returns:
        (weighted sum, dict of unweighted terms plus "fc_stats").
    """
    total, terms = 0.0, {}
    if weights.temporal > 0:
        terms["temporal"] = temporal_consistency_loss(x_a, x_hat_b)
        total = total + weights.temporal * terms["temporal"]
    if weights.fc > 0 and atlas is not None:
        S, T = x_a.shape[:2]
        stats: dict = {}
        terms["fc"] = fc_loss(
            atlas.extract_roi_timeseries(x_a.reshape(S * T, *x_a.shape[2:])),
            atlas.extract_roi_timeseries(x_hat_b.reshape(S * T, *x_hat_b.shape[2:])),
            mask_strategy=fc_cfg.mask_strategy, threshold=fc_cfg.threshold, top_k=fc_cfg.top_k,
            percentile=fc_cfg.percentile, stats=stats,
        )
        terms["fc_stats"] = stats
        total = total + weights.fc * terms["fc"]
    return total, terms


def train_one_epoch(ctx: TrainContext, loader: DataLoader, weights: LossWeights, epoch: int, cfg: DictConfig,
                    sequence_mode: bool, show_progress: bool = True) -> dict[str, float]:
    """One pass over the training loader: a generator step per batch, a discriminator step every
    `d_update_every` batches.

    Args:
        ctx: Model, optimisers, replay buffers and optional ROI discriminator / atlases.
        loader: Training loader (grade or PSC dataset).
        weights: Loss weights for this epoch (after warmup).
        epoch: Epoch number, used to reshuffle the dataset.
        cfg: The `train` config group.
        sequence_mode: Batches are (1, S, T, ...) sequences instead of (B, T, ...) chunks.
        show_progress: Show the tqdm bar (main rank only).

    Returns:
        Epoch means of every logged term; D terms are averaged over D updates only.
    """
    model, raw = ctx.model, _unwrap(ctx.model)
    model.train()
    roi_cfg = cfg.roi
    sums: dict[str, float] = defaultdict(float)

    # the grade dataset reshuffles its larger domain per epoch; the PSC datasets advance an A queue
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    else:
        loader.dataset.on_epoch_start()

    n = {"batch": 0, "d": 0, "r1": 0, "fc": 0, "roi": 0}
    last_d_a = last_d_b = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True,
                disable=not show_progress)
    for i, batch in enumerate(pbar):
        if cfg.max_train_batches is not None and i >= cfg.max_train_batches:
            break
        path_a, path_b = _path_keys(batch)
        x_a, x_b = _batch_inputs(batch, ctx.device, sequence_mode)

        # generator step, discriminators frozen
        _set_requires_grad(raw.discriminator_parameters(), False)
        _set_requires_grad(raw.generator_parameters(), True)
        if ctx.roi_disc is not None:
            _set_requires_grad(ctx.roi_disc.parameters(), False)

        out = model(x_a, x_b)
        g = generator_loss(out, x_a, x_b, weights)
        total_loss = g["total"]
        if sequence_mode:
            atlas = (ctx.atlases[age_group_for_subject(batch["subject_id"][0])]
                     if ctx.atlases is not None else None)
            seq_total, seq_terms = _sequence_losses(x_a, out.x_hat_b, weights, atlas, cfg.sequence.fc)
            total_loss = total_loss + seq_total
            g.update(seq_terms)
        else:
            if ctx.roi_disc is not None:
                fake_roi = roi_timeseries_batch(out.x_hat_b, batch[path_a], ctx.atlases)
                g["roi_adv"] = roi_generator_loss(ctx.roi_disc(fake_roi), roi_cfg.lambda_roi)
                total_loss = total_loss + roi_cfg.w_adv * g["roi_adv"]
            if roi_cfg.w_cycle > 0 and ctx.atlases is not None:
                g["roi_cycle"] = roi_cycle_loss(roi_timeseries_batch(x_a, batch[path_a], ctx.atlases),
                                                roi_timeseries_batch(out.x_cycle_a, batch[path_a], ctx.atlases))
                total_loss = total_loss + roi_cfg.w_cycle * g["roi_cycle"]

        ctx.opt_G.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(raw.generator_parameters(), max_norm=cfg.max_grad_norm).item()
        ctx.opt_G.step()

        # discriminator step on reals vs buffered fakes
        if i % cfg.d_update_every == 0:
            _set_requires_grad(raw.discriminator_parameters(), True)
            if ctx.roi_disc is not None:
                _set_requires_grad(ctx.roi_disc.parameters(), True)
            fake_a = ctx.replay_a.push_and_pop(out.x_hat_a).to(ctx.device)
            fake_b = ctx.replay_b.push_and_pop(out.x_hat_b).to(ctx.device)
            loss_d_a = lsgan_discriminator_loss(raw.D_A(x_a), raw.D_A(fake_a), cfg.label_real, cfg.label_fake)
            loss_d_b = lsgan_discriminator_loss(raw.D_B(x_b), raw.D_B(fake_b), cfg.label_real, cfg.label_fake)

            ctx.opt_D.zero_grad()
            # D_A/D_B grads are averaged by hand below, so DDP's own all-reduce hooks are suppressed
            with model.no_sync() if ctx.is_ddp else contextlib.nullcontext():
                (loss_d_a + loss_d_b).backward()
                if cfg.r1.weight > 0 and n["d"] % cfg.r1.every == 0:
                    r1_a, r1_b = r1_gradient_penalty(raw.D_A, x_a), r1_gradient_penalty(raw.D_B, x_b)
                    # lazy regularisation: scale by `every` to keep the effective strength
                    ((cfg.r1.weight / 2.0) * (r1_a + r1_b) * cfg.r1.every).backward()
                    sums["D_r1"] += (r1_a + r1_b).item()
                    n["r1"] += 1
            if ctx.is_ddp:
                average_gradients(raw.discriminator_parameters(), ctx.world_size)
            ctx.opt_D.step()

            last_d_a, last_d_b = loss_d_a.item(), loss_d_b.item()
            sums["D_A"] += last_d_a
            sums["D_B"] += last_d_b
            sums["D_total"] += (loss_d_a + loss_d_b).item()
            n["d"] += 1

            if ctx.roi_disc is not None and not sequence_mode:
                real_roi = roi_timeseries_batch(x_b, batch[path_b], ctx.atlases)
                fake_roi = roi_timeseries_batch(out.x_hat_b.detach(), batch[path_a], ctx.atlases)
                ctx.opt_D_roi.zero_grad()
                loss_roi = roi_discriminator_loss(ctx.roi_disc(real_roi), ctx.roi_disc(fake_roi), roi_cfg.lambda_roi)
                loss_roi.backward()
                if ctx.is_ddp:  # roi_disc is not DDP-wrapped
                    average_gradients(ctx.roi_disc.parameters(), ctx.world_size)
                ctx.opt_D_roi.step()
                sums["D_roi_total"] += loss_roi.item()
                n["roi"] += 1

        for key in ("adv", "cyc", "idt"):
            sums[f"G_{key}"] += g[key].item()
        sums["G_total"] += total_loss.item()
        sums["grad_norm_G"] += grad_norm
        for key in ("temporal", "fc", "roi_adv", "roi_cycle"):
            if key in g:
                sums[f"G_{key}"] += g[key].item()
        if "fc_stats" in g:
            sums["G_fc_n_retained"] += g["fc_stats"]["n_retained"]
            sums["G_fc_retained_frac"] += g["fc_stats"]["retained_fraction"]
            n["fc"] += 1
        for name in ("score_real_a", "score_fake_a", "score_real_b", "score_fake_b"):
            sums[name] += _first_scale_mean(getattr(out, name))
        n["batch"] += 1
        pbar.set_postfix(G=f"{total_loss.item():.3f}", cyc=f"{g['cyc'].item():.3f}", idt=f"{g['idt'].item():.3f}",
                         D_A=f"{last_d_a:.3f}", D_B=f"{last_d_b:.3f}", gradG=f"{grad_norm:.2f}")

    # each term is averaged over the steps that actually produced it, not over all batches
    per_d_update = {"D_A", "D_B", "D_total"}
    divisor = {**dict.fromkeys(per_d_update, n["d"]), "D_r1": n["r1"], "D_roi_total": n["roi"],
               "G_fc_n_retained": n["fc"], "G_fc_retained_frac": n["fc"]}
    return {key: value / max(divisor.get(key, n["batch"]), 1) for key, value in sums.items()}


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, weights: LossWeights, device: torch.device, epoch: int,
             cfg: DictConfig, sequence_mode: bool, clean_grade: str | None = None,
             atlases: dict[str, SchaeferAtlas] | None = None, show_progress: bool = True) -> dict[str, float]:
    """Validation losses and input-vs-corrected fMRI metrics in BOLD units.

    Args:
        model: Unwrapped model.
        loader: Validation loader.
        weights: Loss weights (only needed to evaluate the sequence terms).
        device: Device of the model.
        epoch: Epoch number, for the progress bar.
        cfg: The `train` config group.
        sequence_mode: Batches are sequences.
        clean_grade: Name of the clean grade; adds val_grade1_identity_l1 if present.
        atlases: Age group -> atlas, for the FC term.
        show_progress: Show the tqdm bar.

    Returns:
        "val_*" means over batches. On the grade dataset also the residual |G(x) - x| per grade
        (including Grade 1, never translated in training) and mean D_B scores.
    """
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    n_batches = n_fc = 0
    residual_by_grade: dict[str, list[float]] = defaultdict(list)
    disc_scores: dict[str, list[float]] = defaultdict(list)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]", leave=False, dynamic_ncols=True, disable=not show_progress)
    for i, batch in enumerate(pbar):
        if cfg.max_val_batches is not None and i >= cfg.max_val_batches:
            break
        x_a, x_b = _batch_inputs(batch, device, sequence_mode)
        out = model(x_a, x_b)
        g = generator_loss(out, x_a, x_b, weights)
        sums["cyc"] += g["cyc"].item()
        sums["idt"] += g["idt"].item()

        if sequence_mode:
            atlas = atlases[age_group_for_subject(batch["subject_id"][0])] if atlases is not None else None
            _, seq_terms = _sequence_losses(x_a, out.x_hat_b, weights, atlas, cfg.sequence.fc)
            for key in ("temporal", "fc"):
                if key in seq_terms:
                    sums[key] += seq_terms[key].item()
            if "fc_stats" in seq_terms:
                sums["fc_n_retained"] += seq_terms["fc_stats"]["n_retained"]
                sums["fc_retained_frac"] += seq_terms["fc_stats"]["retained_fraction"]
                n_fc += 1

        if "A_meta" in batch:  # grade dataset: robust (median, scale) normalisation
            meta_a, meta_b = batch["A_meta"], batch["B_meta"]
            # G_B is also applied to clean Grade-1 input, a translation it never makes in training
            residuals = [(meta_a["grade"], out.x_hat_b - x_a), (meta_b["grade"], model.correct(x_b) - x_b)]
            for grades, residual in residuals:
                for grade, l1 in zip(grades, residual.abs().mean(dim=(1, 2, 3, 4)).tolist(), strict=True):
                    residual_by_grade[grade].append(l1)
            disc_scores["real_clean"].append(_first_scale_mean(out.score_real_b))
            disc_scores["raw_corrupted"].append(_first_scale_mean(model.D_B(x_a)))
            disc_scores["corrected"].append(_first_scale_mean(out.score_fake_b))

            brain = x_a != 0
            median, scale = meta_a["median"].to(device), meta_a["scale"].to(device)
            metrics = fmri_metrics(denormalize_chunk(x_a, median, scale, brain),
                                   denormalize_chunk(out.x_hat_b, median, scale, brain))
        else:  # PSC datasets
            mean_vol = (batch["mean_vol_A"].squeeze(0) if sequence_mode else batch["mean_vol_A"]).to(device)
            metrics = fmri_metrics(psc_denormalise(x_a, mean_vol), psc_denormalise(out.x_hat_b, mean_vol))
        for key, value in metrics.items():
            sums[key] += value
        n_batches += 1
        pbar.set_postfix(cyc=f"{g['cyc'].item():.3f}", tsnr=f"{metrics['tsnr_improvement']:+.3f}")

    results = {f"val_{key}": value / max(n_fc if key.startswith("fc_") else n_batches, 1)
               for key, value in sums.items()}
    for grade, values in residual_by_grade.items():
        results[f"val_residual_l1_{grade.replace(' ', '')}"] = float(np.mean(values))
    if clean_grade in residual_by_grade:
        results["val_grade1_identity_l1"] = float(np.mean(residual_by_grade[clean_grade]))
    for key, values in disc_scores.items():
        results[f"val_disc_score_{key}"] = float(np.mean(values))
    return results
