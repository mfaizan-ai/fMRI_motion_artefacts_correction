"""Full training run from a resolved Hydra config."""
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as torch_dist
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from moco.atlas import load_atlases
from moco.data.loaders import build_train_val_loaders
from moco.evaluation.metrics import val_score
from moco.losses.gan import LossWeights
from moco.models.build import build_model
from moco.models.roi_discriminator import MultiScaleROITemporalDiscriminator
from moco.training.checkpoint import load_checkpoint, save_checkpoint
from moco.training.distributed import setup_distributed
from moco.training.loop import TrainContext, train_one_epoch, validate
from moco.training.replay_buffer import ReplayBuffer
from moco.training.schedule import build_scheduler, epoch_loss_weights
from moco.utils.logs import CSVLogger, setup_logging
from moco.utils.run_info import save_run_info
from moco.utils.seed import seed_everything

log = logging.getLogger(__name__)

TRAIN_FIELDS = ["epoch", "lr_G", "lr_D", "G_adv", "G_cyc", "G_idt", "G_total", "D_A", "D_B", "D_total", "D_r1",
                "grad_norm_G", "score_real_a", "score_fake_a", "score_real_b", "score_fake_b"]
VAL_FIELDS = ["epoch", "val_cyc", "val_idt"] + [
    f"val_{m}_{s}" for m in ("dvars", "tsnr", "gs_std") for s in ("input", "corrected", "improvement")
] + ["val_smoothness_input", "val_smoothness_corrected", "val_smoothness_ratio", "val_score"]


def _csv_fields(cfg: DictConfig) -> tuple[list[str], list[str]]:
    train_fields, val_fields = list(TRAIN_FIELDS), list(VAL_FIELDS)
    if cfg.data.sequence_mode:
        train_fields += ["G_temporal", "G_fc", "G_fc_n_retained", "G_fc_retained_frac"]
        val_fields += ["val_temporal", "val_fc", "val_fc_n_retained", "val_fc_retained_frac"]
    if cfg.train.roi.enabled:
        train_fields += ["G_roi_adv", "D_roi_total"]
    if cfg.train.roi.w_cycle > 0:
        train_fields += ["G_roi_cycle"]
    if cfg.data.name == "grade":
        grades = [cfg.data.clean_grade, *cfg.data.grades_b]
        val_fields += [f"val_residual_l1_{g.replace(' ', '')}" for g in grades]
        val_fields += ["val_grade1_identity_l1", "val_disc_score_real_clean", "val_disc_score_raw_corrupted",
                       "val_disc_score_corrected"]
    return train_fields, val_fields


def _check_config(cfg: DictConfig) -> None:
    if cfg.data.sequence_mode and (cfg.train.roi.enabled or cfg.train.roi.w_cycle > 0):
        raise ValueError("ROI discriminator / ROI cycle loss are chunk-level only, not for sequence mode")
    if cfg.data.name == "grade_fc":
        raise NotImplementedError("grade_fc batches (B, K, T, ...) need the FC-preservation step, not added yet")
    if cfg.model.use_convlstm and cfg.model.name != "spatiotemporal":
        raise ValueError("use_convlstm needs the spatiotemporal model")


def run_training(cfg: DictConfig) -> None:
    _check_config(cfg)
    dist = setup_distributed(cfg.train.ddp_timeout_minutes)
    # a different seed per rank diversifies sampling across GPUs
    seed_everything(cfg.seed + dist.rank, cfg.deterministic)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log" if dist.is_main else None, enabled=dist.is_main)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if dist.is_main:
        save_run_info(run_dir, cfg)
        log.info("run dir %s | device %s | world size %d", run_dir, dist.device, dist.world_size)

    use_wandb = dist.is_main and cfg.wandb.enabled
    if use_wandb:
        os.environ.setdefault("WANDB_MODE", cfg.wandb.mode)
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.run_name, dir=str(run_dir),
                   config=resolved, resume="allow")

    train_fields, val_fields = _csv_fields(cfg)
    train_csv = CSVLogger(run_dir / "train_losses.csv", train_fields)
    val_csv = CSVLogger(run_dir / "val_metrics.csv", val_fields)

    tcfg = cfg.train
    loaders = build_train_val_loaders(cfg.data, tcfg.batch_size, tcfg.num_workers, dist)
    log.info("train batches %d | val batches %d", len(loaders["train"]), len(loaders["val"]))

    raw_model = build_model(cfg.model, cfg.data.in_timepoints, cfg.data.spatial_dims).to(dist.device)
    model = DDP(raw_model, device_ids=[dist.local_rank], find_unused_parameters=True) if dist.is_ddp else raw_model
    log.info("parameters: %s", {k: f"{v:,}" for k, v in raw_model.count_parameters().items()})

    weights = LossWeights(
        adv=tcfg.loss.adv, cyc=tcfg.loss.cyc, idt=tcfg.loss.idt,
        temporal=tcfg.sequence.w_temporal if cfg.data.sequence_mode else 0.0,
        fc=tcfg.sequence.w_fc if cfg.data.sequence_mode else 0.0,
    )
    needs_atlases = (cfg.data.sequence_mode and weights.fc > 0) or tcfg.roi.enabled or tcfg.roi.w_cycle > 0
    atlases = (load_atlases(cfg.atlas, tuple(cfg.data.spatial_dims), cropped=cfg.data.name == "grade")
               if needs_atlases else None)

    betas = (tcfg.optim.beta1, tcfg.optim.beta2)
    opt_G = Adam(raw_model.generator_parameters(), lr=tcfg.optim.lr_G, betas=betas)
    opt_D = Adam(raw_model.discriminator_parameters(), lr=tcfg.optim.lr_D, betas=betas)
    optimisers = {"opt_G": opt_G, "opt_D": opt_D}

    roi_disc = opt_D_roi = None
    if tcfg.roi.enabled:
        if atlases["2mo"].active_labels != atlases["9mo"].active_labels:
            raise ValueError("one ROI discriminator is shared by both age groups, so their ROI sets must match")
        roi_disc = MultiScaleROITemporalDiscriminator(n_rois=atlases["2mo"].n_rois).to(dist.device)
        if dist.is_ddp:
            # not DDP-wrapped, so broadcast rank 0's init explicitly (ranks are seeded differently)
            for p in roi_disc.parameters():
                torch_dist.broadcast(p.data, src=0)
        opt_D_roi = Adam(roi_disc.parameters(), lr=tcfg.roi.lr or tcfg.optim.lr_D, betas=betas)
        optimisers["opt_D_roi"] = opt_D_roi

    sched = tcfg.schedule
    schedulers = {
        name: build_scheduler(opt, tcfg.epochs, sched.warmup_epochs, sched.warmup_start_factor, sched.final_factor)
        for name, opt in (("sched_G", opt_G), ("sched_D", opt_D))
    }

    start_epoch, best_score = 1, float("-inf")
    latest = run_dir / "latest.pt"
    if tcfg.finetune_from:
        ckpt = torch.load(tcfg.finetune_from, map_location=dist.device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        log.info("fine-tuning from %s (weights only, epoch %d)", tcfg.finetune_from, ckpt["epoch"])
    elif tcfg.resume_from or latest.exists():
        start_epoch, best_score = load_checkpoint(Path(tcfg.resume_from or latest), dist.device, raw_model,
                                                  optimisers, schedulers, roi_disc)

    ctx = TrainContext(model=model, opt_G=opt_G, opt_D=opt_D, device=dist.device,
                       replay_a=ReplayBuffer(tcfg.replay_buffer_size), replay_b=ReplayBuffer(tcfg.replay_buffer_size),
                       is_ddp=dist.is_ddp, world_size=dist.world_size, atlases=atlases,
                       roi_disc=roi_disc, opt_D_roi=opt_D_roi)

    def checkpoint(path: Path, epoch: int) -> None:
        save_checkpoint(path, epoch, best_score, resolved, raw_model, optimisers, schedulers, roi_disc)

    log.info("training epochs %d -> %d", start_epoch, tcfg.epochs)
    for epoch in range(start_epoch, tcfg.epochs + 1):
        epoch_start = time.time()
        if dist.is_ddp:
            loaders["train"].sampler.set_epoch(epoch)
        epoch_weights = epoch_loss_weights(weights, epoch, tcfg.loss.warmup_epochs)
        train_metrics = train_one_epoch(ctx, loaders["train"], epoch_weights, epoch, tcfg, cfg.data.sequence_mode,
                                        show_progress=dist.is_main)
        # keep other ranks from entering the next forward while rank 0 validates
        if dist.is_ddp:
            torch_dist.barrier()
        for scheduler in schedulers.values():
            scheduler.step()
        lr_G, lr_D = (schedulers[k].get_last_lr()[0] for k in ("sched_G", "sched_D"))

        if dist.is_main:
            log.info("epoch %03d/%d (%.0fs) G=%.4f cyc=%.4f idt=%.4f D_A=%.4f D_B=%.4f gradG=%.3f lr_G=%.2e",
                     epoch, tcfg.epochs, time.time() - epoch_start, train_metrics["G_total"],
                     train_metrics["G_cyc"], train_metrics["G_idt"], train_metrics["D_A"],
                     train_metrics["D_B"], train_metrics["grad_norm_G"], lr_G)
            train_csv.write({"epoch": epoch, "lr_G": lr_G, "lr_D": lr_D,
                             **{k: f"{v:.6f}" for k, v in train_metrics.items()}})
            if use_wandb:
                wandb.log({"epoch": epoch, "lr_G": lr_G, "lr_D": lr_D,
                           **{f"train/{k}": v for k, v in train_metrics.items()}}, step=epoch)

        if dist.is_main and epoch % tcfg.val_every == 0:
            torch.cuda.empty_cache()
            val_metrics = validate(raw_model, loaders["val"], epoch_weights, dist.device, epoch, tcfg,
                                   cfg.data.sequence_mode, cfg.data.get("clean_grade"), atlases)
            score = val_score({k.removeprefix("val_"): v for k, v in val_metrics.items()}, tcfg.val_score)
            val_metrics["val_score"] = score
            log.info("  val cyc=%.4f idt=%.4f tSNR=%+.4f DVARS=%+.4f GSstd=%+.4f smooth=%.3f score=%.4f",
                     val_metrics["val_cyc"], val_metrics["val_idt"], val_metrics["val_tsnr_improvement"],
                     val_metrics["val_dvars_improvement"], val_metrics["val_gs_std_improvement"],
                     val_metrics["val_smoothness_ratio"], score)
            val_csv.write({"epoch": epoch, **{k: f"{v:.6f}" for k, v in val_metrics.items()}})
            if use_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
            if score > best_score:
                best_score = score
                checkpoint(run_dir / "best_model.pt", epoch)
                log.info("  new best score %.4f", best_score)

        if dist.is_main:
            if epoch % tcfg.save_every == 0:
                checkpoint(run_dir / f"epoch_{epoch:03d}.pt", epoch)
            checkpoint(latest, epoch)
        if dist.is_ddp:
            torch_dist.barrier()

    log.info("training complete, best val score %.4f", best_score)
    if use_wandb:
        wandb.finish()
    if dist.is_ddp:
        torch_dist.destroy_process_group()
