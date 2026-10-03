"""Full training-state checkpoints (model, optimisers, schedulers, RNG)."""
import logging
import random
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def save_checkpoint(path: Path, epoch: int, best_score: float, config: dict, model: torch.nn.Module,
                    optimisers: dict, schedulers: dict, roi_disc: torch.nn.Module | None = None) -> None:
    """optimisers: {"opt_G", "opt_D"[, "opt_D_roi"]}, schedulers: {"sched_G", "sched_D"} (original repo keys)."""
    ckpt = {
        "epoch": epoch,
        "best_score": best_score,
        "config": config,
        "model": model.state_dict(),
        **{name: opt.state_dict() for name, opt in optimisers.items()},
        **{name: sched.state_dict() for name, sched in schedulers.items()},
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if roi_disc is not None:
        ckpt["roi_disc"] = roi_disc.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path: Path, device: torch.device, model: torch.nn.Module, optimisers: dict,
                    schedulers: dict, roi_disc: torch.nn.Module | None = None) -> tuple[int, float]:
    """Restore training state in place; returns (start_epoch, best_score). Works on original-repo checkpoints."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    for name, opt in optimisers.items():
        if name == "opt_D_roi" and "roi_disc" not in ckpt:
            continue
        opt.load_state_dict(ckpt[name])
    for name, sched in schedulers.items():
        sched.load_state_dict(ckpt[name])
    if roi_disc is not None:
        if "roi_disc" in ckpt:
            roi_disc.load_state_dict(ckpt["roi_disc"])
        else:
            log.warning("checkpoint has no roi_disc state; ROI discriminator starts from random init")

    torch.set_rng_state(ckpt["rng_torch"].cpu().byte())
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_python"])
    if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu().byte())
    log.info("resumed from %s (epoch %d, best score %.4f)", path, ckpt["epoch"], ckpt["best_score"])
    return ckpt["epoch"] + 1, ckpt["best_score"]
