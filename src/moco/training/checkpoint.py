"""Full training-state checkpoints (model, optimisers, schedulers, RNG)."""
import logging
import random
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def save_checkpoint(path: Path, epoch: int, best_score: float, config: dict, model: torch.nn.Module,
                    optimisers: dict, schedulers: dict, roi_disc: torch.nn.Module | None = None) -> None:
    """Save everything needed to resume training exactly, including all RNG states.

    Args:
        path: Output `.pt` file.
        epoch: Last completed epoch.
        best_score: Best validation score so far.
        config: Resolved config, stored so the model can be rebuilt from the checkpoint alone.
        model: Unwrapped (non-DDP) model.
        optimisers: {"opt_G", "opt_D"[, "opt_D_roi"]}; keys match the original repo's checkpoints.
        schedulers: {"sched_G", "sched_D"}.
        roi_disc: ROI discriminator, if used.
    """
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
    """Restore training state in place; works on original-repo checkpoints too.

    Args:
        path: Checkpoint to resume from.
        device: Device to map tensors to.
        model, optimisers, schedulers, roi_disc: Objects to load into, keyed as in save_checkpoint.

    Returns:
        (epoch to start from, best validation score so far).
    """
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

    # RNG states must be CPU ByteTensors, whatever map_location did to them
    torch.set_rng_state(ckpt["rng_torch"].cpu().byte())
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_python"])
    if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu().byte())
    log.info("resumed from %s (epoch %d, best score %.4f)", path, ckpt["epoch"], ckpt["best_score"])
    return ckpt["epoch"] + 1, ckpt["best_score"]
