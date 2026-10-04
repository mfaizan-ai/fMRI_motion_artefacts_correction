"""Build a CycleGAN from config, or from any checkpoint (new or from the original repo)."""
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from moco.models.base import CycleGANBase
from moco.models.disentangled import DisentangledCycleGAN
from moco.models.spatiotemporal import SpatioTemporalCycleGAN


def build_model(model_cfg: DictConfig, in_timepoints: int, spatial_dims: tuple) -> CycleGANBase:
    """Instantiate an untrained CycleGAN.

    Args:
        model_cfg: `model` config group; `name` selects "spatiotemporal" or "disentangled".
        in_timepoints: Volumes per chunk (used by the disentangled model only).
        spatial_dims: Padded (H, W, D) input size (used by the spatiotemporal model only).

    Returns:
        The model on CPU, in train mode.

    Raises:
        ValueError: If `model_cfg.name` is unknown.
    """
    common = dict(
        content_base_ch=model_cfg.content_base_ch,
        content_n_res=model_cfg.content_n_res,
        artefact_base_ch=model_cfg.artefact_base_ch,
        global_code_dim=model_cfg.global_code_dim,
        spatial_code_ch=model_cfg.spatial_code_ch,
        disc_base_ch=model_cfg.disc_base_ch,
        num_disc_scales=model_cfg.num_disc_scales,
        residual=model_cfg.residual,
    )
    if model_cfg.name == "spatiotemporal":
        return SpatioTemporalCycleGAN(spatial_dims=tuple(spatial_dims), temporal_k=model_cfg.temporal_k,
                                      use_convlstm=model_cfg.use_convlstm, **common)
    if model_cfg.name == "disentangled":
        return DisentangledCycleGAN(in_timepoints=in_timepoints,
                                    disc_temporal_diffs=model_cfg.disc_temporal_diffs, **common)
    raise ValueError(f"unknown model {model_cfg.name!r}")


def _config_from_legacy_args(args: dict) -> DictConfig:
    """Map the original repo's argparse namespace (checkpoint["args"]) onto this repo's config layout."""
    st = args.get("use_st_model", False)
    return OmegaConf.create({
        "model": {
            "name": "spatiotemporal" if st else "disentangled",
            "content_base_ch": args["content_base_ch"],
            "content_n_res": args["content_n_res"],
            "artefact_base_ch": args["artefact_base_ch"],
            "global_code_dim": args["global_code_dim"],
            "spatial_code_ch": args["spatial_code_ch"],
            "disc_base_ch": args["disc_base_ch"],
            "num_disc_scales": args["num_disc_scales"],
            "residual": args["residual"],
            "temporal_k": 3,  # fixed in the original repo, never an argparse option
            "use_convlstm": args.get("convlstm", False),
            "disc_temporal_diffs": not args.get("no_disc_temporal_diffs", False),
        },
        "data": {
            "in_timepoints": args["in_timepoints"],
            "spatial_dims": [64, 72, 56] if args.get("use_grade_dataset", False) else [80, 96, 72],
        },
    })


def config_from_checkpoint(ckpt: dict) -> DictConfig:
    """Config a checkpoint was trained with: stored `config` (this repo) or mapped legacy `args`."""
    return OmegaConf.create(ckpt["config"]) if "config" in ckpt else _config_from_legacy_args(ckpt["args"])


def load_model(checkpoint_path: str | Path, device: str | torch.device) -> tuple[CycleGANBase, dict]:
    """Rebuild a model from its checkpoint and load the weights.

    Args:
        checkpoint_path: `.pt` file from this repo or the original one.
        device: Device to load the weights and model onto.

    Returns:
        (model in eval mode, raw checkpoint dict with e.g. "epoch" and "config"/"args").
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ckpt)
    model = build_model(cfg.model, cfg.data.in_timepoints, cfg.data.spatial_dims).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt
