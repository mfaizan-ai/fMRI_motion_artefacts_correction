"""Spatially smooth raw or denoised runs: python scripts/smooth.py source=raw|<denoised name> [fwhm_mm=4]"""
from pathlib import Path

import hydra
from omegaconf import DictConfig

from moco.evaluation.denoise import check_output_root
from moco.evaluation.smoothing import run_smoothing
from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info


@hydra.main(config_path="../configs", config_name="smooth", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # check first: logging and run info below would otherwise make the folder non-empty
    check_output_root(Path(cfg.output_root), cfg.overwrite)
    setup_logging(Path(cfg.output_root) / "smooth.log")
    save_run_info(Path(cfg.output_root), cfg)
    run_smoothing(cfg)


if __name__ == "__main__":
    main()
