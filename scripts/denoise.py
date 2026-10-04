"""Correct whole video runs: python scripts/denoise.py checkpoint=<path> output_name=<folder> [runs=all_video]"""
from pathlib import Path

import hydra
from omegaconf import DictConfig

from moco.evaluation.denoise import check_output_root, run_denoise
from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info


@hydra.main(config_path="../configs", config_name="denoise", version_base="1.3")
def main(cfg: DictConfig) -> None:
    check_output_root(Path(cfg.output_root), cfg.overwrite)
    setup_logging(Path(cfg.output_root) / "denoise.log")
    save_run_info(Path(cfg.output_root), cfg)
    run_denoise(cfg)


if __name__ == "__main__":
    main()
