"""Test-split metrics by grade: python scripts/evaluate.py checkpoint=<path> output_dir=<dir>"""
from pathlib import Path

import hydra
from omegaconf import DictConfig

from moco.evaluation.test_set import run_test
from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(Path(cfg.output_dir) / "evaluate.log")
    save_run_info(Path(cfg.output_dir), cfg)
    run_test(cfg)


if __name__ == "__main__":
    main()
