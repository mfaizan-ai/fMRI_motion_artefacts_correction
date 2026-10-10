"""Plot training curves: python scripts/visualization/plot_training.py run_dir=<run folder> output_dir=<dir>"""
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig

from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info
from moco.visualization.training_plots import plot_training

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="plot_training", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    setup_logging(out_dir / "plot_training.log")
    save_run_info(out_dir, cfg)
    name = cfg.title or Path(cfg.run_dir).name
    for stem, fig in plot_training(pd.read_csv(cfg.train_csv), pd.read_csv(cfg.val_csv), name, cfg).items():
        path = out_dir / f"{stem}.png"
        fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("saved %s", path)


if __name__ == "__main__":
    main()
