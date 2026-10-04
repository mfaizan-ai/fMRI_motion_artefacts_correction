"""Plot test metrics by grade: python scripts/visualization/plot_test.py input_dir=<evaluate output_dir>"""
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig

from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info
from moco.visualization.test_visualization import plot_test_metrics

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="plot_test", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    setup_logging(out_dir / "plot_test.log")
    save_run_info(out_dir, cfg)
    fig = plot_test_metrics(pd.read_csv(cfg.metrics_csv), cfg)
    path = out_dir / cfg.output_name
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    log.info("saved %s", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
