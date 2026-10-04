"""Plot pipeline QC across sources: python scripts/visualization/plot_pipeline_qc.py [comparison_name=<name>]"""
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info
from moco.visualization.pipeline_qc_plots import plot_pipeline_qc

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="plot_pipeline_qc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    setup_logging(out_dir / "plot_pipeline_qc.log")
    save_run_info(out_dir, cfg)
    for stem, fig in plot_pipeline_qc(cfg).items():
        path = out_dir / f"{stem}.png"
        fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("saved %s", path)


if __name__ == "__main__":
    main()
