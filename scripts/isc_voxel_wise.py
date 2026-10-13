"""Voxel-wise LOSO ISC, raw vs a denoised source: python scripts/isc_voxel_wise.py source=<denoised name> [order=A]"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from moco.evaluation.isc import selected_orders
from moco.evaluation.isc_voxel_wise import run_order
from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info
from moco.visualization.isc_voxel_plots import make_order_figures

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="isc_voxel_wise", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.qc_root) / cfg.source / cfg.output_subdir
    setup_logging(out_dir / "isc_voxel_wise.log")
    save_run_info(out_dir, cfg)
    for order in selected_orders(cfg):
        order_dir = run_order(cfg, order)
        make_order_figures(order_dir, order, cfg)
        log.info("order %s -> %s", order, order_dir)


if __name__ == "__main__":
    main()
