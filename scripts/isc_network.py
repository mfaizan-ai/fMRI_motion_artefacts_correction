"""Leave-one-out network ISC per video order: python scripts/isc_network.py source=raw|<denoised name> [order=A]"""
import logging

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from moco.evaluation.isc import build_atlas, run_order, selected_orders
from moco.evaluation.pipeline_qc import output_dir
from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info
from moco.visualization.isc_plots import plot_network_isc, plot_network_isfc

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="isc_network", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = output_dir(cfg) / cfg.output_subdir
    setup_logging(out_dir / "isc_network.log")
    save_run_info(out_dir, cfg)
    atlas, roi_names, network_names, network_index = build_atlas(cfg)
    for order in selected_orders(cfg):
        result = run_order(order, atlas, roi_names, network_names, network_index, cfg)
        title = f"Order {order} ({result['video']}), {cfg.source}, N = {result['n_subjects']}"
        figures = {"network_isc": plot_network_isc(result["network"], title, cfg),
                   "network_isfc": plot_network_isfc(result["isfc"], network_names, title, cfg)}
        for stem, fig in figures.items():
            fig.savefig(result["order_dir"] / f"{stem}.png", dpi=cfg.plot.dpi, bbox_inches="tight")
            plt.close(fig)
        log.info("order %s -> %s", order, result["order_dir"])
if __name__ == "__main__":
    main()
