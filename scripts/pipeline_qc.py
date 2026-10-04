"""Pipeline-level QC of raw or denoised runs: python scripts/pipeline_qc.py source=raw|<denoised output_name>"""
import hydra
from omegaconf import DictConfig

from moco.evaluation.denoise import check_output_root
from moco.evaluation.pipeline_qc import output_dir, run_pipeline_qc
from moco.utils.logs import setup_logging
from moco.utils.run_info import save_run_info


@hydra.main(config_path="../configs", config_name="pipeline_qc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = output_dir(cfg)
    check_output_root(out_dir, cfg.overwrite)  # before logging and run info make the folder non-empty
    setup_logging(out_dir / "pipeline_qc.log")
    save_run_info(out_dir, cfg)
    run_pipeline_qc(cfg)


if __name__ == "__main__":
    main()
