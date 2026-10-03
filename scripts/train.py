"""Train a CycleGAN. Single GPU: python scripts/train.py experiment=st_v4
DDP: torchrun --standalone --nproc_per_node=4 scripts/train.py experiment=st_v4"""
import hydra
from omegaconf import DictConfig

from moco.training.trainer import run_training


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
