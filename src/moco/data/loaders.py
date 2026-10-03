"""Train/val DataLoaders for the configured dataset."""
from torch.utils.data import DataLoader, DistributedSampler

from moco.data.fc_groups import FCGroupBatchSampler, FCGroupDataset
from moco.data.grade import FMRIUnpairedGradeDataset
from moco.data.psc import build_psc_dataloaders
from moco.training.distributed import DistInfo
from moco.utils.seed import worker_init_fn


def _grade_dataset(data_cfg, split: str, full_coverage: bool) -> FMRIUnpairedGradeDataset:
    return FMRIUnpairedGradeDataset(
        split=split, chunk_metadata_csv=data_cfg.chunk_metadata_csv, run_stats_csv=data_cfg.run_stats_csv,
        splits_csv=data_cfg.splits_csv, clean_grade=data_cfg.clean_grade, grades_b=list(data_cfg.grades_b),
        padded_h=data_cfg.spatial_dims[0], task=data_cfg.task, flip_prob=data_cfg.flip_prob,
        base_seed=data_cfg.base_seed, full_coverage=full_coverage,
    )


def build_fc_group_loaders(data_cfg, batch_size: int, num_workers: int, dist: DistInfo) -> dict[str, DataLoader]:
    """Batches of K-chunk examples, A and B: (batch, K, T, H, W, D). Every K-chunk set is used once per epoch."""
    loaders = {}
    for split, train in (("train", True), ("val", False)):
        dataset = FCGroupDataset(
            split=split, chunk_metadata_csv=data_cfg.chunk_metadata_csv, run_stats_csv=data_cfg.run_stats_csv,
            splits_csv=data_cfg.splits_csv, clean_grade=data_cfg.clean_grade, grades_b=list(data_cfg.grades_b),
            padded_h=data_cfg.spatial_dims[0], chunks_per_example=data_cfg.chunks_per_example, task=data_cfg.task,
        )
        sampler = FCGroupBatchSampler(
            a_group_sizes=[len(g) for g in dataset.A_groups], a_grades=[g[0]["grade"] for g in dataset.A_groups],
            b_group_sizes=[len(g) for g in dataset.B_groups], k=data_cfg.chunks_per_example,
            batch_size=batch_size, shuffle=train, seed=data_cfg.base_seed,
            rank=dist.rank if train else 0, world_size=dist.world_size if train else 1,
        )
        loaders[split] = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True,
                                    worker_init_fn=worker_init_fn)
    return loaders


def build_train_val_loaders(data_cfg, batch_size: int, num_workers: int, dist: DistInfo) -> dict[str, DataLoader]:
    """Train is sharded across DDP ranks; val is the full set (validation runs on rank 0 only)."""
    if data_cfg.name == "psc":
        loaders = build_psc_dataloaders(data_cfg, batch_size, num_workers, ["train"],
                                        distributed=dist.is_ddp, world_size=dist.world_size, rank=dist.rank)
        loaders.update(build_psc_dataloaders(data_cfg, batch_size, num_workers, ["val"]))
        return loaders
    if data_cfg.name == "grade_fc":
        return build_fc_group_loaders(data_cfg, batch_size, num_workers, dist)

    train_ds = _grade_dataset(data_cfg, "train", full_coverage=False)
    val_ds = _grade_dataset(data_cfg, "val", full_coverage=True)
    sampler = (DistributedSampler(train_ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
               if dist.is_ddp else None)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                            num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=True, worker_init_fn=worker_init_fn),
    }
