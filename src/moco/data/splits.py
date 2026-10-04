"""Subject-level train/val/test assignment, stratified by age group and motion level."""
import numpy as np
import pandas as pd


def subject_profiles(meta: pd.DataFrame, high_motion_grades: list[str], n_motion_bins: int) -> pd.DataFrame:
    """One row per subject_id: age_group, share of chunks in high_motion_grades, and its quantile bin."""
    profiles = meta.groupby("subject_id").agg(
        age_group=("age_group", "first"),
        high_motion_frac=("grade", lambda g: g.isin(high_motion_grades).mean()),
        n_chunks=("grade", "size"),
    )
    # rank first so ties (e.g. many subjects with no high-motion chunks) still split into equal bins
    ranks = profiles["high_motion_frac"].rank(method="first")
    profiles["motion_bin"] = pd.qcut(ranks, n_motion_bins, labels=False)
    return profiles.reset_index()


def assign_splits(profiles: pd.DataFrame, ratios: dict[str, float], seed: int) -> pd.Series:
    """subject_id -> split, allocated proportionally within each (age_group, motion_bin) stratum.

    Subjects are shuffled within a stratum and the i-th of n goes to the split whose cumulative ratio
    first exceeds (i + 0.5) / n, so every stratum is divided in the requested proportions."""
    names = list(ratios)
    bounds = np.cumsum([ratios[s] for s in names])
    if not np.isclose(bounds[-1], 1.0):
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, stratum in profiles.sort_values("subject_id").groupby(["age_group", "motion_bin"], sort=True):
        subjects = stratum["subject_id"].to_numpy()[rng.permutation(len(stratum))]
        for i, subject in enumerate(subjects):
            assignment[subject] = names[int(np.searchsorted(bounds, (i + 0.5) / len(subjects)))]
    return pd.Series(assignment, name="split").rename_axis("subject_id").sort_index()
