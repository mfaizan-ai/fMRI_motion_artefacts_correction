import pandas as pd

from moco.data.splits import assign_splits, subject_profiles

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _meta(n_subjects: int = 60) -> pd.DataFrame:
    rows = []
    for s in range(n_subjects):
        age = "9mo" if s % 3 == 0 else "2mo"
        for c in range(10):
            grade = "Grade 5" if c < s % 7 else "Grade 1"  # subjects differ in their high-motion share
            rows.append(dict(subject_id=f"S{s:02d}", age_group=age, grade=grade))
    return pd.DataFrame(rows)


def _split(seed: int = 0) -> pd.Series:
    profiles = subject_profiles(_meta(), ["Grade 4", "Grade 5", "Grade 6"], n_motion_bins=3)
    return assign_splits(profiles, RATIOS, seed)


def test_every_subject_in_exactly_one_split():
    split = _split()
    assert split.index.is_unique and len(split) == 60
    assert set(split) == set(RATIOS)


def test_ratios_and_strata_respected():
    split = _split()
    assert split.value_counts().to_dict() == {"train": 42, "val": 9, "test": 9}
    ages = _meta().drop_duplicates("subject_id").set_index("subject_id")["age_group"]
    by_age = pd.crosstab(split, ages.loc[split.index])
    assert (by_age.loc["val"] > 0).all() and (by_age.loc["test"] > 0).all()  # both ages in every split


def test_reproducible_with_seed():
    assert _split(0).equals(_split(0))
    assert not _split(0).equals(_split(1))
