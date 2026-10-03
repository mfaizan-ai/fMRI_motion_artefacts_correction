"""Age-appropriate Schaefer-400 atlases on the model's voxel grid and ROI time-series extraction.

FoundCog subject IDs ending in "A" are the ~9-month visit (nihpd-08-11 template), others ~2 months
(nihpd-02-05). Both atlases are pre-warped onto the BOLD common-space grid (97, 116, 79).
"""
import logging
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)

NATIVE_SPATIAL = (97, 116, 79)
N_SCHAEFER_ROIS = 400


def age_group_for_subject(subject_id: str) -> str:
    return "9mo" if subject_id.endswith("A") else "2mo"


class SchaeferAtlas:
    """Atlas brought to `target_spatial`; the active ROI set is fixed at construction so every
    extraction returns the same columns."""

    def __init__(self, atlas_path: str, target_spatial: tuple, name: str = "atlas"):
        self.name = name
        self.target_spatial = tuple(target_spatial)
        native = np.asarray(nib.load(atlas_path).dataobj)
        if native.shape != NATIVE_SPATIAL:
            raise ValueError(f"[{name}] atlas shape {native.shape} != BOLD grid {NATIVE_SPATIAL}: {atlas_path}")
        native_int = np.rint(native).astype(np.int64)
        if not np.allclose(native, native_int, atol=1e-5):
            raise ValueError(f"[{name}] atlas has non-integer labels")

        resampled = self._to_target(native_int)
        assert resampled.shape == self.target_spatial
        labels = sorted(set(np.unique(resampled).tolist()) - {0})
        lost = set(np.unique(native_int).tolist()) - {0} - set(labels)
        if lost:
            log.warning("[%s] %d ROIs vanished going to %s", name, len(lost), self.target_spatial)
        if len(labels) < 0.95 * N_SCHAEFER_ROIS:
            log.warning("[%s] only %d/%d ROIs present", name, len(labels), N_SCHAEFER_ROIS)

        self.active_labels = labels
        self.n_rois = len(labels)
        # voxels outside active ROIs go to a trash bin at index n_rois, dropped after the scatter-sum
        roi_index = np.full(resampled.shape, self.n_rois, dtype=np.int64)
        for i, label in enumerate(labels):
            roi_index[resampled == label] = i
        self.roi_index = torch.from_numpy(roi_index.reshape(-1))
        self.roi_voxel_counts = torch.tensor([(resampled == lb).sum() for lb in labels], dtype=torch.float32)
        log.info("[%s] %d/%d ROIs on grid %s", name, self.n_rois, N_SCHAEFER_ROIS, self.target_spatial)

    def _to_target(self, native: np.ndarray) -> np.ndarray:
        """Nearest-neighbour resize to the target grid."""
        t = torch.from_numpy(native.astype(np.float32))[None, None]
        return F.interpolate(t, size=self.target_spatial, mode="nearest")[0, 0].round().long().numpy()

    def extract_roi_timeseries(self, volumes: Tensor) -> Tensor:
        """volumes: (T, X, Y, Z) on target_spatial -> ROI means (T, n_rois). Differentiable."""
        assert tuple(volumes.shape[-3:]) == self.target_spatial
        T = volumes.shape[0]
        flat = volumes.reshape(T, -1)
        sums = torch.zeros(T, self.n_rois + 1, device=flat.device, dtype=flat.dtype)
        sums = sums.index_add(1, self.roi_index.to(flat.device), flat)[:, : self.n_rois]
        return sums / self.roi_voxel_counts.to(flat.device).clamp(min=1)


class SchaeferAtlasCropped(SchaeferAtlas):
    """Atlas on the grade dataset's grid via the fixed per-age crop used to build the chunks (no resize,
    so all 400 ROIs survive), then zero-padded on axis 0 to the model grid.

    For 9mo ~195 boundary voxels are nonzero, so the zero pad is a small accepted discontinuity."""

    def __init__(self, atlas_path: str, age_group: str, crop_window: dict, target_spatial: tuple):
        self.crop_window = crop_window
        super().__init__(atlas_path, target_spatial, name=f"{age_group}_cropped")

    def _to_target(self, native: np.ndarray) -> np.ndarray:
        (x0, x1), (y0, y1), (z0, z1) = self.crop_window["x"], self.crop_window["y"], self.crop_window["z"]
        cropped = native[x0:x1, y0:y1, z0:z1]
        pad = self.target_spatial[0] - cropped.shape[0]
        return np.pad(cropped, [(pad // 2, pad - pad // 2), (0, 0), (0, 0)])


def load_atlases(atlas_cfg, spatial_dims: tuple, cropped: bool) -> dict[str, SchaeferAtlas]:
    """{"2mo": atlas, "9mo": atlas} on spatial_dims, cropped (grade dataset) or resized (PSC dataset)."""
    if cropped:
        return {group: SchaeferAtlasCropped(atlas_cfg.paths[group], group, atlas_cfg.crop_window[group], spatial_dims)
                for group in ("2mo", "9mo")}
    return {group: SchaeferAtlas(atlas_cfg.paths[group], spatial_dims, name=group) for group in ("2mo", "9mo")}


def subject_id_from_path(path: str) -> str:
    """'ICC103A' from '.../sub-ICC103A_ses-1_...'."""
    match = re.search(r"sub-([A-Za-z0-9]+)_", os.path.basename(path))
    if not match:
        raise ValueError(f"no subject id in {path}")
    return match.group(1)


def roi_timeseries_batch(volumes: Tensor, paths: list[str], atlases: dict[str, SchaeferAtlas]) -> Tensor:
    """volumes: (B, T, X, Y, Z) -> (B, n_rois, T), each sample with its own age-appropriate atlas."""
    series = [atlases[age_group_for_subject(subject_id_from_path(p))].extract_roi_timeseries(volumes[b])
              for b, p in enumerate(paths)]
    return torch.stack(series).permute(0, 2, 1)
