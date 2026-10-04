import numpy as np
import pytest
import torch

from moco.evaluation.denoise import check_output_root, correct_run


class Identity(torch.nn.Module):
    def correct(self, x):
        return x


def test_correct_run_roundtrips_with_identity_model():
    """Chunking, tail filling, axis-0 padding/cropping and (de)normalisation must invert exactly."""
    rng = np.random.default_rng(0)
    volume = rng.uniform(100, 200, size=(60, 8, 6, 13)).astype(np.float32)  # T=13: last chunk has 3 volumes
    volume[:5] = 0
    out = correct_run(volume, median=150.0, scale=20.0, model=Identity(), device="cpu", chunk_t=5, padded_h=64)
    assert out.shape == volume.shape
    np.testing.assert_allclose(out, volume, rtol=1e-5)
    assert np.all(out[:5] == 0)


def test_check_output_root_refuses_non_empty_folder(tmp_path):
    check_output_root(tmp_path / "new", overwrite=False)
    check_output_root(tmp_path, overwrite=False)  # empty dir, as Hydra creates it
    (tmp_path / "denoise_log.csv").write_text("")
    with pytest.raises(SystemExit):
        check_output_root(tmp_path, overwrite=False)
    check_output_root(tmp_path, overwrite=True)


@pytest.mark.parametrize("config_name", ["denoise", "evaluate", "splits", "train"])
def test_hydra_logging_keeps_module_loggers(config_name):
    """Hydra's `disabled` preset silences every `moco.*` logger created at import, leaving job logs empty."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name, return_hydra_config=True)
    for preset in (cfg.hydra.job_logging, cfg.hydra.hydra_logging):
        assert preset.disable_existing_loggers is False
