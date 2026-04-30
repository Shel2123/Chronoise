"""Save/load round-trip for NoiseSeriesDataset."""
from __future__ import annotations

import numpy as np

from chronoise import NoiseSeriesDataset


def test_roundtrip_basic(small_cfg, tmp_path):
    ds = NoiseSeriesDataset(small_cfg)
    path = tmp_path / "ds.npz"
    ds.save(path)

    loaded = NoiseSeriesDataset.load(path)
    assert len(loaded) == len(ds)
    assert loaded.cfg.T == small_cfg.T
    assert loaded.specs == ds.specs

    for i in (0, len(ds) // 2, len(ds) - 1):
        x_a, y_a = ds[i]
        x_b, y_b = loaded[i]
        assert np.array_equal(x_a.numpy(), x_b.numpy())
        assert np.array_equal(y_a.numpy(), y_b.numpy())


def test_roundtrip_with_components(small_cfg, tmp_path):
    ds = NoiseSeriesDataset(small_cfg, keep_components=True)
    path = tmp_path / "ds.npz"
    ds.save(path)

    loaded = NoiseSeriesDataset.load(path)
    assert loaded.keep_components is True
    L_a, N_a = ds.components(0)
    L_b, N_b = loaded.components(0)
    assert np.array_equal(L_a, L_b)
    assert np.array_equal(N_a, N_b)


def test_load_uses_no_pickle(small_cfg, tmp_path):
    """Manifest must round-trip without allow_pickle — guards against regressions."""
    ds = NoiseSeriesDataset(small_cfg)
    path = tmp_path / "ds.npz"
    ds.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert "manifest" in data.files
        assert "X" in data.files and "y" in data.files
