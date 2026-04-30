"""NoiseSeriesDataset: length, shapes, components, and metadata."""
from __future__ import annotations

import numpy as np
import torch

from chronoise import GeneratorConfig, NoiseSeriesDataset


def test_default_grid_size():
    cfg = GeneratorConfig(T=128, n_seeds=10)
    ds = NoiseSeriesDataset(cfg)
    # 6 noise styles x 3 amplitudes x 2 modes x 10 seeds.
    assert len(ds) == 360


def test_custom_grid_size(small_cfg):
    ds = NoiseSeriesDataset(small_cfg)
    # 6 x 3 x 2 x n_seeds with n_seeds=2.
    assert len(ds) == 6 * 3 * 2 * small_cfg.n_seeds


def test_item_is_torch_pair(small_cfg):
    ds = NoiseSeriesDataset(small_cfg)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (small_cfg.T,) and y.shape == (small_cfg.T,)
    assert x.dtype == torch.float32 and y.dtype == torch.int64


def test_components_require_flag(small_cfg):
    ds = NoiseSeriesDataset(small_cfg)
    try:
        ds.components(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("components() must raise without keep_components=True")


def test_components_returned_when_kept(small_cfg):
    ds = NoiseSeriesDataset(small_cfg, keep_components=True)
    L, N = ds.components(0)
    assert L.shape == (small_cfg.T,) and N.shape == (small_cfg.T,)
    assert np.isfinite(L).all() and np.isfinite(N).all()


def test_return_meta(small_cfg):
    ds = NoiseSeriesDataset(small_cfg, return_meta=True)
    x, y, meta = ds[0]
    assert isinstance(meta, dict)
    assert {"noise_kind", "beta", "amplitude", "structural_mode", "seed"} <= meta.keys()


def test_dataloader_collation(small_cfg):
    from torch.utils.data import DataLoader

    ds = NoiseSeriesDataset(small_cfg)
    loader = DataLoader(ds, batch_size=4)
    xb, yb = next(iter(loader))
    assert xb.shape == (4, small_cfg.T)
    assert yb.shape == (4, small_cfg.T)
