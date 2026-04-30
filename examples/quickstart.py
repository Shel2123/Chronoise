"""Quickstart: generate a single realization and a small dataset, save/load."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chronoise import (
    GeneratorConfig,
    NoiseKind,
    NoiseSeriesDataset,
    NoiseSpec,
    generate_series,
)

def main() -> None:
    # One realization with explicit parameters.
    cfg = GeneratorConfig()
    r = generate_series(
        cfg,
        noise=NoiseSpec(NoiseKind.BETA, beta=1.0),
        amplitude=1.0,
        structural_mode=1,
        seed=0,
    )
    print(
        f"Single series: X.shape={r.X.shape}, "
        f"L range=({r.L.min():.2f}, {r.L.max():.2f}), "
        f"y counts: -1={(r.y == -1).sum()}, 0={(r.y == 0).sum()}, +1={(r.y == 1).sum()}"
    )

    # same seed -> identical level across noise types.
    r2 = generate_series(
        cfg,
        noise=NoiseSpec(NoiseKind.EL_NINO),
        amplitude=2.0,
        structural_mode=1,
        seed=0,
    )
    assert np.array_equal(r.L, r2.L), "level should be seed-stable across noise types"
    print("Level trace is seed-stable across noise types: OK")

    # 3) Build a small dataset (T=1024, 2 seeds -> 6 * 3 * 2 * 2 = 72 samples).
    cfg_small = GeneratorConfig(T=1024, n_seeds=2)
    ds = NoiseSeriesDataset(cfg=cfg_small)
    print(f"Dataset size: {len(ds)} samples, X shape per item: {tuple(ds[0][0].shape)}")

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    xb, yb = next(iter(loader))
    print(f"Batch: X={tuple(xb.shape)} dtype={xb.dtype}, y={tuple(yb.shape)} dtype={yb.dtype}")

    # 4) Round-trip save/load.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dataset.npz"
        ds.save(path)
        ds_loaded = NoiseSeriesDataset.load(path)
        assert len(ds_loaded) == len(ds)
        assert torch.allclose(ds_loaded[0][0], ds[0][0])
        print(f"Round-trip save/load OK ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
