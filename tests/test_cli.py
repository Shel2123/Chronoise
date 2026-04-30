"""CLI smoke test."""
from __future__ import annotations

from chronoise import NoiseSeriesDataset
from chronoise.cli import main


def test_cli_builds_and_saves(tmp_path, capsys):
    out = tmp_path / "ds.npz"
    rc = main([str(out), "--T", "256", "--n-seeds", "1"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0

    captured = capsys.readouterr()
    assert "Saved" in captured.out

    # 6 x 3 x 2 x 1 seeds = 36 series, T=256.
    loaded = NoiseSeriesDataset.load(out)
    assert len(loaded) == 36
    assert loaded.cfg.T == 256


def test_cli_keep_components(tmp_path):
    out = tmp_path / "ds.npz"
    rc = main([str(out), "--T", "128", "--n-seeds", "1", "--keep-components"])
    assert rc == 0

    loaded = NoiseSeriesDataset.load(out)
    assert loaded.keep_components is True
    L, N = loaded.components(0)
    assert L.shape == (128,) and N.shape == (128,)
