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


def test_cli_label_mode_binary(tmp_path, capsys):
    import numpy as np

    out = tmp_path / "ds_bin.npz"
    rc = main([str(out), "--T", "128", "--n-seeds", "1", "--label-mode", "binary"])
    assert rc == 0
    assert "label_mode=binary" in capsys.readouterr().out

    loaded = NoiseSeriesDataset.load(out)
    assert loaded.label_mode == "binary"
    x, y = loaded[0]
    assert set(np.unique(y.numpy()).tolist()).issubset({0, 1})


def test_cli_label_mode_signed_default(tmp_path):
    """When --label-mode is omitted, archive defaults to 'signed'."""
    import numpy as np

    out = tmp_path / "ds_default.npz"
    rc = main([str(out), "--T", "128", "--n-seeds", "1"])
    assert rc == 0

    loaded = NoiseSeriesDataset.load(out)
    assert loaded.label_mode == "signed"
    x, y = loaded[0]
    assert set(np.unique(y.numpy()).tolist()).issubset({-1, 0, 1})
