"""Tests for binary bifurcation labels."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chronoise import (
    GeneratorConfig,
    NoiseKind,
    NoiseSeriesDataset,
    NoiseSpec,
    generate_series,
    labels_binary_from_level,
    labels_from_level,
)


# --- pure helper -----------------------------------------------------------


def test_labels_binary_from_level_basic_shape_and_dtype():
    L = np.array([0.0, 0.0, 1.5, 1.5, 0.5, 0.5], dtype=np.float64)
    y = labels_binary_from_level(L)
    assert y.dtype == np.int8
    assert y.shape == L.shape
    # First label is always 0 by construction.
    assert y[0] == 0
    # Bifurcations at index 2 and 4.
    assert y.tolist() == [0, 0, 1, 0, 1, 0]


def test_labels_binary_from_level_range():
    rng = np.random.default_rng(0)
    L = rng.normal(size=512)
    y = labels_binary_from_level(L)
    assert set(np.unique(y).tolist()).issubset({0, 1})


def test_labels_binary_equals_nonzero_of_signed():
    """`y_binary == (y_signed != 0)` must hold exactly for any L."""
    rng = np.random.default_rng(7)
    L = rng.normal(size=256)
    y_sign = labels_from_level(L)
    y_bin = labels_binary_from_level(L)
    assert np.array_equal(y_bin.astype(bool), y_sign != 0)


# --- SeriesResult ----------------------------------------------------------


def _gen(cfg, **kw):
    defaults = dict(
        noise=NoiseSpec(NoiseKind.BETA, beta=1.0),
        amplitude=1.0,
        structural_mode=1,
        seed=0,
    )
    defaults.update(kw)
    return generate_series(cfg, **defaults)


def test_series_result_has_y_binary(small_cfg):
    r = _gen(small_cfg, seed=3)
    assert hasattr(r, "y_binary")
    assert r.y_binary.shape == (small_cfg.T,)
    assert r.y_binary.dtype == np.int8


def test_series_result_y_binary_matches_y(small_cfg):
    r = _gen(small_cfg, seed=11)
    assert np.array_equal(r.y_binary.astype(bool), r.y != 0)


def test_y_binary_values_in_range(small_cfg):
    r = _gen(small_cfg, seed=4)
    assert set(np.unique(r.y_binary).tolist()).issubset({0, 1})
    assert r.y_binary[0] == 0


# --- NoiseSeriesDataset ----------------------------------------------------


def test_dataset_default_label_mode_is_signed(small_cfg):
    ds = NoiseSeriesDataset(small_cfg)
    assert ds.label_mode == "signed"
    x, y = ds[0]
    assert set(np.unique(y.numpy()).tolist()).issubset({-1, 0, 1})


def test_dataset_binary_label_mode_changes_getitem(small_cfg):
    ds = NoiseSeriesDataset(small_cfg, label_mode="binary")
    assert ds.label_mode == "binary"
    x, y = ds[0]
    assert set(np.unique(y.numpy()).tolist()).issubset({0, 1})
    # Shape unchanged.
    assert y.shape == (small_cfg.T,)


def test_dataset_invalid_label_mode_raises(small_cfg):
    with pytest.raises(ValueError):
        NoiseSeriesDataset(small_cfg, label_mode="ternary")  # type: ignore[arg-type]


def test_dataset_labels_helper_signed_and_binary_match(small_cfg):
    ds = NoiseSeriesDataset(small_cfg, label_mode="signed")
    for i in (0, 5, len(ds) - 1):
        y_sign = ds.labels(i, mode="signed")
        y_bin = ds.labels(i, mode="binary")
        assert np.array_equal(y_bin.astype(bool), y_sign != 0)


def test_dataset_labels_default_uses_dataset_mode(small_cfg):
    ds = NoiseSeriesDataset(small_cfg, label_mode="binary")
    y_default = ds.labels(0)
    y_explicit = ds.labels(0, mode="binary")
    assert np.array_equal(y_default, y_explicit)


# --- save/load round-trip --------------------------------------------------


def test_dataset_roundtrip_preserves_binary_labels(small_cfg, tmp_path):
    ds = NoiseSeriesDataset(small_cfg, label_mode="binary")
    path = tmp_path / "ds_bin.npz"
    ds.save(path)
    loaded = NoiseSeriesDataset.load(path)
    assert loaded.label_mode == "binary"
    for i in (0, len(ds) // 2, len(ds) - 1):
        # Both encodings preserved
        assert np.array_equal(
            ds.labels(i, mode="signed"), loaded.labels(i, mode="signed")
        )
        assert np.array_equal(
            ds.labels(i, mode="binary"), loaded.labels(i, mode="binary")
        )


def test_load_label_mode_override(small_cfg, tmp_path):
    """`label_mode` passed to `load` should override the manifest's default."""
    ds = NoiseSeriesDataset(small_cfg, label_mode="signed")
    path = tmp_path / "ds.npz"
    ds.save(path)
    loaded = NoiseSeriesDataset.load(path, label_mode="binary")
    assert loaded.label_mode == "binary"
    x, y = loaded[0]
    assert set(np.unique(y.numpy()).tolist()).issubset({0, 1})


def test_load_legacy_archive_without_y_binary(small_cfg, tmp_path):
    """Archives saved before the y_binary field should still load and derive it."""
    import json

    ds = NoiseSeriesDataset(small_cfg, label_mode="signed")
    path = tmp_path / "legacy.npz"
    # Manually save the archive WITHOUT y_binary to mimic an old file.
    manifest = {
        "cfg": json.loads(_cfg_to_json(ds.cfg)),
        "specs": [s.to_dict() for s in ds.specs],
        "keep_components": False,
    }
    np.savez_compressed(
        path,
        manifest=np.array(json.dumps(manifest)),
        X=ds._X,
        y=ds._y,
    )
    loaded = NoiseSeriesDataset.load(path)
    # Derived y_binary should match nonzero-of-signed for every sample.
    for i in (0, len(ds) - 1):
        sign = loaded.labels(i, mode="signed")
        bin_ = loaded.labels(i, mode="binary")
        assert np.array_equal(bin_.astype(bool), sign != 0)


# Helper: serialize cfg to JSON exactly like the package does.
def _cfg_to_json(cfg) -> str:
    import json
    from chronoise.dataset import _cfg_to_dict  # private but tested here
    return json.dumps(_cfg_to_dict(cfg))
