"""Determinism, seed stability, shapes, and label range for single realizations."""
from __future__ import annotations

import numpy as np
import pytest

from chronoise import (
    GeneratorConfig,
    NoiseKind,
    NoiseSpec,
    generate_series,
)


def _gen(cfg, *, kind=NoiseKind.BETA, beta=1.0, amp=1.0, mode=1, seed=0):
    return generate_series(
        cfg,
        noise=NoiseSpec(kind, beta=beta),
        amplitude=amp,
        structural_mode=mode,
        seed=seed,
    )


def test_same_seed_same_output(small_cfg):
    a = _gen(small_cfg, seed=42)
    b = _gen(small_cfg, seed=42)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.L, b.L)
    assert np.array_equal(a.N, b.N)


def test_different_seed_changes_output(small_cfg):
    a = _gen(small_cfg, seed=0)
    b = _gen(small_cfg, seed=1)
    assert not np.array_equal(a.X, b.X)
    assert not np.array_equal(a.N, b.N)


def test_level_stable_across_noise_kinds(small_cfg):
    """Same seed must yield identical L across noise types — paired-comparison contract."""
    a = _gen(small_cfg, kind=NoiseKind.BETA, beta=1.0, seed=7)
    b = _gen(small_cfg, kind=NoiseKind.BETA, beta=-2.0, seed=7)
    c = _gen(small_cfg, kind=NoiseKind.EL_NINO, seed=7)
    assert np.array_equal(a.L, b.L)
    assert np.array_equal(a.L, c.L)


def test_level_changes_with_seed():
    # Force multiple segments per trace so seeds 0 and 1 both produce non-trivial L.
    cfg = GeneratorConfig(T=512, geom_offset=4, geom_p=0.2)
    a = _gen(cfg, seed=0)
    b = _gen(cfg, seed=1)
    assert not np.array_equal(a.L, b.L)
    assert a.level_trace.breakpoints.size > 2
    assert b.level_trace.breakpoints.size > 2


def test_shapes_and_dtypes(small_cfg):
    r = _gen(small_cfg)
    T = small_cfg.T
    assert r.X.shape == (T,) and r.X.dtype == np.float64
    assert r.y.shape == (T,) and r.y.dtype == np.int8
    assert r.L.shape == (T,) and r.L.dtype == np.float64
    assert r.N.shape == (T,) and r.N.dtype == np.float64


def test_labels_in_valid_range(small_cfg):
    r = _gen(small_cfg)
    assert set(np.unique(r.y).tolist()).issubset({-1, 0, 1})
    # Labels are derived from L diffs; the first label must be 0 by construction.
    assert r.y[0] == 0


@pytest.mark.parametrize("beta", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_beta_grid_runs(small_cfg, beta):
    r = _gen(small_cfg, beta=beta, seed=0)
    assert np.isfinite(r.X).all()


def test_invalid_structural_mode(small_cfg):
    with pytest.raises(ValueError):
        _gen(small_cfg, mode=2, seed=0)


def test_amplitude_zero_is_pure_level(small_cfg):
    r = _gen(small_cfg, amp=0.0, seed=0)
    assert np.array_equal(r.X, r.L)


def test_normalized_noise_is_unit_variance(small_cfg):
    r = _gen(small_cfg, beta=1.0, seed=0)
    assert abs(r.N.mean()) < 1e-9
    assert abs(r.N.std() - 1.0) < 1e-9
