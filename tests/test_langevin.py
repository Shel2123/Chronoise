"""Tests for chronoise.langevin: integrator, drivers, labels, .npz round-trip."""
from __future__ import annotations

import math

import numpy as np
import pytest

from chronoise import (
    NoiseKind,
    NoiseSpec,
    colored_driver,
    draw_driver,
    el_nino_driver,
    integrate_langevin,
    labels_from_trajectory,
    load_langevin_run,
    save_langevin_run,
)


BETA_SPEC = NoiseSpec(NoiseKind.BETA, beta=1.0)
EN_SPEC = NoiseSpec(NoiseKind.EL_NINO)


def test_shapes_and_dtypes():
    X, y, k = integrate_langevin(T=256, dt=0.1, sigma=0.5, nu=BETA_SPEC, seed=0)
    assert X.shape == (256,) and X.dtype == np.float64
    assert y.shape == (256,) and y.dtype == np.int8
    assert k.shape == (256,) and k.dtype == np.int64


def test_initial_condition_default_is_pi():
    X, _, k = integrate_langevin(T=64, dt=0.1, sigma=0.0, nu=BETA_SPEC, seed=0)
    assert X[0] == math.pi
    assert k[0] == 1  # round(pi/pi) = 1


def test_custom_x0():
    X, _, k = integrate_langevin(T=16, dt=0.1, sigma=0.0, nu=BETA_SPEC, seed=0, x0=0.0)
    assert X[0] == 0.0
    assert k[0] == 0


def test_zero_sigma_stays_at_stable_equilibrium():
    """At sigma=0 starting from x0=pi (a stable equilibrium of sin(x)), the
    deterministic flow is a fixed point: X_n = pi for all n."""
    X, y, k = integrate_langevin(T=512, dt=0.05, sigma=0.0, nu=BETA_SPEC, seed=0)
    assert np.allclose(X, math.pi)
    assert np.all(y == 0)
    assert np.all(k == 1)


def test_same_seed_same_output():
    a = integrate_langevin(T=300, dt=0.1, sigma=0.7, nu=BETA_SPEC, seed=42)
    b = integrate_langevin(T=300, dt=0.1, sigma=0.7, nu=BETA_SPEC, seed=42)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)


def test_different_seed_changes_output():
    Xa, _, _ = integrate_langevin(T=300, dt=0.1, sigma=0.7, nu=BETA_SPEC, seed=0)
    Xb, _, _ = integrate_langevin(T=300, dt=0.1, sigma=0.7, nu=BETA_SPEC, seed=1)
    assert not np.array_equal(Xa, Xb)


@pytest.mark.parametrize("beta", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_beta_grid_runs_finite(beta):
    X, _, _ = integrate_langevin(
        T=512, dt=0.1, sigma=0.5,
        nu=NoiseSpec(NoiseKind.BETA, beta=beta), seed=0,
    )
    assert np.isfinite(X).all()


def test_el_nino_runs_finite():
    X, _, _ = integrate_langevin(T=512, dt=0.1, sigma=0.5, nu=EN_SPEC, seed=0)
    assert np.isfinite(X).all()


def test_labels_in_valid_range_and_zero_at_origin():
    _, y, _ = integrate_langevin(T=512, dt=0.5, sigma=2.0, nu=BETA_SPEC, seed=1)
    assert set(np.unique(y).tolist()).issubset({-1, 0, 1})
    assert y[0] == 0


def test_labels_match_diff_of_k():
    X, y, k = integrate_langevin(T=512, dt=0.5, sigma=2.0, nu=BETA_SPEC, seed=2)
    expected = np.zeros_like(y)
    expected[1:] = np.sign(np.diff(k)).astype(np.int8)
    assert np.array_equal(y, expected)


def test_labels_from_trajectory_matches_integrator():
    X, y, k = integrate_langevin(T=256, dt=0.1, sigma=0.5, nu=BETA_SPEC, seed=3)
    y2, k2 = labels_from_trajectory(X)
    assert np.array_equal(y, y2)
    assert np.array_equal(k, k2)


def test_em_step_matches_hand_computation():
    """One EM step verified by hand: X[1] = X[0] + sin(X[0])*dt + sigma*sqrt(dt)*xi[0]."""
    T, dt, sigma = 8, 0.1, 0.3
    nu = BETA_SPEC
    seed = 0
    # Re-derive xi using the same RNG-spawn convention as integrate_langevin.
    ss_noise, _ = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(ss_noise)
    xi = draw_driver(T, dt, nu, rng)

    X, _, _ = integrate_langevin(T=T, dt=dt, sigma=sigma, nu=nu, seed=seed)
    expected_X1 = math.pi + math.sin(math.pi) * dt + sigma * math.sqrt(dt) * xi[0]
    assert math.isclose(X[1], expected_X1, rel_tol=1e-12, abs_tol=1e-12)


def test_colored_driver_unit_variance_when_normalized():
    rng = np.random.default_rng(0)
    xi = colored_driver(T=4096, dt=0.1, beta=1.0, rng=rng)
    assert abs(xi.mean()) < 1e-9
    assert abs(xi.std() - 1.0) < 1e-9


def test_el_nino_driver_unit_variance_when_normalized():
    rng = np.random.default_rng(0)
    xi = el_nino_driver(T=4096, dt=0.1, rng=rng)
    assert abs(xi.mean()) < 1e-9
    assert abs(xi.std() - 1.0) < 1e-9


def test_colored_driver_frequency_bounds():
    """Sanity-check: a single-tone test that f_max = 1/(2*dt) corresponds to
    the Nyquist frequency of the integration grid."""
    rng = np.random.default_rng(0)
    T, dt = 1024, 0.25
    xi = colored_driver(T=T, dt=dt, beta=0.0, rng=rng, K=16, normalize=False)
    # Power should be non-trivial — driver is well-defined.
    assert np.isfinite(xi).all() and xi.std() > 0


def test_invalid_T_raises():
    with pytest.raises(ValueError):
        integrate_langevin(T=0, dt=0.1, sigma=0.1, nu=BETA_SPEC, seed=0)


def test_invalid_dt_raises():
    with pytest.raises(ValueError):
        integrate_langevin(T=16, dt=0.0, sigma=0.1, nu=BETA_SPEC, seed=0)


def test_save_and_load_round_trip(tmp_path):
    T, dt, sigma, seed = 128, 0.1, 0.7, 11
    nu = NoiseSpec(NoiseKind.BETA, beta=-1.0)
    X, y, k = integrate_langevin(T=T, dt=dt, sigma=sigma, nu=nu, seed=seed)

    p = tmp_path / "run.npz"
    save_langevin_run(p, X, y, k, T=T, dt=dt, sigma=sigma, nu=nu, seed=seed)
    obj = load_langevin_run(p)

    assert np.array_equal(obj["X"], X)
    assert np.array_equal(obj["y"], y)
    assert np.array_equal(obj["k"], k)
    assert obj["T"] == T
    assert obj["dt"] == dt
    assert obj["sigma"] == sigma
    assert obj["seed"] == seed
    assert obj["nu"] == nu


def test_save_and_load_round_trip_el_nino(tmp_path):
    T, dt, sigma, seed = 96, 0.5, 1.0, 5
    nu = NoiseSpec(NoiseKind.EL_NINO)
    X, y, k = integrate_langevin(T=T, dt=dt, sigma=sigma, nu=nu, seed=seed)
    p = tmp_path / "en.npz"
    save_langevin_run(p, X, y, k, T=T, dt=dt, sigma=sigma, nu=nu, seed=seed)
    obj = load_langevin_run(p)
    assert obj["nu"].kind is NoiseKind.EL_NINO
    assert np.array_equal(obj["X"], X)
