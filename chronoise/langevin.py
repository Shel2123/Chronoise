"""Langevin trajectory generator (methods.tex Sec. 3).

Integrates the overdamped Langevin SDE

    dx/dt = sin(x) + sigma * xi(t)

by Euler-Maruyama on a uniform grid of step ``dt``, driven by a stationary
unit-variance discrete-time process ``xi`` of one of two families:

* coloured ``1/f^beta`` noise (Timmer-Konig sum of sinusoids), or
* the El-Nino-like biharmonic + AR(1) process,

both standardised to zero mean and unit variance before the ``sigma * sqrt(dt)``
scaling. Frequency support of the coloured driver is set from the integration
grid as ``f_min = 1 / (T * dt)`` and ``f_max = 1 / (2 * dt)``.

Per-sample labels track the index of the nearest equilibrium of the periodic
potential ``V(x) = cos(x)``: ``k_n = round(X_n / pi)`` and
``y_n = sign(k_n - k_{n-1})`` with ``y_0 = 0``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import NoiseKind, NoiseSpec

__all__ = [
    "colored_driver",
    "el_nino_driver",
    "draw_driver",
    "integrate_langevin",
    "labels_from_trajectory",
    "save_langevin_run",
    "load_langevin_run",
]


def _standardize(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std == 0.0:
        return x - x.mean()
    return (x - x.mean()) / std


def colored_driver(
    T: int,
    dt: float,
    beta: float,
    rng: np.random.Generator,
    *,
    K: int = 512,
    normalize: bool = True,
) -> np.ndarray:
    """Coloured 1/f^beta driver, sum of sinusoids (methods.tex Eq. 6).

    Frequencies span ``[1/(T*dt), 1/(2*dt)]`` on a uniform grid of ``K`` points;
    phases ``phi_k ~ U[0, 2*pi)``. Returned series has length ``T``.
    """
    f_min = 1.0 / (T * dt)
    f_max = 1.0 / (2.0 * dt)
    f = np.linspace(f_min, f_max, K, dtype=np.float64)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=K)
    amp = f ** (-beta / 2.0)
    n = np.arange(T, dtype=np.float64)
    args = 2.0 * np.pi * np.outer(n * dt, f) + phi[None, :]
    xi = (np.sin(args) * amp[None, :]).sum(axis=1)
    if normalize:
        xi = _standardize(xi)
    return xi


def el_nino_driver(
    T: int,
    dt: float,
    rng: np.random.Generator,
    *,
    T1: float = 50.0,
    T2: float = 120.0,
    phi_ar: float = 0.7,
    normalize: bool = True,
) -> np.ndarray:
    """El-Nino-like driver (methods.tex Eq. 8).

    sin(2*pi*t/T1) + 0.5*sin(2*pi*t/T2) + AR(1) residual with
    ``eps_n = phi * eps_{n-1} + eta_n``, ``eta_n ~ N(0, 1)``, ``eps_{-1} = 0``.
    Time ``t_n = n * dt`` for ``n = 0..T-1``.
    """
    n = np.arange(T, dtype=np.float64)
    t = n * dt
    base = np.sin(2.0 * np.pi * t / T1) + 0.5 * np.sin(2.0 * np.pi * t / T2)
    eta = rng.standard_normal(T).astype(np.float64)
    eps = np.empty(T, dtype=np.float64)
    prev = 0.0
    for i in range(T):
        prev = phi_ar * prev + eta[i]
        eps[i] = prev
    xi = base + eps
    if normalize:
        xi = _standardize(xi)
    return xi


def draw_driver(
    T: int,
    dt: float,
    nu: NoiseSpec,
    rng: np.random.Generator,
    *,
    K: int = 512,
    T1: float = 50.0,
    T2: float = 120.0,
    phi_ar: float = 0.7,
    normalize: bool = True,
) -> np.ndarray:
    """Dispatch to the driver selected by ``nu``."""
    if nu.kind is NoiseKind.BETA:
        return colored_driver(T, dt, nu.beta, rng, K=K, normalize=normalize)
    if nu.kind is NoiseKind.EL_NINO:
        return el_nino_driver(T, dt, rng, T1=T1, T2=T2, phi_ar=phi_ar, normalize=normalize)
    raise ValueError(f"Unknown noise kind: {nu.kind!r}")


def labels_from_trajectory(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y, k)`` from a trajectory (methods.tex Eq. 10-11).

    ``k_n = round(X_n / pi)`` is the nearest-equilibrium index;
    ``y_n = sign(k_n - k_{n-1})`` for ``n >= 1`` and ``y_0 = 0``.
    """
    k = np.round(X / math.pi).astype(np.int64)
    y = np.zeros(X.shape[0], dtype=np.int8)
    if X.shape[0] > 1:
        y[1:] = np.sign(np.diff(k)).astype(np.int8)
    return y, k


def integrate_langevin(
    T: int,
    dt: float,
    sigma: float,
    nu: NoiseSpec,
    seed: int,
    *,
    K: int = 512,
    x0: float = math.pi,
    T1: float = 50.0,
    T2: float = 120.0,
    phi_ar: float = 0.7,
    normalize_driver: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euler-Maruyama integration of ``dx/dt = sin(x) + sigma * xi(t)``.

    Returns ``(X, y, k_t)`` each of length ``T``:

    * ``X[0] = x0``; ``X[n+1] = X[n] + sin(X[n]) * dt + sigma * sqrt(dt) * xi[n]``
      for ``n = 0 .. T-2`` (methods.tex Eq. 2 / Alg. 1).
    * ``k_t[n] = round(X[n] / pi)``, the index of the nearest equilibrium.
    * ``y[n] = sign(k_t[n] - k_t[n-1])`` with ``y[0] = 0``.

    Two independent PCG64 streams are spawned from ``seed`` via ``SeedSequence``;
    the first drives the noise, the second is reserved (matches methods.tex
    Sec. 3.4).
    """
    if T < 1:
        raise ValueError("T must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    ss_noise, _ss_reserved = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(ss_noise)

    xi = draw_driver(
        T, dt, nu, rng,
        K=K, T1=T1, T2=T2, phi_ar=phi_ar, normalize=normalize_driver,
    )

    X = np.empty(T, dtype=np.float64)
    X[0] = float(x0)
    sqrt_dt = math.sqrt(dt)
    for n in range(T - 1):
        X[n + 1] = X[n] + math.sin(X[n]) * dt + sigma * sqrt_dt * xi[n]

    y, k = labels_from_trajectory(X)
    return X, y, k


def _nu_to_dict(nu: NoiseSpec) -> dict:
    return {"kind": nu.kind.value, "beta": float(nu.beta)}


def _nu_from_dict(d: dict) -> NoiseSpec:
    return NoiseSpec(kind=NoiseKind(d["kind"]), beta=float(d.get("beta", 0.0)))


def save_langevin_run(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    k_t: np.ndarray,
    *,
    T: int,
    dt: float,
    sigma: float,
    nu: NoiseSpec,
    seed: int,
) -> None:
    """Persist a single Langevin run alongside its 5-tuple manifest.

    Writes ``X``, ``y``, ``k`` arrays and a JSON manifest with
    ``(nu, sigma, dt, T, seed)`` into a compressed ``.npz`` archive.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "T": int(T),
        "dt": float(dt),
        "sigma": float(sigma),
        "seed": int(seed),
        "nu": _nu_to_dict(nu),
    }
    np.savez_compressed(
        path,
        X=np.asarray(X),
        y=np.asarray(y),
        k=np.asarray(k_t),
        manifest=np.array(json.dumps(manifest)),
    )


def load_langevin_run(path: str | Path) -> dict:
    """Load a run saved by :func:`save_langevin_run`.

    Returns a dict with keys ``X``, ``y``, ``k``, ``T``, ``dt``, ``sigma``,
    ``seed``, ``nu`` (a :class:`NoiseSpec`).
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        return {
            "X": np.asarray(data["X"]),
            "y": np.asarray(data["y"]),
            "k": np.asarray(data["k"]),
            "T": int(manifest["T"]),
            "dt": float(manifest["dt"]),
            "sigma": float(manifest["sigma"]),
            "seed": int(manifest["seed"]),
            "nu": _nu_from_dict(manifest["nu"]),
        }
