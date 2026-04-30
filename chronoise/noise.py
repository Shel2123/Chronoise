"""Background noise generators: 1/f^beta colored noise and El-Nino-style noise."""
from __future__ import annotations

import numpy as np

from .config import GeneratorConfig


def _frequency_grid(cfg: GeneratorConfig) -> np.ndarray:
    f_low = cfg.f_low if cfg.f_low is not None else 1.0 / cfg.T
    return np.linspace(f_low, cfg.f_high, cfg.K, dtype=np.float64)


def _zero_mean_unit_var(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std == 0.0:
        return x - x.mean()
    return (x - x.mean()) / std


def colored_noise(cfg: GeneratorConfig, beta: float, rng: np.random.Generator) -> np.ndarray:
    """N_beta(t) = sum_k f_k^{-beta/2} * sin(2*pi*f_k*t + phi_k), then standardized."""
    T = cfg.T
    f = _frequency_grid(cfg)                          # (K,)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=cfg.K)   # (K,)
    amp = f ** (-beta / 2.0)                          # (K,)
    t = np.arange(1, T + 1, dtype=np.float64)         # (T,)
    args = 2.0 * np.pi * np.outer(t, f) + phi[None, :]
    N = (np.sin(args) * amp[None, :]).sum(axis=1)
    if cfg.normalize_beta_noise:
        N = _zero_mean_unit_var(N)
    return N


def el_nino_noise(cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """sin(2*pi*t/T1) + 0.5*sin(2*pi*t/T2) + AR(1) residual with phi = phi_ar."""
    T = cfg.T
    t = np.arange(1, T + 1, dtype=np.float64)
    base = np.sin(2.0 * np.pi * t / cfg.T1) + 0.5 * np.sin(2.0 * np.pi * t / cfg.T2)
    eta = rng.standard_normal(T).astype(np.float64)
    eps = np.empty(T, dtype=np.float64)
    prev = 0.0  # eps_0 = 0 -> eps_1 = phi * 0 + eta_1
    phi = cfg.phi_ar
    for i in range(T):
        prev = phi * prev + eta[i]
        eps[i] = prev
    N = base + eps
    if cfg.normalize_el_nino:
        N = _zero_mean_unit_var(N)
    return N
