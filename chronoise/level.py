"""Piecewise-constant level component L_t and its direction labels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeneratorConfig


@dataclass
class LevelTrace:
    L: np.ndarray            # (T,) float64
    breakpoints: np.ndarray  # (M+1,) int64 - tau_0=1, tau_1, ..., tau_M (>= T+1)
    levels: np.ndarray       # (M+1,) float64 - c_0, c_1, ..., c_M
    signs: np.ndarray        # (M,) int8 - s_k for k = 0..M-1
    deltas: np.ndarray       # (M,) float64 - Delta_k


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_level(
    cfg: GeneratorConfig,
    structural_mode: int,
    rng: np.random.Generator,
) -> LevelTrace:
    """Generate L_t for t = 1..T.

    structural_mode == 0: P(s_k = +1) = 0.5  (no seasonal sign bias)
    structural_mode == 1: P(s_k = +1) = sigmoid(gamma * sin(2 * pi * tau_k / T_0))
    """
    if structural_mode not in (0, 1):
        raise ValueError("structural_mode must be 0 or 1")

    T = cfg.T
    L = np.zeros(T, dtype=np.float64)

    c_curr = 0.0
    tau_curr = 1
    levels_rec: list[float] = [c_curr]
    taus_rec: list[int] = [tau_curr]
    signs_rec: list[int] = []
    deltas_rec: list[float] = []

    while tau_curr <= T:
        seg_len = cfg.geom_offset + int(rng.geometric(cfg.geom_p))
        tau_next = tau_curr + seg_len

        if structural_mode == 1:
            p_pos = _sigmoid(cfg.gamma * np.sin(2.0 * np.pi * tau_curr / cfg.T0))
        else:
            p_pos = 0.5
        sign = 1 if rng.uniform() < p_pos else -1
        # Delta_k ~ N+(0, sigma_L^2) — half-normal via |Z|.
        delta = float(abs(rng.normal(0.0, cfg.sigma_L)))

        start = tau_curr - 1
        end = min(tau_next - 1, T)
        L[start:end] = c_curr

        c_curr = c_curr + sign * delta
        tau_curr = tau_next
        levels_rec.append(c_curr)
        taus_rec.append(tau_curr)
        signs_rec.append(sign)
        deltas_rec.append(delta)

    return LevelTrace(
        L=L,
        breakpoints=np.asarray(taus_rec, dtype=np.int64),
        levels=np.asarray(levels_rec, dtype=np.float64),
        signs=np.asarray(signs_rec, dtype=np.int8),
        deltas=np.asarray(deltas_rec, dtype=np.float64),
    )


def labels_from_level(L: np.ndarray) -> np.ndarray:
    """Signed three-class direction labels.

    y_t = sign(L_t - L_{t-1}); y_1 = 0. Values are in {-1, 0, +1}.
    """
    y = np.zeros_like(L, dtype=np.int8)
    diff = np.diff(L)
    y[1:][diff > 0] = 1
    y[1:][diff < 0] = -1
    return y


def labels_binary_from_level(L: np.ndarray) -> np.ndarray:
    """Binary break/no-break labels.

    y_t = 1 if L_t != L_{t-1} else 0; y_1 = 0. Values are in {0, 1}.

    This is the bifurcation-event encoding used in the
    "break vs. no-break" detection task. It is equivalent to
    ``(labels_from_level(L) != 0).astype(np.int8)`` but is exposed as a
    first-class helper to avoid a redundant conversion and to mirror
    the signed variant ``labels_from_level``.
    """
    y = np.zeros_like(L, dtype=np.int8)
    diff = np.diff(L)
    y[1:][diff != 0] = 1
    return y
