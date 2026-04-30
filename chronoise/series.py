"""Single-realization series generation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeneratorConfig, NoiseKind, NoiseSpec
from .level import LevelTrace, generate_level, labels_from_level
from .noise import colored_noise, el_nino_noise


@dataclass
class SeriesResult:
    X: np.ndarray             # (T,) float64 — observed series X_t = L_t + a*N_t
    y: np.ndarray             # (T,) int8 — direction labels in {-1, 0, +1}
    L: np.ndarray             # (T,) float64 — level component
    N: np.ndarray             # (T,) float64 — (standardized) noise component
    amplitude: float
    structural_mode: int
    noise: NoiseSpec
    seed: int
    level_trace: LevelTrace


def _spawn_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Return (rng_level, rng_noise) drawn from independent SeedSequence streams.

    Splitting the seed makes the level trace identical for a given seed across
    different noise types — useful for paired comparisons.
    """
    ss_level, ss_noise = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(ss_level), np.random.default_rng(ss_noise)


def generate_series(
    cfg: GeneratorConfig,
    *,
    noise: NoiseSpec,
    amplitude: float,
    structural_mode: int,
    seed: int,
) -> SeriesResult:
    rng_level, rng_noise = _spawn_rngs(seed)
    trace = generate_level(cfg, structural_mode, rng_level)
    if noise.kind is NoiseKind.BETA:
        N = colored_noise(cfg, noise.beta, rng_noise)
    elif noise.kind is NoiseKind.EL_NINO:
        N = el_nino_noise(cfg, rng_noise)
    else:
        raise ValueError(f"Unknown noise kind: {noise.kind!r}")
    X = trace.L + amplitude * N
    y = labels_from_level(trace.L)
    return SeriesResult(
        X=X,
        y=y,
        L=trace.L,
        N=N,
        amplitude=float(amplitude),
        structural_mode=int(structural_mode),
        noise=noise,
        seed=int(seed),
        level_trace=trace,
    )
