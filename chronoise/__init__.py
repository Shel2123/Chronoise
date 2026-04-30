"""chronoise — synthetic noise-driven time series for ML benchmarking.

The package implements

    X_t = L_t + a * N(t)

where ``L_t`` is a piecewise-constant level with stochastic step lengths and
sign bias, and ``N(t)`` is one of six configurable noise styles (five
``1/f^beta`` colored noises and one El-Niño-style noise). The canonical
dataset is a 6 x 3 x 2 x 10 = 360-element grid.

Quick usage::

    from chronoise import (
        GeneratorConfig, NoiseSpec, NoiseKind,
        generate_series, NoiseSeriesDataset,
    )

    cfg = GeneratorConfig()
    r = generate_series(cfg, noise=NoiseSpec(NoiseKind.BETA, 1.0),
                        amplitude=1.0, structural_mode=1, seed=0)

    ds = NoiseSeriesDataset(cfg)        # builds all 360 series in memory
    ds.save("dataset.npz")
    ds2 = NoiseSeriesDataset.load("dataset.npz")
"""
from .config import GeneratorConfig, NoiseKind, NoiseSpec, default_noise_specs
from .dataset import NoiseSeriesDataset, SampleSpec, default_specs
from .level import LevelTrace, generate_level, labels_from_level
from .noise import colored_noise, el_nino_noise
from .series import SeriesResult, generate_series

__all__ = [
    "GeneratorConfig",
    "NoiseKind",
    "NoiseSpec",
    "default_noise_specs",
    "NoiseSeriesDataset",
    "SampleSpec",
    "default_specs",
    "LevelTrace",
    "generate_level",
    "labels_from_level",
    "colored_noise",
    "el_nino_noise",
    "SeriesResult",
    "generate_series",
]

__version__ = "0.1.0"
