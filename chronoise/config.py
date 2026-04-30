"""Generator configuration and noise-style identifiers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class NoiseKind(str, Enum):
    BETA = "beta"
    EL_NINO = "el_nino"


@dataclass(frozen=True)
class NoiseSpec:
    """Identifier of a noise realization style.

    For ``BETA`` the spectral exponent is taken from ``beta``.
    For ``EL_NINO`` the ``beta`` field is unused.
    """

    kind: NoiseKind
    beta: float = 0.0

    def label(self) -> str:
        if self.kind is NoiseKind.EL_NINO:
            return "el_nino"
        return f"beta={self.beta:g}"


@dataclass(frozen=True)
class GeneratorConfig:
    # Series length
    T: int = 8192

    # Level component L_t
    sigma_L: float = 1.0
    gamma: float = 2.0
    T0: int = 500
    geom_offset: int = 20
    geom_p: float = 1.0 / 200.0

    # Colored 1/f^beta noise
    K: int = 512
    f_low: float | None = None  # default: 1 / T
    f_high: float = 0.5
    normalize_beta_noise: bool = True

    # El-Nino-style noise
    T1: int = 50
    T2: int = 120
    phi_ar: float = 0.7
    normalize_el_nino: bool = True

    # Default product axes for the canonical 6 x 3 x 2 x 10 = 360 dataset
    betas: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    amplitudes: Tuple[float, ...] = (1.0, 2.0, math.pi)
    structural_modes: Tuple[int, ...] = (0, 1)
    n_seeds: int = 10
    seed_base: int = 0


def default_noise_specs(cfg: GeneratorConfig) -> list[NoiseSpec]:
    """Return the canonical 6 noise styles: 5 beta-noise + 1 El-Nino."""
    specs: list[NoiseSpec] = [NoiseSpec(NoiseKind.BETA, b) for b in cfg.betas]
    specs.append(NoiseSpec(NoiseKind.EL_NINO))
    return specs
