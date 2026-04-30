"""PyTorch ``Dataset`` over the 360-element canonical grid + .npz persistence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import GeneratorConfig, NoiseKind, NoiseSpec, default_noise_specs
from .series import generate_series


@dataclass(frozen=True)
class SampleSpec:
    """A single (noise, amplitude, structural_mode, seed) configuration."""

    noise: NoiseSpec
    amplitude: float
    structural_mode: int
    seed: int

    def to_dict(self) -> dict:
        return {
            "noise_kind": self.noise.kind.value,
            "beta": float(self.noise.beta),
            "amplitude": float(self.amplitude),
            "structural_mode": int(self.structural_mode),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SampleSpec":
        return cls(
            noise=NoiseSpec(kind=NoiseKind(d["noise_kind"]), beta=float(d.get("beta", 0.0))),
            amplitude=float(d["amplitude"]),
            structural_mode=int(d["structural_mode"]),
            seed=int(d["seed"]),
        )


def default_specs(cfg: GeneratorConfig) -> list[SampleSpec]:
    """Iterate the canonical grid: 6 noise styles x 3 amplitudes x 2 modes x n_seeds."""
    out: list[SampleSpec] = []
    for noise in default_noise_specs(cfg):
        for amp in cfg.amplitudes:
            for mode in cfg.structural_modes:
                for s in range(cfg.n_seeds):
                    out.append(SampleSpec(
                        noise=noise,
                        amplitude=float(amp),
                        structural_mode=int(mode),
                        seed=cfg.seed_base + s,
                    ))
    return out


class NoiseSeriesDataset(Dataset):
    """In-memory PyTorch dataset of generated series.

    Each item is ``(X, y)`` where ``X`` has shape ``(T,)`` and ``y`` has shape
    ``(T,)`` with values in ``{-1, 0, +1}``. Pass ``return_meta=True`` to also
    receive a per-sample metadata dict (note: this disables default DataLoader
    collation since the dict carries strings).

    Set ``keep_components=True`` to also retain the level component ``L`` and
    the standardized noise component ``N`` (accessed via ``ds.components(idx)``).
    """

    def __init__(
        self,
        cfg: GeneratorConfig | None = None,
        specs: Sequence[SampleSpec] | None = None,
        *,
        keep_components: bool = False,
        return_meta: bool = False,
        x_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64,
    ):
        self.cfg = cfg if cfg is not None else GeneratorConfig()
        self.specs: list[SampleSpec] = list(specs) if specs is not None else default_specs(self.cfg)
        self.keep_components = bool(keep_components)
        self.return_meta = bool(return_meta)
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

        n = len(self.specs)
        T = self.cfg.T
        self._X = np.empty((n, T), dtype=np.float32)
        self._y = np.empty((n, T), dtype=np.int64)
        self._L: np.ndarray | None = np.empty((n, T), dtype=np.float32) if keep_components else None
        self._N: np.ndarray | None = np.empty((n, T), dtype=np.float32) if keep_components else None

        for i, s in enumerate(self.specs):
            r = generate_series(
                self.cfg,
                noise=s.noise,
                amplitude=s.amplitude,
                structural_mode=s.structural_mode,
                seed=s.seed,
            )
            self._X[i] = r.X.astype(np.float32, copy=False)
            self._y[i] = r.y.astype(np.int64, copy=False)
            if keep_components:
                self._L[i] = r.L.astype(np.float32, copy=False)
                self._N[i] = r.N.astype(np.float32, copy=False)

    # ---------------- Dataset interface ----------------

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._X[idx]).to(self.x_dtype)
        y = torch.from_numpy(self._y[idx]).to(self.y_dtype)
        if self.return_meta:
            return x, y, self.specs[idx].to_dict()
        return x, y

    def components(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (L, N) arrays for sample ``idx`` (requires keep_components=True)."""
        if self._L is None or self._N is None:
            raise RuntimeError("Dataset was built without keep_components=True")
        return self._L[idx], self._N[idx]

    # ---------------- Persistence ----------------

    def save(self, path: str | Path) -> None:
        """Save to a single ``.npz`` archive with an embedded JSON manifest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "cfg": _cfg_to_dict(self.cfg),
            "specs": [s.to_dict() for s in self.specs],
            "keep_components": self.keep_components,
        }
        arrays: dict[str, np.ndarray] = {"X": self._X, "y": self._y}
        if self.keep_components:
            arrays["L"] = self._L
            arrays["N"] = self._N
        # Manifest stored as a 0-d unicode array — survives ``allow_pickle=False``.
        np.savez_compressed(path, manifest=np.array(json.dumps(manifest)), **arrays)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        x_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64,
        return_meta: bool = False,
    ) -> "NoiseSeriesDataset":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest"]))
            cfg = _cfg_from_dict(manifest["cfg"])
            specs = [SampleSpec.from_dict(s) for s in manifest["specs"]]
            keep_components = bool(manifest.get("keep_components", "L" in data.files))

            obj = cls.__new__(cls)
            obj.cfg = cfg
            obj.specs = specs
            obj.keep_components = keep_components
            obj.return_meta = return_meta
            obj.x_dtype = x_dtype
            obj.y_dtype = y_dtype
            obj._X = np.asarray(data["X"], dtype=np.float32)
            obj._y = np.asarray(data["y"], dtype=np.int64)
            if keep_components:
                obj._L = np.asarray(data["L"], dtype=np.float32)
                obj._N = np.asarray(data["N"], dtype=np.float32)
            else:
                obj._L = None
                obj._N = None
            return obj


# ---------- (de)serialization helpers for GeneratorConfig ----------

_CFG_KEYS_INT = ("T", "T0", "geom_offset", "K", "T1", "T2", "n_seeds", "seed_base")
_CFG_KEYS_FLOAT = ("sigma_L", "gamma", "geom_p", "f_high", "phi_ar")
_CFG_KEYS_BOOL = ("normalize_beta_noise", "normalize_el_nino")
_CFG_KEYS_TUPLE_FLOAT = ("betas", "amplitudes")
_CFG_KEYS_TUPLE_INT = ("structural_modes",)


def _cfg_to_dict(cfg: GeneratorConfig) -> dict:
    d: dict = {k: getattr(cfg, k) for k in _CFG_KEYS_INT + _CFG_KEYS_FLOAT + _CFG_KEYS_BOOL}
    d["f_low"] = cfg.f_low  # nullable
    for k in _CFG_KEYS_TUPLE_FLOAT + _CFG_KEYS_TUPLE_INT:
        d[k] = list(getattr(cfg, k))
    return d


def _cfg_from_dict(d: dict) -> GeneratorConfig:
    kwargs: dict = {}
    for k in _CFG_KEYS_INT:
        kwargs[k] = int(d[k])
    for k in _CFG_KEYS_FLOAT:
        kwargs[k] = float(d[k])
    for k in _CFG_KEYS_BOOL:
        kwargs[k] = bool(d[k])
    kwargs["f_low"] = None if d.get("f_low") is None else float(d["f_low"])
    for k in _CFG_KEYS_TUPLE_FLOAT:
        kwargs[k] = tuple(float(x) for x in d[k])
    for k in _CFG_KEYS_TUPLE_INT:
        kwargs[k] = tuple(int(x) for x in d[k])
    return GeneratorConfig(**kwargs)
