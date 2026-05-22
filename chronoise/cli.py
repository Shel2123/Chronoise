"""CLI for building the canonical 360-element dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import GeneratorConfig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="chronoise-build",
        description="Generate the canonical chronoise dataset and save to .npz.",
    )
    p.add_argument("output", type=Path, help="Output .npz file")
    p.add_argument("--T", type=int, default=8192, help="Series length (default: 8192)")
    p.add_argument("--n-seeds", type=int, default=10, help="Seeds per configuration")
    p.add_argument("--seed-base", type=int, default=0, help="Starting seed value")
    p.add_argument(
        "--keep-components", action="store_true",
        help="Also store the level (L) and noise (N) components",
    )
    p.add_argument(
        "--label-mode", choices=("signed", "binary"), default="signed",
        help=(
            "Default label encoding stored in the archive: 'signed' for "
            "{-1, 0, +1} direction labels (default) or 'binary' for "
            "{0, 1} bifurcation labels. Both arrays are always persisted; "
            "this flag only selects which one __getitem__ returns by default."
        ),
    )
    args = p.parse_args(argv)

    from .dataset import NoiseSeriesDataset

    cfg = GeneratorConfig(T=args.T, n_seeds=args.n_seeds, seed_base=args.seed_base)
    ds = NoiseSeriesDataset(
        cfg=cfg,
        keep_components=args.keep_components,
        label_mode=args.label_mode,
    )
    ds.save(args.output)
    print(
        f"Saved {len(ds)} series of length {cfg.T} to {args.output} "
        f"(default label_mode={args.label_mode})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
