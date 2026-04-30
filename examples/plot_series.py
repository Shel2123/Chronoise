"""Plot a generated series X_t together with the level L_t and labels y_t.

Usage::

    PYTHONPATH=. python examples/plot_series.py            # default config
    PYTHONPATH=. python examples/plot_series.py --noise el_nino --a 3.14159
    PYTHONPATH=. python examples/plot_series.py --beta -2 --mode 0 --seed 7 \\
        --save out.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from chronoise import (
    GeneratorConfig,
    NoiseKind,
    NoiseSpec,
    generate_series,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noise", default="beta", choices=["beta", "el_nino"],
                   help="Noise kind (default: beta)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Spectral exponent for beta noise (default: 1.0)")
    p.add_argument("--a", "--amplitude", dest="amplitude", type=float, default=1.0,
                   help="Noise amplitude a (default: 1.0; try 1, 2, or pi=3.14159)")
    p.add_argument("--mode", type=int, default=1, choices=[0, 1],
                   help="Structural mode S (default: 1)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--T", type=int, default=10000,
                   help="Series length (default: 2048 for readable plots)")
    p.add_argument("--save", type=Path, default=None,
                   help="If given, save the figure to this path instead of showing it")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Headless backend if we are saving to file (works without a display).
    import matplotlib
    if args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = GeneratorConfig(T=args.T)
    if args.noise == "el_nino":
        spec = NoiseSpec(NoiseKind.EL_NINO)
    else:
        spec = NoiseSpec(NoiseKind.BETA, beta=args.beta)

    r = generate_series(
        cfg,
        noise=spec,
        amplitude=args.amplitude,
        structural_mode=args.mode,
        seed=args.seed,
    )

    t = np.arange(1, cfg.T + 1)
    X, L, y = r.X, r.L, r.y
    up = np.where(y == 1)[0]
    down = np.where(y == -1)[0]

    fig, (ax_x, ax_y) = plt.subplots(
        nrows=2, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
        dpi=300
    )

    # ---- Top: X_t, L_t, transition markers --------------------------------
    ax_x.plot(t, X, lw=0.6, color="0.55", label=r"$X_t$")
    ax_x.plot(t, L, lw=1.5, color="C0", label=r"$L_t$")
    if up.size:
        ax_x.scatter(t[up], L[up], s=22, marker="^", color="C2",
                     zorder=3, label=r"$y_t=+1$")
    if down.size:
        ax_x.scatter(t[down], L[down], s=22, marker="v", color="C3",
                     zorder=3, label=r"$y_t=-1$")
    title = (
        f"noise={spec.label()}, amplitude={args.amplitude:g}, "
        f"S={args.mode}, seed={args.seed}, T={cfg.T} "
        f"(up={up.size}, down={down.size})"
    )
    ax_x.set_title(title)
    ax_x.set_ylabel("value")
    ax_x.legend(loc="upper right", ncol=4, fontsize=9, framealpha=0.85)
    ax_x.grid(True, alpha=0.3)

    # ---- Bottom: y_t as a step plot ---------------------------------------
    ax_y.step(t, y, where="mid", lw=0.8, color="0.25")
    ax_y.set_yticks([-1, 0, 1])
    ax_y.set_ylim(-1.4, 1.4)
    ax_y.set_xlabel("t")
    ax_y.set_ylabel(r"$y_t$")
    ax_y.grid(True, alpha=0.3)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=130, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
