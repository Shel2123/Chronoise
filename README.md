# chronoise

Synthetic noise-driven time series with controllable structure for ML benchmarks.

Each series is generated as

```
X_t = L_t + a * N(t)
```

where `L_t` is a piecewise-constant level with stochastic step lengths and an
optional seasonal sign bias, and `N(t)` is one of six configurable noise styles
(five `1/f^beta` colored noises and one El-Nino-style noise). The canonical
dataset is a `6 x 3 x 2 x 10 = 360`-element grid.

## Install

```bash
pip install chronoise
```

Requires Python `>=3.10`, `numpy>=1.23`, `torch>=2.0`.

## Usage

```python
from chronoise import (
    GeneratorConfig, NoiseKind, NoiseSpec,
    generate_series, NoiseSeriesDataset,
)

cfg = GeneratorConfig()

# Single realization.
r = generate_series(
    cfg,
    noise=NoiseSpec(NoiseKind.BETA, beta=1.0),
    amplitude=1.0,
    structural_mode=1,
    seed=0,
)
# r.X        : (T,) float64 observed series
# r.y        : (T,) int8 signed direction labels in {-1, 0, +1}
# r.y_binary : (T,) int8 binary bifurcation labels in {0, 1}
# r.L, r.N   : level and standardized noise components

# Full canonical dataset (360 series in memory).
ds = NoiseSeriesDataset(cfg)                  # default: signed labels {-1, 0, +1}
ds.save("dataset.npz")
ds = NoiseSeriesDataset.load("dataset.npz")

x, y = ds[0]  # torch tensors

# Switch to binary bifurcation labels for the break-vs-no-break task:
ds_bin = NoiseSeriesDataset(cfg, label_mode="binary")
x, y = ds_bin[0]                              # y in {0, 1}

# Both encodings are always stored — access either without rebuilding:
y_signed = ds.labels(0, mode="signed")
y_binary = ds.labels(0, mode="binary")
```

For a runnable example see `examples/quickstart.py`.

## CLI

```bash
chronoise-build dataset.npz --T 8192 --n-seeds 10
```

Add `--keep-components` to also store the level (`L`) and noise (`N`)
components in the archive. Add `--label-mode {signed,binary}` to choose the
default label encoding (both arrays are always persisted; this flag only
selects what `__getitem__` returns):

```bash
chronoise-build dataset.npz --label-mode binary
```

## Configuration

All knobs live on `GeneratorConfig` (`chronoise/config.py`): series length,
level statistics (`sigma_L`, `gamma`, `T0`, segment-length geometric
parameters), colored-noise frequency grid (`K`, `f_low`, `f_high`),
El-Nino-style noise (`T1`, `T2`, `phi_ar`), and the canonical product axes
(`betas`, `amplitudes`, `structural_modes`, `n_seeds`).

Series with the same `seed` share the same level trace across noise types,
which is convenient for paired comparisons.

## Example

![Example](https://raw.githubusercontent.com/Shel2123/Chronoise/main/examples/example1.jpg)

## License
This project is licensed under the [MIT License](LICENSE).
