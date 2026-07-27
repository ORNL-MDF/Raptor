<img
  src="https://raw.githubusercontent.com/ORNL-MDF/raptor-media/main/images/Raptor-wordmark.svg"
  alt="Raptor"
>

# Raptor

Raptor predicts lack-of-fusion porosity in Laser Powder Bed Fusion (LPBF)
builds. It sweeps a time-dependent melt-pool cross-section along ordered scan
vectors and returns an Int8 voxel field containing defects, melted interiors,
boundaries, and repeated boundary interactions.

The optimized implementation is designed for large CPU calculations. It uses
an implicit voxel grid, independent width/depth/height spectra, packed
spectral histories, spatial candidate indexing, conservative vertical pruning,
and a parallel Numba kernel.

For the complete mathematics, numerical-error analysis, implementation,
optimization decisions, validation, and scaling results, see
[RAPTOR_METHODS.md](RAPTOR_METHODS.md).

## Method overview

Each melt-pool dimension \(Q_d\), for
\(d\in\{W,D,H\}\), is represented independently:

$$
Q_{d,j}(\tau)=
\sum_k A_{d,k}\cos\left(
2\pi f_{d,k}\tau+\phi_{d,j,k}+2\pi f_{d,k}t_{0,j}
\right).
$$

At transverse coordinate \(y\) and build coordinate \(z\), Raptor evaluates
the modified Lamé function

$$
F(y,z)=
\left(\frac{y}{W/2}\right)^2+
\left(\frac{|z|}{B(z)}\right)^{n(z)},
$$

where \(B=H\) above the scan plane and \(B=D\) below it. The nominal melt
region satisfies \(F<1\). A one-voxel radial band identifies boundaries.
Depth and height exponents may be any positive value; the transverse width
exponent is fixed at two.

The production kernel currently requires horizontal scan vectors with local
build direction aligned to global \(z\).

## Installation

Raptor requires Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

No C or C++ extension is built during installation. Numba compiles the
performance-critical kernels on first use and caches them.

For development:

```bash
python -m pip install -e ".[test,dev]"
pre-commit run --all-files
python -m pytest -q
```

## JIT cache and cold start

Production signatures can be compiled before a simulation:

```bash
export NUMBA_CACHE_DIR=/path/to/persistent/raptor-numba-cache
raptor-warm-cache --cache-dir "$NUMBA_CACHE_DIR"
```

Use a separate cache for each Python/Numba version and CPU model. Cache
warming changes startup time only; it does not change numerical results.

## Examples

Run the quick synthetic API example:

```bash
cd examples/synthetic_api_example
python run_synthetic_rve.py
```

It evaluates 1,030,301 voxels and writes `rve.vti` and
`rve_morphology.csv`. The documented billion-voxel profile is available
through:

```bash
python benchmark_scaling.py \
    --profile reference \
    --study strong \
    --threads 8 16 32 64
```

Run the YAML-driven CLI example from the repository root:

```bash
raptor examples/cli_example/input.yaml
```

The complete configuration is in
[examples/cli_example/input.yaml](examples/cli_example/input.yaml).
Relative data and output paths are resolved from the YAML file's directory.

## Python API

This constant-dimension example shows the complete computational workflow:

```python
import numpy as np

from raptor.api import (
    compute_porosity,
    create_grid,
    create_melt_pool,
    create_path_vectors,
    write_vtk,
)

minimum = np.array([0.0, 0.0, 0.0])
maximum = np.array([5.0e-4, 5.0e-4, 5.0e-4])
bounds = np.array([minimum, maximum])
resolution = 5.0e-6

grid = create_grid(resolution, bound_box=bounds)
vectors = create_path_vectors(
    bounds,
    power=370.0,
    scan_speed=1.7,
    hatch_spacing=140.0e-6,
    layer_height=30.0e-6,
    rotation=67.0,
    scan_extension=5.0e-4,
    extra_layers=0,
)

width = np.array([[148.0e-6, 0.0, 0.0]])
depth = np.array([[118.0e-6, 0.0, 0.0]])
height = np.array([[30.0e-6, 0.0, 0.0]])
melt_pool = create_melt_pool(
    {
        "width": (width, None, 1.0, 2.0),
        "depth": (depth, None, 1.0, 1.0),
        "height": (height, None, 1.0, 1.0),
    },
    enable_random_phases=False,
)

phase = compute_porosity(
    grid,
    vectors,
    melt_pool,
    random_seed=42,
    # memory_limit_mb=4096,  # Optional per-process allocation for ensembles.
)
write_vtk(grid.origin, grid.resolution, phase, "rve.vti")
```

Time-series arrays use columns `[time, value]`. Direct spectral arrays use
`[amplitude, frequency, phase]`. Width, depth, and height may have different
mode counts and phases.

## Outputs

The phase codes are:

| Code | Meaning |
|---:|---|
| 0 | no retained melt-pool interaction; treated as defect |
| 1 | melted interior |
| 2 | one-voxel melt-pool boundary |
| 3 | repeated boundary interaction |

`write_vtk` produces compressed VTK ImageData compatible with ParaView.
`compute_morphology` and `write_morphology` calculate and save connected-pore
properties. `visualize` displays phase-zero defects with PyVista.

## Numerical and performance controls

The defaults are intended for ordinary use:

- `spectral_error_fraction=0.25` bounds guarded spectral and interpolation
  error to one quarter of a voxel.
- `tile_width=None` selects the tested automatic spatial-index width.
- `memory_limit_mb=None` first checks `RAPTOR_MEMORY_LIMIT_MB`; if neither is
  set, Raptor uses 80% of memory currently available to the process. Linux
  cgroup limits are honored.

The memory budget is per process and shared by all Numba threads; it is not
divided by the core count. API and YAML values take precedence over the
environment. Despite the user-facing `MB` name, values are converted using
`1024**2` bytes so allocation is deterministic.

When the complete packed spectral table and phase field fit, Raptor uses the
fast resident path. Under a smaller budget it streams contiguous vector
batches in scan order. Sampling density, independent dimension histories, and
phase labels remain unchanged. If even the phase field, fixed workspace, and
one vector history cannot fit, the error prints copy-ready API, YAML, and
environment settings. The automatic 20% headroom is for Python, libraries,
output, and changes in system load; an explicit limit receives no additional
fractional reduction. Set a smaller per-job value when running ensembles.

Configure the same per-process limit in one place:

```python
phase = compute_porosity(grid, vectors, melt_pool, memory_limit_mb=4096)
```

```yaml
parameters:
  memory_limit_mb: 4096
```

or, for an ensemble launcher:

```bash
export RAPTOR_MEMORY_LIMIT_MB=4096
```

The documented billion-voxel case uses an approximately 957 MB Int8 phase
field and a 12.10 MB full spectral table. Table size depends primarily on
active scan histories, spectral curvature, and the requested error fraction.

## Reference performance

On the documented virtualized 64-core AMD EPYC 7702 allocation, the
1,003,003,001-voxel profile achieved a median 1.210 s mask time, or
828.6 million voxels per second, at 64 threads. Smaller controlled comparisons
were 20.4–30.1 times faster than main commit
`e6e7fb3794fc2ae977023d4464148ce676128369`. These measurements include
spectral-table construction but exclude VTK and morphology and should not be
generalized to other hardware without remeasurement.

## Current constraints

- Production scan vectors must be horizontal.
- Width shape exponent must equal two.
- Evaluated dimensions must remain finite and positive.
- The guarded Float32 spectral approximation must satisfy the selected
  voxel-relative error budget.
- Phase updates are ordered; changing scan-vector order may change boundary
  and intersection labels.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests should pass:

```bash
pre-commit run --all-files
NUMBA_NUM_THREADS=2 python -m pytest -q
python -m build
```

## Reference data

The example melt-pool measurements are from:

Miner, Justin; Narra, Sneha Prabha (2024). *Dataset of Melt Pool Variability
Measurements for Powder Bed Fusion - Laser Beam of Ti-6Al-4V*. Carnegie Mellon
University. <https://doi.org/10.1184/R1/25696293.v1>

## Contributors

- Vamsi Subraveti, Vanderbilt University
- John Coleman, Oak Ridge National Laboratory
- Çağlar Oskay, Vanderbilt University
- Alex Plotkowski, Oak Ridge National Laboratory

## License

Raptor is distributed under the BSD 3-Clause [License](LICENSE).
