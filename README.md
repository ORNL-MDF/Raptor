<img src="https://raw.githubusercontent.com/ORNL-MDF/raptor-media/main/images/Raptor-wordmark.svg" alt="image">

---

Raptor is a Python-based simulation tool for estimating porosity-related defects in Laser Powder Bed Fusion (LPBF) additive manufacturing processes. It uses a computationally efficient geometric approach to model the dynamic melt pool and identify regions of unmelted material, which correspond to lack-of-fusion pores. The core of Raptor is a geometric model of the melt pool cross-section whose dimensions (width, depth, and height) oscillate over time. By analyzing the volume swept by this dynamic melt pool along the laser scan paths, Raptor generates a 3D map of the final part's porosity.

Journal-ready algorithm, numerical-method, optimization, validation, and
scaling documentation is available in the
[`docs/` directory](https://github.com/ORNL-MDF/Raptor/tree/main/docs).

## License

This project is licensed under the BSD 3-Clause [License](LICENSE).

## Contributors

*   Vamsi Subraveti, Vanderbilt University, vamsi.r.subraveti@vanderbilt.edu
*   John Coleman, Oak Ridge National Laboratory, colemanjs@ornl.gov
*   Çağlar Oskay, Vanderbilt University, caglar.oskay@vanderbilt.edu
*   Alex Plotkowski, Oak Ridge National Laboratory, plotkowskiaj@ornl.gov

## How It Works

**Raptor predicts porosity by following a multi-step process:**

*  **Domain Voxelization**: A 3D bounding box, or Representative Volume Element (RVE), is defined and discretized into a uniform grid of voxels.
*  **Scan Path Ingestion**: Scan path data is used to calculating the timing and trajectory for each laser vector.
*  **Dynamic Melt Pool Definition**: For each melt pool dimension (width, depth, height), the input time-series data is converted into a Fourier series (a sum of cosine functions). This creates a dynamic, time-dependent model of the melt pool's cross-sectional shape, which is modeled using modified Lamé curves. To capture stochastic process variations, a random phase shift can be applied to the Fourier series for each scan vector.
*  **Melt Mask Calculation**: The core of the simulation iterates through each voxel in the domain. For each scan vector that passes near the voxel, it calculates the instantaneous melt pool shape and determines the voxel state. The voxel is either unmelted (0), on the interior of the melt pool (1), on the boundary of the melt pool (2), or at an intersection of melt pool boundaries (3). The voxel state updates dynamically as the scan vectors that interact with it successively melt / interact with it on the melt pool boundary. The outcome of this process is a simulated volume of overlapping melt pools, flagging melted voxels, boundary voxels, intersection voxels, and voxels that make up the defect structure. The core geometry computations are executed with a high-performance parallel kernel, Just-In-Time (JIT) compiled with Numba. This enables the rapid analysis of large, industrially-relevant domains.
*  **Porosity Prediction**: Any voxel that is not melted by the end of the simulation is flagged as porosity.
*  **Boundary Tracking**: Voxels that are close to local melt pool boundaries are flagged as boundaries.
*  **Intersection Tracking**: A boundary update applied to a voxel already carrying a boundary or intersection code is flagged as an intersection point.
*  **Analysis and Output**: The final 3D volume is saved in the binary VTK ImageData (`.vti`) format. The morphological characteristics (e.g., volume, surface area, equivalent diameter) of contiguous pore structures can be quantified using the `scikit-image` library, and saved to a `.csv` file.

<figure style="text-align:center;">
  <img
    src="https://raw.githubusercontent.com/ORNL-MDF/raptor-media/main/images/example_phases_annotated.gif"
    alt="example animation"
    style="width:60%; height:auto;"
  >
  <figcaption>
    Visualization of the melt pool overlaps with defects, interior, boundaries and intersections tracked.
  </figcaption>
</figure>

## Installation

Raptor requires Python 3.10 or newer. The following Python packages are installed
with the package:
```bash
    numpy, numba, scipy, matplotlib, pyyaml, vtk, scikit-image, pandas, pyvista
```

*   **NumPy**: For numerical operations and array manipulation.
*   **Numba**: For JIT compilation and performance acceleration.
*   **PyYAML**: For reading and parsing YAML configuration files
*   **VTK**: For writing the output porosity map in `.vti` format
*   **scikit-image**: For calculating pore morphologies.
*   **pandas**: For writing morphology information to .csv
*   **pyvista**: For visualization of `.vti` results.

You can install all runtime dependencies and Raptor itself by running
`python -m pip install .` in the cloned Raptor directory. Contributors can
install the development checks with
`python -m pip install -e ".[test,dev]"`.

It's highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage these dependencies.

### Numba cache and cold-start setup

Raptor is a Python package and does not require the user to build a C/C++
extension. Its performance-critical CPU kernels are compiled by Numba when
they are first used and then cached. For development, use an editable
installation so the imported source and its cache identity remain stable:

```bash
python -m pip install -e .
```

For containers, batch jobs, or compute nodes, select a persistent directory
and precompile the production signatures before launching simulations:

```bash
export NUMBA_CACHE_DIR=/path/to/persistent/raptor-numba-cache
raptor-warm-cache --cache-dir "$NUMBA_CACHE_DIR"
```

Use a separate cache for each Python/Numba version and CPU model. Cache warming
moves compilation out of the first simulation; it does not change the
numerical result. The optimized CPU kernel operates on horizontal scan vectors
and reports a clear error if a different vector orientation is supplied.

### Run the synthetic example

From a fresh clone, the complete installation, precompilation, and example
workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export NUMBA_CACHE_DIR="$PWD/.numba-cache"
raptor-warm-cache --cache-dir "$NUMBA_CACHE_DIR"
cd examples/synthetic_api_example
python run_synthetic_rve.py
```

The cache-warming command is optional: without it, the first simulation
compiles the kernels automatically. The synthetic example contains about one
billion voxels, uses approximately 1.35 GiB of peak working memory for the
melt-mask calculation, and writes `rve.vti` plus `rve_morphology.csv` in the
example directory. Its elapsed time depends strongly on CPU core count, memory
bandwidth, and whether the Numba cache is warm.

## Usage

The project is organized into several modules:

*   `cli.py`: Handles command-line argument parsing and manages the main simulation workflow.
*   `api.py`: Provides high-level functions for creating the grid, melt pool, running the simulation, and writing output files.
*   `core.py`: Contains the core Numba-accelerated functions for calculating the melt mask.
*   `structures.py`: Defines the main data structures for the simulation (`Grid`, `MeltPool`, `PathVector`).
*   `io.py`: Contains functions for reading and parsing input files (scan paths, melt pool data).
*   `utilities.py`: Includes helper classes, such as the `ScanPathBuilder` for generating scan strategies.

Raptor can be used in two primary ways: through its Command-Line Interface (CLI) for quick, configuration-driven simulations, or as a Python Library (API) for integration into custom scripts and simulation workflows.

### 1. Command-Line Interface (CLI)

The CLI usage requires scan path files corresponding to build information. These scan path files can be generated with the `ScanPathBuilder` class in `raptor.utilities`. The CLI is controlled by a single YAML input file that defines all inputs, parameters, and outputs. The CLI example contains a single scan path to show functionality and observe the fluctuations of the simulated melt pool. The API example is recommended for a more descriptive simulation of undermelting-induced defects.

**How to Run (CLI):**

1.  **Prepare Inputs**: Create scan path files, melt pool data files, and a `input.yaml` file (detailed below).
2.  **Execute Script**: Run the following command from your terminal, providing the path to your configuration file:
    ```bash
    raptor path/to/your/input.yaml
    ```
3.  **Check Outputs**:
    *   Progress will be printed to the console.
    *   The 3D porosity map is saved to the `.vti` file specified in the config.
    *   The pore morphology data is saved to the `.csv` file (if configured -- the example does not save the morphology information.).

#### CLI Input: The `input.yaml` File

Running Raptor from the CLI requires a YAML input file to specify all parameters.

**Example `input.yaml`:**
```yaml
# List of scan path files (relative to this config file's location)
scan_paths:
  - "scan_paths/layer_01.txt"
  - "scan_paths/layer_02.txt"

parameters:
  layer_height: 5.0e-5      # Layer height in meters
  voxel_resolution: 5.0e-6  # Voxel resolution in meters
  enable_random_segment_phase: true # Use random phases for melt pool oscillations per vector
  random_seed: 42            # Optional reproducible segment phases
  tile_width: null           # Optional physical tile width in meters
  spectral_error_fraction: 0.25
  max_spectral_table_bytes: 268435456

# Melt pool dimension data. Can be 'time_series' or 'spectral_components'.
melt_pool_data:
  width:
    type: "time_series"
    file_name: "melt_pool_data/width_timeseries.txt"
    nmodes: 10          # Number of Fourier modes to extract
    scale: 1.0          # Multiplicative scaling factor for amplitudes
  depth:
    type: "time_series"
    file_name: "melt_pool_data/depth_timeseries.txt"
    nmodes: 10
    scale: 1.0
    shape: 2.0          # Shape factor 'n' for the Lame curve (n=2 is elliptical)
  height:
    type: "time_series"
    file_name: "melt_pool_data/height_timeseries.txt"
    nmodes: 10
    scale: 1.0
    shape: 0.5          # n=0.5 is bell-shaped

# Representative Volume Element (RVE) defining the simulation domain
rve:
  min_point: [0.0, 0.0, 0.0]          # [x, y, z] minimum corner in meters
  max_point: [0.001, 0.001, 0.0001]   # [x, y, z] maximum corner in meters

# Output settings
output:
  vtk:
    file_name: "porosity_map.vti"
  morphology:
    file_name: "morphology.csv"
    # Properties from scikit-image regionprops
    # See: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    fields:
      - "label"
      - "area"
      - "equivalent_diameter_area"
```

#### Configuration Details:
* **Scan Path Files**: Each file in `scan_paths` should be a space-delimited text file. **The first line is treated as a header and is skipped.**
   **Format per line:**
   `mode x y z power parameter`

   *   `mode`: Integer (`0` for line raster, `1` for point source/delay).
   *   `x, y, z`: Float endpoint coordinates in meters.
   *   `power`: Float laser power in Watts (for information, not used in the geometry model).
   *   `parameter`: Float (`mode=0`: scanning speed in m/s; `mode=1`: duration in seconds).

   **Example `layer_01.txt`:**
   ```
   # Mode X_end(m) Y_end(m) Z_end(m) Power(W) Speed(m/s)_or_Time(s)
   1 0.000 0.000 0.000 0   0.00001
   0 0.001 0.000 0.000 200 0.8
   0 0.001 0.0001 0.000 200 0.8
   ```

   * The RVE min and max points *filter the scan paths for those that are near* the box defined by `min_point` and `max_point`; a large number of scan path files (such as from a part-scale build) can be downselected using this parameter setting.

* **Melt Pool Data Files**: These files provide the data for the `melt_pool_data` section of the config.
   *   If `type: "time_series"`, the file should be a two-column text or CSV file: `[time, value]`.
   *   If `type: "spectral_components"`, the file should be a three-column text or CSV file: `[amplitude, frequency, phase]`.
   *   Each dimension must contain finite values, use a positive scale and shape factor, begin with a positive zero-frequency component, and evaluate to positive physical dimensions. Signed coefficients are normalized to an equivalent shifted-phase form and are included conservatively when constructing the interaction envelope.
   *   The transverse width exponent is fixed at `2.0`; depth and height may use other positive shape factors.

### 2. Python Library (API)

The API allows for programmatic parameter studies, custom workflows, and integration with other tools. The core functionality of Raptor can be called by scripting with the API library. An example is provided in `examples/api_example/rve.py`, which is an RVE simulation of defects in 500µm edge length cube.

The following is a breakdown of the main steps for running a simulation programmatically.

#### Step 1: Create the Voxel Grid
First, define the simulation domain (RVE) by specifying its minimum and maximum coordinates and the desired voxel resolution. The `create_grid` function then generates the grid object.
```python
import numpy as np
from raptor.api import create_grid

# 1. Create voxel grid for the representative volume element (RVE)
min_point = np.array([0.0, 0.0, 0.0])
max_point = np.array([5.0e-4, 5.0e-4, 5.0e-4])
bound_box = np.array([min_point, max_point])
voxel_resolution = 5.0e-6

grid = create_grid(voxel_resolution, bound_box=bound_box)
```

#### Step 2:  Generate Scan Path Vectors
Use the `ScanPathBuilder` utility to programmatically generate a scan strategy. This builder takes process parameters like power, speed, and hatch spacing to create a list of PathVector objects for the simulation.

```python
from raptor.utilities import ScanPathBuilder

# 2. Create path vectors through the representative volume element (RVE)
power = 370
velocity = 1.7
hatch_spacing = 140e-6
layer_height = 30e-6
rotation = 67
scan_extension = max(max_point - min_point)
extra_layers = 0

scan_path_builder = ScanPathBuilder(
    bound_box,
    power,
    velocity,
    hatch_spacing,
    layer_height,
    rotation,
    scan_extension,
    extra_layers,
)

scan_path_builder.generate_layers()
path_vectors = scan_path_builder.process_vectors()
```

#### Step 3:  Define the Melt Pool
Load the melt pool dimension data (in this case, from a text file) and use the create_melt_pool function to construct the MeltPool object. The API allows you to set scaling factors and shape parameters for each dimension.

```python
from pathlib import Path
from raptor.io import read_data
from raptor.api import create_melt_pool

# 3. Create melt pools given a width sequence
SCRIPT_DIR = Path(__file__).resolve().parent
melt_pool_data_path = (
    SCRIPT_DIR / ".." / "data" / "meltPoolData" / "ULI_v1700_theta0_widths.txt"
)
width_data = read_data(melt_pool_data_path)
n_modes = 50

# scale melt pool data by constant factor
width_scale = 1.0
depth_scale = 0.8
height_scale = 0.4

# assign shape to melt pool and cap (1 = parabola, 2 = ellipse)
width_shape = 2  # The transverse exponent is fixed at two.
height_shape = 1
depth_shape = 1

melt_pool_dict = {
    "width": (width_data, n_modes, width_scale, width_shape),
    "depth": (width_data, n_modes, depth_scale, depth_shape),
    "height": (width_data, n_modes, height_scale, height_shape),
}

melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=False)
```

#### Step 4:  Compute Porosity
With the grid, path vectors, and melt pool defined, call the main compute_porosity function. This runs the core Numba-accelerated simulation and returns the final 3D porosity field as a NumPy array.

```python
from raptor.api import compute_porosity

# 4. Compute porosity using melt pool mask
porosity = compute_porosity(
    grid,
    path_vectors,
    melt_pool,
    # Optional numerical and resource policies:
    spectral_error_fraction=0.25,
    max_spectral_table_bytes=256 * 1024**2,
    # None uses the automatic tile heuristic.
    tile_width=None,
)
```

`spectral_error_fraction` bounds the independent width, depth, and height
spectral-table errors relative to the voxel spacing. Smaller values construct
denser tables and can be rejected when the guarded Float32 cosine error alone
exceeds the requested tolerance. `max_spectral_table_bytes=None` removes the
table-memory limit. `tile_width` is specified in metres and affects spatial
indexing performance only; an explicit value overrides the automatic
80-micrometre target and its 16-voxel minimum. These keyword-only controls do
not create additional Numba signatures.

#### Step 5:  Write Results to a VTK File
Use the write_vtk helper function to save the resulting porosity NumPy array to
a `.vti` file for visualization in tools like ParaView. The phase codes are
0 for defects, 1 for melted interiors, 2 for boundaries, and 3 for repeated
boundary interactions. ParaView's threshold feature can isolate any phase.

```python
from raptor.api import write_vtk

# 5. Write porosity field to .VTI
write_vtk(grid.origin, grid.resolution, porosity, "rve.vti")
```

#### Step 6:  Compute and Write Morphology Descriptors
Optionally use the `compute_morphology` and `write_morphology` functions to compute global descriptors such as volume, equivalent diameter, etc. For a full list of possible descriptors, see https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops.

```python
from raptor.api import compute_morphology, write_morphology

# 6. Compute morphology
morphology = compute_morphology(porosity, voxel_resolution, ['area', 'equivalent_diameter_area'])
write_morphology(morphology, "rve_morphology.csv")
```
#### Step 7:  Visualize the Output
Optionally use the `visualize` function to open an interactive window via
`pyvista`. The helper thresholds phase-zero defect voxels automatically. The
VTK direction matrix preserves the physical Raptor axes without a full array
transpose.

```python
from raptor.api import visualize

#7. Visualize using PyVista
visualize("./rve.vti")
```
To visualize the example output, uncomment the `visualize("./rve.vti")` line.


## References
The melt pool measurements in the examples are scans performed in Ti6Al4V from the following study:
* Miner, Justin; Narra, Sneha Prabha (2024). Dataset of Melt Pool Variability Measurements for Powder Bed Fusion - Laser Beam of Ti-6Al-4V. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/25696293.v1
