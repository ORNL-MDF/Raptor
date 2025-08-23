# RAPTOR

RAPTOR is a Python-based simulation tool for estimating porosity-related defects in Laser Powder Bed Fusion (LPBF) additive manufacturing processes. It uses a computationally efficient geometric approach to model the dynamic melt pool and identify regions of unmelted material, which correspond to lack-of-fusion pores. The core of RAPTOR is a geometric model of the melt pool cross-section whose dimensions (width, depth, and height) oscillate over time. By analyzing the volume swept by this dynamic melt pool along the laser scan paths, RAPTOR generates a 3D map of the final part's porosity.

## License

This project is licensed under the BSD 3-Clause [License](LICENSE).

## Contributors

*   Vamsi Subraveti, Vanderbilt University, vamsi.r.subraveti@vanderbilt.edu
*   John Coleman, Oak Ridge National Laboratory, colemanjs@ornl.gov
*   Çağlar Oskay, Vanderbilt University, caglar.oskay@vanderbilt.edu
*   Alex Plotkowski, Oak Ridge National Laboratory, plotkowskiaj@ornl.gov

## How It Works

**RAPTOR predicts porosity by following a multi-step process:**

*  **Domain Voxelization**: A 3D bounding box, or Representative Volume Element (RVE), is defined and discretized into a uniform grid of voxels.
*  **Scan Path Ingestion**: Scan path data is used to calculating the timing and trajectory for each laser vector.
*  **Dynamic Melt Pool Definition**: For each melt pool dimension (width, depth, height), the input time-series data is converted into a Fourier series (a sum of cosine functions). This creates a dynamic, time-dependent model of the melt pool's cross-sectional shape, which is modeled using modified Lamé curves. To capture stochastic process variations, a random phase shift can be applied to the Fourier series for each scan vector.
*  **Melt Mask Calculation**: The core of the simulation iterates through each voxel in the domain. For each scan vector that passes near the voxel, it calculates the instantaneous melt pool shape and determines if the voxel is inside the melt pool mask. This process, is executed with a high-performance parallel kernel, Just-In-Time (JIT) compiled with Numba. This enables the rapid analysis of large, industrially-relevant domains.
*  **Porosity Prediction**: Any voxel that is not melted by the end of the simulation is flagged as porosity.
*  **Analysis and Output**: The final 3D porosity field is saved in the binary VTK ImageData (`.vti`) format. The morphological characteristics (e.g., volume, surface area, equivalent diameter) of contiguous pore structures can be quantified using the `scikit-image` library, and saved to a `.csv` file.

## Installation

RAPTOR requires requires Python 3 (tested with Python 3.8+). The following Python packages are necessary:

*   **NumPy**: For numerical operations and array manipulation.
    ```bash
    pip install numpy
    ```
*   **Numba**: For JIT compilation and performance acceleration.
    ```bash
    pip install numba
    ```
*   **PyYAML**: For reading and parsing YAML configuration files.
    ```bash
    pip install pyyaml
    ```
*   **VTK**: For writing the output porosity map in `.vti` format.
    ```bash
    pip install vtk
    ```

*   **scikit-image**: For calculating pore morphologies.
    ```bash
    pip install scikit-image
    ```

You can install all dependencies using pip:
```bash
pip install numpy numba pyyaml vtk scikit-image
```
It's highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage these dependencies.

## Usage

The project is organized into several modules:

*   `cli.py`: Handles command-line argument parsing and manages the main simulation workflow.
*   `api.py`: Provides high-level functions for creating the grid, melt pool, running the simulation, and writing output files.
*   `core.py`: Contains the core Numba-accelerated functions for calculating the melt mask.
*   `structures.py`: Defines the main data structures for the simulation (`Grid`, `MeltPool`, `PathVector`).
*   `io.py`: Contains functions for reading and parsing input files (scan paths, melt pool data).
*   `utilities.py`: Includes helper classes, such as the `ScanPathBuilder` for generating scan strategies.

RAPTOR can be used in two primary ways: through its Command-Line Interface (CLI) for quick, configuration-driven simulations, or as a Python Library (API) for integration into custom scripts and more complex workflows.

### 1. Command-Line Interface (CLI)

The CLI is the simplest way to run a simulation. It is controlled by a single YAML configuration file that defines all inputs, parameters, and outputs.

**How to Run (CLI):**

1.  **Prepare Inputs**: Create scan path files, melt pool data files, and a `config.yaml` file (detailed below).
2.  **Execute Script**: Run the following command from your terminal, providing the path to your configuration file:
    ```bash
    raptor path/to/your/config.yaml
    ```
3.  **Check Outputs**:
    *   Progress will be printed to the console.
    *   The 3D porosity map is saved to the `.vti` file specified in the config.
    *   The pore morphology data is saved to the `.csv` file (if configured).

#### CLI Input: The `config.yaml` File

Running RAPTOR from the CLI requires a YAML configuration file to specify all parameters.

**Example `config.yaml`:**
```yaml
# List of scan path files (relative to this config file's location)
scan_paths:
  - "scan_paths/layer_01.txt"
  - "scan_paths/layer_02.txt"

parameters:
  layer_height: 5.0e-5      # Layer height in meters
  voxel_resolution: 5.0e-6  # Voxel resolution in meters
  enable_random_segment_phase: true # Use random phases for melt pool oscillations per vector

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
      - "extent"
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

* **Melt Pool Data Files**: These files provide the data for the `melt_pool_data` section of the config.
   *   If `type: "time_series"`, the file should be a two-column text or CSV file: `[time, value]`.
   *   If `type: "spectral_components"`, the file should be a three-column text or CSV file: `[amplitude, frequency, phase]`.

### 2. Python Library (API)

For advanced use cases, RAPTOR's core functions can be imported directly into your Python scripts. This allows for programmatic parameter studies, custom workflows, and integration with other tools. An example is provided in examples/api_example/rve.py.

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
hatch_spacing = 130e-6
layer_height = 50e-6
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
width_shape = 2  # placeholder
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
)
```

#### Step 5:  Write Results to a VTK File
Finally, use the write_vtk helper function to save the resulting porosity NumPy array to a .vti file for visualization in tools like ParaView.

```python
from raptor.api import write_vtk

# 5. Write porosity field to .VTI
write_vtk(grid.origin, grid.resolution, porosity, "rve.vti")
```
