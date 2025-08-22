# RAPTOR - Rapid Geometric-based Porosity Prediction for LPBF

RAPTOR (Rapid Geometric-based Porosity Prediction) is a Python-based simulation tool for estimating porosity-related defects in Laser Powder Bed Fusion (LPBF) additive manufacturing processes. It uses a computationally efficient geometric approach to model the dynamic melt pool and identify regions of unmelted powder, which correspond to lack-of-fusion pores.

The core of RAPTOR is a geometric model of the melt pool cross-section whose dimensions (width, depth, and height) oscillate over time. By analyzing the volume swept by this dynamic melt pool along the laser scan paths, the tool generates a 3D map of the final part's porosity.

## Features

*   **Geometric Melt Pool Modeling**: Models the melt pool cross-section using a modified Lamé curve. The width, depth, and height of the melt pool are dynamic, driven by a Fourier series representation derived from experimental data or other simulations.
*   **Flexible Input**: Reads scan path data from standard text files and accepts melt pool dimension data as either raw time-series or pre-computed spectral components (amplitude, frequency, phase).
*   **Stochastic Simulation**: Supports random phase shifts for each scan vector's melt pool oscillations, enabling the simulation of stochastic process variations.
*   **High Performance**: Utilizes Numba for Just-In-Time (JIT) compilation and parallel execution of the core computational kernel, enabling rapid analysis of large domains.
*   **Porosity and Morphology Analysis**:
    *   Outputs a 3D porosity map in the binary VTK ImageData format (`.vti`), which can be visualized in software like [ParaView](https://www.paraview.org/) or [VisIt](https://visit.llnl.gov/).
    *   Identifies individual pores and calculates their morphological properties (e.g., volume, surface area, equivalent diameter) using `scikit-image`, exporting the results to a `.csv` file.
*   **Scan Path Generation**: Includes a utility class (`ScanPathBuilder`) to programmatically generate island or stripe scan strategies with inter-layer rotation.

## How It Works

RAPTOR predicts porosity by following a multi-step computational process:

1.  **Domain Voxelization**: A 3D bounding box, or Representative Volume Element (RVE), is defined and discretized into a uniform grid of voxels.
2.  **Scan Path Ingestion**: The tool reads scan path data from input files, calculating the precise timing and trajectory of the laser for each vector.
3.  **Dynamic Melt Pool Definition**: For each dimension (width, depth, height), the input time-series data is converted into a Fourier series (a sum of cosine functions). This creates a dynamic, time-dependent model of the melt pool's cross-sectional shape.
4.  **Melt Mask Calculation**: The core of the simulation iterates through each voxel in the domain. For each scan vector that passes nearby, it calculates the exact melt pool shape at that moment in time and determines if the voxel's center falls inside it. This process is heavily accelerated with Numba.
5.  **Porosity Prediction**: Any voxel that is not melted by the end of the simulation is flagged as porosity.
6.  **Analysis and Output**: The final 3D porosity field is saved as a `.vti` file. If requested, the tool then identifies contiguous pore regions and computes their geometric characteristics, saving them to a `.csv` file.

## Dependencies

This script requires Python 3 (tested with Python 3.8+). The following packages are necessary:

*   **NumPy**: For numerical operations.
*   **Numba**: For JIT compilation and performance.
*   **PyYAML**: For parsing the YAML configuration file.
*   **VTK**: For writing the output `.vti` file.
*   **scikit-image**: For calculating pore morphology and features.

You can install all dependencies using pip. It is highly recommended to use a virtual environment (`venv` or `conda`).

## Input Data

RAPTOR is controlled by a single YAML configuration file.

### 1. Configuration File (`config.yaml`)

The YAML file specifies all simulation parameters, input file paths, and output settings.

**Example `config.yaml`:**

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
    shape: 2.0          # Shape factor n for the Lame curve (n=2 is elliptical)
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
      
### 2. Scan Path Files

Each file in `scan_paths` should be a space-delimited text file. **The first line is treated as a header and is skipped.**

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

### 3. Melt Pool Data Files

These files provide the data for the `melt_pool_data` section of the config.

*   If `type: "time_series"`, the file should be a two-column text or CSV file: `[time, value]`.
*   If `type: "spectral_components"`, the file should be a three-column text or CSV file: `[amplitude, frequency, phase]`.

## How to Run

# Using RAPTOR from the command line (CLI)

1.  **Prepare Inputs**: Create your scan path files, melt pool data files, and a `config.yaml` file to orchestrate the simulation.
2.  **Activate Environment**: Activate the Python virtual environment where the dependencies are installed.
3.  **Execute Script**: Run the script from the command line, providing the path to your configuration file. Assuming the entry point is `raptor`:
    ```bash
    raptor path/to/your/config.yaml
    ```
4.  **Check Outputs**:
    *   Progress will be printed to the console.
    *   The 3D porosity map will be saved to the `.vti` file specified in the config.
    *   If configured, the pore morphology data will be saved to the `.csv` file.

# Using RAPTOR as a Python Library (API)
In addition to the command-line interface, RAPTOR is designed to be used as a Python library. You can import its functions into your own scripts to build custom workflows, perform parameter studies, or integrate porosity prediction into a larger simulation chain.

An example demonstrating this usage is provided in `examples/api_example/rve.py`.

### API Usage Example

The following is a breakdown of the `rve.py` example, which shows the main steps for running a simulation programmatically.

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

# 4. Compute porosity using conic section / superellipse curves for melt pool mask
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

## Code Structure

The project is organized into several modules:

*   `cli.py`: Handles command-line argument parsing and manages the main simulation workflow.
*   `api.py`: Provides high-level functions for creating the grid, melt pool, running the simulation, and writing output files.
*   `core.py`: Contains the core Numba-accelerated functions for calculating the melt mask.
*   `structures.py`: Defines the main data structures for the simulation (`Grid`, `MeltPool`, `PathVector`).
*   `io.py`: Contains functions for reading and parsing input files (scan paths, melt pool data).
*   `utilities.py`: Includes helper classes, such as the `ScanPathBuilder` for generating scan strategies.

## License

This project is licensed under the BSD 3-Clause [License](LICENSE).

Copyright (C) 2025, Oak Ridge National Laboratory    
All rights reserved.

## Contributors

*   Vamsi Subraveti, Vanderbilt University, vamsi.r.subraveti@vanderbilt.edu
*   John Coleman, ORNL, colemanjs@ornl.gov
