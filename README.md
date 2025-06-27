# RAPTOR - Rapid Geometric-based Porosity Prediction for LPBF

The purpose of this code is to estimate porosity-related defects in Laser Powder Bed Fusion (LPBF) additive manufacturing processes. It analyzes scan path data and process parameters to generate a 3D porosity map.

## Features

*   Reads scan path data from text files.
*   Models melt pool geometry using Bezier curves with oscillating dimensions.
*   Supports random phase shifts per scan vector for stochastic effects.
*   Utilizes Numba for JIT compilation and parallel execution of computationally intensive tasks, significantly speeding up calculations.
*   Processes the computational domain in Z-axis chunks to manage memory usage effectively, allowing for larger problem sizes.
*   Outputs a 3D porosity map in the binary VTK ImageData format (`.vti`), suitable for visualization in software like ParaView or VisIt.

## Contributors

*   Vamsi Subraveti, Vanderbilt University, vamsi.r.subraveti@vanderbilt.edu
*   John Coleman, ORNL, colemanjs@ornl.gov

## License

This code is intended for internal ORNL use only. Please discuss with the authors before sharing externally.

## Dependencies

This script requires Python 3 (tested with Python 3.8+). The following Python packages are necessary:

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

You can install all dependencies using pip:
```bash
pip install numpy numba pyyaml vtk
```
It's highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage these dependencies.

## Input Data

### 1. Configuration File (YAML)

The script requires a YAML configuration file that specifies all input parameters, scan path file locations, and output settings.

**Example `config.yaml`:**
```yaml
# List of scan path files
scan_paths:
  - "layer_01.txt"
  - "layer_02.txt"
  # ... more layers

# Output VTK filename (will be a .vti file)
output_vtk_filename: "results/porosity_map.vti"

parameters:
  layer_height: 5.0e-5  # Layer height in meters
  voxel_resolution: 5.0e-6     # Default: 5e-6 (resolution in meters)
  bezier_points_per_half: 20   # Default: 20 (must be >= 2)
  enable_random_segment_phase: true # Default: true

melt_pool_data:
  width:
      type: "time_series"
      file_name: "widths.txt"
      nmodes: 10.0
      scale: 1.0

  depth:
      type: "time_series"
      file_name: "depthss.txt"
      nmodes: 10.0
      scale: 1.0
  hump:
      type: "time_series"
      file_name: "humps.txt"
      nmodes: 10.0
      scale: 1.0
rve:
  min_point: [0.002, -100.0e-6, -100.0e-6]
  max_point: [0.006, 100.0e-6, 100.0e-6]
```

### 2. Scan Path Files

Each file listed in `scan_paths` (e.g., `layer_01.txt`) should be a space-delimited text file where each row represents a scan vector or a delay.
The **first line is assumed to be a header and is skipped**.

**Format per line (after header):**
`mode x y z power parameter`

*   `mode`: Integer.
    *   `0`: Line raster.
    *   `1`: Point source.
*   `x, y, z`: Float. Endpoint coordinates of the segment in meters.
*   `power`: Float. Laser power in Watts (used for information, not directly in current porosity model geometry).
*   `parameter`: Float.
    *   If `mode == 0`: Scanning speed in meters/second.
    *   If `mode == 1`: Duration of the point source in seconds.

**Example `layer_01.txt` content:**
```
# Header: mode x_end y_end z_end power speed_or_time
1 0.000 0.000 0.000 0   0
0 0.000 0.008 0.000 200 0.8
```

## How to Run

1.  **Prepare your input files**:
    *   Create your scan path text files (one per layer).
    *   Create a YAML configuration file (`config.yaml` or similar) pointing to your scan paths and specifying all parameters.

2.  **Activate your Python environment** (if using one) where the dependencies are installed.

3.  **Execute the script** from the command line, providing the path to your YAML configuration file as an argument.
    Assuming your Python script is named `raptor_porosity.py`:

    ```bash
    raptor path/to/your/config.yaml
    ```

4.  **Output**:
    *   The script will print progress information to the console.
    *   The final 3D porosity map will be saved as a `.vti` file at the location specified by `output_vtk_filename` in your YAML config.
    *   This `.vti` file can be opened with visualization software like [ParaView](https://www.paraview.org/) or [VisIt](https://visit.llnl.gov/).

## Script Structure Overview

*   **`main()`**: Parses command-line arguments (the config file path), loads the YAML configuration, validates parameters, and orchestrates the simulation by calling `compute_porosity_vtk`.
*   **`compute_porosity(...)`**: The core function that:
    *   Reads all scan path files using `read_scan_path`.
    *   Processes segments, calculates times, and determines active exposure vectors.
    *   Precomputes Bezier basis functions and segment AABBs (Axis-Aligned Bounding Boxes).
    *   Defines the overall voxel grid and parameters for chunked processing.
    *   Iterates through Z-chunks of the grid:
        *   Calls `compute_melt_mask_structured` (Numba JIT-compiled function) for each chunk to determine melted voxels.
    *   Assembles the full porosity map from chunk results.
*   **`write_vtk(...)`**:
    *   Writes the final porosity map to a binary `.vti` file using the VTK library.
*   **Numba JIT-compiled functions**:
    *   `compute_melt_mask(...)`: Determines if voxels within a chunk are melted based on proximity to scan vectors and melt pool geometry. Leverages `prange` for parallel execution.
    *   `bezier_vertices(...)`: Calculates melt pool cross-section.
    *   `point_in_poly(...)`: Point-in-polygon test.
*   **Helper functions**:
    *   `read_scan_path(...)`: Parses individual scan path files.
    *   `local_frame_2d(...)`: Computes local coordinate frames for scan vectors.
*   **`PathSegment` Dataclass**: Represents a single segment from a scan path file.

## Further Development / Notes

*   The current model for melt pool geometry and its oscillation provides a foundational framework. More complex physics or material-specific parameters could be incorporated.
