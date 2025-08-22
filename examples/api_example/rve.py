import numpy as np
from pathlib import Path
from raptor.io import read_data
from raptor.api import (
    create_path_vectors,
    create_melt_pool,
    create_grid,
    compute_porosity,
    write_vtk,
)
from raptor.utilities import ScanPathBuilder

# 1. Create voxel grid for the representative volume element (RVE)
min_point = np.array([0.0, 0.0, 0.0])
max_point = np.array([5.0e-4, 5.0e-4, 5.0e-4])
bound_box = np.array([min_point, max_point])
voxel_resolution = 2.5e-6

grid = create_grid(voxel_resolution, bound_box=bound_box)

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

# 3. Create melt pools given a width sequence
SCRIPT_DIR = Path(__file__).resolve().parent
melt_pool_data_path = (
    SCRIPT_DIR / ".." / "data" / "meltPoolData" / "ULI_v1700_theta0_widths.txt"
)
width_data = read_data(melt_pool_data_path)
width_scale, depth_scale, height_scale = 1.0, 0.8, 0.4
n_modes = 50
height_shape = 1 # parabola
depth_shape = 1 # parabola
melt_pool_dict = {
    "width": (width_data, n_modes, width_scale, 2),
    "depth": (width_data, n_modes, depth_scale, depth_shape),
    "height": (width_data, n_modes, height_scale, height_shape),
}


melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=False)

# 4. Compute porosity using conic section / superellipse curves for melt pool mask
porosity = compute_porosity(
    grid,
    path_vectors,
    melt_pool,
)

# 5. Write porosity field to .VTI
write_vtk(grid.origin, grid.resolution, porosity, "rve.vti")
