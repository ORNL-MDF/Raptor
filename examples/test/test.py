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
from raptor.structures import Bezier

# 1. Create voxel grid for the representative volume element (RVE)
min_point = np.array([0.0, 0.0, 0.0])
max_point = np.array([5.0e-4, 5.0e-4, 5.0e-4])
bound_box = np.array([min_point, max_point])
voxel_resolution = 5.0e-6

grid = create_grid(voxel_resolution, bound_box=bound_box)

# 2. Create path vectors through the representative volume element (RVE)
power = 370
velocity = 1.7
hatch_spacing = 130e-6
layer_height = 30e-6
rotation = 67
scan_extension = max(max_point - min_point)
extra_layers = 7

path_vectors = create_path_vectors(
    bound_box,
    power,
    velocity,
    hatch_spacing,
    layer_height,
    rotation,
    scan_extension,
    extra_layers,
)

# 3. Create melt pools given a width sequence
SCRIPT_DIR = Path(__file__).resolve().parent
melt_pool_data_path = (
    SCRIPT_DIR / ".." / "data" / "meltPoolData" / "ULI_v1700_theta0_widths.txt"
)
width_data = read_data(melt_pool_data_path)
width_scale, depth_scale, height_scale = 1.0, 0.8, 0.4
n_modes = 2
melt_pool_dict = {
    "width": (width_data, width_scale, n_modes),
    "depth": (width_data, depth_scale, n_modes),
    "height": (width_data, height_scale, n_modes),
}

melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

# 4. Compute porosity using Bezier curves for melt pool mask
porosity = compute_porosity(
    grid,
    path_vectors,
    melt_pool,
    Bezier(n_points=20),
)

# 5. Write porosity field to .VTI
write_vtk(grid.origin, grid.resolution, porosity, "rve.vti")
