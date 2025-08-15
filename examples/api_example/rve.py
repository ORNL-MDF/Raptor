import numpy as np
from raptor.io import read_data
from raptor.api import (
    create_scan_paths,
    create_melt_pool,
    compute_porosity,
    write_vtk,
)

# 1. Build scan path and select active scan vectors in RVE
RVE_dimensions = np.array([500e-6, 500e-6, 500e-6])
power = 370
velocity = 1.7
hatch_spacing = 130e-6
layer_height = 30e-6
rotation = 67
overhang_hatch = 500e-6
additional_layers = 7

path_vectors = create_scan_paths(
    RVE_dimensions,
    power,
    velocity,
    hatch_spacing,
    layer_height,
    rotation,
    overhang_hatch,
    additional_layers,
    "foo",
)

scan_path_builder.construct_vectors()
path_vectors = scan_path_builder.process_vectors()

# 2. Read melt pool width data data from files
melt_pool_data_path = "../data/meltPoolData/ULI_v1700_theta0_widths.txt"
width_data = read_data(melt_pool_data_path)

# 3. Create melt pools where depths and heights are scaled from width sequence
width_scale = 1.0
depth_scale = 0.8
height_scale = 0.4

n_modes = 2

melt_pool_dict = {
    "width": (width_data, width_scale, n_modes),
    "depth": (width_data, depth_scale, n_modes),
    "height": (width_data, height_scale, n_modes),
}
melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

# 4. Compute porosity field
voxel_resolution = 5.0e-6
grid = Grid(path_vectors, voxel_resolution)


n_points_bezier = 20
bezier = Bezier(n_points_bezier)

origin, porosity = compute_porosity(
    grid,
    path_vectors,
    melt_pool,
    bezier,
)

# 5. Write porosity field to .VTI
write_vtk(origin, voxel_resolution, porosity, "rve.vti")
