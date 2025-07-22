import numpy as np
from raptor.io import read_data
from raptor.api import (
    generate_scanpaths,
    construct_meltpool,
    compute_porosity,
    write_vtk,
)

"""
This script is a backbone for the api usage of Raptor to generate RVE ensembles.
Loosely, the procedure can be broken down as follows:
 1. Define scan strategy (RVE size, hatch spacing, layer thickness ...)
 2. Define melt pool from a data source
 3. Compute defect structures
 4. Output desired data
"""

# 1. Defining RVE parameters; scan strategy is
rvedims = np.array([1000e-6, 1000e-6, 1000e-6])  # 0.5 mm edge cube.
power = 370  # Laser power (W)
velocity = 1.7  # Laser velocity (m/s)
hatch = 110e-6  # Hatch spacing (m)
layer_thickness = 30e-6  # Layer height (m)
rotation = 67  # Inter-layer rotation (degrees)
overhang_hatch = 500e-6  # Distance from edge of RVE
# that scanpaths are extended(m)
additional_layers = 7  # Additional layers built past
# zmax of RVE(none)
voxel_res = 2.5e-6  # Voxel resolution (m)
n_bezier = 20  # Number of points used in Bezier curve evaluaton
# Passing in a dummy filename since we won't write any files in this example.
ssb = generate_scanpaths(
    rvedims,
    power,
    velocity,
    hatch,
    layer_thickness,
    rotation,
    overhang_hatch,
    additional_layers,
    "foo",
)
# All vectors available for RVE
ssb.construct_vectors()
# Downselect and compute local times
active_vectors = ssb.process_vectors()

# 2. Defining melt pool from data
mp_datapath = "../data/meltPoolData/ULI_v1700_theta0_widths.txt"
width_data = read_data(mp_datapath)
dw, hw = 0.8, 0.45
# Defining the cap and depth as the same here; will get rescaled in construct_meltpool
nmodes = 50
mp_data_dict = {
    "width": (width_data, 1, nmodes),
    "depth": (width_data, 0.7, nmodes),
    "hump": (width_data, 0.4, int(nmodes / 2)),
}
mp = construct_meltpool(mp_data_dict, en_rand_ph=True)
# Compute porosity
spatter_radius = 100e-6
spatter_centroid = np.ones(3)*250e-6
origin, porosity = compute_porosity(
    active_vectors, voxel_res, n_bezier, mp, spatter_centroid,spatter_radius,ssb.rveBoundBox
)
# Write vti
write_vtk(origin, voxel_res, porosity, "rve.vti")
