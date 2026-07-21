# =============================================================================
# Copyright (c) 2025 Oak Ridge National Laboratory
#
# All rights reserved.
#
# This file is part of Raptor.
#
# For details, see the top-level LICENSE file at:
# https://github.com/ORNL-MDF/Raptor/LICENSE
# =============================================================================
import numpy as np

from raptor.api import (
    compute_morphology,
    compute_porosity,
    compute_spectral_components,
    create_grid,
    create_melt_pool,
    create_path_vectors,
    visualize,
    write_morphology,
    write_vtk,
)
from raptor.utilities import (
    MeltPoolFilter,
    plot_melt_pool_signal,
    reconstruct_spectral_signal,
)


# 0. User defined process parameters
LASER_POWER = 370.0
SCAN_SPEED = 1.7
HATCH_SPACING = 140.0e-6
LAYER_HEIGHT = 30.0e-6
SCAN_ROTATION = 67.0

MELT_POOL_WIDTH = 148.0e-6
MELT_POOL_DEPTH = 118.0e-6
MELT_POOL_HEIGHT = 30.0e-6
MELT_POOL_WIDTH_STD_DEV = 18.0e-6
MELT_POOL_LENGTH = 300.0e-6

HEIGHT_SHAPE_FACTOR = 1.0
DEPTH_SHAPE_FACTOR = 1.0

RVE_MIN_POINT = np.array([0.0, 0.0, 0.0])
RVE_MAX_POINT = np.array([5.0e-4, 5.0e-4, 5.0e-4])
VOXEL_RESOLUTION = 5.0e-6

VTK_OUTPUT = "rve.vti"
MORPHOLOGY_OUTPUT = "rve_morphology.csv"
WIDTH_DATA_PLOT = "melt_pool_width_timeseries.png"
ENABLE_VISUALIZATION = False


def build_melt_pool():
    # Create melt pools from convolution filter

    # Instantiate object
    mp_filter = MeltPoolFilter(
        MELT_POOL_WIDTH,
        MELT_POOL_WIDTH_STD_DEV,
        SCAN_SPEED,
        VOXEL_RESOLUTION,
        random_seed=42,
    )

    # Define physical scales
    mp_filter.add_effect("melt_pool", [MELT_POOL_LENGTH, None, 1])

    # Generate stochastic melt pool
    mp_filter.initialize()
    width_data = mp_filter.generate_fluctuations(1.0, mp_filter.n_points, mp_filter.t)
    width_spectral = compute_spectral_components(width_data, tolerance=VOXEL_RESOLUTION)
    reconstructed_width = reconstruct_spectral_signal(width_data[:, 0], width_spectral)
    reconstruction_rmse = np.sqrt(
        np.mean((width_data[:, 1] - reconstructed_width) ** 2)
    )
    print(
        "Melt-pool signal: "
        f"duration={mp_filter.duration:.6e} s, "
        f"samples={mp_filter.n_points}, "
        f"n_modes={width_spectral.shape[0]}, "
        f"reconstruction_rmse={reconstruction_rmse:.6e} m"
    )
    plot_melt_pool_signal(
        width_data,
        width_spectral,
        MELT_POOL_WIDTH,
        MELT_POOL_WIDTH_STD_DEV,
        WIDTH_DATA_PLOT,
        value_label="Melt-pool width (µm)",
    )

    # scale melt pool data by constant factor
    depth_scale = MELT_POOL_DEPTH / MELT_POOL_WIDTH
    height_scale = MELT_POOL_HEIGHT / MELT_POOL_WIDTH

    # assign shape to melt pool and cap (1 = parabola, 2 = ellipse)
    melt_pool_dict = {
        "width": (width_data, None, 1.0, 2.0),
        "depth": (
            width_data,
            None,
            depth_scale,
            DEPTH_SHAPE_FACTOR,
        ),
        "height": (
            width_data,
            None,
            height_scale,
            HEIGHT_SHAPE_FACTOR,
        ),
    }

    return create_melt_pool(
        melt_pool_dict, enable_random_phases=True, tolerance=VOXEL_RESOLUTION
    )


def main():
    # 1. Create voxel grid for the representative volume element (RVE)
    bound_box = np.array([RVE_MIN_POINT, RVE_MAX_POINT])
    grid = create_grid(VOXEL_RESOLUTION, bound_box=bound_box)

    # 2. Create path vectors through the representative volume element (RVE)
    path_vectors = create_path_vectors(
        bound_box,
        LASER_POWER,
        SCAN_SPEED,
        HATCH_SPACING,
        LAYER_HEIGHT,
        SCAN_ROTATION,
        scan_extension=max(RVE_MAX_POINT - RVE_MIN_POINT),
        extra_layers=10,
    )

    # 3. Create melt pools from convolution filter
    melt_pool = build_melt_pool()

    # 4. Compute porosity using conic section / superellipse curves for melt pool mask
    porosity = compute_porosity(grid, path_vectors, melt_pool, jit_warmup=True)

    # 5. Write porosity field to .VTI
    write_vtk(grid.origin, grid.resolution, porosity, VTK_OUTPUT)

    # 6. Compute morphology
    morphology = compute_morphology(
        porosity,
        VOXEL_RESOLUTION,
        ["area", "equivalent_diameter_area"],
    )
    write_morphology(morphology, MORPHOLOGY_OUTPUT)

    # 7. Optionally visualize using PyVista (requires a graphical display)
    if ENABLE_VISUALIZATION:
        visualize(VTK_OUTPUT)


if __name__ == "__main__":
    main()
