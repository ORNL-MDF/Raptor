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
import matplotlib.pyplot as plt

from raptor.api import (
    compute_morphology,
    compute_porosity,
    create_grid,
    create_melt_pool,
    create_path_vectors,
    visualize,
    write_morphology,
    write_vtk,
)
from raptor.utilities import MeltPoolFilter


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

VARIANCE_SIGNIFICANCE = 0.025  # \alpha value for significance of variance test
CI_TOLERANCE_WINDOW = 0.01  # relative tolerance window around the user supplied std

N_SPECTRAL_MODES = 50
HEIGHT_SHAPE_FACTOR = 1.0
DEPTH_SHAPE_FACTOR = 1.0

RVE_MIN_POINT = np.array([0.0, 0.0, 0.0])
RVE_MAX_POINT = np.array([10.0e-4, 10.0e-4, 5.0e-4])
VOXEL_RESOLUTION = 5.0e-6

VTK_OUTPUT = "rve.vti"
MORPHOLOGY_OUTPUT = "rve_morphology.csv"
WIDTH_DATA_PLOT = "melt_pool_width_timeseries.png"


def plot_width_data(width_data):
    time_ms = width_data[:, 0] * 1.0e3
    width_um = width_data[:, 1] * 1.0e6
    mean_um = MELT_POOL_WIDTH * 1.0e6
    std_dev_um = MELT_POOL_WIDTH_STD_DEV * 1.0e6

    plot_limits = (
        min(width_um.min(), mean_um - 4.0 * std_dev_um),
        max(width_um.max(), mean_um + 4.0 * std_dev_um),
    )
    gaussian_width = np.linspace(*plot_limits, 500)
    gaussian_density = np.exp(-0.5 * ((gaussian_width - mean_um) / std_dev_um) ** 2) / (
        std_dev_um * np.sqrt(2.0 * np.pi)
    )

    style = {
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
    with plt.rc_context(style):
        fig, (time_axis, distribution_axis) = plt.subplots(
            1,
            2,
            figsize=(6.5, 3),
            constrained_layout=True,
        )

        time_axis.plot(time_ms, width_um, color="#0072B2", linewidth=0.8)
        time_axis.axhline(
            mean_um,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Specified mean",
        )
        time_axis.set_xlabel("Time (ms)")
        time_axis.set_ylabel("Melt-pool width (µm)")
        time_axis.legend(frameon=False, loc="upper right")

        distribution_axis.hist(
            width_um,
            bins=40,
            density=True,
            color="#56B4E9",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.75,
            label="Filtered samples",
        )
        distribution_axis.plot(
            gaussian_width,
            gaussian_density,
            color="#D55E00",
            linewidth=2.0,
            label=f"Gaussian (µ={mean_um:.0f}, σ={std_dev_um:.0f} µm)",
        )
        distribution_axis.set_xlim(plot_limits)
        distribution_axis.set_xlabel("Melt-pool width (µm)")
        distribution_axis.set_ylabel("Probability density (µm⁻¹)")
        distribution_axis.legend(frameon=False, loc="upper right")

        for label, axis in zip(("(a)", "(b)"), (time_axis, distribution_axis)):
            axis.text(
                0.02,
                0.96,
                label,
                transform=axis.transAxes,
                fontweight="bold",
                va="top",
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(direction="out")

        fig.savefig(WIDTH_DATA_PLOT, dpi=300, facecolor="white")
        plt.close(fig)


def build_melt_pool():
    # Create melt pools from convolution filter

    # Instantiate object
    mp_filter = MeltPoolFilter(
        MELT_POOL_WIDTH,
        MELT_POOL_WIDTH_STD_DEV,
        SCAN_SPEED,
        VARIANCE_SIGNIFICANCE,
        CI_TOLERANCE_WINDOW,
        VOXEL_RESOLUTION,
    )

    # Define physical scales
    mp_filter.add_effect("melt_pool", [MELT_POOL_LENGTH, None, 1])

    # Generate stochastic melt pool
    mp_filter.initialize()
    width_data = mp_filter.generate_fluctuations(1, mp_filter.n_points, mp_filter.t)
    plot_width_data(width_data)

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
        extra_layers=0,
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

    # 7. Visualize using PyVista
    visualize(VTK_OUTPUT)


if __name__ == "__main__":
    main()
