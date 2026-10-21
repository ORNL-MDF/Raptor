import argparse
import json
import logging
from pathlib import Path

import numpy as np

# Intersect Imports
from intersect_sdk import (
    HierarchyConfig,
    IntersectClient,
    IntersectClientConfig,
    default_intersect_lifecycle_loop,
)

from defect_model_common import (
    AnalysisMode,
    run_raptor,
    process_raptor_data,
    ActiveLearningOrchestrator,
)

# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------
LASER_POWER_WATTS = 195
LASER_VELOCITY_M_S = 1.083

HATCH_BOUNDS = (60e-6, 140e-6)
LAYER_HEIGHT_BOUNDS = (20e-6, 90e-6)

BOUNDS = [HATCH_BOUNDS, LAYER_HEIGHT_BOUNDS]
NUM_DIMS = len(BOUNDS)

VOXEL_RESOLUTION_M = 5.0e-6  # reference 5.0e-6
RVE_LENGTH_M = 2e-3
QUERY_VOLUME_MM3 = (
    3 * 8.0
)  # decrease query_volume_mm3 factor * rve_volume to speed up

MIN_LEN_DEFECTS = 50


# -----------------------------------------------------------------------------
# AL PARAMETERS
# -----------------------------------------------------------------------------
ANALYZE = AnalysisMode.CVAR
CVAR_LEVEL = 0.01  # level for CVAR analysis

INITIAL_DATA_SIZE = 1  # size of the initial data batch >=1
MAX_ITERATIONS = (
    200  # total number of points to acquire (after initial_dataset)
)

BACKEND = "sable"  # "sable" or "sklearn"
STATISTICS_YERR = (
    "yerr"  # either a noise value, e.g., 1e-2 or "yerr" for the data noise
)
BATCHSIZE = 5  # batch size for planning (>=1, 1 is single acquisition)

# if positive, dial only suggests points on n_acquire_grid^dim grid
N_ACQUIRE_GRID = -1  # -1 or positive number

# grid size for plotting and saving
N_GRIDS = (80, 70)


def meshgrid_2d():
    grids_1d = [
        np.linspace(*bound, ngrid) for bound, ngrid in zip(BOUNDS, N_GRIDS)
    ]
    x1, x2 = np.meshgrid(*grids_1d)
    return np.stack((x1.reshape(-1), x2.reshape(-1)), axis=1)


INPUT_GRID = meshgrid_2d()

# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------


class ActiveLearningOrchestrator2D(ActiveLearningOrchestrator):

    def _get_data_point(self, x_suggested):
        self.logger.info(
            f"Iteration {self.iteration_count}: "
            f"DIAL suggests HS={x_suggested[0]*1e6:.2f}um, "
            f"LH={x_suggested[1]*1e6:.2f}."
        )

        bounds = np.asarray(self.bounds)
        x = np.clip(x_suggested, bounds[:, 0], bounds[:, 1]).tolist()

        raptor_data = run_raptor(
            self.mp_interpolator,
            x[0],
            layer_thickness_m=x[1],
            query_volume_mm3=QUERY_VOLUME_MM3,
            laser_velocity_m_s=LASER_VELOCITY_M_S,
            laser_power_watts=LASER_POWER_WATTS,
            rve_length_m=RVE_LENGTH_M,
            voxel_resolution_m=VOXEL_RESOLUTION_M,
        )
        y, yerr = process_raptor_data(
            raptor_data,
            analyze=ANALYZE,
            min_len_defects=MIN_LEN_DEFECTS,
            cvar_level=CVAR_LEVEL,
        )
        return x, y, yerr, raptor_data

    def _save_dataset(self, *args):
        with open("raptor_data_2.json", "w") as outfile:
            json.dump(
                self.dataset_raptor,
                outfile,
                indent="",
            )

        np.savez(
            "defect_model_surrogate_2.npz",
            mean_grid=self.mean_grid,
            variance_grid=self.variance_grid,
            dataset_x=self.dataset_x,
            dataset_y=self.dataset_y,
            dataset_yerr=self.dataset_yerr,
            bounds=self.bounds,
            n_grids=N_GRIDS,
            laser_power=LASER_POWER_WATTS,
            laser_velocity=LASER_VELOCITY_M_S,
            scaler=np.array(
                self.scaler, dtype=object
            ),  # save the scaler that was used
        )

        live_plot = True
        if live_plot:
            y_norm_grid, yerr_norm_grid = args
            do_live_plot(self, y_norm_grid, yerr_norm_grid)


# ----
# plotting
# ----
def do_live_plot(
    obj: ActiveLearningOrchestrator,
    y_norm_grid: np.ndarray,
    yerr_norm_grid: np.ndarray,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_filename = f"raw_mean_std_{obj.iteration_count}.png"

    fig, ax_tl = plt.subplots(
        1, 1, figsize=(7, 6), subplot_kw={"projection": "3d"}
    )

    n_grids = (N_GRIDS[1], N_GRIDS[0])
    points_unit = np.asarray(obj.input_scaler.to_unit(obj.input_grid))
    x_grid = [
        points_unit[:, 0].reshape(n_grids),
        points_unit[:, 1].reshape(n_grids),
    ]
    y_pred = y_norm_grid.reshape(n_grids) / 3.0
    y_band = yerr_norm_grid.reshape(n_grids) / 3.0

    ax_tl.plot_surface(
        x_grid[0], x_grid[1], y_pred, cmap="viridis"
    )  # type: ignore[attr-defined]

    CONTOUR_OFFSET = -4.0
    contour_plot_tl = ax_tl.contour(
        x_grid[0], x_grid[1], y_band, linestyles="solid", offset=CONTOUR_OFFSET
    )
    cbar = fig.colorbar(contour_plot_tl, ax=ax_tl)
    cbar.set_label("Predicted error of surrogate model.")
    ax_tl.set_xlim((0, 1))
    ax_tl.set_ylim((0, 1))
    ax_tl.set_zlim((CONTOUR_OFFSET, 1))  # type: ignore[attr-defined]
    ax_tl.set_title("learned")

    # ── Acquired training data on the truth panel ──
    x_train = np.asarray(obj.input_scaler.to_unit(obj.dataset_x))
    y_norm, yerr_norm = obj.scaler.scale(obj.dataset_y, obj.dataset_yerr)
    # y_train = np.asarray(y_norm)
    ye = np.asarray(yerr_norm)
    if x_train.shape[0] > 0:
        dotsize = (
            10.0 * ye / np.max(ye)
            if np.max(ye) > 0
            else 10.0 * np.ones_like(ye)
        )
        ax_tl.scatter(
            x_train[:, 0],
            x_train[:, 1],
            np.full(ye.shape, CONTOUR_OFFSET),
            s=dotsize,
            alpha=1,
            zorder=10,
            color="tab:orange",
            label="Acquired values",
        )  # type: ignore[misc]

    ax_tl.view_init(elev=25, azim=130 + 180)  # type: ignore[attr-defined]
    ax_tl.set_title(output_filename)
    plt.tight_layout()
    output_path = Path("live_plots")
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / output_filename, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated client")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    with Path(args.config).open("rb") as f:
        from_config_file = json.load(f)

    active_learning = ActiveLearningOrchestrator2D(
        service_destination=HierarchyConfig(
            **from_config_file["intersect-hierarchy"]
        ).hierarchy_string("."),
        bounds=BOUNDS,
        n_acquire_grid=N_ACQUIRE_GRID,
        initial_data_size=INITIAL_DATA_SIZE,
        batch_size=BATCHSIZE,
        max_iterations=MAX_ITERATIONS,
        input_grid=INPUT_GRID.tolist(),
        backend=BACKEND,
        statistics_yerr=STATISTICS_YERR,
    )

    config = IntersectClientConfig(
        initial_message_event_config=active_learning.assemble_message(
            "initialize_workflow"
        ),
        **from_config_file["intersect"],
    )

    client = IntersectClient(
        config=config,
        user_callback=active_learning,
    )

    default_intersect_lifecycle_loop(
        client,
    )
