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
LAYER_THICKNESS = 40e-6

BOUNDS = [
    HATCH_BOUNDS,
]
NUM_DIMS = len(BOUNDS)

VOXEL_RESOLUTION_M = 5.0e-6  # reference 5.0e-6
RVE_LENGTH_M = 2e-3
QUERY_VOLUME_MM3 = (
    3 * 8.0
)  # decrease query_volume_mm3 factor * rve_volume to speed up

MIN_LEN_DEFECTS = 100

# -----------------------------------------------------------------------------
# AL PARAMETERS
# -----------------------------------------------------------------------------
ANALYZE = AnalysisMode.CVAR
CVAR_LEVEL = 0.01  # level for CVAR analysis

INITIAL_DATA_SIZE = 1  # size of the initial data batch >=1
MAX_ITERATIONS = 20  # total number of points to acquire (after initial_dataset)

BACKEND = "sable"  # "sable" or "sklearn"
STATISTICS_YERR = (
    "yerr"  # either a noise value, e.g., 1e-2 or "yerr" for the data noise
)
BATCHSIZE = 1  # batch size for planning (>=1, 1 is single acquisition)

# if positive, dial only suggests points on n_acquire_grid^dim grid
N_ACQUIRE_GRID = -1  # -1 or positive number

# grid size for plotting and saving
N_GRIDS = (150,)

INPUT_GRID = np.linspace(BOUNDS[0][0], BOUNDS[0][1], N_GRIDS[0]).reshape(-1, 1)


# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------
class ActiveLearningOrchestrator1D(ActiveLearningOrchestrator):

    def _get_data_point(self, x_suggested):
        self.logger.info(
            f"Iteration {self.iteration_count}: "
            f"DIAL suggests HS={x_suggested[0]*1e6:.2f}um."
        )

        bounds = np.asarray(self.bounds)
        x = np.clip(x_suggested, bounds[:, 0], bounds[:, 1]).tolist()

        raptor_data = run_raptor(
            self.mp_interpolator,
            x[0],
            layer_thickness_m=LAYER_THICKNESS,
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
        with open("raptor_data_1.json", "w") as outfile:
            json.dump(
                self.dataset_raptor,
                outfile,
                indent="",
            )

        np.savez(
            "defect_model_surrogate_1.npz",
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

    active_learning = ActiveLearningOrchestrator1D(
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
