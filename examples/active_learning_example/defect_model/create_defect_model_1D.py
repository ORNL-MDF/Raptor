import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Raptor Imports
from numba import njit
from raptor.io import read_data
from raptor.api import (
    create_grid,
    create_melt_pool,
    compute_porosity,
    compute_morphology,
)
from raptor.utilities import ScanPathBuilder, MeltPoolFilter

# Intersect Imports
from intersect_sdk import (
    INTERSECT_JSON_VALUE,
    HierarchyConfig,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
    default_intersect_lifecycle_loop,
)

# Dial Imports
from dial_dataclass import (
    DialInputPredictions,
    DialInputSingleOtherStrategy,
    DialWorkflowCreationParamsClient,
    DialWorkflowDatasetUpdate,
)


# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------
WIDTH_MEAN = 148.0e-6
WIDTH_STD = 18.0e-6

BOUNDS = [[60e-6, 200e-6]]  # Hatch spacing in meters
UNIT_BOUNDS = [[0.0, 1.0]]
NUM_DIMS = len(BOUNDS)

INITIAL_DATA_SIZE = 8  # Number of initial LHS points
MAX_ITERATIONS = 25  # Number of active learning (AL) iterations

SEED = 42


@njit
def seed_numba(seed):
    np.random.seed(seed)


MESHGRID_SIZE = 101

INITIAL_POINTS_TO_PREDICT = np.linspace(
    BOUNDS[0][0], BOUNDS[0][1], MESHGRID_SIZE
).reshape(-1, 1)


# -----------------------------------------------------------------------------
# RAPTOR
# -----------------------------------------------------------------------------
def compute_mean_and_std(defects: np.ndarray):
    if len(defects) == 0:
        y = 0.0
        yerr = np.nan
    elif len(defects) == 1:
        y = defects[0]
        yerr = defects[0]
    else:
        y = np.mean(defects)
        yerr = np.std(defects, ddof=1) / np.sqrt(len(defects))

    return y, yerr


def run_raptor(
    hatch_spacing_m: float,
    layer_thickness_m: float = 50e-6,
    query_volume_mm3: float = 1.0,
    voxel_resolution_m: float = 5.0e-6,
    metric_names: list[str] = ["equivalent_diameter_area"],
):
    """Executes a single Raptor simulation."""
    # Re-seed at the start of every ensemble for independent determinism
    random.seed(SEED)
    np.random.seed(SEED)
    seed_numba(SEED)

    # 1. Create voxel grid
    rve_min_point = np.array([0.0, 0.0, 0.0])
    rve_max_point = np.array([5e-4, 5e-4, 5e-4])
    rve_bounding_box = np.array([rve_min_point, rve_max_point])

    grid = create_grid(voxel_resolution=voxel_resolution_m, bound_box=rve_bounding_box)

    # 2. Build the laser scan path
    laser_power_watts = 370.0
    laser_velocity_m_per_s = 1.7
    layer_rotation_angle_deg = 67.0
    scan_extension_distance_m = max(rve_max_point - rve_min_point)
    num_extra_layers = 0

    scan_path_builder = ScanPathBuilder(
        rve_bounding_box,
        laser_power_watts,
        laser_velocity_m_per_s,
        hatch_spacing_m,
        layer_thickness_m,
        layer_rotation_angle_deg,
        scan_extension_distance_m,
        num_extra_layers,
    )
    scan_path_builder.generate_layers()
    path_vectors = scan_path_builder.process_vectors()

    # 3. Create stochastic melt pool model using convolution filter
    frequency = 250000
    duration = 0.08
    sampling_rate = [frequency, duration]

    melt_pool_filter = MeltPoolFilter(
        WIDTH_MEAN, WIDTH_STD, laser_velocity_m_per_s, sampling_rate
    )
    melt_pool_filter.add_effect("melt_pool", [800e-6, None, 1])
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(1)

    # 4. Create stochastic geometry model for melt pool mask
    melt_pool_dict = {
        "width": (
            width_data,
            50,  # number of modes in series expansion
            1.0,  # scale
            2,  # shape: ellipse
        ),
        "depth": (
            width_data,
            50,  # number of modes in series expansion
            0.8,  # scale
            1,  # shape: parabola
        ),
        "height": (
            width_data,
            50,  # number of modes in series expansion
            0.4,  # scale
            1,  # shape: parabola
        ),
    }
    melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

    # 5. Run Raptor and calculate defect metrics
    rve_volume_mm3 = (1e3 * (rve_max_point - rve_min_point)).prod()
    num_rves = int(np.ceil(query_volume_mm3 / rve_volume_mm3))

    defect_metrics = []
    for i in range(num_rves):
        porosity = compute_porosity(grid, path_vectors, melt_pool, False)
        metrics = compute_morphology(porosity, grid.resolution, metric_names)
        defect_metrics.append(metrics)

    combined_metrics = {}
    for name in metric_names:
        arrays_to_concat = [m[name] for m in defect_metrics if name in m]

        if arrays_to_concat:
            combined_metrics[name] = np.concatenate(arrays_to_concat)
        else:
            combined_metrics[name] = np.array([])

    mean, std = compute_mean_and_std(combined_metrics["equivalent_diameter_area"])

    print(f"Hatch Spacing: {hatch_spacing_m*1e6:.1f}um | Mean Pore Size: {mean:.2f}um")
    return mean


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def x_to_unit(X):
    X = np.asarray(X, dtype=float)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    return (X - lo) / (hi - lo + 1e-12)


def x_from_unit(U):
    U = np.asarray(U, dtype=float)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    return U * (hi - lo) + lo


def get_data_point(x_suggested):
    """
    Runs the suggested point.
    """
    x_ = [np.clip(x_suggested[0], BOUNDS[0][0], BOUNDS[0][1])]
    y_ = run_raptor(x_[0])

    return x_, y_


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def plot_1d_surrogate(mean_grid, variance_grid, dataset_x, dataset_y):
    Xg = np.linspace(BOUNDS[0][0], BOUNDS[0][1], MESHGRID_SIZE)

    train_x = np.array(dataset_x).flatten()
    train_y = np.array(dataset_y)

    plt.figure(figsize=(8, 5))
    plt.plot(Xg, mean_grid.flatten(), "b-", label="GP Mean")
    std = np.sqrt(variance_grid.flatten())

    plt.fill_between(
        Xg,
        mean_grid.flatten() - 2 * std,
        mean_grid.flatten() + 2 * std,
        color="blue",
        alpha=0.2,
        label="95% Conf",
    )

    plt.scatter(
        train_x,
        train_y,
        c="red",
        marker="x",
        label="training points",
    )

    plt.xlabel("Hatch Spacing (m)")
    plt.ylabel("Mean Defect Diameter")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("raptor_surrogate.png")
    plt.close()


# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------
class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination
        self.iteration_count = 0
        self.workflow_id = ""

        logger.info(f"Performing cold start with {INITIAL_DATA_SIZE} points...")
        lhs = qmc.LatinHypercube(d=NUM_DIMS, seed=SEED)
        self.dataset_x = qmc.scale(
            lhs.random(n=INITIAL_DATA_SIZE),
            [b[0] for b in BOUNDS],
            [b[1] for b in BOUNDS],
        ).tolist()

        self.dataset_y = [run_raptor(x[0]) for x in self.dataset_x]

        self.dataset_x_unit = x_to_unit(self.dataset_x).tolist()

        self.bounds_unit = UNIT_BOUNDS

    def assemble_message(
        self, operation: str, **kwargs: Any
    ) -> IntersectClientCallback:
        payload = None
        if operation == "initialize_workflow":
            payload = DialWorkflowCreationParamsClient(
                dataset_x=self.dataset_x_unit,
                dataset_y=self.dataset_y,
                bounds=self.bounds_unit,
                kernel="matern",
                length_per_dimension=True,
                y_is_good=False,
                backend="sklearn",
                seed=SEED,
                preprocess_standardize=False,
            )
        elif operation == "update_workflow_with_data":
            payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                **kwargs,
            )
        elif operation == "get_next_point":
            payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy="expected_improvement",
                bounds=self.bounds_unit,
            )
        elif operation == "get_surrogate_values":
            points_to_predict_unit = x_to_unit(INITIAL_POINTS_TO_PREDICT)
            payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=points_to_predict_unit,
            )

        logger.info(f"✉️ Sending: dial.{operation}")
        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.service_destination,
                    operation=f"dial.{operation}",
                    payload=payload,
                )
            ]
        )

    def __call__(
        self,
        _source: str,
        operation: str,
        has_error: bool,
        payload: INTERSECT_JSON_VALUE,
    ) -> IntersectClientCallback:

        if has_error:
            print("============ERROR==============", file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            print(file=sys.stderr)
            raise Exception

        if operation == "dial.initialize_workflow":
            self.workflow_id = payload
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.update_workflow_with_data":
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.get_surrogate_values":
            self.mean_grid = np.array(payload[0])
            self.variance = np.array(payload[1])

            if self.iteration_count >= MAX_ITERATIONS:
                plot_1d_surrogate(
                    self.mean_grid, self.variance, self.dataset_x, self.dataset_y
                )
                logger.info("Max iterations reached. Optimization complete.")
                raise Exception("DONE")
            return self.assemble_message("get_next_point")

        if operation == "dial.get_next_point":
            x_suggested_unit = np.array(payload).reshape(1, -1)
            x_suggested = x_from_unit(x_suggested_unit)[0].tolist()

            logger.info(
                f"Iteration {self.iteration_count}: "
                f"DIAL suggests HS={x_suggested[0]*1e6:.2f}um"
            )

            new_x, new_y = get_data_point(x_suggested)

            self.dataset_x.append(new_x)
            self.dataset_y.append(new_y)

            new_x_unit = x_to_unit(new_x).flatten().tolist()
            self.dataset_x_unit.append(new_x_unit)

            self.iteration_count += 1

            plot_1d_surrogate(
                self.mean_grid, self.variance, self.dataset_x, self.dataset_y
            )

            return self.assemble_message(
                "update_workflow_with_data",
                next_x=new_x_unit,
                next_y=float(new_y),
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

    active_learning = ActiveLearningOrchestrator(
        service_destination=HierarchyConfig(
            **from_config_file["intersect-hierarchy"]
        ).hierarchy_string(".")
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

    default_intersect_lifecycle_loop(client)
