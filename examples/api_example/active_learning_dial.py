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
from raptor.io import read_data
from raptor.api import (
    create_grid,
    create_melt_pool,
    compute_porosity,
    compute_morphology,
)
from raptor.utilities import ScanPathBuilder

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
BOUNDS = [[60e-6, 200e-6]]  # Hatch spacing in meters
UNIT_BOUNDS = [[0.0, 1.0]]
NUM_DIMS = len(BOUNDS)

INITIAL_DATA_SIZE = 8   # Number of initial LHS points
MAX_ITERATIONS = 20     # Number of active learning iterations
MESHGRID_SIZE = 101

# Ensemble parameters (sampling around suggested point)
ENSEMBLE_SIZE = 8
LOCAL_FRAC = 0.1

INITIAL_POINTS_TO_PREDICT = np.linspace(
    BOUNDS[0][0],
    BOUNDS[0][1],
    MESHGRID_SIZE
).reshape(-1, 1)

# History tracking
HISTORY_X: list[list[float]] = []
HISTORY_Y: list[float] = []


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
    rve_min_point = np.array([0.0, 0.0, 0.0])
    rve_max_point = np.array([5e-4, 5e-4, 5e-4])
    rve_bounding_box = np.array([rve_min_point, rve_max_point])

    grid = create_grid(
        voxel_resolution=voxel_resolution_m,
        bound_box=rve_bounding_box
    )

    SCRIPT_DIR = Path(__file__).resolve().parent
    melt_pool_data_path = (
        SCRIPT_DIR / ".." / "data" / "meltPoolData" / "ULI_v1700_theta0_widths.txt"
    )
    base_width_data = read_data(melt_pool_data_path)
    melt_pool_dict = {
        "width": (
            base_width_data,
            50,
            1.0,
            2,
        ),
        "depth": (
            base_width_data,
            50,
            0.8,
            1,
        ),
        "height": (
            base_width_data,
            50,
            0.4,
            1,
        ),
    }
    melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

    # Build the laser paths
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

    # Run Raptor and calculate defect metrics
    rve_volume_mm3 = (1e3 * (rve_max_point - rve_min_point)).prod()
    num_rves = int(np.ceil(query_volume_mm3 / rve_volume_mm3))

    defect_metrics = []
    for i in range(num_rves):
        porosity = compute_porosity(
            grid,
            path_vectors,
            melt_pool,
            False  # jit warmup
        )
        metrics = compute_morphology(porosity, grid.resolution, metric_names)
        defect_metrics.append(metrics)

    combined_metrics = {}
    for name in metric_names:
        arrays_to_concat = [m[name] for m in defect_metrics if name in m]

        if arrays_to_concat:
            combined_metrics[name] = np.concatenate(arrays_to_concat)
        else:
            combined_metrics[name] = np.array([])

    mean, _ = compute_mean_and_std(combined_metrics["equivalent_diameter_area"])

    print("Hatch spacing: ", hatch_spacing_m, "Mean pore size: ", mean)
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
    Runs the suggested point plus a few local variations to account for simulation noise.
    """
    ensemble_xs = [x_suggested]

    span = BOUNDS[0][1] - BOUNDS[0][0]
    for _ in range(ENSEMBLE_SIZE - 1):
        delta = (random.random() - 0.5) * 2 * LOCAL_FRAC * span
        new_x = np.clip(
            x_suggested[0] + delta,
            BOUNDS[0][0],
            BOUNDS[0][1]
        )
        ensemble_xs.append([new_x])

    results_y = []
    for x_vec in ensemble_xs:
        y = run_raptor(x_vec[0])
        HISTORY_X.append(x_vec)
        HISTORY_Y.append(y)
        results_y.append(y)

    # Return the best (lowest defect diameter) found in the local ensemble
    best_idx = np.argmin(results_y)
    return ensemble_xs[best_idx], results_y[best_idx]


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def plot_1d_surrogate(mean_grid, variance_grid, dataset_x, dataset_y):
    Xg = np.linspace(BOUNDS[0][0], BOUNDS[0][1], MESHGRID_SIZE)
    hx = np.array(HISTORY_X).flatten()
    hy = np.array(HISTORY_Y)

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
    plt.scatter(hx, hy, c="black", s=20, alpha=0.5, label="All Samples")
    plt.scatter(
        np.array(dataset_x).flatten(),
        dataset_y,
        c="red",
        marker="x",
        label="Active Learning Selections",
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
        lhs = qmc.LatinHypercube(d=NUM_DIMS)
        self.dataset_x = qmc.scale(
            lhs.random(n=INITIAL_DATA_SIZE),
            [b[0] for b in BOUNDS],
            [b[1] for b in BOUNDS],
        ).tolist()

        self.dataset_y = []
        for x in self.dataset_x:
            y = run_raptor(x[0])
            self.dataset_y.append(y)
            HISTORY_X.append(x)
            HISTORY_Y.append(y)

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
                seed=-1,
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

        print("in call")

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
                    self.mean_grid,
                    self.variance,
                    self.dataset_x,
                    self.dataset_y
                )
                logger.info("Max iterations reached. Optimization complete.")
                raise Exception("DONE")
            return self.assemble_message("get_next_point")

        if operation == "dial.get_next_point":
            x_suggested_unit = np.array(payload).reshape(1, -1)
            x_suggested_raw = x_from_unit(x_suggested_unit)[0].tolist()

            logger.info(
                f"Iteration {self.iteration_count}: "
                f"DIAL suggests HS={x_suggested_raw[0]*1e6:.2f}um"
            )

            best_x, best_y = get_data_point(x_suggested_raw)

            self.dataset_x.append(best_x)
            self.dataset_y.append(best_y)
            best_x_unit = x_to_unit(best_x).flatten().tolist()
            self.dataset_x_unit.append(best_x_unit)

            self.iteration_count += 1

            plot_1d_surrogate(
                self.mean_grid,
                self.variance,
                self.dataset_x,
                self.dataset_y
            )

            return self.assemble_message(
                "update_workflow_with_data",
                next_x=best_x_unit,
                next_y=float(best_y),
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

    try:
        with Path(args.config).open("rb") as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical("unable to load config file: %s", str(e))
        sys.exit(1)

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

    default_intersect_lifecycle_loop(
        client,
    )
