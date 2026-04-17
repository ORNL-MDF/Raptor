import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc
from scipy.interpolate import RegularGridInterpolator

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
LASER_POWER_WATTS = 200.0
LASER_VELOCITY_M_S = 1.0

BOUNDS = [[70e-6, 170e-6]]
UNIT_BOUNDS = [[0.0, 1.0]]
NUM_DIMS = len(BOUNDS)

INITIAL_DATA_SIZE = 20
MAX_ITERATIONS = 35

SEED = 42


@njit
def seed_numba(seed):
    np.random.seed(seed)


MESHGRID_SIZE = 150

INITIAL_POINTS_TO_PREDICT = np.linspace(
    BOUNDS[0][0], BOUNDS[0][1], MESHGRID_SIZE
).reshape(-1, 1)

MELT_POOL_SURROGATE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "melt_pool_model", "melt_pool_surrogates.npz"
    )
)


# -----------------------------------------------------------------------------
# RAPTOR UTILITIES
# -----------------------------------------------------------------------------
class MeltPoolInterpolator:
    def __init__(self, filepath: str):
        data = np.load(filepath)
        self.v_axis = data["velocity"]
        self.p_axis = data["power"]

        self.features = [
            "depth_mean",
            "depth_std",
            "width_mean",
            "width_std",
            "height_mean",
            "height_std",
        ]

        self.interpolators = {}
        for f in self.features:
            self.interpolators[f] = RegularGridInterpolator(
                (self.v_axis, self.p_axis), data[f]
            )

    def query(self, velocity: float, power: float):
        point = np.array([[velocity, power]])
        return {f: float(self.interpolators[f](point)[0]) for f in self.features}


def run_raptor(
    hatch_spacing_m: float,
    mp_interpolator: MeltPoolInterpolator,
    layer_thickness_m: float = 40e-6,
    query_volume_mm3: float = 1.0,
    voxel_resolution_m: float = 5e-6,
    metric_names: list[str] = ["equivalent_diameter_area"],
):
    random.seed(SEED)
    np.random.seed(SEED)
    seed_numba(SEED)

    mp_stats = mp_interpolator.query(LASER_VELOCITY_M_S, LASER_POWER_WATTS)

    rve_min_point = np.array([0.0, 0.0, 0.0])
    rve_max_point = np.array([1e-3, 1e-3, 1e-3])
    rve_bounding_box = np.array([rve_min_point, rve_max_point])

    grid = create_grid(voxel_resolution=voxel_resolution_m, bound_box=rve_bounding_box)

    scan_path_builder = ScanPathBuilder(
        rve_bounding_box,
        LASER_POWER_WATTS,
        LASER_VELOCITY_M_S,
        hatch_spacing_m,
        layer_thickness_m,
        67.0,
        max(rve_max_point - rve_min_point),
        0,
    )
    scan_path_builder.generate_layers()
    path_vectors = scan_path_builder.process_vectors()

    melt_pool_filter = MeltPoolFilter(
        mp_stats["width_mean"],
        mp_stats["width_std"],
        LASER_VELOCITY_M_S,
        [250000, 0.08],
    )

    length_scale = 10.0 * mp_stats["depth_mean"]
    melt_pool_filter.add_effect("melt_pool", [length_scale, None, 1])
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(1)

    melt_pool_dict = {
        "width": (width_data, 50, 1.0, 2),
        "depth": (width_data, 50, mp_stats["depth_mean"] / mp_stats["width_mean"], 1),
        "height": (width_data, 50, mp_stats["height_mean"] / mp_stats["width_mean"], 1),
    }
    melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

    porosity = compute_porosity(grid, path_vectors, melt_pool, False)
    metrics = compute_morphology(porosity, grid.resolution, metric_names)

    combined_defects = metrics["equivalent_diameter_area"]

    if len(combined_defects) > 0:
        max_pore = np.max(combined_defects)
    else:
        max_pore = 0.0

    logger.info(f"Hatch: {hatch_spacing_m*1e6:.1f}um | Max Pore: {max_pore*1e6:.2f}um")

    return float(np.log1p(max_pore))


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


def get_data_point(x_suggested, mp_interpolator):
    x_ = [np.clip(x_suggested[0], BOUNDS[0][0], BOUNDS[0][1])]
    y_ = run_raptor(x_[0], mp_interpolator)
    return x_, y_


# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------
class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination
        self.iteration_count = 0
        self.workflow_id = ""

        self.mp_interpolator = MeltPoolInterpolator(MELT_POOL_SURROGATE_PATH)

        logger.info(f"Performing cold start with {INITIAL_DATA_SIZE} points...")
        lhs = qmc.LatinHypercube(d=NUM_DIMS, seed=SEED)
        lhs_samples = qmc.scale(
            lhs.random(n=INITIAL_DATA_SIZE), [BOUNDS[0][0]], [BOUNDS[0][1]]
        )
        bounds_points = np.array([[BOUNDS[0][0]], [BOUNDS[0][1]]])

        self.dataset_x = np.vstack([lhs_samples, bounds_points]).tolist()
        self.dataset_y = [
            run_raptor(x[0], self.mp_interpolator) for x in self.dataset_x
        ]

        self.y_mean = np.mean(self.dataset_y)
        self.y_std = np.std(self.dataset_y) if np.std(self.dataset_y) > 0 else 1.0

        self.dataset_x_unit = x_to_unit(self.dataset_x).tolist()
        self.bounds_unit = UNIT_BOUNDS

    def assemble_message(
        self, operation: str, **kwargs: Any
    ) -> IntersectClientCallback:
        payload = None
        if operation == "initialize_workflow":
            y_norm = ((np.array(self.dataset_y) - self.y_mean) / self.y_std).tolist()
            payload = DialWorkflowCreationParamsClient(
                dataset_x=self.dataset_x_unit,
                dataset_y=y_norm,
                bounds=self.bounds_unit,
                kernel="matern",
                length_per_dimension=False,
                y_is_good=False,
                backend="sklearn",
                seed=SEED,
                preprocess_standardize=False,
                backend_args={"alpha": 1e-3},
            )
        elif operation == "update_workflow_with_data":
            next_y_norm = (kwargs["next_y"] - self.y_mean) / self.y_std
            kwargs["next_y"] = float(next_y_norm)
            payload = DialWorkflowDatasetUpdate(workflow_id=self.workflow_id, **kwargs)
        elif operation == "get_next_point":
            payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy="expected_improvement",
                bounds=self.bounds_unit,
            )
        elif operation == "get_surrogate_values":
            points_unit = x_to_unit(INITIAL_POINTS_TO_PREDICT).tolist()
            payload = DialInputPredictions(
                workflow_id=self.workflow_id, points_to_predict=points_unit
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
            print(operation, payload, file=sys.stderr)
            raise Exception

        if operation == "dial.initialize_workflow":
            self.workflow_id = payload
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.update_workflow_with_data":
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.get_surrogate_values":
            mean_norm = np.array(payload[0])
            var_norm = np.array(payload[1])

            self.mean_grid = (mean_norm * self.y_std) + self.y_mean
            self.variance = var_norm * (self.y_std**2)

            np.savez(
                "defect_model_surrogate.npz",
                mean_grid=self.mean_grid,
                variance_grid=self.variance,
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=BOUNDS,
                laser_power=LASER_POWER_WATTS,
                laser_velocity=LASER_VELOCITY_M_S,
            )

            if self.iteration_count >= MAX_ITERATIONS:
                logger.info(
                    "Active Learning Complete. Surrogate saved to 'defect_model_surrogate.npz'."
                )
                raise Exception("DONE")
            return self.assemble_message("get_next_point")

        if operation == "dial.get_next_point":
            x_suggested_unit = np.array(payload).reshape(1, -1)
            x_suggested = x_from_unit(x_suggested_unit)[0].tolist()

            logger.info(
                f"Iteration {self.iteration_count}: "
                f"DIAL suggests HS={x_suggested[0]*1e6:.2f}um"
            )

            new_x, new_y_log = get_data_point(x_suggested, self.mp_interpolator)

            self.dataset_x.append(new_x)
            self.dataset_y.append(new_y_log)

            new_x_unit = x_to_unit(new_x).flatten().tolist()
            self.dataset_x_unit.append(new_x_unit)

            self.iteration_count += 1

            return self.assemble_message(
                "update_workflow_with_data", next_x=new_x_unit, next_y=float(new_y_log)
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

    default_intersect_lifecycle_loop(
        client,
    )
