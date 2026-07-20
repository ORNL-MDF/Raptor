import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
from scipy.interpolate import RegularGridInterpolator

# Raptor Imports
from raptor.api import (
    create_grid,
    create_melt_pool,
    compute_porosity,
    compute_morphology,
)
from raptor.utilities import ScanPathBuilder, MeltPoolFilter

# Intersect Imports
from intersect_sdk import (
    INTERSECT_RESPONSE_VALUE,
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
    Normal,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------
LASER_POWER_WATTS = 200.0
LASER_VELOCITY_M_S = 1.0

BOUNDS = ((70e-6, 140e-6),)
UNIT_BOUNDS = ((0.0, 1.0),)
NUM_DIMS = len(BOUNDS)

INITIAL_DATA_SIZE = 1
MAX_ITERATIONS = 40

SEED = 42

VOXEL_RESOLUTION_M = 5e-6
RVE_LENGTH_M = 1e-3
QUERY_VOLUME_MM3 = 3.0  # decrease query_volume_mm3 from 10 to speed up

MIN_LEN_DEFECTS = 50
ANALYZE_MAX = False
BACKEND = "sable"  # "sable" or "sklearn"

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
    query_volume_mm3: float = QUERY_VOLUME_MM3,
    voxel_resolution_m: float = 5e-6,
    metric_names: list[str] = ["equivalent_diameter_area"],
):
    # Query melt pool statistics for processing conditions
    mp_stats = mp_interpolator.query(LASER_VELOCITY_M_S, LASER_POWER_WATTS)

    # Create representative volume element (RVE)
    rve_min_point = np.array([0.0, 0.0, 0.0])
    rve_max_point = np.array([RVE_LENGTH_M, RVE_LENGTH_M, RVE_LENGTH_M])
    rve_bounding_box = np.array([rve_min_point, rve_max_point])

    grid = create_grid(voxel_resolution=voxel_resolution_m, bound_box=rve_bounding_box)

    # Create scan path in RVE
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

    # Create stochastic melt pool model
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

    ellipse = 2
    parabola = 1

    num_modes = 50

    melt_pool_dict = {
        "width": (width_data, num_modes, 1.0, ellipse),
        "depth": (
            width_data,
            num_modes,
            mp_stats["depth_mean"] / mp_stats["width_mean"],
            parabola,
        ),
        "height": (
            width_data,
            num_modes,
            mp_stats["height_mean"] / mp_stats["width_mean"],
            parabola,
        ),
    }
    melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

    # Run simulations for all RVEs
    single_rve_volume_mm3 = np.prod((rve_bounding_box[1] - rve_bounding_box[0]) * 1e3)

    num_rves = int(np.ceil(query_volume_mm3 / single_rve_volume_mm3))

    logger.info(
        f"Query Volume: {query_volume_mm3} mm3 "
        f"| RVE Volume: {single_rve_volume_mm3:.4f} mm3"
    )
    logger.info(f"Running {num_rves} RVE simulations...")

    outputs = []
    for i in range(num_rves):
        porosity = compute_porosity(grid, path_vectors, melt_pool, jit_warmup=0)
        metrics = compute_morphology(porosity, grid.resolution, metric_names)
        outputs.append(metrics)

    combined_outputs = {}
    for name in metric_names:
        arrays = [out[name] for out in outputs if name in out]
        if arrays:
            combined_outputs[name] = np.concatenate(arrays)
        else:
            combined_outputs[name] = np.array([])

    # Package inputs and outputs
    inputs = {
        "hatch_spacing_m": hatch_spacing_m,
        "layer_thickness_m": layer_thickness_m,
        "query_volume_mm3": query_volume_mm3,
        "voxel_resolution_m": voxel_resolution_m,
        "num_rves": num_rves,
    }

    raptor_data = {"inputs": inputs, "outputs": combined_outputs}

    return raptor_data


def process_raptor_data(raptor_data):

    voxel_resolution_m = raptor_data["inputs"]["voxel_resolution_m"]
    combined_defects = raptor_data["outputs"]["equivalent_diameter_area"]

    min_len_defects = MIN_LEN_DEFECTS
    if len(combined_defects) < min_len_defects:
        # add some pores below the voxel_resolution, to deal with empty lists due to finite resolution
        # TODO: discuss how we should handle the fact that pores blow voxel_resolution can not be resolved
        random.seed()  # explicitly call rng seeding to make sure this is truly random
        n_extra_defects = min_len_defects - len(combined_defects)
        more_defects = (voxel_resolution_m * np.random.rand(n_extra_defects)).tolist()
        combined_defects = combined_defects.tolist() + more_defects

    max_pore = np.max(combined_defects)
    mean_pore = np.mean(combined_defects)
    std_pore = np.std(combined_defects, ddof=1)
    # compute the standard error, the standard deviation of the mean (Monte-Carlo error)
    std_err_pore = std_pore / np.sqrt(len(combined_defects))

    logger.info(
        f"Found {len(combined_defects)} defects: "
        f"Hatch: {raptor_data['inputs']['hatch_spacing_m']*1e6:.1f}um | Mean and Max Pore: {mean_pore*1e6:.2f}, {max_pore*1e6:.2f}um, "
        f"Learning {'Max' if ANALYZE_MAX else 'Mean'}."
    )

    # TODO: discuss and refine the data analysis and extreme value statistics
    return_mean = not ANALYZE_MAX
    if return_mean:
        y, yerr = float(mean_pore), float(std_err_pore)
    else:
        # return the maximum pore size, use the standard deveiation as approximate error estimate
        y, yerr = float(max_pore), float(std_pore)

    return y, yerr


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def x_to_unit(x):
    x = np.asarray(x, dtype=float)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    return (x - lo) / (hi - lo + 1e-12)


def x_from_unit(x):
    x = np.asarray(x, dtype=float)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    return x * (hi - lo) + lo


def get_data_point(x_suggested, mp_interpolator):
    x_ = [np.clip(x_suggested[0], BOUNDS[0][0], BOUNDS[0][1])]
    raptor_data = run_raptor(
        x_[0],
        mp_interpolator,
        voxel_resolution_m=VOXEL_RESOLUTION_M,
    )
    y_, yerr_ = process_raptor_data(raptor_data)
    return x_, y_, yerr_, raptor_data


# The unlist_list transformer
def unlist_list(func):
    def wrapper_func(self, *args):
        args_np = [np.asarray(y, dtype=float) for y in args]
        res_np = func(self, *args_np)
        return [y.tolist() for y in res_np]

    return wrapper_func


@dataclass
class ScalerLog1p:
    y_prescale: float = 1.0
    y_postscale: float = 1.0

    @unlist_list
    def scale(self, y, yerr):
        y_scale = np.log1p(y / self.y_prescale) / self.y_postscale
        yerr_scale = yerr / (y + self.y_prescale) / self.y_postscale
        return y_scale, yerr_scale

    @unlist_list
    def unscale(self, y_scale, yerr_scale):
        y = self.y_prescale * np.expm1(self.y_postscale * y_scale)
        yerr = yerr_scale * (y + self.y_prescale) * self.y_postscale
        return y, yerr


@dataclass
class ScalerOutputFocus:
    y_low: float = 0.5
    y_high: float = 1.5
    focus: float = 1.0

    def params(self):
        y_mean = (self.y_low + self.y_high) / 2.0
        y_diff = (1 / self.focus) * (self.y_high - self.y_low) / 2.0
        return y_mean, y_diff

    @unlist_list
    def scale(self, y, yerr):
        y_mean, y_diff = self.params()
        y_norm = (y - y_mean) / y_diff
        yerr_norm = yerr / y_diff
        y_scale = np.asinh(y_norm)
        yerr_scale = 1 / np.sqrt(1 + y_norm**2) * yerr_norm
        return y_scale, yerr_scale

    @unlist_list
    def unscale(self, y_scale, yerr_scale):
        y_mean, y_diff = self.params()
        y_norm = np.sinh(y_scale)
        yerr_norm = np.sqrt(1 + y_norm**2) * yerr_scale
        y = y_diff * y_norm + y_mean
        yerr = y_diff * yerr_norm
        return y, yerr


scaler_reg = {
    "log1p": ScalerLog1p,
    "output_focus": ScalerOutputFocus,
}


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
        self.dataset_raptor = [
            run_raptor(
                x[0],
                self.mp_interpolator,
                voxel_resolution_m=VOXEL_RESOLUTION_M,
            )
            for x in self.dataset_x
        ]
        # pre-process the raw data to extract y value and yerr
        dataset_statistics = [
            process_raptor_data(raptor_data) for raptor_data in self.dataset_raptor
        ]
        self.dataset_y, self.dataset_yerr = [
            list(tuple) for tuple in zip(*dataset_statistics)
        ]

        scaler = "output_focus"
        if scaler == "lop1p":
            # scaling factor for output data transformation, pre-scaling, and post-scaling after log transform
            # crucially, scling the outputs also scales the error bar, which influences the acquisition strategy
            pre_to_post_scale_ratio = 0.05
            y_prescale = pre_to_post_scale_ratio * np.max(self.dataset_y)
            y_postscale = np.log1p(1 / pre_to_post_scale_ratio)
            self.scaler = scaler_reg[scaler](
                y_prescale=y_prescale, y_postscale=y_postscale
            )
        elif scaler == "output_focus":
            D_CRIT_LIST = [10e-6, 20e-6, 40e-6]
            # [y_low, y_high] roghly outlines the "interesting" output region
            y_low = min(D_CRIT_LIST)
            y_high = max(D_CRIT_LIST)
            # focus is a scaling parameter that allows to zoom in (focus > 1) or zoom out (focus < 1) for the target region
            focus = 2.0
            self.scaler = scaler_reg[scaler](y_low=y_low, y_high=y_high, focus=focus)

        self.dataset_x_unit = x_to_unit(self.dataset_x).tolist()
        self.bounds_unit = UNIT_BOUNDS

    def assemble_message(
        self, operation: str, **kwargs: Any
    ) -> IntersectClientCallback:
        payload = None
        if operation == "initialize_workflow":
            # normalize and transform the output data
            y_norm, yerr_norm = self.scaler.scale(self.dataset_y, self.dataset_yerr)
            # configure the output statistics and combined dataset
            self.labels_y = ["y", "yerr"]
            self.statistics_y = Normal(loc="y", scale="yerr")
            initial_dataset_y = list(zip(y_norm, yerr_norm))

            # prior variance of the kernel (how large is uncertainty without data)
            prior_std = 1.5
            prior_variance = prior_std**2

            self.backend = BACKEND
            if self.backend == "sklearn":
                self.kernel = "matern"
                # nondimensionalized GP lengthscale, on the normalized x data
                length_scale = 0.2
                self.kernel_args: dict[str, Any] = {
                    "length_scale": length_scale,
                    "constant_value": prior_variance,
                }
                self.backend_args = {}

            elif self.backend == "sable":
                self.kernel = "rbf"
                self.kernel_args = {
                    # x range of the data
                    # the bounds are always [0, 1], since dial currently normalizes the input
                    "x_range": self.bounds_unit[0],
                    # sigma range of valid lengthscales
                    "sigma_range": [1e-3, 0.5],
                    # smoothness hyperparameter gamma
                    # (0. means the minimum degree of smoothness, i.e. continuous;
                    #  1. is once differentiable, etc. )
                    "gamma": 0.5,
                }
                self.backend_args = {
                    # memory size for number of features:
                    # needs to be large enough, but becomes slower with more features
                    "n_features": 10000,
                    # prior standard deviation
                    "prior_std": prior_std,
                    # algorithm hyperparameters
                    # p is degree of adaptivity (p=2 is a GP, p=1 is fully sparse)
                    "p": 1.25,
                    # number of optimization steps (needs to be large enough, but slows performance)
                    "n_iter_irls": 100,
                }

            payload = DialWorkflowCreationParamsClient(
                dataset_x=self.dataset_x_unit,
                dataset_y=initial_dataset_y,
                labels_y=self.labels_y,
                statistics_y=self.statistics_y,
                bounds=self.bounds_unit,
                kernel=self.kernel,
                y_is_good=False,
                backend=self.backend,
                kernel_args=self.kernel_args,
                backend_args=self.backend_args,
                seed=SEED,
                preprocess_standardize=False,
            )

        elif operation == "update_workflow_with_data":
            try:
                next_x = kwargs["next_x"]
                next_y = kwargs["next_y"]
            except Exception as error:
                print(f"could not extract next datapoint for update: {error}")

            # normalize / transform the output data
            y, yerr = next_y
            y_norm, yerr_norm = self.scaler.scale(y, yerr)
            next_y = [y_norm, yerr_norm]

            payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                backend_args=self.backend_args,
                next_x=next_x,
                next_y=next_y,
            )
        elif operation == "get_next_point":
            payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy="upper_confidence_bound",
                strategy_args={"exploit": 0.0, "explore": 1.0},
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
        _has_error: bool,
        payload: INTERSECT_RESPONSE_VALUE,
    ) -> IntersectClientCallback:

        if _has_error:
            print("============ERROR==============", file=sys.stderr)
            print(operation, payload, file=sys.stderr)
            raise Exception

        if operation == "dial.initialize_workflow":
            self.workflow_id = payload
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.update_workflow_with_data":
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.get_surrogate_values":
            try:
                means = payload["values"]
                stddevs = payload["stddevs"]
            except Exception as error:
                print(f"Could not read surrogate values from payload: {error}")

            y_norm_grid = np.array(means)
            yerr_norm_grid = np.array(stddevs)

            # rescale / transform data back to original units for saving
            y_grid, yerr_grid = self.scaler.unscale(y_norm_grid, yerr_norm_grid)

            self.mean_grid = np.asarray(y_grid)
            self.variance_grid = np.asarray(yerr_grid) ** 2

            np.savez(
                "defect_model_surrogate.npz",
                mean_grid=self.mean_grid,
                variance_grid=self.variance_grid,
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                dataset_yerr=self.dataset_yerr,
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
            try:
                data = payload["data"]
            except Exception as error:
                print(f"Could not read next point from payload: {error}")

            x_suggested_unit = np.array(data).reshape(1, -1)
            x_suggested = x_from_unit(x_suggested_unit)[0].tolist()

            logger.info(
                f"Iteration {self.iteration_count}: "
                f"DIAL suggests HS={x_suggested[0]*1e6:.2f}um"
            )

            new_x, new_y, new_yerr, new_raptor_data = get_data_point(
                x_suggested, self.mp_interpolator
            )

            self.dataset_x.append(new_x)
            self.dataset_raptor.append(new_raptor_data)
            self.dataset_y.append(new_y)
            self.dataset_yerr.append(new_yerr)

            # determine the next data (x, y) for the update message
            next_y = [float(new_y), float(new_yerr)]
            next_x = x_to_unit(new_x).flatten().tolist()

            self.dataset_x_unit.append(next_x)

            self.iteration_count += 1

            return self.assemble_message(
                "update_workflow_with_data",
                next_x=next_x,
                next_y=next_y,
            )

        else:
            err_msg = f"Unknown operation received: {operation}"
            raise Exception(err_msg)  # noqa: TRY002


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
