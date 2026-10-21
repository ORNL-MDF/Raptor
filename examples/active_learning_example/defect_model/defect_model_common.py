import json
import logging
import sys
import random
import time
from pathlib import Path
from typing import Any
from enum import Enum

import numpy as np
from scipy.stats import qmc
from scipy.interpolate import RegularGridInterpolator

# Raptor Imports
from raptor.api import (
    create_grid,
    create_melt_pool,
    create_path_vectors,
    compute_porosity,
    compute_morphology,
)
from raptor.utilities import MeltPoolFilter

# Intersect Imports
from intersect_sdk import (
    INTERSECT_RESPONSE_VALUE,
    IntersectClientCallback,
    IntersectDirectMessageParams,
)

# Dial Imports
from dial_dataclass import (
    DialInputPredictions,
    DialInputSingleOtherStrategy,
    DialInputMultipleOtherStrategy,
    DialWorkflowCreationParamsClient,
    DialWorkflowDatasetUpdates,
    Normal,
)

from scalers import SCALER_REGISTRY, InputScaler
from statistics_utilities import (
    estimate_lognormal_direct,
    estimate_lognormal_MCMC,
    bootstrap_cvar,
    estimate_lognormal_cvar,
    plot_defect_distribution,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# GLOBAL DEFAULT USER PARAMETERS
# -----------------------------------------------------------------------------
LASER_POWER_WATTS = 195
LASER_VELOCITY_M_S = 1.083

LAYER_THICKNESS = 40e-6

VOXEL_RESOLUTION_M = 5.0e-6  # reference 5.0e-6
RVE_LENGTH_M = 2e-3
QUERY_VOLUME_MM3 = (
    3 * 8.0
)  # decrease query_volume_mm3 factor * rve_volume to speed up

MIN_LEN_DEFECTS = 50

SEED = 42

MELT_POOL_SURROGATE_PATH = (
    Path(__file__).parent.parent
    / "melt_pool_model"
    / "melt_pool_surrogates.npz"
)


# -----------------------------------------------------------------------------
# RAPTOR UTILITIES
# -----------------------------------------------------------------------------
class MeltPoolInterpolator:
    def __init__(self, filepath: Path | str):
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
        return {
            f: float(self.interpolators[f](point)[0]) for f in self.features
        }


def run_raptor(
    mp_interpolator: MeltPoolInterpolator,
    hatch_spacing_m: float,
    layer_thickness_m: float = LAYER_THICKNESS,
    query_volume_mm3: float = QUERY_VOLUME_MM3,
    laser_velocity_m_s: float = LASER_VELOCITY_M_S,
    laser_power_watts: float = LASER_POWER_WATTS,
    rve_length_m: float = RVE_LENGTH_M,
    voxel_resolution_m: float = VOXEL_RESOLUTION_M,
    metric_names: list[str] = ["equivalent_diameter_area", "area"],
):
    # Query melt pool statistics for processing conditions
    mp_stats = mp_interpolator.query(laser_velocity_m_s, laser_power_watts)

    # Create representative volume element (RVE)
    rve_min_point = np.array([0.0, 0.0, 0.0])
    rve_max_point = np.array([rve_length_m] * 3)
    rve_bounding_box = np.array([rve_min_point, rve_max_point])

    grid = create_grid(
        voxel_resolution=voxel_resolution_m, bound_box=rve_bounding_box
    )

    # Create scan path in RVE
    # Pad the scan region to limit edge effects inside the rotated RVE.
    scan_extension_m = 5.0 * max(rve_max_point - rve_min_point)
    path_vectors = create_path_vectors(
        rve_bounding_box,
        laser_power_watts,
        laser_velocity_m_s,
        hatch_spacing_m,
        layer_thickness_m,
        67.0,
        scan_extension_m,
        10,
    )

    # Create stochastic melt pool model
    melt_pool_filter = MeltPoolFilter(
        mp_stats["width_mean"],
        mp_stats["width_std"],
        laser_velocity_m_s,
        voxel_resolution_m,
    )

    length_scale = 10.0 * mp_stats["depth_mean"]
    melt_pool_filter.add_effect("melt_pool", [length_scale, None, 1])
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(
        1, melt_pool_filter.n_points, melt_pool_filter.t
    )

    ellipse = 2
    parabola = 1

    # auto-select modes and signal duration
    num_modes = None
    mode_rmse = voxel_resolution_m

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

    melt_pool = create_melt_pool(
        melt_pool_dict,
        enable_random_phases=True,
        tolerance=mode_rmse,
    )

    # Run simulations for all RVEs
    single_rve_volume_mm3 = np.prod(
        (rve_bounding_box[1] - rve_bounding_box[0]) * 1e3
    )

    num_rves = int(np.ceil(query_volume_mm3 / single_rve_volume_mm3))

    logger.info(
        f"Query Volume: {query_volume_mm3} mm3 "
        f"| RVE Volume: {single_rve_volume_mm3:.4f} mm3"
    )
    logger.info(f"Running {num_rves} RVE simulations...")

    outputs = []
    for i in range(num_rves):
        random_seed = SEED + i
        porosity = compute_porosity(
            grid,
            path_vectors,
            melt_pool,
            random_seed=random_seed,
        )
        metrics = compute_morphology(porosity, grid.resolution, metric_names)

        # turn into list and add seed
        metrics = {k: array.tolist() for k, array in metrics.items()}
        metrics["seed"] = random_seed

        outputs.append(metrics)

    # Package inputs and outputs
    raptor_data = {
        "hatch_spacing_m": hatch_spacing_m,
        "layer_thickness_m": layer_thickness_m,
        "query_volume_mm3": query_volume_mm3,
        "voxel_resolution_m": voxel_resolution_m,
        "num_rves": num_rves,
        "rve_metrics": outputs,
    }
    return raptor_data


class AnalysisMode(str, Enum):
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MAX = "max"
    LOG_MEAN = "log_mean"
    LOG_CVAR = "log_cvar"
    CVAR = "cvar"


def process_raptor_data(
    raptor_data,
    analyze: AnalysisMode,
    min_len_defects=MIN_LEN_DEFECTS,
    cvar_level=0.05,
):
    voxel_resolution_m = raptor_data["voxel_resolution_m"]
    hatch_spacing = raptor_data["hatch_spacing_m"]
    layer_thickness = raptor_data["layer_thickness_m"]

    rve_metrics = raptor_data["rve_metrics"]
    combined_defects_list: list[float] = []
    combined_areas_list: list[float] = []
    max_defects_list: list[float] = []
    for metrics in rve_metrics:
        defects = metrics["equivalent_diameter_area"]
        areas = metrics["area"]

        if len(defects) < min_len_defects:
            # Add sub-resolution pores when thedefect list is too short.
            # TODO: refine how to represent pores below the voxel resolution.
            # Explicitly seed the RNG from system entropy.
            rng = np.random.default_rng()
            n_extra_defects = min_len_defects - len(defects)
            mu_subgrid = voxel_resolution_m / 2
            sigma_subgrid = voxel_resolution_m / 6
            more_defects = rng.lognormal(
                np.log(mu_subgrid), sigma_subgrid / mu_subgrid, n_extra_defects
            )
            defects = defects + more_defects.tolist()
            areas = areas + (more_defects**3 * np.pi / 6).tolist()

        combined_defects_list.extend(defects)
        combined_areas_list.extend(areas)
        max_defects_list.append(max(defects))

    combined_defects = np.array(combined_defects_list)
    combined_areas = np.array(combined_areas_list)

    # direct analysis of mean, max and statistics
    max_pore = np.max(combined_defects)
    mean_pore = np.mean(combined_defects)
    std_pore = np.std(combined_defects, ddof=1)
    # Compute the standard error of the mean (Monte Carlo error).
    sem_pore = std_pore / np.sqrt(len(combined_defects))

    # compue the mean of max. RVE pore and SEM
    max_rve_pore = np.mean(max_defects_list)
    sem_max_rve_pore = (
        np.std(max_defects_list, ddof=1) / np.sqrt(len(max_defects_list))
        if len(max_defects_list) > 1
        else std_pore
    )

    # weighted mean
    w_mean_pore = np.average(combined_defects, weights=combined_areas)
    w_std_pore = np.sqrt(
        np.average(
            (combined_defects - w_mean_pore) ** 2, weights=combined_areas
        )
    )
    w_sem_pore = w_std_pore / np.sqrt(len(combined_defects))

    # estimate distribution parameters for lognormal pore size distribution
    # Converting to microns for numerical stability
    sort_defect = np.sort(np.array(combined_defects) * 1e6)

    # direct estimate
    lognormal_params = estimate_lognormal_direct(sort_defect)
    (log_mean, log_sem), (log_std, log_sev) = lognormal_params

    # MCMC estimate
    run_MCMC = False
    if run_MCMC:
        logger.info("running MCMC")
        lognormal_params_MCMC = estimate_lognormal_MCMC(sort_defect)

        (log_mean_pore, log_sem_pore), (log_std_pore, log_sev_pore) = (
            lognormal_params_MCMC
        )

        # Approach 1 and 2 should give the same answer
        print(f"-{len(combined_defects)}-\texpl.,\tMCMC")
        print(f"mean:\t{log_mean:.3f},\t{log_mean_pore:.3f}")
        print(f"std:\t{log_std:.3f},\t{log_std_pore:.3f}")
        print(f"sem:\t{log_sem:.3f},\t{log_sem_pore:.3f}")
        print(f"sev:\t{log_sev:.3f},\t{log_sev_pore:.3f}")

    # Use the lognormal estimates to estimate CVAR with error
    mean_cvar, err_cvar = estimate_lognormal_cvar(
        lognormal_params, cvar_level=cvar_level
    )
    logger.info(
        "estimated CVAR based on lognormal distr: "
        f"{mean_cvar=:.3f}, {err_cvar=:0.3f}"
    )

    # Use the bootstrap to estimate CVAR with error
    mean_cvar_bs, err_cvar_bs = bootstrap_cvar(
        sort_defect, cvar_level=cvar_level
    )
    logger.info(
        f"bootstrapped direct CVAR: {mean_cvar_bs=:.3f}, {err_cvar_bs=:0.3f}"
    )

    # transform back to meters
    mean_cvar = mean_cvar.item() / 1e6
    err_cvar = err_cvar.item() / 1e6
    mean_cvar_bs = mean_cvar_bs.item() / 1e6
    err_cvar_bs = err_cvar_bs.item() / 1e6

    logger.info("plotting pore distribution")
    output_filename = (
        f"pore_{hatch_spacing*1e6:.1f}_{layer_thickness*1e6:.1f}.png"
    )
    plot_defect_distribution(
        output_filename,
        sort_defect,
        (mean_pore, std_pore),
        (max_rve_pore, sem_max_rve_pore),
        lognormal_params,
        (
            (mean_cvar, err_cvar)
            if analyze == "log_cvar"
            else (mean_cvar_bs, err_cvar_bs)
        ),
    )

    # renormalization and exponential transform, to compare and inspect values
    mean_lognormal = np.exp(log_mean).item() / 1e6
    sem_lognormal = log_sem.item() * mean_lognormal
    logger.info(
        f"Found {len(combined_defects)} defects: "
        f"Hatch: {hatch_spacing*1e6:.1f}um, "
        f"LT: {layer_thickness*1e6:.1f}um\n | "
        f"Mean, weighted mean and max Pore: {mean_pore*1e6:.2f},"
        f" {w_mean_pore*1e6:.2f}, {max_pore*1e6:.2f}um\n | "
        f"Estimated mean_lognormal: {mean_lognormal*1e6:.6f}, "
        f"sem_lognormal: {sem_lognormal*1e6:.6f}\n | "
        f"Learning {analyze}."
    )

    if analyze == "mean":
        y, yerr = float(mean_pore), float(sem_pore)
    elif analyze == "log_mean":
        y, yerr = float(mean_lognormal), float(sem_lognormal)
    elif analyze == "log_cvar":
        y, yerr = float(mean_cvar), float(err_cvar)
    elif analyze == "cvar":
        y, yerr = float(mean_cvar_bs), float(err_cvar_bs)
    elif analyze == "weighted_mean":
        y, yerr = float(w_mean_pore), float(w_sem_pore)
    elif analyze == "max":
        y, yerr = float(max_rve_pore), float(sem_max_rve_pore)

    if analyze.startswith("log") or analyze == "cvar":
        # Statistics lose meaning when max defect approaches the RVE length.
        # return a large enough value with high certainty
        cutoff_for_max_pore = RVE_LENGTH_M / 10.0
        if y > cutoff_for_max_pore:
            # or max_pore > cutoff_for_max_pore:
            y = cutoff_for_max_pore
            yerr = y * 1e-4

    return y, yerr


# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------
class ActiveLearningOrchestrator:
    def __init__(
        self,
        service_destination: str,
        bounds: list[tuple[float, float]],
        n_acquire_grid: int,
        initial_data_size: int,
        batch_size: int,
        max_iterations: int,
        input_grid: list[list[float]],
        backend: str,
        statistics_yerr: str,
    ):
        self.service_destination = service_destination
        self.bounds = bounds
        self.n_acquire_grid = n_acquire_grid
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.input_grid = input_grid
        self.backend = backend
        self.statistics_yerr = statistics_yerr
        self.logger = logger

        self.iteration_count = 0
        self.workflow_id = ""

        self.mp_interpolator = MeltPoolInterpolator(MELT_POOL_SURROGATE_PATH)

        self.time_log = [
            (time.perf_counter(), "Initialization and initial dataset.")
        ]

        self._init_dataset(initial_data_size)

        scaler = "output_focus_log"
        if scaler == "lop1p":
            # Scaling factors before and after the log transform.
            # Output scaling also affects the acquisition-strategy error bar.
            pre_to_post_scale_ratio = 0.05
            y_prescale = pre_to_post_scale_ratio * np.max(self.dataset_y)
            y_postscale = np.log1p(1 / pre_to_post_scale_ratio)
            self.scaler = SCALER_REGISTRY[scaler](
                y_prescale=y_prescale, y_postscale=y_postscale
            )
        elif scaler.startswith("output_focus"):
            D_CRIT_LIST = [20e-6, 40e-6]
            # [y_low, y_high] roughly outlines the "interesting" output region
            y_low = min(D_CRIT_LIST)
            y_high = max(D_CRIT_LIST)
            # Focus zooms in (> 1) or out (< 1) on the target region.
            focus = 3.0
            self.scaler = SCALER_REGISTRY[scaler](
                y_low=y_low, y_high=y_high, focus=focus
            )

        self.input_scaler = InputScaler(bounds=list(self.bounds))
        self.dataset_x_unit = self.input_scaler.to_unit(self.dataset_x)
        self.bounds_unit = list(
            zip(*self.input_scaler.to_unit(list(zip(*self.bounds))))
        )

        # acquire on a grid, or on the whole input cube
        if n_acquire_grid > 0:
            self.discrete_measurements = [n_acquire_grid] * len(self.bounds)
        else:
            self.discrete_measurements = []

    def _get_data_point(self, x: list[float]):
        logger.info(f"Iteration {self.iteration_count}: " "Not implemented!.")
        raise NotImplementedError

    def _init_dataset(
        self,
        initial_data_size: int,
    ):
        logger.info(f"Performing cold start with {initial_data_size} points...")
        bounds_np = np.array(self.bounds)
        num_dims = len(self.bounds)
        lhs = qmc.LatinHypercube(d=num_dims, seed=SEED)
        lhs_samples = qmc.scale(
            lhs.random(n=initial_data_size), bounds_np[:, 0], bounds_np[:, 1]
        )

        self.dataset_x = lhs_samples.tolist()
        initial_dataset = [self._get_data_point(x) for x in self.dataset_x]
        (
            self.dataset_x,
            self.dataset_y,
            self.dataset_yerr,
            self.dataset_raptor,
        ) = [list(tup) for tup in zip(*initial_dataset)]

    def _save_dataset(self, *args):
        raise NotImplementedError

    def _handle_surrogate_values(self, means, stddevs):
        y_norm_grid = np.array(means)
        yerr_norm_grid = np.array(stddevs)

        # rescale / transform data back to original units for saving
        y_grid, yerr_grid = self.scaler.unscale(y_norm_grid, yerr_norm_grid)

        self.mean_grid = np.asarray(y_grid)
        self.variance_grid = np.asarray(yerr_grid) ** 2

        self.time_log.append((time.perf_counter(), "Saving data."))

        self._save_dataset(y_norm_grid, yerr_norm_grid)

    def _handle_one_new_input(self, x_suggested):
        self.time_log.append(
            (time.perf_counter(), "Running Raptor to evaluate output data.")
        )

        new_x, new_y, new_yerr, new_raptor_data = self._get_data_point(
            x_suggested
        )

        self.dataset_x.append(new_x)
        self.dataset_raptor.append(new_raptor_data)
        self.dataset_y.append(new_y)
        self.dataset_yerr.append(new_yerr)
        self.dataset_x_unit.append(self.input_scaler.to_unit(new_x))

        self.iteration_count += 1

    def assemble_message(
        self, operation: str, **kwargs: Any
    ) -> IntersectClientCallback:
        payload = None
        if operation == "initialize_workflow":
            # normalize and transform the output data
            y_norm, yerr_norm = self.scaler.scale(
                self.dataset_y, self.dataset_yerr
            )
            # configure the output statistics and combined dataset
            self.labels_y = ["y", "yerr"]
            initial_dataset_y = list(zip(y_norm, yerr_norm))

            # configure the statistics used for learning
            self.statistics_y = Normal(loc="y", scale=self.statistics_yerr)

            # Prior kernel variance (uncertainty without data).
            prior_std = 1.5
            prior_variance = prior_std**2

            if self.backend == "sklearn":
                self.kernel = "matern"
                # nondimensionalized GP lengthscale, on the normalized x data
                length_scale = 0.5
                self.kernel_args: dict[str, Any] = {
                    "length_scale": length_scale,
                    "constant_value": prior_variance,
                }
                self.backend_args = {}

            elif self.backend == "sable":
                self.kernel = "rbf"
                self.kernel_args = {
                    # x range of the data
                    # DIAL currently normalizes the bounds to [0, 1].
                    "x_range": (0, 1),
                    # sigma range of valid lengthscales
                    "sigma_range": [5e-2, 0.5],
                    # smoothness hyperparameter gamma
                    # 0 is continuous, 1 is once differentiable, and so on.
                    "gamma": 0.3,
                }
                self.backend_args = {
                    # memory size for number of features:
                    # More features increase capacity and runtime.
                    "n_features": 2000,
                    # prior standard deviation
                    "prior_std": prior_std,
                    # algorithm hyperparameters
                    # p=2 is a GP; p=1 is fully sparse.
                    "p": 1.0,
                    # More optimization steps improve fit but cost runtime.
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
                seed=-1,
                preprocess_standardize=False,
            )

        elif operation == "update_workflow_with_batch_data":
            try:
                next_x = kwargs["next_x"]
                next_y = kwargs["next_y"]
            except Exception as error:
                print(f"could not extract next datapoints for update: {error}")

            payload = DialWorkflowDatasetUpdates(
                workflow_id=self.workflow_id,
                backend_args=self.backend_args,
                next_x_list=next_x,
                next_y_list=next_y,
            )
        elif operation == "get_next_point":
            payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy="upper_confidence_bound",
                strategy_args={"exploit": 0.0, "explore": 1.0},
                discrete_measurements=bool(self.discrete_measurements),
                discrete_measurement_grid_size=self.discrete_measurements,
                bounds=self.bounds_unit,
            )
        elif operation == "get_next_points":
            payload = DialInputMultipleOtherStrategy(
                workflow_id=self.workflow_id,
                points=self.batch_size,
                batch_strategy="believer",
                strategy="upper_confidence_bound",
                strategy_args={"exploit": 0.0, "explore": 1.0},
                discrete_measurements=bool(self.discrete_measurements),
                discrete_measurement_grid_size=self.discrete_measurements,
                bounds=self.bounds_unit,
            )
        elif operation == "get_surrogate_values":
            points_unit = self.input_scaler.to_unit(self.input_grid)
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
            self.time_log.append(
                (time.perf_counter(), "Asking DIAL for surrogate eval.")
            )
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.update_workflow_with_batch_data":
            self.time_log.append(
                (time.perf_counter(), "Asking DIAL for surrogate eval.")
            )
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.get_surrogate_values":
            try:
                means = payload["values"]
                stddevs = payload["stddevs"]
            except Exception as error:
                print(f"Could not read surrogate values from payload: {error}")

            self._handle_surrogate_values(means, stddevs)

            # Log timings:
            newevent = (
                time.perf_counter(),
                f"Asking DIAL for {self.batch_size} next points x.",
            )
            for (t0, e0), (t1, e1_) in zip(
                self.time_log,
                self.time_log[1:] + [newevent],
            ):
                timespan = t1 - t0
                logger.info(f"Timelog: took {timespan:6.2f}s for '{e0}'")

            self.time_log = [newevent]

            if self.iteration_count >= self.max_iterations:
                logger.info("Active Learning Complete.")
                raise Exception("DONE")

            if self.batch_size == 1:
                return self.assemble_message("get_next_point")
            else:
                return self.assemble_message("get_next_points")

        if (
            operation == "dial.get_next_points"
            or operation == "dial.get_next_point"
        ):
            try:
                data = payload["data"]
            except Exception as error:
                print(f"Could not read next point from payload: {error}")

            x_suggested_unit = np.array(data).reshape(
                self.batch_size, len(self.bounds)
            )
            x_suggested = self.input_scaler.from_unit(x_suggested_unit)

            for x in x_suggested:
                self._handle_one_new_input(x)

            # prepare collected data for sending
            next_x = x_suggested_unit.tolist()
            n_acquired = len(x_suggested)
            next_y_raw = self.dataset_y[-n_acquired:]
            next_yerr_raw = self.dataset_yerr[-n_acquired:]

            # normalize / transform the output data
            y_norm, yerr_norm = self.scaler.scale(next_y_raw, next_yerr_raw)
            next_y = [list(item) for item in zip(y_norm, yerr_norm)]

            self.time_log.append(
                (time.perf_counter(), "Sending new (x,y) data to DIAL.")
            )
            return self.assemble_message(
                "update_workflow_with_batch_data",
                next_x=next_x,
                next_y=next_y,
            )

        else:
            err_msg = f"Unknown operation received: {operation}"
            raise Exception(err_msg)  # noqa: TRY002
