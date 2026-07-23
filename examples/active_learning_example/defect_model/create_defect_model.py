import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
from scipy.stats import qmc
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as st

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

VOXEL_RESOLUTION_M = 5.0e-6
RVE_LENGTH_M = 1e-3
QUERY_VOLUME_MM3 = 5.0  # decrease query_volume_mm3 from 10 to speed up

MIN_LEN_DEFECTS = 50


class AnalysisMode(StrEnum):
    MEAN = auto()
    MAX = auto()
    LOG_MEAN = auto()
    LOG_CVAR = auto()


ANALYZE = AnalysisMode.LOG_CVAR

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
    hatch_spacing = raptor_data["inputs"]["hatch_spacing_m"]

    min_len_defects = MIN_LEN_DEFECTS
    if len(combined_defects) < min_len_defects:
        # add some pores below the voxel_resolution, to deal with empty lists due to finite resolution
        # TODO: discuss how we should handle the fact that pores blow voxel_resolution can not be resolved
        random.seed()  # explicitly call rng seeding to make sure this is truly random
        n_extra_defects = min_len_defects - len(combined_defects)
        mu_subgrid = voxel_resolution_m / 2
        sigma_subgrid = voxel_resolution_m / 4
        more_defects = np.random.lognormal(
            np.log(mu_subgrid), sigma_subgrid / mu_subgrid, n_extra_defects
        ).tolist()
        combined_defects = combined_defects.tolist() + more_defects

    # direct analysis of mean, max and statistics
    max_pore = np.max(combined_defects)
    mean_pore = np.mean(combined_defects)
    std_pore = np.std(combined_defects, ddof=1)
    # compute the standard error of the mean (SEM), the standard deviation of the mean (Monte-Carlo error)
    sem_pore = std_pore / np.sqrt(len(combined_defects))

    # estimate distribution parameters for lognormal pore size distribution
    # Converting to microns for numerical stability
    sort_defect = np.sort(np.array(combined_defects) * 1e6)

    # direct estimate
    (log_mean, log_sem), (log_std, log_sev) = estimate_lognormal_direct(sort_defect)

    # MCMC estimate
    logger.info("running MCMC")
    (log_mean_pore, log_sem_pore), (log_std_pore, log_sev_pore) = (
        estimate_lognormal_MCMC(sort_defect)
    )

    # Approach 1 and 2 should give the same answer
    print(f"-{len(combined_defects)}-\texpl.,\tMCMC")
    print(f"mean:\t{log_mean:.3f},\t{log_mean_pore:.3f}")
    print(f"std:\t{log_std:.3f},\t{log_std_pore:.3f}")
    print(f"sem:\t{log_sem:.3f},\t{log_sem_pore:.3f}")
    print(f"sev:\t{log_sev:.3f},\t{log_sev_pore:.3f}")

    # TODO: if we want to transform these values back to meters we need to account for different scaling of sev
    #       for now, I will do it after further use below.

    def estimate_cvar(defects_list, level=0.05):
        "Estimate the conditional value at risk from a finite sample."
        n_defects = len(defects_list)
        n_bad_defects = n_defects * level
        remainder = n_bad_defects - np.floor(n_bad_defects)
        n_bad_defects = int(np.floor(n_bad_defects))
        weights = np.concat(([remainder], np.ones(n_bad_defects)))
        weights /= np.sum(weights)
        defects_sort = np.sort(np.asarray(defects_list))
        cvar = np.sum(weights * defects_sort[-n_bad_defects - 1 :])
        return cvar

    def bootstrap_defects(n_defects):
        while True:
            # sample a big defect sample from a random realization of the estimated density
            log_mu = log_mean + log_sem * np.random.randn(1)
            log_s2 = log_std**2 + log_sev * np.random.randn(1)
            log_sigma = np.sqrt(log_s2)
            log_samples = log_mu + log_sigma * np.random.randn(n_defects)
            yield np.exp(log_samples)

    cvar_level = 0.2

    def bootstrap_cvar(n_defects=1000, max_bootstrap=1000):
        cvar_array = np.zeros((max_bootstrap, 1))
        for n_bs, sample in enumerate(bootstrap_defects(n_defects)):
            cvar = estimate_cvar(sample, cvar_level)
            cvar_array[n_bs] = cvar
            if n_bs > 1:
                mean_cvar = np.mean(cvar_array[:n_bs])
                std_cvar = np.std(cvar_array[:n_bs], ddof=1)
            if n_bs + 1 >= max_bootstrap:
                return mean_cvar, std_cvar

    ## Use the lognormal estimates to bootstrap cvar
    mean_cvar, err_cvar = bootstrap_cvar()
    # transform back to meters
    print(f"naive CVAR: {estimate_cvar(sort_defect, cvar_level):0.3f}")
    print(
        f"bootstrapped cvar based on lognormal distr: {mean_cvar=:.3f}, {err_cvar=:0.3f}"
    )
    mean_cvar = mean_cvar.item() / 1e6
    err_cvar = err_cvar.item() / 1e6

    def plot_pore_distr():
        logger.info("plotting pore distribution")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        defect_mesh = np.linspace(0.01, (mean_pore + 3 * std_pore) * 1e6, 500)
        ax.plot(
            defect_mesh,
            st.norm.pdf(defect_mesh, loc=mean_pore * 1e6, scale=std_pore * 1e6),
            color="tab:blue",
            linewidth=2,
            label="Standard Gaussian estimate",
        )
        ax.plot(
            defect_mesh,
            st.norm.pdf(np.log(defect_mesh), loc=log_mean, scale=log_std) / defect_mesh,
            color="tab:green",
            linewidth=2,
            label="Log Gaussian estimate",
        )
        ax.plot(
            defect_mesh,
            st.gaussian_kde(sort_defect)(defect_mesh),
            color="tab:orange",
            linewidth=2,
            label="Gaussian KDE",
        )
        ax.scatter(
            sort_defect,
            np.zeros(sort_defect.shape),
            color="black",
            marker="+",
            s=15,
            alpha=0.6,
            label=f"Pore size data {hatch_spacing*1e6:.1f}um",
        )
        ax.axvline(mean_cvar * 1e6, color="k", linestyle="-", label="CVAR")
        ax.axvline(
            (mean_cvar + err_cvar) * 1e6, color="k", linestyle=":", label="CVAR+"
        )
        ax.axvline(
            (mean_cvar - err_cvar) * 1e6, color="k", linestyle=":", label="CVAR-"
        )
        ax.axvline(np.max(sort_defect), color="b", linestyle="-", label="maximum")
        ax.legend()
        ax.set_xlabel("defect size")
        ax.set_ylabel("probability density")
        plt.tight_layout()
        output_path = Path("pore_plots")
        output_path.mkdir(exist_ok=True)
        output_filename = f"pore_{hatch_spacing*1e6:.1f}.png"
        plt.savefig(output_path / output_filename, dpi=300)
        plt.close()

    plot_pore_distr()

    # renormalization and exponential transform, to compare and inspect values
    mean_lognormal = np.exp(log_mean).item() / 1e6
    sem_lognormal = log_sem.item() * mean_lognormal
    logger.info(
        f"Found {len(combined_defects)} defects: "
        f"Hatch: {hatch_spacing*1e6:.1f}um | Mean and Max Pore: {mean_pore*1e6:.2f}, {max_pore*1e6:.2f}um\n | "
        f"Estimated mean_lognormal: {mean_lognormal*1e6:.6f}, sem_lognormal: {sem_lognormal*1e6:.6f}\n | "
        f"Estimated CVAR({cvar_level:.0%}): {mean_cvar*1e6:.6f}, err_CVAR {err_cvar*1e6:.6f}\n | "
        f"Learning {ANALYZE}."
    )

    # TODO: discuss and refine the data analysis and extreme value statistics
    if ANALYZE == "mean":
        y, yerr = float(mean_pore), float(sem_pore)
    elif ANALYZE == "log_mean":
        y, yerr = float(mean_lognormal), float(sem_lognormal)
    elif ANALYZE == "log_cvar":
        y, yerr = float(mean_cvar), float(err_cvar)
    elif ANALYZE == "max":
        # return the maximum pore size, use the standard deviation as approximate error estimate
        y, yerr = float(max_pore), float(std_pore)

    return y, yerr


# -----------------------------------------------------------------------------
# STATISTICS UTILITIES
# -----------------------------------------------------------------------------
def estimate_lognormal_direct(norm_defect):
    "Approach 1: directly estimate the parameters using standard formulas bases on log transform"
    log_norm_defect = np.log(norm_defect)
    log_mean_defect = np.mean(log_norm_defect)
    log_std_defect = np.std(log_norm_defect, ddof=1)
    log_sem_defect = np.sqrt(1.0 / len(norm_defect)) * log_std_defect

    # formula to estimate variance of the sample variance, requires estimate of fourth moment
    def var_of_sample_var():
        n = len(norm_defect)
        coeff_n = (n / (n - 1)) * (n / (n - 2)) * (n / (n - 3))
        moment4 = coeff_n * np.mean((log_norm_defect - log_mean_defect) ** 4)
        res = (moment4 - (n - 3) / (n - 1) * log_std_defect**4) / n
        return res

    # standard error of the variance (sev)
    log_sev_defect = np.sqrt(var_of_sample_var())

    # this simpler formula is only correct when log(norm_defect) is exactly normally distributed
    # log_sev_defect = np.sqrt(2.0 / (len(norm_defect) - 1)) * log_std_defect**2

    return (log_mean_defect, log_sem_defect), (log_std_defect, log_sev_defect)


def estimate_lognormal_MCMC(norm_defect):
    # Approach 2: using a lightweight mcmc approach assuming a lognormal underlying distribution
    # target y - E[µ] in posterior, yerr - sqrt(Var[µ]) in posterior
    trace = run_metropolis_hastings(
        norm_defect,
        iterations=5000,
        proposal_widths=np.array([1, 1]),
    )
    burnin = 1000
    trace = trace.T[:, burnin:]  # discard burn-in samples

    # extract statistics from MCMC trace
    log_mean_pore = np.mean(trace[0])
    log_sem_pore = np.std(trace[0], ddof=1)
    # square root of the mean variance
    log_std_pore = np.sqrt(np.mean(trace[1] ** 2))
    # standard deviation of the variance
    log_sev_pore = np.std(trace[1] ** 2, ddof=1)

    return (log_mean_pore, log_sem_pore), (log_std_pore, log_sev_pore)


def log_prior_lognormal(params):
    mu, sigma = params
    if sigma <= 0:
        return -np.inf  # log(0)
    mu_prior = st.norm.logpdf(mu, loc=0, scale=10)  # Example prior for mean
    sigma_prior = st.norm.logpdf(sigma, loc=1, scale=5)  # Example prior for std
    return mu_prior + sigma_prior


def loglikelihood_lognormal(params, data):
    mu, sigma = params
    if sigma <= 0:
        return -np.inf  # log(0)
    return np.sum(st.lognorm.logpdf(data, s=sigma, scale=np.exp(mu)))


def log_posterior_lognormal(params, data):
    return loglikelihood_lognormal(params, data) + log_prior_lognormal(params)


def run_metropolis_hastings(
    data, iterations=10000, proposal_widths=np.array([1.0, 2.0])
):
    # Initial guesses
    current_params = np.array([1, 1])  # Example initial guess
    current_log_post = log_posterior_lognormal(current_params, data)

    trace = []

    for i in range(iterations):
        # Propose new parameters (Random Walk)
        proposal = current_params + np.random.normal(
            0, proposal_widths, size=current_params.shape
        )

        proposal_log_post = log_posterior_lognormal(proposal, data)

        # Acceptance ratio
        ratio = np.exp((proposal_log_post - current_log_post))

        if np.random.rand() < ratio:
            current_params = proposal
            current_log_post = proposal_log_post

        trace.append(current_params)

    return np.array(trace)


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
