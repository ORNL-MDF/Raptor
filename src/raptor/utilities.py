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
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from .structures import PathVector
from scipy.signal import butter, sosfilt
from scipy.stats import chi2


class ScanPathBuilder:
    """
    Handles scan strategy generation from process parameters using explicit boundaries.
    """

    def __init__(
        self,
        bound_box: np.ndarray,
        power: float,
        scan_speed: float,
        hatch_spacing: float,
        layer_height: float,
        rotation: float,
        scan_extension: float,
        extra_layers: int,
    ):
        """
        Initializes the builder with geometric and process parameters.

        Args:
            min_point: The [x, y, z] minimum corner of the part volume.
            max_point: The [x, y, z] maximum corner of the part volume.
            power: Laser power in Watts.
            scan_speed: Scan speed in m/s.
            hatch_spacing: Distance between adjacent scan vectors.
            layer_height: Thickness of each layer.
            rotation: Inter-layer rotation angle in degrees.
            scan_extension: Extra length to add to scan vectors beyond the part boundary.
            extra_layers: Extra layers to generate above the defined part volume.
        """
        self.min_point = bound_box[0]
        self.max_point = bound_box[1]

        self.power = power
        self.scan_speed = scan_speed
        self.hatch_spacing = hatch_spacing
        self.layer_height = layer_height
        self.rotation = np.deg2rad(rotation)
        self.scan_extension = scan_extension
        self.extra_layers = extra_layers

        self.dimensions = self.max_point - self.min_point

        self.center_of_rotation = (self.min_point[:2] + self.max_point[:2]) / 2.0
        self.nlayers = np.int16(
            (self.dimensions[2] // self.layer_height + 1) + self.extra_layers
        )

        self.layers = {}
        self.path_vector_layers = {}

    def generate_layers(self):
        """
        Generates all layers by rotating the base layer.
        """

        # 1. Generate the base layer aligned nominally with [1,0,0]
        xmin = self.min_point[0] - self.scan_extension
        xmax = self.max_point[0] + self.scan_extension

        ymin = self.min_point[1] - self.scan_extension
        ymax = self.max_point[1] + self.scan_extension

        ys = np.arange(ymin, ymax, self.hatch_spacing)
        starts = np.vstack([np.ones_like(ys) * xmin, ys]).transpose()
        ends = np.vstack([np.ones_like(ys) * xmax, ys]).transpose()
        self.layers[0] = [starts, ends]

        # 2. Generate the kth layer by rotating the base layer
        for k in range(1, self.nlayers + 1):
            angle = k * self.rotation
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

            starts = np.array(
                [
                    np.matmul(rotation_matrix, s - self.center_of_rotation)
                    + self.center_of_rotation
                    for s in self.layers[0][0]
                ]
            )
            ends = np.array(
                [
                    np.matmul(rotation_matrix, e - self.center_of_rotation)
                    + self.center_of_rotation
                    for e in self.layers[0][1]
                ]
            )
            self.layers[k] = [starts, ends]

    def process_vectors(self):
        """
        Creates and processes PathVector objects from the generated layers.
        """

        if not self.layers.keys():
            print("No layers generated. Aborting.")
            return
        time_offset = 0.0
        rve_bound_box = np.array([self.min_point, self.max_point])
        # constructing vectors.
        for layer_key, (layer_start, layer_end) in self.layers.items():
            if layer_start.size == 0:
                self.path_vector_layers[layer_key] = []
                continue
            active_vectors = []
            layer_time = time_offset
            for start_xy, end_xy in zip(layer_start, layer_end):
                # defining start, end points and start, end times
                vector_start = np.array(
                    [start_xy[0], start_xy[1], layer_key * self.layer_height]
                )
                vector_end = np.array(
                    [end_xy[0], end_xy[1], layer_key * self.layer_height]
                )
                vector_length = np.linalg.norm(vector_end - vector_start)
                scan_duration = (
                    vector_length / self.scan_speed if self.scan_speed > 1e-12 else 0.0
                )
                if layer_key >= 1 and not active_vectors:
                    start_time = self.path_vector_layers[layer_key - 1][-1].start_time
                else:
                    start_time = layer_time
                end_time = start_time + scan_duration
                # PathVector object instantiation
                path_vector = PathVector(vector_start, vector_end, start_time, end_time)
                active_vectors.append(path_vector)
                layer_time = end_time
            self.path_vector_layers[layer_key] = active_vectors
            time_offset = active_vectors[-1].end_time if active_vectors else 0.0

        # condensing and returning all vectors
        all_vectors = []
        for layer_key, layer_vectors in self.path_vector_layers.items():
            for vec in layer_vectors:
                # not currently filtering --> OPTIMIZE HERE
                vec.set_coordinate_frame()
                all_vectors.append(vec)
        return all_vectors

    def write_layers(self, output_name, mode="layers"):
        """
        Writes the generated raw scan paths to text files.

        Args:
            output_name: Base name for the output files.
            mode: "layers" to write separate files for each layer, "all" to write a single file with all layers.
        """
        if mode == "all":
            all_layers = []
        for l_key, (l_start, l_end) in self.layers.items():
            if l_start.size == 0:
                continue

            se_pairs = [
                np.vstack(
                    [
                        np.hstack([1, s, l_key * self.layer_height, 0, 0]),
                        np.hstack(
                            [
                                0,
                                e,
                                l_key * self.layer_height,
                                self.power,
                                self.scan_speed,
                            ]
                        ),
                    ]
                )
                for s, e in zip(l_start, l_end)
            ]

            all_paths = np.vstack(se_pairs)
            if mode == "all":
                all_layers.append(all_paths)
                continue
            header_str = "Mode X(m) Y(m) Z(m) Power(W) tParam"
            filename = f"{output_name}_layer_{l_key}.txt"

            np.savetxt(
                filename,
                all_paths,
                fmt="%.6f",
                delimiter=" ",
                header=header_str,
                comments="",
            )
            print(f"Wrote file {filename}")

        if mode == "all" and all_layers:
            all_layers = np.vstack(all_layers)
            header_str = "Mode X(m) Y(m) Z(m) Power(W) tParam"
            filename = f"{output_name}.txt"
            np.savetxt(
                filename,
                all_layers,
                fmt="%.6f",
                delimiter=" ",
                header=header_str,
                comments="",
            )
            print(f"Wrote file {filename}")


class MeltPoolFilter:
    def __init__(
        self,
        mu: float,
        sigma: float,
        scan_speed: float,
        voxel_resolution: float,
        *,
        confidence: float = 0.95,
        ci_relative_width: float = 0.10,
        correlation_tolerance: float = 0.10,
        random_seed: Optional[int] = None,
    ):
        """Generate a Gaussian melt-pool history from physical length scales.

        Sampling is tied to the spatial grid through ``fs = scan_speed /
        voxel_resolution``.  ``initialize`` chooses the shortest duration that
        satisfies the requested variance precision while resolving each
        effect's characteristic wavelength. ``correlation_tolerance`` controls
        the relative wavelength-resolution check; it is not the FFT RMSE.
        """
        self.mu, self.sigma = mu, sigma
        self.scan_speed = scan_speed
        self.voxel_resolution = voxel_resolution
        self.fs = self.scan_speed / self.voxel_resolution
        self.confidence = confidence
        self.ci_relative_width = ci_relative_width
        self.correlation_tolerance = correlation_tolerance
        self.rng = np.random.default_rng(random_seed)
        self.physical_effects = {}

    def add_effect(self, effect_name: str, effect_params: list):
        """
        Adds a physical effect {effect_name} with parameters
        length_scale_m,frequency_hz,sigma_weight = effect_params
        to the MeltPoolFiltration.physical_effects dictionary.
        """
        length_scale_m, frequency_hz, sigma_weight = effect_params
        self.physical_effects[effect_name] = {
            "length_scale_m": length_scale_m,
            "frequency_hz": frequency_hz,
            "sigma_weight": sigma_weight,
        }

    def initialize(self) -> None:
        """Choose the minimum planned duration.

        Duration selection is deterministic.  It uses the autocorrelation of
        the configured linear filters to approximate the effective degrees of
        freedom of a Gaussian-process variance estimate.
        """
        # Calculate frequencies from length scales
        for params in self.physical_effects.values():
            if params["length_scale_m"] is not None:
                params["frequency_hz"] = self.scan_speed / params["length_scale_m"]

        # The complete passband, rather than only its center, must satisfy
        # Nyquist.  The same limits are used by bandpass_filter below.
        highest_passband_frequency = max(
            1.5 * p["frequency_hz"] for p in self.physical_effects.values()
        )
        if highest_passband_frequency >= self.fs / 2.0:
            raise ValueError(
                "The filter passband exceeds the Nyquist frequency. "
                "Decrease voxel_resolution or increase the physical length scale."
            )

        # Normalize sigma weights so the variances sum correctly
        weights = np.array([p["sigma_weight"] for p in self.physical_effects.values()])
        sum_of_sq_weights = np.sum(weights**2)
        self.normalization_factor = np.sqrt(sum_of_sq_weights)

        for params in self.physical_effects.values():
            params["sigma_contribution"] = (
                params["sigma_weight"] / self.normalization_factor
            ) * self.sigma

        max_timescale = max(
            1.0 / p["frequency_hz"] for p in self.physical_effects.values()
        )
        minimum_periods = max(4, int(np.ceil(1.0 / self.correlation_tolerance)))
        minimum_duration = minimum_periods * max_timescale
        lower_points = max(3, int(np.ceil(minimum_duration * self.fs)) + 1)

        # The IIR response decays over a small number of characteristic periods;
        # fifty periods provides a conservative autocorrelation horizon without
        # making it scale with the final time-series duration.
        autocorrelation = self._model_autocorrelation()

        upper_points = lower_points
        for _ in range(16):
            effective_n = self._effective_sample_size(upper_points, autocorrelation)
            planned_ci = self._variance_ci(self.sigma**2, effective_n)
            if planned_ci["precision_satisfied"]:
                break
            upper_points = 2 * upper_points - 1
        else:
            raise RuntimeError(
                "Unable to satisfy the variance precision within "
                "16 duration refinements."
            )

        # Precision improves monotonically with sample count for the fixed model
        # autocorrelation, so binary search gives the minimum accepted count.
        left, right = lower_points, upper_points
        while left < right:
            midpoint = (left + right) // 2
            effective_n = self._effective_sample_size(midpoint, autocorrelation)
            if self._variance_ci(self.sigma**2, effective_n)["precision_satisfied"]:
                right = midpoint
            else:
                left = midpoint + 1

        self.n_points = left
        self.duration = (self.n_points - 1) / self.fs
        self.t = np.arange(self.n_points, dtype=np.float64) / self.fs
        self.planned_effective_n = self._effective_sample_size(
            self.n_points, autocorrelation
        )
        self.planned_variance_ci = self._variance_ci(
            self.sigma**2, self.planned_effective_n
        )

    def _model_autocorrelation(self) -> np.ndarray:
        """Return the normalized autocorrelation implied by the shared driver."""
        minimum_frequency = min(
            params["frequency_hz"] for params in self.physical_effects.values()
        )
        response_points = max(256, int(np.ceil(50.0 * self.fs / minimum_frequency)))
        impulse = np.zeros(response_points, dtype=np.float64)
        impulse[0] = 1.0
        combined_response = np.zeros(response_points, dtype=np.float64)

        for params in self.physical_effects.values():
            response = self.bandpass_filter(
                impulse,
                params["frequency_hz"],
                1.0,
                self.fs,
            )
            response_norm = np.sqrt(np.sum(response**2))
            combined_response += params["sigma_contribution"] * response / response_norm

        n_fft = 1 << (2 * response_points - 1).bit_length()
        response_fft = np.fft.rfft(combined_response, n=n_fft)
        autocovariance = np.fft.irfft(
            response_fft * np.conjugate(response_fft), n=n_fft
        )[:response_points]
        return autocovariance / autocovariance[0]

    @staticmethod
    def _effective_sample_size(n_samples: int, autocorrelation: np.ndarray) -> float:
        """Approximate effective sample count for a Gaussian variance estimate."""
        max_lag = min(n_samples - 1, autocorrelation.size - 1)
        if max_lag < 1:
            return float(n_samples)
        lags = np.arange(1, max_lag + 1, dtype=np.float64)
        finite_sample_weights = 1.0 - lags / n_samples
        correlation_penalty = 1.0 + 2.0 * np.sum(
            finite_sample_weights * autocorrelation[1 : max_lag + 1] ** 2
        )
        return float(np.clip(n_samples / correlation_penalty, 2.0, n_samples))

    def bandpass_filter(self, data, f0, bandwidth_fraction, fs, order=4):
        """Applies a bandpass filter around a center frequency f0."""
        lowcut = f0 * (1.0 - bandwidth_fraction / 2.0)
        highcut = f0 * (1.0 + bandwidth_fraction / 2.0)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if not 0.0 < low < high < 1.0:
            raise ValueError(
                "Bandpass range must lie strictly between 0 Hz and the "
                "Nyquist frequency. Increase the sampling frequency."
            )
        sos = butter(order, [low, high], btype="band", output="sos")
        return sosfilt(sos, data)

    def generate_fluctuations(self, noise_scale, n_points, t):
        base_white_noise = self.rng.normal(loc=0.0, scale=noise_scale, size=n_points)
        final_series = np.zeros(n_points)
        self.component_series = {}

        # Create each component series, scale it, and add to the final series
        for name, params in self.physical_effects.items():
            component_noise = self.bandpass_filter(
                data=base_white_noise,
                f0=params["frequency_hz"],
                bandwidth_fraction=1,
                fs=self.fs,
            )

            component_noise -= np.mean(component_noise)
            std_dev = np.std(component_noise)
            if not np.isfinite(std_dev) or std_dev == 0.0:
                raise ValueError("Unable to generate finite melt-pool fluctuations.")
            scaled_component = component_noise * (
                params["sigma_contribution"] / std_dev
            )
            self.component_series[name] = scaled_component
            final_series += scaled_component

        # Shared noise can correlate filtered effects, so normalizing their
        # individual variances does not generally normalize the variance of
        # their sum. Apply one common correction to preserve relative weights.
        aggregate_std = np.std(final_series)
        if not np.isfinite(aggregate_std) or aggregate_std == 0.0:
            raise ValueError("Unable to generate finite melt-pool fluctuations.")
        covariance_scale = self.sigma / aggregate_std
        final_series *= covariance_scale
        for name in self.component_series:
            self.component_series[name] *= covariance_scale

        # Adding the mean
        final_series += self.mu

        return np.column_stack([t, final_series])

    def evaluate_variance_ci(self, data):
        """
        Evaluates the confidence interval for the variance of the data.
        Returns (lower_bound, upper_bound) for the variance.
        """
        n = len(data)
        sample_variance = np.var(data, ddof=1)

        # Compute autocorrelation to estimate effective sample size
        autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode="full")
        autocorr = autocorr[autocorr.size // 2 :] / autocorr[autocorr.size // 2]

        # Effective sample size
        zero_crossings = np.where(autocorr < 0)[0]
        cutoff = zero_crossings[0] if zero_crossings.size > 0 else len(autocorr)
        effective_n = n / (1 + 2 * np.sum(autocorr[1:cutoff]))
        effective_n = int(max(1, min(effective_n, n)))

        return self._variance_ci(sample_variance, effective_n)

    def _variance_ci(self, sample_variance: float, effective_n: float) -> dict:
        """Construct the approximate Gaussian-process variance interval."""
        dof = effective_n - 1.0
        alpha = 1 - self.confidence
        chi2_lower = chi2.ppf(alpha / 2, dof)
        chi2_upper = chi2.ppf(1 - alpha / 2, dof)

        lower_bound = dof * sample_variance / chi2_upper
        upper_bound = dof * sample_variance / chi2_lower

        target_within_ci = lower_bound <= self.sigma**2 <= upper_bound
        target_std_lower = self.sigma * (1.0 - self.ci_relative_width)
        target_std_upper = self.sigma * (1.0 + self.ci_relative_width)
        precision_satisfied = (
            np.sqrt(lower_bound) >= target_std_lower
            and np.sqrt(upper_bound) <= target_std_upper
        )

        return {
            "sample_variance": sample_variance,
            "effective_n": effective_n,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "target_within_ci": target_within_ci,
            "precision_satisfied": precision_satisfied,
        }


def reconstruct_spectral_signal(
    time_values: np.ndarray, spectral_components: np.ndarray
) -> np.ndarray:
    """Evaluate a ``[amplitude, frequency, phase]`` cosine expansion."""
    time_values = np.asarray(time_values, dtype=np.float64)
    spectral_components = np.asarray(spectral_components, dtype=np.float64)
    if spectral_components.ndim != 2 or spectral_components.shape[1] != 3:
        raise ValueError("spectral_components must have shape (n_modes, 3).")
    reconstructed = np.zeros_like(time_values)
    for amplitude, frequency, phase in spectral_components:
        reconstructed += amplitude * np.cos(
            2.0 * np.pi * frequency * time_values + phase
        )
    return reconstructed


def plot_melt_pool_signal(
    time_series: np.ndarray,
    spectral_components: np.ndarray,
    mean: float,
    standard_deviation: float,
    output_path: Union[str, Path],
    *,
    time_scale: float = 1.0e3,
    value_scale: float = 1.0e6,
    time_label: str = "Time (ms)",
    value_label: str = "Melt-pool dimension (µm)",
    bins: int = 40,
) -> Path:
    """Write a 6.5-by-3 inch, 300 dpi signal and distribution figure."""
    data = np.asarray(time_series, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("time_series must have shape (n, 2).")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")

    time_values = data[:, 0]
    signal = data[:, 1]
    reconstructed = reconstruct_spectral_signal(time_values, spectral_components)
    plot_time = time_values * time_scale
    plot_signal = signal * value_scale
    plot_reconstructed = reconstructed * value_scale
    plot_mean = mean * value_scale
    plot_std = standard_deviation * value_scale

    distribution_limits = (
        min(plot_signal.min(), plot_mean - 4.0 * plot_std),
        max(plot_signal.max(), plot_mean + 4.0 * plot_std),
    )
    gaussian_values = np.linspace(*distribution_limits, 500)
    gaussian_density = np.exp(
        -0.5 * ((gaussian_values - plot_mean) / plot_std) ** 2
    ) / (plot_std * np.sqrt(2.0 * np.pi))

    style = {
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
    }
    with plt.rc_context(style):
        figure, (signal_axis, distribution_axis) = plt.subplots(
            1, 2, figsize=(6.5, 3.0), constrained_layout=True
        )
        signal_axis.plot(
            plot_time, plot_signal, color="#0072B2", linewidth=0.9, label="Generated"
        )
        signal_axis.plot(
            plot_time,
            plot_reconstructed,
            color="#D55E00",
            linewidth=0.9,
            linestyle="--",
            label=f"Truncated FFT ({len(spectral_components)} modes)",
        )
        signal_axis.set_xlabel(time_label)
        signal_axis.set_ylabel(value_label)
        signal_axis.legend(frameon=False, loc="upper right")

        distribution_axis.hist(
            plot_signal,
            bins=bins,
            density=True,
            color="#56B4E9",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.8,
            label="Generated samples",
        )
        distribution_axis.plot(
            gaussian_values,
            gaussian_density,
            color="#D55E00",
            linewidth=1.8,
            label=f"Gaussian (µ={plot_mean:.0f}, σ={plot_std:.0f})",
        )
        distribution_axis.axvline(
            plot_mean,
            color="black",
            linewidth=0.8,
            linestyle=":",
            label="Specified mean",
        )
        distribution_axis.set_xlim(distribution_limits)
        distribution_axis.set_xlabel(value_label)
        distribution_axis.set_ylabel("Probability density")
        distribution_axis.legend(frameon=False, loc="upper right")

        for panel, axis in zip(("(a)", "(b)"), (signal_axis, distribution_axis)):
            axis.text(
                0.02,
                0.96,
                panel,
                transform=axis.transAxes,
                fontweight="bold",
                va="top",
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(direction="out")

        output = Path(output_path)
        figure.savefig(output, dpi=300, facecolor="white")
        plt.close(figure)
    return output
