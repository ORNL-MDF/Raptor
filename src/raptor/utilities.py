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
from typing import List, Tuple
from .structures import PathVector
from scipy.signal import lfilter, butter


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
        self, mu: float, sigma: float, scan_speed: float, timeseries_params: list
    ):
        """
        Filtration of disparate fluctuation scales to infer a melt pool oscillations sequence.
        Uses scipy.butter to convolve scales of fluctuations together.
        Initialization parameters:
        mu: mean melt pool dimension
        sigma: target standard deviation of the fluctuations
        scan_speed: speed in m/s
        timeseries_params: list of [fs,duration]
        """
        # statistical properties
        self.mu, self.sigma = mu, sigma
        # process parameters
        self.scan_speed = scan_speed
        # timeseries related properties
        self.fs, self.duration = timeseries_params
        self.dt = 1 / self.fs
        self.n_points = int(self.duration / self.dt)
        self.t = np.arange(0, self.duration, self.dt)
        # parametric representations of fluctuation scales
        self.physical_effects = {}  # contains scale description and parameters

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

    def initialize(self):
        # Calculate frequencies from length scales
        for name, params in self.physical_effects.items():
            if params["length_scale_m"] is not None:
                params["frequency_hz"] = self.scan_speed / params["length_scale_m"]

        # Check for Nyquist limit violations
        max_freq = max(p["frequency_hz"] for p in self.physical_effects.values())
        if max_freq > self.fs / 2:
            raise ValueError(
                f"Error: Maximum frequency ({max_freq/1000:.1f} kHz) exceeds Nyquist limit ({self.fs/2000:.1f} kHz). Increase sampling rate 'fs'."
            )

        # Normalize sigma weights so the variances sum correctly
        weights = np.array([p["sigma_weight"] for p in self.physical_effects.values()])
        sum_of_sq_weights = np.sum(weights**2)
        self.normalization_factor = np.sqrt(sum_of_sq_weights)

        for name, params in self.physical_effects.items():
            params["sigma_contribution"] = (
                params["sigma_weight"] / self.normalization_factor
            ) * self.sigma

    def bandpass_filter(self, data, f0, bandwidth_fraction, fs, order=4):
        """Applies a bandpass filter around a center frequency f0."""
        lowcut = f0 * (1 - bandwidth_fraction / 2)
        highcut = f0 * (1 + bandwidth_fraction / 2)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if high >= 1.0:
            high = 0.999
        if low <= 0.0001:
            low = 0.0001
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, data)

    def generate_fluctuations(self, noise_scale):
        base_white_noise = np.random.normal(
            loc=0, scale=noise_scale, size=self.n_points
        )
        final_series = np.zeros(self.n_points)
        self.component_series = {}

        # Create each component series, scale it, and add to the final series
        for name, params in self.physical_effects.items():
            component_noise = self.bandpass_filter(
                data=base_white_noise,
                f0=params["frequency_hz"],
                bandwidth_fraction=1,
                fs=self.fs,
            )

            std_dev = np.std(component_noise)
            scaled_component = component_noise * (
                params["sigma_contribution"] / std_dev
            )
            self.component_series[name] = scaled_component
            final_series += scaled_component

        # Adding the mean
        final_series += self.mu

        return np.column_stack([self.t, final_series])
