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
"""Accuracy-controlled spectral evaluation and packed history tables."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numba import njit, prange

from .structures import MeltPool, PathVector


_FLOAT32_EPSILON = np.finfo(np.float32).eps
# Baseline absolute-error allowance for the DirectXMath degree-10 cosine
# polynomial. Planning adds range-reduction and accumulation allowances, then
# rejects spectra that cannot satisfy the requested fraction of a voxel.
_COSINE_POLYNOMIAL_ERROR = 3.0e-7
DEFAULT_SPECTRAL_ERROR_FRACTION = 0.25


def validate_spectral_error_fraction(spectral_error_fraction: float) -> None:
    """Validate the fraction of one voxel available to spectral error."""
    if (
        not np.isfinite(spectral_error_fraction)
        or spectral_error_fraction <= 0.0
        or spectral_error_fraction > 1.0
    ):
        raise ValueError(
            "spectral_error_fraction must be finite and in the interval "
            "(0, 1]."
        )


@dataclass(frozen=True)
class SpectralHistoryPlan:
    """Accuracy-derived arrays shared by resident and streamed tables."""

    durations: np.ndarray
    point_counts: np.ndarray
    width_phases: np.ndarray
    depth_phases: np.ndarray
    height_phases: np.ndarray
    width_amplitudes: np.ndarray
    width_frequencies: np.ndarray
    depth_amplitudes: np.ndarray
    depth_frequencies: np.ndarray
    height_amplitudes: np.ndarray
    height_frequencies: np.ndarray
    table_error_bound: float

    @property
    def total_points(self) -> int:
        return sum(int(count) for count in self.point_counts)

    @property
    def table_bytes(self) -> int:
        return self.total_points * 3 * np.dtype(np.float32).itemsize

    @property
    def fixed_bytes(self) -> int:
        arrays = (
            self.durations,
            self.point_counts,
            self.width_phases,
            self.depth_phases,
            self.height_phases,
            self.width_amplitudes,
            self.width_frequencies,
            self.depth_amplitudes,
            self.depth_frequencies,
            self.height_amplitudes,
            self.height_frequencies,
        )
        return sum(array.nbytes for array in arrays)


def spectral_approximation_error_bound(
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
    max_duration: float,
) -> float:
    """Conservatively bound float32 range reduction and polynomial error."""
    if amplitudes.size == 0:
        return 0.0
    max_angle = (
        2.0 * np.pi * max_duration * np.max(np.abs(frequencies)) + 2.0 * np.pi
    )
    cosine_error = _COSINE_POLYNOMIAL_ERROR + (
        16.0 * _FLOAT32_EPSILON * (max_angle + 1.0)
    )
    accumulation_error = 2.0 * amplitudes.size * _FLOAT32_EPSILON
    relative_error = min(2.0, cosine_error + accumulation_error)
    return float(np.sum(np.abs(amplitudes)) * relative_error)


@njit(cache=True, inline="always", fastmath=True)
def approximate_cosine(value: np.float32) -> np.float32:
    """Evaluate cosine with a range-reduced float32 minimax polynomial."""
    pi = np.float32(np.pi)
    half_pi = np.float32(np.pi / 2.0)
    two_pi = np.float32(2.0 * np.pi)
    reduced = value - np.floor(value / two_pi + np.float32(0.5)) * two_pi
    sign = np.float32(1.0)
    if reduced > half_pi:
        reduced = pi - reduced
        sign = np.float32(-1.0)
    elif reduced < -half_pi:
        reduced = -pi - reduced
        sign = np.float32(-1.0)

    squared = reduced * reduced
    polynomial = np.float32(-2.6051615e-7)
    polynomial = np.float32(2.4760495e-5) + squared * polynomial
    polynomial = np.float32(-1.3888378e-3) + squared * polynomial
    polynomial = np.float32(4.1666638e-2) + squared * polynomial
    polynomial = np.float32(-4.9999997e-1) + squared * polynomial
    polynomial = np.float32(1.0) + squared * polynomial
    return sign * polynomial


@njit(cache=True, inline="always", fastmath=True)
def evaluate_spectra_fast(
    time: float,
    vector_index: int,
    width_phases: np.ndarray,
    depth_phases: np.ndarray,
    height_phases: np.ndarray,
    width_amplitudes: np.ndarray,
    width_frequencies: np.ndarray,
    depth_amplitudes: np.ndarray,
    depth_frequencies: np.ndarray,
    height_amplitudes: np.ndarray,
    height_frequencies: np.ndarray,
):
    """Evaluate independent width, depth, and height spectra."""
    two_pi_t = np.float32(2.0 * np.pi) * np.float32(time)
    width = np.float32(0.0)
    for mode_index in range(width_amplitudes.shape[0]):
        width += width_amplitudes[mode_index] * approximate_cosine(
            two_pi_t * width_frequencies[mode_index]
            + width_phases[vector_index, mode_index]
        )

    depth = np.float32(0.0)
    for mode_index in range(depth_amplitudes.shape[0]):
        depth += depth_amplitudes[mode_index] * approximate_cosine(
            two_pi_t * depth_frequencies[mode_index]
            + depth_phases[vector_index, mode_index]
        )

    height = np.float32(0.0)
    for mode_index in range(height_amplitudes.shape[0]):
        height += height_amplitudes[mode_index] * approximate_cosine(
            two_pi_t * height_frequencies[mode_index]
            + height_phases[vector_index, mode_index]
        )
    return width, depth, height


@njit(cache=True, parallel=True, fastmath=True)
def build_spectral_tables(
    durations: np.ndarray,
    table_offsets: np.ndarray,
    width_phases: np.ndarray,
    depth_phases: np.ndarray,
    height_phases: np.ndarray,
    width_amplitudes: np.ndarray,
    width_frequencies: np.ndarray,
    depth_amplitudes: np.ndarray,
    depth_frequencies: np.ndarray,
    height_amplitudes: np.ndarray,
    height_frequencies: np.ndarray,
) -> np.ndarray:
    """Precompute packed, independently evaluated dimension histories."""
    table = np.empty((table_offsets[-1], 3), dtype=np.float32)
    for vector_index in prange(durations.size):
        table_start = table_offsets[vector_index]
        point_count = table_offsets[vector_index + 1] - table_start
        inverse_intervals = 1.0 / (point_count - 1)
        for point_index in range(point_count):
            time = durations[vector_index] * point_index * inverse_intervals
            table[table_start + point_index] = evaluate_spectra_fast(
                time,
                vector_index,
                width_phases,
                depth_phases,
                height_phases,
                width_amplitudes,
                width_frequencies,
                depth_amplitudes,
                depth_frequencies,
                height_amplitudes,
                height_frequencies,
            )
    return table


@njit(cache=True, inline="always", fastmath=True)
def evaluate_spectral_table(
    time_fraction: float,
    vector_index: int,
    table_offsets: np.ndarray,
    table: np.ndarray,
):
    """Linearly interpolate one vector's packed dimension history."""
    table_start = table_offsets[vector_index]
    point_count = table_offsets[vector_index + 1] - table_start
    position = time_fraction * (point_count - 1)
    lower = int(position)
    upper = min(lower + 1, point_count - 1)
    fraction = np.float32(position - lower)
    inverse_fraction = np.float32(1.0) - fraction
    width = (
        table[table_start + lower, 0] * inverse_fraction
        + table[table_start + upper, 0] * fraction
    )
    depth = (
        table[table_start + lower, 1] * inverse_fraction
        + table[table_start + upper, 1] * fraction
    )
    height = (
        table[table_start + lower, 2] * inverse_fraction
        + table[table_start + upper, 2] * fraction
    )
    return width, depth, height


def prepare_spectral_history_plan(
    resolution: float,
    melt_pool: MeltPool,
    path_vectors: List[PathVector],
    *,
    spectral_error_fraction: float = DEFAULT_SPECTRAL_ERROR_FRACTION,
) -> SpectralHistoryPlan:
    """Plan globally identical resident or streamed spectral histories.

    The accuracy-derived sample count is fixed for every path vector before
    memory batching is considered. Streaming therefore changes table residency,
    not sampling density or the requested numerical error bound.
    """
    validate_spectral_error_fraction(spectral_error_fraction)
    width_amplitudes64 = np.ascontiguousarray(
        melt_pool.width_oscillations[:, 0]
    )
    width_frequencies64 = np.ascontiguousarray(
        melt_pool.width_oscillations[:, 1]
    )
    depth_amplitudes64 = np.ascontiguousarray(
        melt_pool.depth_oscillations[:, 0]
    )
    depth_frequencies64 = np.ascontiguousarray(
        melt_pool.depth_oscillations[:, 1]
    )
    height_amplitudes64 = np.ascontiguousarray(
        melt_pool.height_oscillations[:, 0]
    )
    height_frequencies64 = np.ascontiguousarray(
        melt_pool.height_oscillations[:, 1]
    )
    start_times = np.array([vector.start_time for vector in path_vectors])
    durations = np.array([vector.duration for vector in path_vectors])

    width_phases64 = np.remainder(
        np.array([vector.width_phases for vector in path_vectors])
        + 2.0 * np.pi * start_times[:, None] * width_frequencies64[None, :],
        2.0 * np.pi,
    )
    depth_phases64 = np.remainder(
        np.array([vector.depth_phases for vector in path_vectors])
        + 2.0 * np.pi * start_times[:, None] * depth_frequencies64[None, :],
        2.0 * np.pi,
    )
    height_phases64 = np.remainder(
        np.array([vector.height_phases for vector in path_vectors])
        + 2.0 * np.pi * start_times[:, None] * height_frequencies64[None, :],
        2.0 * np.pi,
    )
    max_duration = np.max(np.abs(durations), initial=0.0)
    dimension_spectral_errors = (
        spectral_approximation_error_bound(
            width_amplitudes64,
            width_frequencies64,
            max_duration,
        ),
        spectral_approximation_error_bound(
            depth_amplitudes64,
            depth_frequencies64,
            max_duration,
        ),
        spectral_approximation_error_bound(
            height_amplitudes64,
            height_frequencies64,
            max_duration,
        ),
    )
    spectral_error_bound = max(dimension_spectral_errors)
    error_budget = resolution * spectral_error_fraction
    if spectral_error_bound > error_budget:
        raise ValueError(
            "The optimized float32 spectral kernel cannot satisfy the "
            f"{error_budget * 1.0e6:.3f} µm error budget; its conservative "
            f"bound is {spectral_error_bound * 1.0e6:.3f} µm."
        )

    dimension_data = (
        (
            width_amplitudes64,
            width_frequencies64,
            dimension_spectral_errors[0],
        ),
        (
            depth_amplitudes64,
            depth_frequencies64,
            dimension_spectral_errors[1],
        ),
        (
            height_amplitudes64,
            height_frequencies64,
            dimension_spectral_errors[2],
        ),
    )
    max_time_step = np.inf
    for amplitudes, frequencies, approximation_error in dimension_data:
        remaining_error = error_budget - approximation_error
        curvature = np.sum(
            np.abs(amplitudes) * (2.0 * np.pi * frequencies) ** 2
        )
        if remaining_error <= 0.0:
            raise ValueError("No interpolation error budget remains.")
        if curvature > 0.0:
            max_time_step = min(
                max_time_step,
                np.sqrt(8.0 * remaining_error / curvature),
            )

    if np.isfinite(max_time_step):
        interval_counts = np.ceil(np.abs(durations) / max_time_step)
        if (
            not np.isfinite(interval_counts).all()
            or np.max(interval_counts, initial=0.0) >= np.iinfo(np.int64).max
        ):
            raise ValueError("Spectral table interval count is not finite.")
        point_counts = interval_counts.astype(np.int64) + 1
    else:
        point_counts = np.full(len(path_vectors), 2, dtype=np.int64)

    point_counts = np.maximum(point_counts, 2)
    total_points = sum(int(count) for count in point_counts)
    if total_points > np.iinfo(np.int64).max:
        raise MemoryError("The optimized spectral table is too large to index.")

    effective_time_step = np.max(
        np.divide(
            np.abs(durations),
            point_counts - 1,
            out=np.zeros_like(durations),
            where=point_counts > 1,
        ),
        initial=0.0,
    )
    table_error_bound = max(
        approximation_error
        + np.sum(np.abs(amplitudes) * (2.0 * np.pi * frequencies) ** 2)
        * effective_time_step**2
        / 8.0
        for amplitudes, frequencies, approximation_error in dimension_data
    )
    return SpectralHistoryPlan(
        durations=np.ascontiguousarray(durations),
        point_counts=np.ascontiguousarray(point_counts),
        width_phases=np.ascontiguousarray(width_phases64, dtype=np.float32),
        depth_phases=np.ascontiguousarray(depth_phases64, dtype=np.float32),
        height_phases=np.ascontiguousarray(height_phases64, dtype=np.float32),
        width_amplitudes=np.ascontiguousarray(
            width_amplitudes64,
            dtype=np.float32,
        ),
        width_frequencies=np.ascontiguousarray(
            width_frequencies64,
            dtype=np.float32,
        ),
        depth_amplitudes=np.ascontiguousarray(
            depth_amplitudes64,
            dtype=np.float32,
        ),
        depth_frequencies=np.ascontiguousarray(
            depth_frequencies64,
            dtype=np.float32,
        ),
        height_amplitudes=np.ascontiguousarray(
            height_amplitudes64,
            dtype=np.float32,
        ),
        height_frequencies=np.ascontiguousarray(
            height_frequencies64,
            dtype=np.float32,
        ),
        table_error_bound=float(table_error_bound),
    )


def build_spectral_history_batch(
    plan: SpectralHistoryPlan,
    start: int,
    stop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build and validate one contiguous range from the global plan."""
    point_counts = plan.point_counts[start:stop]
    offsets = np.zeros(stop - start + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(point_counts)
    table = build_spectral_tables(
        plan.durations[start:stop],
        offsets,
        plan.width_phases[start:stop],
        plan.depth_phases[start:stop],
        plan.height_phases[start:stop],
        plan.width_amplitudes,
        plan.width_frequencies,
        plan.depth_amplitudes,
        plan.depth_frequencies,
        plan.height_amplitudes,
        plan.height_frequencies,
    )
    if not np.isfinite(table).all():
        raise ValueError(
            "The evaluated melt-pool dimension histories contain "
            "non-finite values."
        )
    invalid_dimensions = np.min(table, axis=0) <= 0.0
    if np.any(invalid_dimensions):
        names = np.array(("width", "depth", "height"))
        raise ValueError(
            "The evaluated melt-pool histories must remain positive; "
            "invalid dimensions: " + ", ".join(names[invalid_dimensions]) + "."
        )
    return offsets, table
