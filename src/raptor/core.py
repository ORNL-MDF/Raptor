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
import os
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numba import get_num_threads, njit, prange

from .structures import MeltPool, PathVector


_FLOAT32_EPSILON = np.finfo(np.float32).eps
# Baseline absolute-error allowance for the DirectXMath degree-10 cosine
# polynomial. Range-reduction and accumulation allowances are added below.
_COSINE_POLYNOMIAL_ERROR = 3.0e-7
DEFAULT_SPECTRAL_ERROR_FRACTION = 0.25
DEFAULT_TARGET_TILE_WIDTH = 80.0e-6
_MINIMUM_AUTOMATIC_TILE_SIZE = 16
_MEBIBYTE = 1024**2
_AUTO_MEMORY_FRACTION = 0.80
_MEMORY_LIMIT_ENVIRONMENT = "RAPTOR_MEMORY_LIMIT_MB"


def _validate_spectral_controls(
    spectral_error_fraction: float,
    memory_limit_mb: Optional[int],
) -> None:
    if (
        not np.isfinite(spectral_error_fraction)
        or spectral_error_fraction <= 0.0
        or spectral_error_fraction > 1.0
    ):
        raise ValueError(
            "spectral_error_fraction must be finite and in the interval "
            "(0, 1]."
        )
    if memory_limit_mb is not None and (
        isinstance(memory_limit_mb, (bool, np.bool_))
        or not isinstance(memory_limit_mb, Integral)
        or memory_limit_mb < 1
    ):
        raise ValueError("memory_limit_mb must be a positive integer or None.")


def _read_cgroup_remaining_bytes() -> Optional[int]:
    """Return the remaining Linux control-group memory, when constrained."""
    candidates = (
        (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory.current"),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    remaining = []
    for limit_path, usage_path in candidates:
        try:
            limit_text = limit_path.read_text().strip()
            if limit_text == "max":
                continue
            limit_bytes = int(limit_text)
            usage_bytes = int(usage_path.read_text().strip())
        except (OSError, ValueError):
            continue
        if limit_bytes >= 1 << 60:
            continue
        remaining.append(max(0, limit_bytes - usage_bytes))
    return min(remaining) if remaining else None


def _available_memory_bytes() -> int:
    """Return memory currently available to this process."""
    import psutil

    available_bytes = int(psutil.virtual_memory().available)
    cgroup_remaining = _read_cgroup_remaining_bytes()
    if cgroup_remaining is not None:
        available_bytes = min(available_bytes, cgroup_remaining)
    return available_bytes


def _parse_environment_memory_limit() -> Optional[int]:
    value = os.getenv(_MEMORY_LIMIT_ENVIRONMENT)
    if value is None:
        return None
    try:
        limit_mb = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_MEMORY_LIMIT_ENVIRONMENT} must be a positive integer."
        ) from error
    if limit_mb < 1:
        raise ValueError(
            f"{_MEMORY_LIMIT_ENVIRONMENT} must be a positive integer."
        )
    return limit_mb


def _resolve_memory_budget(memory_limit_mb: Optional[int]) -> Tuple[int, str]:
    """Resolve API, environment, or automatic memory into bytes."""
    _validate_spectral_controls(
        DEFAULT_SPECTRAL_ERROR_FRACTION,
        memory_limit_mb,
    )
    source = "api"
    resolved_mb = memory_limit_mb
    if resolved_mb is None:
        resolved_mb = _parse_environment_memory_limit()
        source = "environment"
    if resolved_mb is not None:
        budget_bytes = int(resolved_mb) * _MEBIBYTE
        cgroup_remaining = _read_cgroup_remaining_bytes()
        if cgroup_remaining is not None:
            budget_bytes = min(budget_bytes, cgroup_remaining)
            if budget_bytes < int(resolved_mb) * _MEBIBYTE:
                source += "+cgroup"
        return budget_bytes, source

    available_bytes = _available_memory_bytes()
    return int(available_bytes * _AUTO_MEMORY_FRACTION), "automatic"


@dataclass
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


@dataclass
class MemoryExecutionPlan:
    """Resolved core-array budget and contiguous spectral vector batches."""

    budget_bytes: int
    budget_source: str
    fixed_bytes: int
    full_table_bytes: int
    peak_bytes: int
    mode: str
    batches: List[Tuple[int, int]]


@njit(cache=True, inline="always", fastmath=True)
def classify_horizontal_melt_and_boundary(
    y_sqr: float,
    y_term: float,
    a: float,
    z: float,
    height: float,
    depth: float,
    height_shape_factor: float,
    depth_shape_factor: float,
    resolution: float,
):
    """Classify with fixed-y terms hoisted out of a horizontal z column."""
    if z >= 0.0:
        b = height
        n = height_shape_factor
    else:
        b = depth
        n = depth_shape_factor
    radius_sqr = y_sqr + z * z
    epsilon = 1.0e-12
    if radius_sqr < resolution * resolution:
        center_distance = -min(a, height, depth)
        return (
            center_distance < epsilon,
            abs(center_distance) - resolution <= epsilon,
        )
    radius = np.sqrt(radius_sqr)
    inverse_radius = 1.0 / radius
    z_scaled = abs(z) / b
    if n == 1.0:
        z_term = z_scaled
    elif n == 2.0:
        z_term = z_scaled * z_scaled
    else:
        z_term = z_scaled**n
    melt_scale = max(0.0, 1.0 - epsilon * inverse_radius)
    boundary_delta = (resolution + epsilon) * inverse_radius
    lower_scale = max(0.0, 1.0 - boundary_delta)
    upper_scale = 1.0 + boundary_delta
    if n == 1.0:
        melt_value = y_term * melt_scale * melt_scale + z_term * melt_scale
        lower_value = y_term * lower_scale * lower_scale + z_term * lower_scale
        upper_value = y_term * upper_scale * upper_scale + z_term * upper_scale
    elif n == 2.0:
        base_value = y_term + z_term
        melt_value = base_value * melt_scale * melt_scale
        lower_value = base_value * lower_scale * lower_scale
        upper_value = base_value * upper_scale * upper_scale
    else:
        melt_value = y_term * melt_scale * melt_scale + z_term * melt_scale**n
        lower_value = (
            y_term * lower_scale * lower_scale + z_term * lower_scale**n
        )
        upper_value = (
            y_term * upper_scale * upper_scale + z_term * upper_scale**n
        )
    return melt_value < 1.0, lower_value <= 1.0 and upper_value >= 1.0


@njit(cache=True, inline="always", fastmath=True)
def conservative_vertical_interval(
    y: float,
    width: float,
    height: float,
    depth: float,
    height_shape_factor: float,
    depth_shape_factor: float,
    resolution: float,
):
    """Bound all melt/radial-boundary points using a one-voxel inflation."""
    if (
        width <= 0.0
        or height <= 0.0
        or depth <= 0.0
        or height_shape_factor <= 0.0
        or depth_shape_factor <= 0.0
    ):
        return 0, 0.0, 0.0
    a = width / 2.0
    inflated_y = max(0.0, abs(y) - resolution - 1.0e-12)
    if inflated_y > a:
        return 2, 0.0, 0.0
    remaining = max(0.0, 1.0 - (inflated_y / a) ** 2)
    if height_shape_factor == 1.0:
        height_scale = remaining
    elif height_shape_factor == 2.0:
        height_scale = np.sqrt(remaining)
    else:
        height_scale = remaining ** (1.0 / height_shape_factor)
    if depth_shape_factor == 1.0:
        depth_scale = remaining
    elif depth_shape_factor == 2.0:
        depth_scale = np.sqrt(remaining)
    else:
        depth_scale = remaining ** (1.0 / depth_shape_factor)
    inflation = resolution + 1.0e-12
    return (
        1,
        -depth * depth_scale - inflation,
        height * height_scale + inflation,
    )


def _prepare_spectral_history_plan(
    resolution: float,
    melt_pool: MeltPool,
    path_vectors: List[PathVector],
    *,
    spectral_error_fraction: float = DEFAULT_SPECTRAL_ERROR_FRACTION,
) -> SpectralHistoryPlan:
    """Plan globally identical resident or streamed spectral histories."""
    _validate_spectral_controls(
        spectral_error_fraction,
        None,
    )
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
    start_times = np.array([p.start_time for p in path_vectors])
    durations = np.array([p.duration for p in path_vectors])

    width_phases64 = np.remainder(
        np.array([p.width_phases for p in path_vectors])
        + 2.0 * np.pi * start_times[:, None] * width_frequencies64[None, :],
        2.0 * np.pi,
    )
    depth_phases64 = np.remainder(
        np.array([p.depth_phases for p in path_vectors])
        + 2.0 * np.pi * start_times[:, None] * depth_frequencies64[None, :],
        2.0 * np.pi,
    )
    height_phases64 = np.remainder(
        np.array([p.height_phases for p in path_vectors])
        + 2.0 * np.pi * start_times[:, None] * height_frequencies64[None, :],
        2.0 * np.pi,
    )
    max_duration = np.max(np.abs(durations), initial=0.0)
    dimension_spectral_errors = (
        _spectral_approximation_error_bound(
            width_amplitudes64, width_frequencies64, max_duration
        ),
        _spectral_approximation_error_bound(
            depth_amplitudes64, depth_frequencies64, max_duration
        ),
        _spectral_approximation_error_bound(
            height_amplitudes64, height_frequencies64, max_duration
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
            width_amplitudes64, dtype=np.float32
        ),
        width_frequencies=np.ascontiguousarray(
            width_frequencies64, dtype=np.float32
        ),
        depth_amplitudes=np.ascontiguousarray(
            depth_amplitudes64, dtype=np.float32
        ),
        depth_frequencies=np.ascontiguousarray(
            depth_frequencies64, dtype=np.float32
        ),
        height_amplitudes=np.ascontiguousarray(
            height_amplitudes64, dtype=np.float32
        ),
        height_frequencies=np.ascontiguousarray(
            height_frequencies64, dtype=np.float32
        ),
        table_error_bound=float(table_error_bound),
    )


def _build_spectral_history_batch(
    plan: SpectralHistoryPlan,
    start: int,
    stop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build and validate one contiguous vector range from a global plan."""
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


def build_spectral_history_table(
    resolution: float,
    melt_pool: MeltPool,
    path_vectors: List[PathVector],
    *,
    spectral_error_fraction: float = DEFAULT_SPECTRAL_ERROR_FRACTION,
):
    """Build a complete table for accuracy analysis and diagnostics."""
    plan = _prepare_spectral_history_plan(
        resolution,
        melt_pool,
        path_vectors,
        spectral_error_fraction=spectral_error_fraction,
    )
    spectral_table_offsets, spectral_table = _build_spectral_history_batch(
        plan,
        0,
        len(path_vectors),
    )
    diagnostics = {
        "spectral_table_points": int(spectral_table.shape[0]),
        "spectral_table_bytes": int(plan.table_bytes),
        "spectral_error_bound": plan.table_error_bound,
    }
    return spectral_table_offsets, spectral_table, diagnostics


def _insufficient_memory_error(
    budget_bytes: int,
    minimum_bytes: int,
    budget_source: str,
) -> MemoryError:
    minimum_mb = (minimum_bytes + _MEBIBYTE - 1) // _MEBIBYTE
    return MemoryError(
        "Raptor cannot fit the phase field and required core workspace.\n\n"
        f"Resolved memory budget: {budget_bytes / _MEBIBYTE:.2f} MB "
        f"({budget_source})\n"
        f"Minimum required memory: {minimum_bytes / _MEBIBYTE:.2f} MB\n\n"
        "Increase the per-process memory budget to at least:\n"
        f"  Python API: memory_limit_mb={minimum_mb}\n"
        f"  YAML: memory_limit_mb: {minimum_mb}\n"
        f"  Environment: {_MEMORY_LIMIT_ENVIRONMENT}={minimum_mb}\n\n"
        "Raptor did not reduce spectral accuracy."
    )


def _plan_contiguous_batches(
    point_counts: np.ndarray,
    maximum_table_bytes: int,
) -> List[Tuple[int, int]]:
    """Partition ordered vectors without changing per-vector table density."""
    bytes_per_point = 3 * np.dtype(np.float32).itemsize
    batches = []
    start = 0
    batch_bytes = 0
    for vector_index, point_count in enumerate(point_counts):
        vector_bytes = int(point_count) * bytes_per_point
        if vector_bytes > maximum_table_bytes:
            return []
        if (
            vector_index > start
            and batch_bytes + vector_bytes > maximum_table_bytes
        ):
            batches.append((start, vector_index))
            start = vector_index
            batch_bytes = 0
        batch_bytes += vector_bytes
    if start < point_counts.size:
        batches.append((start, point_counts.size))
    return batches


def _plan_memory_execution(
    budget_bytes: int,
    budget_source: str,
    fixed_bytes: int,
    spectral_plan: SpectralHistoryPlan,
    streaming_workspace_bytes: int,
) -> MemoryExecutionPlan:
    """Choose the resident table or ordered streaming under one budget."""
    offsets_bytes = (spectral_plan.point_counts.size + 1) * np.dtype(
        np.int64
    ).itemsize
    resident_peak = fixed_bytes + spectral_plan.table_bytes + offsets_bytes
    if resident_peak <= budget_bytes:
        return MemoryExecutionPlan(
            budget_bytes,
            budget_source,
            fixed_bytes,
            spectral_plan.table_bytes,
            resident_peak,
            "resident",
            [(0, spectral_plan.point_counts.size)],
        )

    streaming_fixed = fixed_bytes + streaming_workspace_bytes
    maximum_table_bytes = budget_bytes - streaming_fixed - offsets_bytes
    batches = _plan_contiguous_batches(
        spectral_plan.point_counts,
        maximum_table_bytes,
    )
    if not batches:
        largest_vector_bytes = (
            int(np.max(spectral_plan.point_counts, initial=0))
            * 3
            * np.dtype(np.float32).itemsize
        )
        minimum_bytes = streaming_fixed + offsets_bytes + largest_vector_bytes
        raise _insufficient_memory_error(
            budget_bytes,
            minimum_bytes,
            budget_source,
        )

    largest_batch_bytes = max(
        sum(int(count) for count in spectral_plan.point_counts[start:stop])
        * 3
        * np.dtype(np.float32).itemsize
        for start, stop in batches
    )
    return MemoryExecutionPlan(
        budget_bytes,
        budget_source,
        fixed_bytes,
        spectral_plan.table_bytes,
        streaming_fixed + offsets_bytes + largest_batch_bytes,
        "streamed",
        batches,
    )


def _filter_candidate_index_range(
    candidate_offsets: np.ndarray,
    candidate_indices: np.ndarray,
    start: int,
    stop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retain one ordered global-vector range and remap it to local indices."""
    counts = np.empty(candidate_offsets.size - 1, dtype=np.int64)
    for tile_index in range(counts.size):
        tile_start = candidate_offsets[tile_index]
        tile_stop = candidate_offsets[tile_index + 1]
        tile_indices = candidate_indices[tile_start:tile_stop]
        counts[tile_index] = np.count_nonzero(
            (tile_indices >= start) & (tile_indices < stop)
        )

    offsets = np.empty(candidate_offsets.size, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)
    indices = np.empty(offsets[-1], dtype=np.int32)
    for tile_index in range(counts.size):
        tile_start = candidate_offsets[tile_index]
        tile_stop = candidate_offsets[tile_index + 1]
        tile_indices = candidate_indices[tile_start:tile_stop]
        selected = tile_indices[(tile_indices >= start) & (tile_indices < stop)]
        output_start = offsets[tile_index]
        indices[output_start : output_start + selected.size] = selected - start
    return offsets, indices


def compute_melt_mask_grid(
    origin: np.ndarray,
    voxel_shape: Tuple[int, int, int],
    resolution: float,
    melt_pool: MeltPool,
    path_vectors: List[PathVector],
    *,
    tile_size: int,
    candidate_offsets: np.ndarray,
    candidate_indices: np.ndarray,
    spectral_error_fraction: float = DEFAULT_SPECTRAL_ERROR_FRACTION,
    memory_limit_mb: Optional[int] = None,
    diagnostics: Optional[dict] = None,
    report: bool = False,
) -> np.ndarray:
    """Compute a regular RVE with the optimized horizontal table kernel."""
    _validate_spectral_controls(
        spectral_error_fraction,
        memory_limit_mb,
    )
    budget_bytes, budget_source = _resolve_memory_budget(memory_limit_mb)

    origin = np.asarray(origin, dtype=np.float64)
    voxel_shape_array = np.asarray(voxel_shape, dtype=np.int64)
    n_voxels = (
        int(voxel_shape_array[0])
        * int(voxel_shape_array[1])
        * int(voxel_shape_array[2])
    )
    if not path_vectors:
        phase_field_bytes = n_voxels * np.dtype(np.int8).itemsize
        if phase_field_bytes > budget_bytes:
            raise _insufficient_memory_error(
                budget_bytes,
                phase_field_bytes,
                budget_source,
            )
        memory_diagnostics = {
            "memory_budget_bytes": int(budget_bytes),
            "memory_budget_source": budget_source,
            "estimated_fixed_memory_bytes": int(phase_field_bytes),
            "estimated_peak_memory_bytes": int(phase_field_bytes),
            "execution_mode": "resident",
            "spectral_table_batches": 0,
        }
        melt_mask = np.zeros(n_voxels, dtype=np.int8)
        if diagnostics is not None:
            diagnostics.update(
                {
                    "spectral_table_points": 0,
                    "spectral_table_bytes": 0,
                    "spectral_error_bound": 0.0,
                    **memory_diagnostics,
                }
            )
        if report:
            print(" -> Spectral table: no active path vectors.")
            print(
                " -> Memory plan: budget="
                f"{budget_bytes / _MEBIBYTE:.2f} MB "
                f"({budget_source}), estimated peak="
                f"{phase_field_bytes / _MEBIBYTE:.2f} MB, mode=resident."
            )
            print(
                " -> Kernel: horizontal implicit/table; "
                f"Numba threads={get_num_threads()}."
            )
        return melt_mask

    height_shape_factor = melt_pool.height_shape_factor
    depth_shape_factor = melt_pool.depth_shape_factor

    start_points = np.array([p.start_point for p in path_vectors])
    distances = np.array([p.distance for p in path_vectors])
    inv_distance_sqr = np.array([p.inv_distance_sqr for p in path_vectors])
    e0 = np.array([p.e0 for p in path_vectors])
    e1 = np.array([p.e1 for p in path_vectors])
    e2 = np.array([p.e2 for p in path_vectors])
    L0_sqr = np.array([p.L0_sqr for p in path_vectors])
    L1_sqr = np.array([p.L1_sqr for p in path_vectors])
    AABB = np.array([p.AABB for p in path_vectors])
    centroids = np.array([p.centroid for p in path_vectors])

    horizontal_paths = (
        np.all(distances[:, 2] == 0.0)
        and np.all(e2[:, 0] == 0.0)
        and np.all(e2[:, 1] == 0.0)
        and np.all(e2[:, 2] == 1.0)
    )
    if not horizontal_paths:
        raise ValueError(
            "The optimized Raptor kernel requires horizontal path vectors "
            "whose local z axes match the grid z axis."
        )
    del e2

    lower_z = (
        np.floor((AABB[:, 4] - origin[2]) / resolution).astype(np.int64) - 1
    )
    upper_z = (
        np.ceil((AABB[:, 5] - origin[2]) / resolution).astype(np.int64) + 1
    )
    lower_z = np.maximum(lower_z, 0)
    upper_z = np.minimum(upper_z, voxel_shape_array[2] - 1)

    spectral_plan = _prepare_spectral_history_plan(
        resolution,
        melt_pool,
        path_vectors,
        spectral_error_fraction=spectral_error_fraction,
    )
    phase_field_bytes = n_voxels * np.dtype(np.int8).itemsize
    fixed_arrays = (
        origin,
        voxel_shape_array,
        candidate_offsets,
        candidate_indices,
        start_points,
        distances,
        inv_distance_sqr,
        e0,
        e1,
        L0_sqr,
        L1_sqr,
        AABB,
        centroids,
        lower_z,
        upper_z,
    )
    fixed_bytes = (
        phase_field_bytes
        + spectral_plan.fixed_bytes
        + sum(array.nbytes for array in fixed_arrays)
    )
    streaming_workspace_bytes = (
        2 * candidate_offsets.nbytes + candidate_indices.nbytes
    )
    memory_plan = _plan_memory_execution(
        budget_bytes,
        budget_source,
        fixed_bytes,
        spectral_plan,
        streaming_workspace_bytes,
    )
    table_diagnostics = {
        "spectral_table_points": int(spectral_plan.total_points),
        "spectral_table_bytes": int(spectral_plan.table_bytes),
        "spectral_error_bound": spectral_plan.table_error_bound,
        "memory_budget_bytes": int(memory_plan.budget_bytes),
        "memory_budget_source": memory_plan.budget_source,
        "estimated_fixed_memory_bytes": int(memory_plan.fixed_bytes),
        "estimated_peak_memory_bytes": int(memory_plan.peak_bytes),
        "execution_mode": memory_plan.mode,
        "spectral_table_batches": len(memory_plan.batches),
    }
    if diagnostics is not None:
        diagnostics.update(table_diagnostics)

    if report:
        print(
            " -> Spectral table plan: "
            f"{spectral_plan.total_points} samples, "
            f"{spectral_plan.table_bytes / _MEBIBYTE:.2f} MB full table, "
            "error bound="
            f"{spectral_plan.table_error_bound * 1.0e6:.3f} µm."
        )
        print(
            " -> Memory plan: budget="
            f"{memory_plan.budget_bytes / _MEBIBYTE:.2f} MB "
            f"({memory_plan.budget_source}), fixed="
            f"{memory_plan.fixed_bytes / _MEBIBYTE:.2f} MB, estimated peak="
            f"{memory_plan.peak_bytes / _MEBIBYTE:.2f} MB, "
            f"mode={memory_plan.mode}, batches={len(memory_plan.batches)}."
        )
        print(
            " -> Kernel: horizontal implicit/table; "
            f"Numba threads={get_num_threads()}."
        )

    melt_mask = np.zeros(n_voxels, dtype=np.int8)
    for start, stop in memory_plan.batches:
        spectral_table_offsets, spectral_table = _build_spectral_history_batch(
            spectral_plan,
            start,
            stop,
        )
        if memory_plan.mode == "resident":
            batch_candidate_offsets = candidate_offsets
            batch_candidate_indices = candidate_indices
        else:
            (
                batch_candidate_offsets,
                batch_candidate_indices,
            ) = _filter_candidate_index_range(
                candidate_offsets,
                candidate_indices,
                start,
                stop,
            )
        compute_melt_mask_kernel(
            resolution,
            melt_mask,
            origin,
            voxel_shape_array,
            tile_size,
            batch_candidate_offsets,
            batch_candidate_indices,
            start_points[start:stop],
            e0[start:stop],
            e1[start:stop],
            L0_sqr[start:stop],
            L1_sqr[start:stop],
            AABB[start:stop],
            lower_z[start:stop],
            upper_z[start:stop],
            spectral_table_offsets,
            spectral_table,
            centroids[start:stop],
            distances[start:stop],
            inv_distance_sqr[start:stop],
            height_shape_factor,
            depth_shape_factor,
        )
    return melt_mask


def _spectral_approximation_error_bound(
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
    max_duration: float,
) -> float:
    """Conservatively bound float32 polynomial spectral error."""
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


@njit(cache=True, parallel=True)
def collect_zero_indices(values: np.ndarray, zero_count: int) -> np.ndarray:
    """Collect flat defect indices without a full-size temporary mask."""
    if values.size < 1_000_000:
        indices = np.empty(zero_count, dtype=np.int64)
        cursor = 0
        for index in range(values.size):
            if values[index] == 0:
                indices[cursor] = index
                cursor += 1
        return indices

    target_chunk_size = 1 << 20
    n_chunks = min(
        256,
        max(
            1,
            (values.size + target_chunk_size - 1) // target_chunk_size,
        ),
    )
    chunk_size = (values.size + n_chunks - 1) // n_chunks
    chunk_counts = np.zeros(n_chunks, dtype=np.int64)
    for chunk in prange(n_chunks):
        start = chunk * chunk_size
        stop = min(values.size, start + chunk_size)
        count = 0
        for index in range(start, stop):
            count += values[index] == 0
        chunk_counts[chunk] = count

    chunk_offsets = np.empty(n_chunks + 1, dtype=np.int64)
    chunk_offsets[0] = 0
    for chunk in range(n_chunks):
        chunk_offsets[chunk + 1] = chunk_offsets[chunk] + chunk_counts[chunk]

    indices = np.empty(zero_count, dtype=np.int64)
    for chunk in prange(n_chunks):
        start = chunk * chunk_size
        stop = min(values.size, start + chunk_size)
        cursor = chunk_offsets[chunk]
        for index in range(start, stop):
            if values[index] == 0:
                indices[cursor] = index
                cursor += 1
    return indices


@njit(cache=True, parallel=True)
def count_phase_codes(values: np.ndarray) -> np.ndarray:
    """Count Raptor's four phase codes in one parallel array pass."""
    phase_0 = 0
    phase_1 = 0
    phase_2 = 0
    phase_3 = 0
    for index in prange(values.size):
        value = values[index]
        phase_0 += value == 0
        phase_1 += value == 1
        phase_2 += value == 2
        phase_3 += value == 3
    return np.array((phase_0, phase_1, phase_2, phase_3), dtype=np.int64)


@njit(cache=True, inline="always")
def _find_component_root(parent: np.ndarray, node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(cache=True)
def label_sparse_defects(
    flat_indices: np.ndarray,
    shape: Tuple[int, int, int],
    min_size: int = 2,
):
    """Label sparse defect coordinates with 26-neighbor connectivity."""
    n_points = flat_indices.size
    parent = np.arange(n_points, dtype=np.int32)
    component_size = np.ones(n_points, dtype=np.int32)
    ny = shape[1]
    nz = shape[2]
    yz_size = ny * nz

    for point_index in range(n_points):
        flat_index = flat_indices[point_index]
        x = flat_index // yz_size
        remainder = flat_index - x * yz_size
        y = remainder // nz
        z = remainder - y * nz

        for dx in range(-1, 1):
            neighbor_x = x + dx
            if neighbor_x < 0:
                continue
            for dy in range(-1, 2):
                neighbor_y = y + dy
                if neighbor_y < 0 or neighbor_y >= ny:
                    continue
                for dz in range(-1, 2):
                    if dx == 0 and (dy > 0 or (dy == 0 and dz >= 0)):
                        continue
                    neighbor_z = z + dz
                    if neighbor_z < 0 or neighbor_z >= nz:
                        continue
                    neighbor_flat = (
                        neighbor_x * yz_size + neighbor_y * nz + neighbor_z
                    )
                    neighbor_position = np.searchsorted(
                        flat_indices, neighbor_flat
                    )
                    if (
                        neighbor_position >= point_index
                        or flat_indices[neighbor_position] != neighbor_flat
                    ):
                        continue
                    root = _find_component_root(parent, point_index)
                    neighbor_root = _find_component_root(
                        parent, neighbor_position
                    )
                    if root == neighbor_root:
                        continue
                    if component_size[root] < component_size[neighbor_root]:
                        root, neighbor_root = neighbor_root, root
                    parent[neighbor_root] = root
                    component_size[root] += component_size[neighbor_root]

    root_labels = np.full(n_points, -1, dtype=np.int32)
    point_labels = np.full(n_points, -1, dtype=np.int32)
    n_components = 0
    for point_index in range(n_points):
        root = _find_component_root(parent, point_index)
        if component_size[root] < min_size:
            continue
        if root_labels[root] < 0:
            root_labels[root] = n_components
            n_components += 1
        point_labels[point_index] = root_labels[root]
    return point_labels, n_components


def build_spatial_index(
    origin: np.ndarray,
    voxel_shape: Tuple[int, int, int],
    resolution: float,
    path_vectors: List[PathVector],
    tile_size: Optional[int] = None,
):
    """Build an ordered, melt-envelope-aware tile candidate index."""
    shape = np.asarray(voxel_shape, dtype=np.int64)
    grid_min = np.asarray(origin, dtype=np.float64)
    grid_max = grid_min + (shape - 1) * resolution
    if tile_size is None:
        tile_size = max(
            _MINIMUM_AUTOMATIC_TILE_SIZE,
            int(round(DEFAULT_TARGET_TILE_WIDTH / resolution)),
        )
        tile_size = min(tile_size, int(max(shape[0], shape[1])))
    elif tile_size < 1:
        raise ValueError("tile_size must be a positive integer.")

    active_vectors = []
    for vector in path_vectors:
        bounds = vector.AABB
        if (
            bounds[1] < grid_min[0]
            or bounds[0] > grid_max[0]
            or bounds[3] < grid_min[1]
            or bounds[2] > grid_max[1]
            or bounds[5] < grid_min[2]
            or bounds[4] > grid_max[2]
        ):
            continue
        active_vectors.append(vector)

    tile_shape = (shape[:2] + tile_size - 1) // tile_size
    n_tiles = int(tile_shape[0] * tile_shape[1])
    tile_candidates = [[] for _ in range(n_tiles)]

    for vector_index, vector in enumerate(active_vectors):
        bounds = vector.AABB
        lower = np.array([bounds[0], bounds[2]])
        upper = np.array([bounds[1], bounds[3]])
        lower_voxel = np.ceil((lower - grid_min[:2]) / resolution).astype(
            np.int64
        )
        upper_voxel = np.floor((upper - grid_min[:2]) / resolution).astype(
            np.int64
        )
        lower_voxel = np.maximum(lower_voxel, 0)
        upper_voxel = np.minimum(upper_voxel, shape[:2] - 1)
        lower_tile = lower_voxel // tile_size
        upper_tile = upper_voxel // tile_size

        for tx in range(lower_tile[0], upper_tile[0] + 1):
            for ty in range(lower_tile[1], upper_tile[1] + 1):
                tile_index = tx * tile_shape[1] + ty
                tile_candidates[int(tile_index)].append(vector_index)

    candidate_offsets = np.empty(n_tiles + 1, dtype=np.int64)
    candidate_offsets[0] = 0
    for tile_index, candidates in enumerate(tile_candidates):
        candidate_offsets[tile_index + 1] = candidate_offsets[tile_index] + len(
            candidates
        )
    candidate_indices = np.empty(candidate_offsets[-1], dtype=np.int32)
    for tile_index, candidates in enumerate(tile_candidates):
        start = candidate_offsets[tile_index]
        candidate_indices[start : start + len(candidates)] = candidates

    return (
        active_vectors,
        shape,
        tile_size,
        candidate_offsets,
        candidate_indices,
    )


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
    """Evaluate independent spectra with guarded float32 polynomial math."""
    two_pi_t = np.float32(2.0 * np.pi) * np.float32(time)
    width = np.float32(0.0)
    for k in range(width_amplitudes.shape[0]):
        width += width_amplitudes[k] * approximate_cosine(
            two_pi_t * width_frequencies[k] + width_phases[vector_index, k]
        )

    depth = np.float32(0.0)
    for k in range(depth_amplitudes.shape[0]):
        depth += depth_amplitudes[k] * approximate_cosine(
            two_pi_t * depth_frequencies[k] + depth_phases[vector_index, k]
        )

    height = np.float32(0.0)
    for k in range(height_amplitudes.shape[0]):
        height += height_amplitudes[k] * approximate_cosine(
            two_pi_t * height_frequencies[k] + height_phases[vector_index, k]
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
    """Linearly interpolate a packed dimension history."""
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


@njit(cache=True, parallel=True, fastmath=True)
def compute_melt_mask_kernel(
    resolution: float,
    melt_mask: np.ndarray,
    origin: np.ndarray,
    voxel_shape: np.ndarray,
    tile_size: int,
    candidate_offsets: np.ndarray,
    candidate_indices: np.ndarray,
    start_points: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    L0_sqr: np.ndarray,
    L1_sqr: np.ndarray,
    AABB: np.ndarray,
    lower_z_bounds: np.ndarray,
    upper_z_bounds: np.ndarray,
    spectral_table_offsets: np.ndarray,
    spectral_table: np.ndarray,
    centroids: np.ndarray,
    distances: np.ndarray,
    inv_distance_sqr: np.ndarray,
    height_shape_factor: np.float64,
    depth_shape_factor: np.float64,
) -> np.ndarray:
    """Evaluate horizontal paths and their independent spectral histories."""
    n_tile_y = (voxel_shape[1] + tile_size - 1) // tile_size
    n_columns = voxel_shape[0] * voxel_shape[1]
    nz = voxel_shape[2]

    for column_index in prange(n_columns):
        voxel_x = column_index // voxel_shape[1]
        voxel_y = column_index - voxel_x * voxel_shape[1]
        tile_index = (voxel_x // tile_size) * n_tile_y + voxel_y // tile_size
        candidate_start = candidate_offsets[tile_index]
        candidate_end = candidate_offsets[tile_index + 1]
        vx = origin[0] + voxel_x * resolution
        vy = origin[1] + voxel_y * resolution

        for candidate_position in range(candidate_start, candidate_end):
            vector_index = candidate_indices[candidate_position]
            if (
                vx < AABB[vector_index, 0]
                or vx > AABB[vector_index, 1]
                or vy < AABB[vector_index, 2]
                or vy > AABB[vector_index, 3]
            ):
                continue

            vec_cx = vx - centroids[vector_index, 0]
            vec_cy = vy - centroids[vector_index, 1]
            dot_e0_xy = (
                vec_cx * e0[vector_index, 0] + vec_cy * e0[vector_index, 1]
            )
            if dot_e0_xy * dot_e0_xy > L0_sqr[vector_index]:
                continue
            dot_e1_xy = (
                vec_cx * e1[vector_index, 0] + vec_cy * e1[vector_index, 1]
            )
            if dot_e1_xy * dot_e1_xy > L1_sqr[vector_index]:
                continue

            time_fraction = 0.0
            if inv_distance_sqr[vector_index] > 0.0:
                vec_sx = vx - start_points[vector_index, 0]
                vec_sy = vy - start_points[vector_index, 1]
                dot_dist = (
                    vec_sx * distances[vector_index, 0]
                    + vec_sy * distances[vector_index, 1]
                )
                time_fraction = dot_dist * inv_distance_sqr[vector_index]
            time_fraction = max(0.0, min(1.0, time_fraction))
            width, depth, height = evaluate_spectral_table(
                time_fraction,
                vector_index,
                spectral_table_offsets,
                spectral_table,
            )

            path_x = (
                start_points[vector_index, 0]
                + time_fraction * distances[vector_index, 0]
            )
            path_y = (
                start_points[vector_index, 1]
                + time_fraction * distances[vector_index, 1]
            )
            vec_path_x = vx - path_x
            vec_path_y = vy - path_y
            local_y = (
                vec_path_x * e0[vector_index, 0]
                + vec_path_y * e0[vector_index, 1]
            )
            local_y_sqr = local_y * local_y
            half_width = width / 2.0
            y_scaled = local_y / half_width
            y_term = y_scaled * y_scaled

            (
                interval_status,
                local_z_lower,
                local_z_upper,
            ) = conservative_vertical_interval(
                local_y,
                width,
                height,
                depth,
                height_shape_factor,
                depth_shape_factor,
                resolution,
            )
            if interval_status == 2:
                continue

            lower_z = lower_z_bounds[vector_index]
            upper_z = upper_z_bounds[vector_index]
            if interval_status == 1:
                path_z = start_points[vector_index, 2]
                dynamic_lower_z = (
                    int(
                        np.floor(
                            (path_z + local_z_lower - origin[2]) / resolution
                        )
                    )
                    - 1
                )
                dynamic_upper_z = (
                    int(
                        np.ceil(
                            (path_z + local_z_upper - origin[2]) / resolution
                        )
                    )
                    + 1
                )
                lower_z = max(lower_z, dynamic_lower_z)
                upper_z = min(upper_z, dynamic_upper_z)
                if lower_z > upper_z:
                    continue

            path_z = start_points[vector_index, 2]
            for voxel_z in range(lower_z, upper_z + 1):
                flat_index = column_index * nz + voxel_z
                vz = origin[2] + voxel_z * resolution
                if vz < AABB[vector_index, 4] or vz > AABB[vector_index, 5]:
                    continue
                local_z = vz - path_z
                (
                    is_voxel_melted,
                    is_voxel_boundary,
                ) = classify_horizontal_melt_and_boundary(
                    local_y_sqr,
                    y_term,
                    half_width,
                    local_z,
                    height,
                    depth,
                    height_shape_factor,
                    depth_shape_factor,
                    resolution,
                )
                previous_phase = melt_mask[flat_index]
                if is_voxel_melted:
                    melt_mask[flat_index] = 1
                if is_voxel_boundary:
                    melt_mask[flat_index] = 2
                if previous_phase > 1 and is_voxel_boundary:
                    melt_mask[flat_index] = 3

    return melt_mask
