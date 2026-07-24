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
from numba import get_num_threads, njit, prange
from numbers import Integral
from typing import List, Optional, Tuple
from .structures import MeltPool, PathVector


_FLOAT32_EPSILON = np.finfo(np.float32).eps
# Baseline absolute-error allowance for the DirectXMath degree-10 cosine
# polynomial. Range-reduction and accumulation allowances are added below.
_COSINE_POLYNOMIAL_ERROR = 3.0e-7
DEFAULT_SPECTRAL_ERROR_FRACTION = 0.25
DEFAULT_MAX_SPECTRAL_TABLE_BYTES = 256 * 1024**2
DEFAULT_TARGET_TILE_WIDTH = 80.0e-6
_MINIMUM_AUTOMATIC_TILE_SIZE = 16


def _validate_spectral_controls(
    spectral_error_fraction: float,
    max_spectral_table_bytes: Optional[int],
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
    if max_spectral_table_bytes is not None and (
        isinstance(max_spectral_table_bytes, (bool, np.bool_))
        or not isinstance(max_spectral_table_bytes, Integral)
        or max_spectral_table_bytes < 1
    ):
        raise ValueError(
            "max_spectral_table_bytes must be a positive integer or None."
        )


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


def build_spectral_history_table(
    resolution: float,
    melt_pool: MeltPool,
    path_vectors: List[PathVector],
    *,
    spectral_error_fraction: float = DEFAULT_SPECTRAL_ERROR_FRACTION,
    max_spectral_table_bytes: Optional[int] = DEFAULT_MAX_SPECTRAL_TABLE_BYTES,
):
    """Build independent packed width, depth, and height histories."""
    _validate_spectral_controls(
        spectral_error_fraction,
        max_spectral_table_bytes,
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
    table_bytes = total_points * 3 * np.dtype(np.float32).itemsize
    if (
        max_spectral_table_bytes is not None
        and table_bytes > max_spectral_table_bytes
    ):
        raise MemoryError(
            "The optimized spectral table would require "
            f"{table_bytes / 1024**2:.2f} MiB; the limit is "
            f"{max_spectral_table_bytes / 1024**2:.2f} MiB."
        )

    spectral_table_offsets = np.zeros(len(path_vectors) + 1, dtype=np.int64)
    spectral_table_offsets[1:] = np.cumsum(point_counts)
    spectral_table = build_spectral_tables(
        durations,
        spectral_table_offsets,
        width_phases64.astype(np.float32),
        depth_phases64.astype(np.float32),
        height_phases64.astype(np.float32),
        width_amplitudes64.astype(np.float32),
        width_frequencies64.astype(np.float32),
        depth_amplitudes64.astype(np.float32),
        depth_frequencies64.astype(np.float32),
        height_amplitudes64.astype(np.float32),
        height_frequencies64.astype(np.float32),
    )
    if not np.isfinite(spectral_table).all():
        raise ValueError(
            "The evaluated melt-pool dimension histories contain "
            "non-finite values."
        )
    invalid_dimensions = np.min(spectral_table, axis=0) <= 0.0
    if np.any(invalid_dimensions):
        names = np.array(("width", "depth", "height"))
        raise ValueError(
            "The evaluated melt-pool histories must remain positive; "
            "invalid dimensions: " + ", ".join(names[invalid_dimensions]) + "."
        )

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
    diagnostics = {
        "spectral_table_points": int(spectral_table.shape[0]),
        "spectral_table_bytes": int(table_bytes),
        "spectral_error_bound": float(table_error_bound),
    }
    return spectral_table_offsets, spectral_table, diagnostics


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
    max_spectral_table_bytes: Optional[int] = DEFAULT_MAX_SPECTRAL_TABLE_BYTES,
    diagnostics: Optional[dict] = None,
    report: bool = False,
) -> np.ndarray:
    """Compute a regular RVE with the optimized horizontal table kernel."""
    _validate_spectral_controls(
        spectral_error_fraction,
        max_spectral_table_bytes,
    )

    origin = np.asarray(origin, dtype=np.float64)
    voxel_shape_array = np.asarray(voxel_shape, dtype=np.int64)
    n_voxels = (
        int(voxel_shape_array[0])
        * int(voxel_shape_array[1])
        * int(voxel_shape_array[2])
    )
    melt_mask = np.zeros(n_voxels, dtype=np.int8)
    if not path_vectors:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "spectral_table_points": 0,
                    "spectral_table_bytes": 0,
                    "spectral_error_bound": 0.0,
                }
            )
        if report:
            print(" -> Spectral table: no active path vectors.")
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

    lower_z = (
        np.floor((AABB[:, 4] - origin[2]) / resolution).astype(np.int64) - 1
    )
    upper_z = (
        np.ceil((AABB[:, 5] - origin[2]) / resolution).astype(np.int64) + 1
    )
    lower_z = np.maximum(lower_z, 0)
    upper_z = np.minimum(upper_z, voxel_shape_array[2] - 1)

    (
        spectral_table_offsets,
        spectral_table,
        table_diagnostics,
    ) = build_spectral_history_table(
        resolution,
        melt_pool,
        path_vectors,
        spectral_error_fraction=spectral_error_fraction,
        max_spectral_table_bytes=max_spectral_table_bytes,
    )
    if diagnostics is not None:
        diagnostics.update(table_diagnostics)

    if report:
        print(
            " -> Spectral table: "
            f"{spectral_table.shape[0]} samples, "
            f"{table_diagnostics['spectral_table_bytes'] / 1024**2:.2f} MiB, "
            "error bound="
            f"{table_diagnostics['spectral_error_bound'] * 1.0e6:.3f} µm."
        )
        print(
            " -> Kernel: horizontal implicit/table; "
            f"Numba threads={get_num_threads()}."
        )

    return compute_melt_mask_kernel(
        resolution,
        melt_mask,
        origin,
        voxel_shape_array,
        tile_size,
        candidate_offsets,
        candidate_indices,
        start_points,
        e0,
        e1,
        L0_sqr,
        L1_sqr,
        AABB,
        lower_z,
        upper_z,
        spectral_table_offsets,
        spectral_table,
        centroids,
        distances,
        inv_distance_sqr,
        height_shape_factor,
        depth_shape_factor,
    )


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
