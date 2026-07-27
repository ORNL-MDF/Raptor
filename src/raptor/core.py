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
"""Spatial indexing, melt geometry, and the production voxel kernel."""

from typing import List, Optional, Tuple

import numpy as np
from numba import get_num_threads, njit, prange

from .resources import (
    MEMORY_UNIT_BYTES,
    filter_candidate_index_range,
    insufficient_memory_error,
    plan_memory_execution,
    resolve_memory_budget,
)
from .spectral import (
    DEFAULT_SPECTRAL_ERROR_FRACTION,
    build_spectral_history_batch,
    evaluate_spectral_table,
    prepare_spectral_history_plan,
)
from .structures import MeltPool, PathVector


# Performance-only spatial-index defaults; neither changes melt geometry.
DEFAULT_TARGET_TILE_WIDTH = 80.0e-6
_MINIMUM_AUTOMATIC_TILE_SIZE = 16
_GEOMETRY_EPSILON = 1.0e-12


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
    """Classify melt interior and the one-voxel radial boundary.

    The implicit surface is evaluated at ``r``, ``r - resolution``, and
    ``r + resolution``. This retains continuous radial-distance semantics for
    any positive vertical shape factor without an iterative root solve.
    """
    if z >= 0.0:
        b = height
        n = height_shape_factor
    else:
        b = depth
        n = depth_shape_factor
    radius_sqr = y_sqr + z * z
    epsilon = _GEOMETRY_EPSILON
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
    inflated_y = max(0.0, abs(y) - resolution - _GEOMETRY_EPSILON)
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
    inflation = resolution + _GEOMETRY_EPSILON
    return (
        1,
        -depth * depth_scale - inflation,
        height * height_scale + inflation,
    )


def build_spatial_index(
    origin: np.ndarray,
    voxel_shape: Tuple[int, int, int],
    resolution: float,
    path_vectors: List[PathVector],
    tile_size: Optional[int] = None,
):
    """Build ordered path candidates for rectangular tiles in the x-y plane."""
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
    """Compute a regular RVE with the horizontal packed-table kernel.

    Spectral accuracy is planned globally before the memory policy selects
    either a resident table or ordered vector batches. Both modes use identical
    samples and update the shared phase field in the same vector order.
    """
    budget_bytes, budget_source = resolve_memory_budget(memory_limit_mb)

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
            raise insufficient_memory_error(
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
                f"{budget_bytes / MEMORY_UNIT_BYTES:.2f} MB "
                f"({budget_source}), estimated peak="
                f"{phase_field_bytes / MEMORY_UNIT_BYTES:.2f} MB, "
                "mode=resident."
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

    spectral_plan = prepare_spectral_history_plan(
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
    memory_plan = plan_memory_execution(
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
            f"{spectral_plan.table_bytes / MEMORY_UNIT_BYTES:.2f} MB "
            "full table, "
            "error bound="
            f"{spectral_plan.table_error_bound * 1.0e6:.3f} µm."
        )
        print(
            " -> Memory plan: budget="
            f"{memory_plan.budget_bytes / MEMORY_UNIT_BYTES:.2f} MB "
            f"({memory_plan.budget_source}), fixed="
            f"{memory_plan.fixed_bytes / MEMORY_UNIT_BYTES:.2f} MB, "
            "estimated peak="
            f"{memory_plan.peak_bytes / MEMORY_UNIT_BYTES:.2f} MB, "
            f"mode={memory_plan.mode}, batches={len(memory_plan.batches)}."
        )
        print(
            " -> Kernel: horizontal implicit/table; "
            f"Numba threads={get_num_threads()}."
        )

    melt_mask = np.zeros(n_voxels, dtype=np.int8)
    for start, stop in memory_plan.batches:
        spectral_table_offsets, spectral_table = build_spectral_history_batch(
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
            ) = filter_candidate_index_range(
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
    """Apply ordered horizontal paths to complete voxel z-columns.

    Each path reads independently evaluated width, depth, and height histories.
    Parallelism is across columns, so each phase-field element has one writer.
    """
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
