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
"""Exact, ordered spatial indexing for path-vector candidates."""

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from .structures import PathVector


# These values affect performance only; they do not alter melt geometry.
DEFAULT_TARGET_TILE_WIDTH = 80.0e-6
_MINIMUM_AUTOMATIC_TILE_SIZE = 16
_GEOMETRY_EPSILON = 1.0e-12


@njit(cache=True, inline="always")
def _tile_intersects_path_obb(
    delta_x: float,
    delta_y: float,
    tile_half_x: float,
    tile_half_y: float,
    transverse_half_length: float,
    longitudinal_half_length: float,
    e0_x: float,
    e0_y: float,
    e1_x: float,
    e1_y: float,
) -> bool:
    """Test an axis-aligned tile against a path's oriented rectangle."""
    abs_e0_x = abs(e0_x)
    abs_e0_y = abs(e0_y)
    abs_e1_x = abs(e1_x)
    abs_e1_y = abs(e1_y)
    tolerance = _GEOMETRY_EPSILON

    path_half_x = (
        transverse_half_length * abs_e0_x + longitudinal_half_length * abs_e1_x
    )
    if abs(delta_x) > path_half_x + tile_half_x + tolerance:
        return False

    path_half_y = (
        transverse_half_length * abs_e0_y + longitudinal_half_length * abs_e1_y
    )
    if abs(delta_y) > path_half_y + tile_half_y + tolerance:
        return False

    transverse_distance = delta_x * e0_x + delta_y * e0_y
    transverse_tile_radius = tile_half_x * abs_e0_x + tile_half_y * abs_e0_y
    if (
        abs(transverse_distance)
        > transverse_half_length + transverse_tile_radius + tolerance
    ):
        return False

    longitudinal_distance = delta_x * e1_x + delta_y * e1_y
    longitudinal_tile_radius = tile_half_x * abs_e1_x + tile_half_y * abs_e1_y
    return (
        abs(longitudinal_distance)
        <= longitudinal_half_length + longitudinal_tile_radius + tolerance
    )


@njit(cache=True, inline="always")
def _candidate_tile_y_range(
    delta_x: float,
    centroid_y: float,
    maximum_tile_half_x: float,
    maximum_tile_half_y: float,
    transverse_half_length: float,
    longitudinal_half_length: float,
    e0_x: float,
    e0_y: float,
    e1_x: float,
    e1_y: float,
    first_regular_center_y: float,
    tile_stride: float,
    lower_tile_y: int,
    upper_tile_y: int,
) -> Tuple[int, int]:
    """Conservatively bound intersecting tile rows for one tile column."""
    tolerance = _GEOMETRY_EPSILON
    path_half_x = (
        transverse_half_length * abs(e0_x)
        + longitudinal_half_length * abs(e1_x)
    )
    if abs(delta_x) > path_half_x + maximum_tile_half_x + tolerance:
        return 1, 0

    path_half_y = (
        transverse_half_length * abs(e0_y)
        + longitudinal_half_length * abs(e1_y)
    )
    lower_y = centroid_y - path_half_y - maximum_tile_half_y - tolerance
    upper_y = centroid_y + path_half_y + maximum_tile_half_y + tolerance

    transverse_radius = (
        transverse_half_length
        + maximum_tile_half_x * abs(e0_x)
        + maximum_tile_half_y * abs(e0_y)
        + tolerance
    )
    transverse_constant = delta_x * e0_x
    if e0_y == 0.0:
        if abs(transverse_constant) > transverse_radius:
            return 1, 0
    else:
        transverse_lower = (-transverse_radius - transverse_constant) / e0_y
        transverse_upper = (transverse_radius - transverse_constant) / e0_y
        if transverse_lower > transverse_upper:
            transverse_lower, transverse_upper = (
                transverse_upper,
                transverse_lower,
            )
        lower_y = max(lower_y, centroid_y + transverse_lower)
        upper_y = min(upper_y, centroid_y + transverse_upper)

    longitudinal_radius = (
        longitudinal_half_length
        + maximum_tile_half_x * abs(e1_x)
        + maximum_tile_half_y * abs(e1_y)
        + tolerance
    )
    longitudinal_constant = delta_x * e1_x
    if e1_y == 0.0:
        if abs(longitudinal_constant) > longitudinal_radius:
            return 1, 0
    else:
        longitudinal_lower = (
            -longitudinal_radius - longitudinal_constant
        ) / e1_y
        longitudinal_upper = (
            longitudinal_radius - longitudinal_constant
        ) / e1_y
        if longitudinal_lower > longitudinal_upper:
            longitudinal_lower, longitudinal_upper = (
                longitudinal_upper,
                longitudinal_lower,
            )
        lower_y = max(lower_y, centroid_y + longitudinal_lower)
        upper_y = min(upper_y, centroid_y + longitudinal_upper)

    if lower_y > upper_y:
        return 1, 0

    # The last tile may be narrower and have a shifted center. Expanding the
    # regular-center interval by one row encloses it; the SAT test remains the
    # final authority for every emitted candidate.
    search_lower = (
        int(np.ceil((lower_y - first_regular_center_y) / tile_stride)) - 1
    )
    search_upper = (
        int(np.floor((upper_y - first_regular_center_y) / tile_stride)) + 1
    )
    return (
        max(lower_tile_y, search_lower),
        min(upper_tile_y, search_upper),
    )


@njit(cache=True)
def _build_exact_tile_index(
    origin: np.ndarray,
    shape: np.ndarray,
    resolution: float,
    tile_size: int,
    bounds: np.ndarray,
    centroids: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    transverse_half_lengths: np.ndarray,
    longitudinal_half_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ordered CSR candidates using exact tile-OBB intersections."""
    tile_count_x = (shape[0] + tile_size - 1) // tile_size
    tile_count_y = (shape[1] + tile_size - 1) // tile_size
    tile_centers_x = np.empty(tile_count_x, dtype=np.float64)
    tile_centers_y = np.empty(tile_count_y, dtype=np.float64)
    tile_half_lengths_x = np.empty(tile_count_x, dtype=np.float64)
    tile_half_lengths_y = np.empty(tile_count_y, dtype=np.float64)

    for tile_x in range(tile_count_x):
        lower_x = tile_x * tile_size
        upper_x = min(shape[0] - 1, lower_x + tile_size - 1)
        tile_centers_x[tile_x] = (
            origin[0] + 0.5 * (lower_x + upper_x) * resolution
        )
        tile_half_lengths_x[tile_x] = 0.5 * (upper_x - lower_x) * resolution
    for tile_y in range(tile_count_y):
        lower_y = tile_y * tile_size
        upper_y = min(shape[1] - 1, lower_y + tile_size - 1)
        tile_centers_y[tile_y] = (
            origin[1] + 0.5 * (lower_y + upper_y) * resolution
        )
        tile_half_lengths_y[tile_y] = 0.5 * (upper_y - lower_y) * resolution
    maximum_tile_half_x = np.max(tile_half_lengths_x)
    maximum_tile_half_y = np.max(tile_half_lengths_y)
    first_regular_center_y = tile_centers_y[0]
    tile_stride = tile_size * resolution

    vector_count = bounds.shape[0]
    lower_tiles = np.empty((vector_count, 2), dtype=np.int64)
    upper_tiles = np.empty((vector_count, 2), dtype=np.int64)
    for vector_index in range(vector_count):
        lower_x = max(
            0,
            int(np.ceil((bounds[vector_index, 0] - origin[0]) / resolution)),
        )
        upper_x = min(
            shape[0] - 1,
            int(np.floor((bounds[vector_index, 1] - origin[0]) / resolution)),
        )
        lower_y = max(
            0,
            int(np.ceil((bounds[vector_index, 2] - origin[1]) / resolution)),
        )
        upper_y = min(
            shape[1] - 1,
            int(np.floor((bounds[vector_index, 3] - origin[1]) / resolution)),
        )
        lower_tiles[vector_index, 0] = lower_x // tile_size
        lower_tiles[vector_index, 1] = lower_y // tile_size
        upper_tiles[vector_index, 0] = upper_x // tile_size
        upper_tiles[vector_index, 1] = upper_y // tile_size

    tile_counts = np.zeros(tile_count_x * tile_count_y, dtype=np.int64)
    for vector_index in range(vector_count):
        for tile_x in range(
            lower_tiles[vector_index, 0],
            upper_tiles[vector_index, 0] + 1,
        ):
            delta_x = tile_centers_x[tile_x] - centroids[vector_index, 0]
            lower_tile_y, upper_tile_y = _candidate_tile_y_range(
                delta_x,
                centroids[vector_index, 1],
                maximum_tile_half_x,
                maximum_tile_half_y,
                transverse_half_lengths[vector_index],
                longitudinal_half_lengths[vector_index],
                e0[vector_index, 0],
                e0[vector_index, 1],
                e1[vector_index, 0],
                e1[vector_index, 1],
                first_regular_center_y,
                tile_stride,
                lower_tiles[vector_index, 1],
                upper_tiles[vector_index, 1],
            )
            for tile_y in range(
                lower_tile_y,
                upper_tile_y + 1,
            ):
                delta_y = tile_centers_y[tile_y] - centroids[vector_index, 1]
                if _tile_intersects_path_obb(
                    delta_x,
                    delta_y,
                    tile_half_lengths_x[tile_x],
                    tile_half_lengths_y[tile_y],
                    transverse_half_lengths[vector_index],
                    longitudinal_half_lengths[vector_index],
                    e0[vector_index, 0],
                    e0[vector_index, 1],
                    e1[vector_index, 0],
                    e1[vector_index, 1],
                ):
                    tile_index = tile_x * tile_count_y + tile_y
                    tile_counts[tile_index] += 1

    candidate_offsets = np.empty(tile_counts.size + 1, dtype=np.int64)
    candidate_offsets[0] = 0
    for tile_index in range(tile_counts.size):
        candidate_offsets[tile_index + 1] = (
            candidate_offsets[tile_index] + tile_counts[tile_index]
        )

    candidate_indices = np.empty(candidate_offsets[-1], dtype=np.int32)
    cursors = candidate_offsets[:-1].copy()
    for vector_index in range(vector_count):
        for tile_x in range(
            lower_tiles[vector_index, 0],
            upper_tiles[vector_index, 0] + 1,
        ):
            delta_x = tile_centers_x[tile_x] - centroids[vector_index, 0]
            lower_tile_y, upper_tile_y = _candidate_tile_y_range(
                delta_x,
                centroids[vector_index, 1],
                maximum_tile_half_x,
                maximum_tile_half_y,
                transverse_half_lengths[vector_index],
                longitudinal_half_lengths[vector_index],
                e0[vector_index, 0],
                e0[vector_index, 1],
                e1[vector_index, 0],
                e1[vector_index, 1],
                first_regular_center_y,
                tile_stride,
                lower_tiles[vector_index, 1],
                upper_tiles[vector_index, 1],
            )
            for tile_y in range(
                lower_tile_y,
                upper_tile_y + 1,
            ):
                delta_y = tile_centers_y[tile_y] - centroids[vector_index, 1]
                if _tile_intersects_path_obb(
                    delta_x,
                    delta_y,
                    tile_half_lengths_x[tile_x],
                    tile_half_lengths_y[tile_y],
                    transverse_half_lengths[vector_index],
                    longitudinal_half_lengths[vector_index],
                    e0[vector_index, 0],
                    e0[vector_index, 1],
                    e1[vector_index, 0],
                    e1[vector_index, 1],
                ):
                    tile_index = tile_x * tile_count_y + tile_y
                    cursor = cursors[tile_index]
                    candidate_indices[cursor] = vector_index
                    cursors[tile_index] = cursor + 1
    return candidate_offsets, candidate_indices


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

    if len(active_vectors) > np.iinfo(np.int32).max:
        raise OverflowError("The spatial index cannot address every vector.")
    if active_vectors:
        bounds = np.ascontiguousarray(
            [vector.AABB[:4] for vector in active_vectors],
            dtype=np.float64,
        )
        centroids = np.ascontiguousarray(
            [vector.centroid[:2] for vector in active_vectors],
            dtype=np.float64,
        )
        e0 = np.ascontiguousarray(
            [vector.e0[:2] for vector in active_vectors],
            dtype=np.float64,
        )
        e1 = np.ascontiguousarray(
            [vector.e1[:2] for vector in active_vectors],
            dtype=np.float64,
        )
        transverse_half_lengths = np.ascontiguousarray(
            [vector.L0 for vector in active_vectors],
            dtype=np.float64,
        )
        longitudinal_half_lengths = np.ascontiguousarray(
            [vector.L1 for vector in active_vectors],
            dtype=np.float64,
        )
    else:
        bounds = np.empty((0, 4), dtype=np.float64)
        centroids = np.empty((0, 2), dtype=np.float64)
        e0 = np.empty((0, 2), dtype=np.float64)
        e1 = np.empty((0, 2), dtype=np.float64)
        transverse_half_lengths = np.empty(0, dtype=np.float64)
        longitudinal_half_lengths = np.empty(0, dtype=np.float64)

    candidate_offsets, candidate_indices = _build_exact_tile_index(
        grid_min,
        shape,
        resolution,
        tile_size,
        bounds,
        centroids,
        e0,
        e1,
        transverse_half_lengths,
        longitudinal_half_lengths,
    )
    return (
        active_vectors,
        shape,
        tile_size,
        candidate_offsets,
        candidate_indices,
    )
