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
"""Memory-efficient phase counting and sparse defect labeling."""

from typing import Tuple

import numpy as np
from numba import njit, prange


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
    """Return a union-find root while compressing the traversed path."""
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
    """Label sorted sparse defect indices with 26-neighbor connectivity."""
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
                        flat_indices,
                        neighbor_flat,
                    )
                    if (
                        neighbor_position >= point_index
                        or flat_indices[neighbor_position] != neighbor_flat
                    ):
                        continue
                    root = _find_component_root(parent, point_index)
                    neighbor_root = _find_component_root(
                        parent,
                        neighbor_position,
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
