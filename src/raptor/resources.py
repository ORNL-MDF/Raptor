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
"""Memory-budget resolution and spectral-table execution planning."""

import os
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .spectral import SpectralHistoryPlan


MEMORY_UNIT_BYTES = 1024**2
# Leave capacity for Python, optional morphology/output, and the host system.
_AUTOMATIC_MEMORY_FRACTION = 0.80
_MEMORY_LIMIT_ENVIRONMENT = "RAPTOR_MEMORY_LIMIT_MB"


def validate_memory_limit(memory_limit_mb: Optional[int]) -> None:
    """Validate an optional per-process memory limit in binary megabytes."""
    if memory_limit_mb is not None and (
        isinstance(memory_limit_mb, (bool, np.bool_))
        or not isinstance(memory_limit_mb, Integral)
        or memory_limit_mb < 1
    ):
        raise ValueError("memory_limit_mb must be a positive integer or None.")


def read_cgroup_remaining_bytes() -> Optional[int]:
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


def available_memory_bytes() -> int:
    """Return memory currently available to this process."""
    import psutil

    available_bytes = int(psutil.virtual_memory().available)
    cgroup_remaining = read_cgroup_remaining_bytes()
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


def resolve_memory_budget(memory_limit_mb: Optional[int]) -> Tuple[int, str]:
    """Resolve API, environment, or automatic memory into bytes."""
    validate_memory_limit(memory_limit_mb)
    source = "api"
    resolved_mb = memory_limit_mb
    if resolved_mb is None:
        resolved_mb = _parse_environment_memory_limit()
        source = "environment"
    if resolved_mb is not None:
        requested_bytes = int(resolved_mb) * MEMORY_UNIT_BYTES
        budget_bytes = requested_bytes
        cgroup_remaining = read_cgroup_remaining_bytes()
        if cgroup_remaining is not None:
            budget_bytes = min(budget_bytes, cgroup_remaining)
            if budget_bytes < requested_bytes:
                source += "+cgroup"
        return budget_bytes, source

    available_bytes = available_memory_bytes()
    return int(available_bytes * _AUTOMATIC_MEMORY_FRACTION), "automatic"


@dataclass
class MemoryExecutionPlan:
    """Core-array budget and contiguous spectral-vector batches."""

    budget_bytes: int
    budget_source: str
    fixed_bytes: int
    full_table_bytes: int
    peak_bytes: int
    mode: str
    batches: List[Tuple[int, int]]


def insufficient_memory_error(
    budget_bytes: int,
    minimum_bytes: int,
    budget_source: str,
) -> MemoryError:
    """Build an actionable error without reducing spectral accuracy."""
    minimum_mb = (
        minimum_bytes + MEMORY_UNIT_BYTES - 1
    ) // MEMORY_UNIT_BYTES
    return MemoryError(
        "Raptor cannot fit the phase field and required core workspace.\n\n"
        f"Resolved memory budget: "
        f"{budget_bytes / MEMORY_UNIT_BYTES:.2f} MB "
        f"({budget_source})\n"
        f"Minimum required memory: "
        f"{minimum_bytes / MEMORY_UNIT_BYTES:.2f} MB\n\n"
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
    """Partition ordered vectors without changing table density."""
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


def plan_memory_execution(
    budget_bytes: int,
    budget_source: str,
    fixed_bytes: int,
    spectral_plan: SpectralHistoryPlan,
    streaming_workspace_bytes: int,
) -> MemoryExecutionPlan:
    """Choose resident or ordered streamed execution under one budget.

    ``fixed_bytes`` includes the phase field and immutable kernel inputs.
    Resident mode allocates the complete packed spectral table. Streamed mode
    preserves vector order and table density while limiting peak table storage.
    """
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
        raise insufficient_memory_error(
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


def filter_candidate_index_range(
    candidate_offsets: np.ndarray,
    candidate_indices: np.ndarray,
    start: int,
    stop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retain a global-vector range and remap it to local indices."""
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
