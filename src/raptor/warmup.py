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
"""Precompile and persist Raptor's production Numba signatures."""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence


def warm_numba_cache(
    *,
    include_morphology: bool = True,
) -> Dict[str, float]:
    """Compile the kernels used by the optimized CPU workflow.

    Shapes are deliberately tiny because Numba specializes on array dtype,
    dimensionality, and memory layout rather than the number of voxels or
    spectral modes. The resulting cache entries therefore serve the full RVE.
    """
    import numpy as np

    from .core import (
        build_spectral_tables,
        collect_zero_indices,
        compute_melt_mask_kernel,
        count_phase_codes,
        label_sparse_defects,
    )

    timings: Dict[str, float] = {}

    durations = np.array([1.0e-6], dtype=np.float64)
    table_offsets = np.array([0, 2], dtype=np.int64)
    phases = np.zeros((1, 1), dtype=np.float32)
    amplitudes = np.array([100.0e-6], dtype=np.float32)
    frequencies = np.zeros(1, dtype=np.float32)

    start = time.perf_counter()
    table = build_spectral_tables(
        durations,
        table_offsets,
        phases,
        phases,
        phases,
        amplitudes,
        frequencies,
        amplitudes,
        frequencies,
        amplitudes,
        frequencies,
    )
    timings["spectral_table"] = time.perf_counter() - start

    resolution = 5.0e-6
    origin = np.zeros(3, dtype=np.float64)
    voxel_shape = np.array([2, 2, 2], dtype=np.int64)
    melt_mask = np.zeros(8, dtype=np.int8)
    candidate_offsets = np.array([0, 1], dtype=np.int64)
    candidate_indices = np.array([0], dtype=np.int32)
    start_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    e0 = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    e1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    half_width = float(amplitudes[0]) / 2.0
    L0_sqr = np.array([half_width * half_width], dtype=np.float64)
    L1_sqr = np.array([(resolution / 2.0) ** 2], dtype=np.float64)
    AABB = np.array(
        [
            [
                -half_width,
                resolution + half_width,
                -half_width,
                half_width,
                -100.0e-6,
                100.0e-6,
            ]
        ],
        dtype=np.float64,
    )
    lower_z = np.array([0], dtype=np.int64)
    upper_z = np.array([1], dtype=np.int64)
    centroids = np.array([[resolution / 2.0, 0.0, 0.0]], dtype=np.float64)
    distances = np.array([[resolution, 0.0, 0.0]], dtype=np.float64)
    inv_distance_sqr = np.array([1.0 / resolution**2], dtype=np.float64)

    start = time.perf_counter()
    compute_melt_mask_kernel(
        resolution,
        melt_mask,
        origin,
        voxel_shape,
        2,
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
        table_offsets,
        table,
        centroids,
        distances,
        inv_distance_sqr,
        1.0,
        1.0,
    )
    timings["production_mask"] = time.perf_counter() - start

    start = time.perf_counter()
    count_phase_codes(melt_mask)
    timings["phase_histogram"] = time.perf_counter() - start

    if include_morphology:
        start = time.perf_counter()
        values = np.array([0, 1, 1], dtype=np.int8)
        defect_indices = collect_zero_indices(values, 1)
        label_sparse_defects(defect_indices, (3, 1, 1), 2)
        timings["morphology"] = time.perf_counter() - start

    return timings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompile Raptor's optimized CPU kernels into a persistent "
            "Numba cache."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Persistent cache directory. This sets NUMBA_CACHE_DIR before "
            "Numba is imported."
        ),
    )
    parser.add_argument(
        "--skip-morphology",
        action="store_true",
        help="Compile only melt-mask and phase-counting kernels.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)

    # Import after applying NUMBA_CACHE_DIR so decorators select that location.
    import numba

    configured_cache = (
        numba.config.CACHE_DIR
        or "source-adjacent __pycache__ (set --cache-dir for persistence)"
    )
    print(f"Numba cache: {configured_cache}")
    print(
        f"Target: {numba.config.CPU_NAME or 'native CPU'}, "
        f"threads={numba.get_num_threads()}"
    )

    total_start = time.perf_counter()
    timings = warm_numba_cache(
        include_morphology=not args.skip_morphology,
    )
    for name, elapsed in timings.items():
        print(f" -> {name}: {elapsed:.3f}s")
    elapsed = time.perf_counter() - total_start
    print(f"Raptor cache warmup complete ({elapsed:.3f}s).")
    return 0


def run() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


if __name__ == "__main__":
    run()
