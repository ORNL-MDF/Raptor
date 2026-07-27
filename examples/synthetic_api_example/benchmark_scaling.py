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
"""Benchmark strong/weak scaling, tile width, and spectral accuracy."""

import argparse
import csv
import gc
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, set_num_threads

from raptor.api import create_grid, create_path_vectors
from raptor.core import (
    build_spatial_index,
    build_spectral_history_table,
    compute_melt_mask_grid,
    evaluate_spectral_table,
)
from raptor.warmup import warm_numba_cache

from run_synthetic_rve import (
    HATCH_SPACING,
    LASER_POWER,
    LAYER_HEIGHT,
    RANDOM_SEED,
    RVE_MAX_POINT,
    RVE_MIN_POINT,
    SCAN_ROTATION,
    SCAN_SPEED,
    VOXEL_RESOLUTION,
    build_melt_pool,
)


@dataclass
class Problem:
    """Prepared synthetic RVE without a tile index."""

    grid: object
    vectors: list
    melt_pool: object
    active_vectors: list


@dataclass
class Geometry:
    """RVE geometry selected for a benchmark profile."""

    minimum: np.ndarray
    maximum: np.ndarray
    resolution: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Raptor using the run_synthetic_rve.py configuration."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("example", "reference"),
        default="example",
        help=(
            "Use the quick-run example geometry or the documented "
            "billion-voxel reference geometry."
        ),
    )
    parser.add_argument(
        "--study",
        choices=("strong", "weak", "accuracy"),
        default="strong",
        help="Study to execute.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help=(
            "Numba threads. Weak scaling pairs each value with an x-y area "
            "proportional to its fraction of the largest value."
        ),
    )
    parser.add_argument(
        "--tile-widths-um",
        type=float,
        nargs="+",
        default=[80.0],
        help="Physical tile widths to evaluate, in micrometres.",
    )
    parser.add_argument(
        "--spectral-error-fractions",
        type=float,
        nargs="+",
        default=[0.25],
        help="Voxel-relative spectral error budgets.",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        help=(
            "Optional per-process memory budget. Omit for automatic sizing; "
            "small budgets exercise ordered spectral-table streaming."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Mask calls per performance configuration.",
    )
    parser.add_argument(
        "--settling-runs",
        type=int,
        default=1,
        help="Initial calls excluded from each reported median.",
    )
    parser.add_argument(
        "--accuracy-samples",
        type=int,
        default=10_000,
        help="Random vector/time evaluations per spectral accuracy budget.",
    )
    parser.add_argument(
        "--phase-accuracy",
        action="store_true",
        help=(
            "Also build full phase fields and compare budgets with the "
            "strictest requested budget."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV destination for raw measurements.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if args.settling_runs < 0 or args.settling_runs >= args.repeats:
        raise ValueError(
            "--settling-runs must be non-negative and smaller than --repeats."
        )
    if any(value < 1 for value in args.threads):
        raise ValueError("All --threads values must be positive.")
    if any(
        not np.isfinite(value) or value <= 0.0 for value in args.tile_widths_um
    ):
        raise ValueError("All --tile-widths-um values must be positive.")
    if any(
        not np.isfinite(value) or value <= 0.0 or value > 1.0
        for value in args.spectral_error_fractions
    ):
        raise ValueError(
            "All spectral error fractions must be in the interval (0, 1]."
        )
    if args.accuracy_samples < 1:
        raise ValueError("--accuracy-samples must be positive.")
    if args.memory_limit_mb is not None and args.memory_limit_mb < 1:
        raise ValueError("--memory-limit-mb must be positive.")


def select_geometry(profile):
    """Resolve the quick-run or documented reference geometry."""
    if profile == "reference":
        return Geometry(
            np.zeros(3, dtype=np.float64),
            np.full(3, 1.0e-3, dtype=np.float64),
            1.0e-6,
        )
    return Geometry(
        np.array(RVE_MIN_POINT, dtype=np.float64),
        np.array(RVE_MAX_POINT, dtype=np.float64),
        VOXEL_RESOLUTION,
    )


def prepare_problem(melt_pool, geometry, xy_scale=1.0):
    """Prepare a geometrically scaled synthetic problem."""
    bound_box = np.array(
        [geometry.minimum, geometry.maximum],
        dtype=np.float64,
    )
    xy_span = bound_box[1, :2] - bound_box[0, :2]
    bound_box[1, :2] = bound_box[0, :2] + xy_scale * xy_span
    grid = create_grid(geometry.resolution, bound_box=bound_box)
    vectors = create_path_vectors(
        bound_box,
        LASER_POWER,
        SCAN_SPEED,
        HATCH_SPACING,
        LAYER_HEIGHT,
        SCAN_ROTATION,
        scan_extension=float(np.max(xy_scale * xy_span)),
        extra_layers=10,
    )
    phase_rng = np.random.RandomState(RANDOM_SEED)
    for vector in vectors:
        vector.set_melt_pool_properties(melt_pool, phase_rng)
    active, *_ = build_spatial_index(
        grid.origin,
        grid.shape,
        grid.resolution,
        vectors,
    )
    return Problem(grid, vectors, melt_pool, active)


def build_index(problem, tile_width_um):
    """Build and characterize one tile index."""
    requested_width = tile_width_um * 1.0e-6
    tile_size = max(
        1,
        int(round(requested_width / problem.grid.resolution)),
    )
    start = time.perf_counter()
    active, _, tile_size, offsets, indices = build_spatial_index(
        problem.grid.origin,
        problem.grid.shape,
        problem.grid.resolution,
        problem.vectors,
        tile_size=tile_size,
    )
    elapsed = time.perf_counter() - start
    counts = np.diff(offsets)
    metrics = {
        "tile_width_um": tile_width_um,
        "tile_size_voxels": tile_size,
        "resolved_tile_width_um": (tile_size * problem.grid.resolution * 1.0e6),
        "index_seconds": elapsed,
        "tiles": counts.size,
        "candidate_entries": indices.size,
        "candidate_mean": float(np.mean(counts)),
        "candidate_p95": float(np.percentile(counts, 95)),
        "candidate_max": int(np.max(counts, initial=0)),
    }
    return active, tile_size, offsets, indices, metrics


def run_performance_case(
    problem,
    study,
    thread_count,
    tile_width_um,
    error_fraction,
    repeats,
    settling_runs,
    expected_checksums,
    xy_scale,
    memory_limit_mb,
):
    """Run one thread/tile/error configuration."""
    active, tile_size, offsets, indices, index_metrics = build_index(
        problem,
        tile_width_um,
    )
    set_num_threads(thread_count)
    samples = []
    rows = []
    checksum_key = (
        tuple(problem.grid.shape),
        float(error_fraction),
    )
    histogram = None
    for run_index in range(repeats):
        diagnostics = {}
        start = time.perf_counter()
        mask = compute_melt_mask_grid(
            problem.grid.origin,
            problem.grid.shape,
            problem.grid.resolution,
            problem.melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=offsets,
            candidate_indices=indices,
            spectral_error_fraction=error_fraction,
            memory_limit_mb=memory_limit_mb,
            diagnostics=diagnostics,
            report=run_index == 0,
        )
        elapsed = time.perf_counter() - start
        checksum = hashlib.sha256(memoryview(mask)).hexdigest()
        expected = expected_checksums.setdefault(checksum_key, checksum)
        if checksum != expected:
            raise RuntimeError(
                "Phase checksum changed with thread count or tile width."
            )
        if run_index == repeats - 1:
            histogram = np.bincount(mask, minlength=4).tolist()
        samples.append(elapsed)
        row = {
            "study": study,
            "xy_scale": xy_scale,
            "threads": thread_count,
            "voxels": problem.grid.n_voxels,
            "padded_vectors": len(problem.vectors),
            "active_vectors": len(active),
            "spectral_error_fraction": error_fraction,
            "run": run_index + 1,
            "settling": int(run_index < settling_runs),
            "seconds": elapsed,
            "throughput_mvox_s": (problem.grid.n_voxels / elapsed / 1.0e6),
            "checksum": checksum,
            **index_metrics,
            **diagnostics,
        }
        rows.append(row)
        del mask
        gc.collect()

    retained = samples[settling_runs:]
    median = float(np.median(retained))
    print(
        f"scale={xy_scale:.4f}, threads={get_num_threads():2d}, "
        f"tile={tile_size:3d} voxels, error={error_fraction:.4f}: "
        f"median={median:.6f}s, "
        f"throughput={problem.grid.n_voxels / median / 1.0e6:.2f} Mvox/s"
    )
    print(f" -> histogram={histogram}, checksum={expected}")
    return rows


def run_performance_study(args, melt_pool, geometry):
    rows = []
    expected_checksums = {}
    maximum_threads = max(args.threads)
    if args.study == "strong":
        problems = [
            (
                1.0,
                args.threads,
                prepare_problem(melt_pool, geometry),
            )
        ]
    else:
        problems = []
        for thread_count in args.threads:
            xy_scale = np.sqrt(thread_count / maximum_threads)
            problems.append(
                (
                    xy_scale,
                    [thread_count],
                    prepare_problem(melt_pool, geometry, xy_scale),
                )
            )

    for xy_scale, thread_counts, problem in problems:
        print(
            f"Prepared {problem.grid.n_voxels} voxels, "
            f"{len(problem.vectors)} padded vectors, "
            f"scale={xy_scale:.4f}."
        )
        for tile_width_um in args.tile_widths_um:
            for error_fraction in args.spectral_error_fractions:
                for thread_count in thread_counts:
                    rows.extend(
                        run_performance_case(
                            problem,
                            args.study,
                            thread_count,
                            tile_width_um,
                            error_fraction,
                            args.repeats,
                            args.settling_runs,
                            expected_checksums,
                            xy_scale,
                            args.memory_limit_mb,
                        )
                    )
        del problem
        gc.collect()
    return rows


def adjusted_spectral_data(problem):
    """Return Float64 spectra with phases shifted to vector-local time."""
    melt_pool = problem.melt_pool
    start_times = np.array(
        [vector.start_time for vector in problem.active_vectors]
    )
    durations = np.array([vector.duration for vector in problem.active_vectors])
    dimension_data = []
    for name in ("width", "depth", "height"):
        oscillations = getattr(melt_pool, f"{name}_oscillations")
        amplitudes = np.ascontiguousarray(oscillations[:, 0])
        frequencies = np.ascontiguousarray(oscillations[:, 1])
        phases = np.remainder(
            np.array(
                [
                    getattr(vector, f"{name}_phases")
                    for vector in problem.active_vectors
                ]
            )
            + 2.0 * np.pi * start_times[:, None] * frequencies[None, :],
            2.0 * np.pi,
        )
        dimension_data.append((amplitudes, frequencies, phases))
    return durations, dimension_data


@njit(cache=False)
def evaluate_spectra_reference(
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
):
    """Evaluate independent spectra with direct Float64 cosine calls."""
    result = np.zeros(3, dtype=np.float64)
    two_pi_t = 2.0 * np.pi * time
    for mode in range(width_amplitudes.size):
        result[0] += width_amplitudes[mode] * np.cos(
            two_pi_t * width_frequencies[mode]
            + width_phases[vector_index, mode]
        )
    for mode in range(depth_amplitudes.size):
        result[1] += depth_amplitudes[mode] * np.cos(
            two_pi_t * depth_frequencies[mode]
            + depth_phases[vector_index, mode]
        )
    for mode in range(height_amplitudes.size):
        result[2] += height_amplitudes[mode] * np.cos(
            two_pi_t * height_frequencies[mode]
            + height_phases[vector_index, mode]
        )
    return result


@njit(cache=False)
def measure_spectral_errors(
    vector_indices,
    fractions,
    durations,
    table_offsets,
    table,
    width_amplitudes,
    width_frequencies,
    width_phases,
    depth_amplitudes,
    depth_frequencies,
    depth_phases,
    height_amplitudes,
    height_frequencies,
    height_phases,
):
    """Compare table interpolation with direct Float64 cosine evaluation."""
    maximum = np.zeros(3, dtype=np.float64)
    sum_squares = np.zeros(3, dtype=np.float64)
    for sample_index in range(vector_indices.size):
        vector_index = vector_indices[sample_index]
        fraction = fractions[sample_index]
        direct = evaluate_spectra_reference(
            fraction * durations[vector_index],
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
        approximate = evaluate_spectral_table(
            fraction,
            vector_index,
            table_offsets,
            table,
        )
        for dimension in range(3):
            error = abs(direct[dimension] - approximate[dimension])
            maximum[dimension] = max(maximum[dimension], error)
            sum_squares[dimension] += error * error
    return maximum, np.sqrt(sum_squares / vector_indices.size)


def phase_accuracy_fields(problem, args, fractions):
    if not args.phase_accuracy:
        return {}
    active, tile_size, offsets, indices, _ = build_index(
        problem,
        args.tile_widths_um[0],
    )
    set_num_threads(max(args.threads))
    reference = None
    comparisons = {}
    for fraction in fractions:
        mask = compute_melt_mask_grid(
            problem.grid.origin,
            problem.grid.shape,
            problem.grid.resolution,
            problem.melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=offsets,
            candidate_indices=indices,
            spectral_error_fraction=fraction,
            memory_limit_mb=args.memory_limit_mb,
        )
        if reference is None:
            reference = mask
            comparisons[fraction] = (0, 0)
            continue
        changed = int(np.count_nonzero(mask != reference))
        defect_changed = int(np.count_nonzero((mask == 0) != (reference == 0)))
        comparisons[fraction] = (changed, defect_changed)
        del mask
        gc.collect()
    return comparisons


def run_accuracy_study(args, melt_pool, geometry):
    problem = prepare_problem(melt_pool, geometry)
    durations, dimension_data = adjusted_spectral_data(problem)
    rng = np.random.default_rng(RANDOM_SEED)
    vector_indices = rng.integers(
        0,
        len(problem.active_vectors),
        size=args.accuracy_samples,
        dtype=np.int64,
    )
    fractions = rng.random(args.accuracy_samples)
    requested_fractions = sorted(set(args.spectral_error_fractions))

    rows = []
    accepted_fractions = []
    for error_fraction in requested_fractions:
        start = time.perf_counter()
        try:
            offsets, table, diagnostics = build_spectral_history_table(
                problem.grid.resolution,
                problem.melt_pool,
                problem.active_vectors,
                spectral_error_fraction=error_fraction,
            )
        except (MemoryError, ValueError) as error:
            rows.append(
                {
                    "study": "accuracy",
                    "status": f"rejected: {error}",
                    "spectral_error_fraction": error_fraction,
                    "accuracy_samples": args.accuracy_samples,
                    "table_build_seconds": time.perf_counter() - start,
                    "spectral_table_points": "",
                    "spectral_table_bytes": "",
                    "spectral_error_bound": "",
                    "width_max_error_um": "",
                    "depth_max_error_um": "",
                    "height_max_error_um": "",
                    "width_rmse_um": "",
                    "depth_rmse_um": "",
                    "height_rmse_um": "",
                    "phase_labels_changed": "",
                    "defect_status_changed": "",
                }
            )
            print(f"error={error_fraction:.4f}: rejected ({error}).")
            continue
        build_seconds = time.perf_counter() - start
        maximum, rmse = measure_spectral_errors(
            vector_indices,
            fractions,
            durations,
            offsets,
            table,
            dimension_data[0][0],
            dimension_data[0][1],
            dimension_data[0][2],
            dimension_data[1][0],
            dimension_data[1][1],
            dimension_data[1][2],
            dimension_data[2][0],
            dimension_data[2][1],
            dimension_data[2][2],
        )
        row = {
            "study": "accuracy",
            "status": "accepted",
            "spectral_error_fraction": error_fraction,
            "accuracy_samples": args.accuracy_samples,
            "table_build_seconds": build_seconds,
            **diagnostics,
            "width_max_error_um": maximum[0] * 1.0e6,
            "depth_max_error_um": maximum[1] * 1.0e6,
            "height_max_error_um": maximum[2] * 1.0e6,
            "width_rmse_um": rmse[0] * 1.0e6,
            "depth_rmse_um": rmse[1] * 1.0e6,
            "height_rmse_um": rmse[2] * 1.0e6,
            "phase_labels_changed": "",
            "defect_status_changed": "",
        }
        rows.append(row)
        accepted_fractions.append(error_fraction)
        print(
            f"error={error_fraction:.4f}: "
            f"table={diagnostics['spectral_table_points']} points, "
            f"max errors={maximum * 1.0e6} µm."
        )
        del table
        gc.collect()

    phase_comparisons = phase_accuracy_fields(
        problem,
        args,
        accepted_fractions,
    )
    for row in rows:
        fraction = row["spectral_error_fraction"]
        if fraction not in phase_comparisons:
            continue
        changed, defect_changed = phase_comparisons[fraction]
        row["phase_labels_changed"] = changed
        row["defect_status_changed"] = defect_changed
        print(
            f" -> error={fraction:.4f}: phase labels changed={changed}, "
            f"defect status changed={defect_changed}."
        )
    return rows


def write_rows(rows, output_path):
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Measurements written to: {output_path}")


def main():
    args = parse_args()
    validate_args(args)
    warm_numba_cache(include_morphology=False)
    geometry = select_geometry(args.profile)
    melt_pool = build_melt_pool(geometry.resolution)
    if args.study == "accuracy":
        rows = run_accuracy_study(args, melt_pool, geometry)
    else:
        rows = run_performance_study(args, melt_pool, geometry)
    write_rows(rows, args.output)


if __name__ == "__main__":
    main()
