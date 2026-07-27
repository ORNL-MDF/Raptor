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
"""Tests for the current melt-mask geometry and compiled core kernel."""

import numpy as np
import pytest

import raptor.core as core
import raptor.resources as resources
from raptor.core import (
    classify_horizontal_melt_and_boundary,
    conservative_vertical_interval,
    compute_melt_mask_grid,
)
from raptor.spectral import (
    build_spectral_tables,
    evaluate_spectra_fast,
    evaluate_spectral_table,
    prepare_spectral_history_plan,
    spectral_approximation_error_bound,
)
from raptor.structures import Grid, MeltPool, PathVector
from raptor.spatial import build_spatial_index


RESOLUTION = 5.0e-6
WIDTH = 200.0e-6
HEIGHT = 100.0e-6
DEPTH = 80.0e-6


@pytest.fixture
def constant_melt_pool():
    width = np.array([[WIDTH, 0.0, 0.0]], dtype=np.float64)
    depth = np.array([[DEPTH, 0.0, 0.0]], dtype=np.float64)
    height = np.array([[HEIGHT, 0.0, 0.0]], dtype=np.float64)
    return MeltPool(
        width,
        depth,
        height,
        WIDTH,
        DEPTH,
        HEIGHT,
        2.0,
        2.0,
        2.0,
        False,
    )


@pytest.fixture
def path_vector():
    vector = PathVector(
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0e-3, 0.0, 0.0], dtype=np.float64),
        0.0,
        1.0e-3,
    )
    vector.set_coordinate_frame()
    return vector


def radial_signed_distance(y, z, shape_factor):
    """Independent bisection reference for the radial boundary distance."""
    radius = np.hypot(y, z)
    half_width = WIDTH / 2.0
    if radius < RESOLUTION:
        return -min(half_width, HEIGHT, DEPTH)
    vertical_scale = HEIGHT if z >= 0.0 else DEPTH
    direction_y = y / radius
    direction_z = abs(z) / radius
    lower = 0.0
    upper = max(half_width, vertical_scale)
    while (upper * direction_y / half_width) ** 2 + (
        upper * direction_z / vertical_scale
    ) ** shape_factor < 1.0:
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        value = (midpoint * direction_y / half_width) ** 2 + (
            midpoint * direction_z / vertical_scale
        ) ** shape_factor
        if value < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    return radius - 0.5 * (lower + upper)


def classify_reference(y, z, shape_factor):
    signed_distance = radial_signed_distance(y, z, shape_factor)
    return (
        signed_distance < 1.0e-12,
        abs(signed_distance) - RESOLUTION <= 1.0e-12,
    )


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
    result = []
    for phases, amplitudes, frequencies in (
        (width_phases, width_amplitudes, width_frequencies),
        (depth_phases, depth_amplitudes, depth_frequencies),
        (height_phases, height_amplitudes, height_frequencies),
    ):
        angles = 2.0 * np.pi * time * frequencies + phases[vector_index]
        result.append(float(np.sum(amplitudes * np.cos(angles))))
    return tuple(result)


class TestOptimizedGeometry:
    @pytest.mark.parametrize(
        ("y", "z"),
        [
            (0.0, 0.0),
            (WIDTH / 2.0, 0.0),
            (-WIDTH / 2.0, 0.0),
            (0.0, HEIGHT),
            (0.0, -DEPTH),
        ],
    )
    def test_axis_and_center_classification(self, y, z):
        expected = classify_reference(y, z, 2.0)
        half_width = WIDTH / 2.0
        actual = classify_horizontal_melt_and_boundary(
            y * y,
            (y / half_width) ** 2,
            half_width,
            z,
            HEIGHT,
            DEPTH,
            2.0,
            2.0,
            RESOLUTION,
        )
        assert actual == expected

    @pytest.mark.parametrize("shape_factor", [0.5, 1.0, 2.0, 10.0])
    def test_classifier_matches_radial_reference(self, shape_factor):
        rng = np.random.default_rng(123)
        for y, z in rng.uniform(-250.0e-6, 250.0e-6, size=(200, 2)):
            expected = classify_reference(y, z, shape_factor)
            a = WIDTH / 2.0
            horizontal = classify_horizontal_melt_and_boundary(
                y * y,
                (y / a) ** 2,
                a,
                z,
                HEIGHT,
                DEPTH,
                shape_factor,
                shape_factor,
                RESOLUTION,
            )
            assert horizontal == expected

    @pytest.mark.parametrize("shape_factor", [0.5, 1.0, 2.0, 10.0])
    def test_conservative_interval_never_excludes_interactions(
        self, shape_factor
    ):
        rng = np.random.default_rng(789)
        for y, z in rng.uniform(-250.0e-6, 250.0e-6, size=(1000, 2)):
            is_melted, is_boundary = classify_reference(y, z, shape_factor)
            status, lower, upper = conservative_vertical_interval(
                y,
                WIDTH,
                HEIGHT,
                DEPTH,
                shape_factor,
                shape_factor,
                RESOLUTION,
            )
            if is_melted or is_boundary:
                assert status == 1
                assert lower <= z <= upper


class TestPathVectorValidation:
    @pytest.mark.parametrize(
        ("start", "end", "start_time", "end_time"),
        [
            (np.zeros(2), np.zeros(3), 0.0, 1.0),
            (np.array([np.nan, 0.0, 0.0]), np.zeros(3), 0.0, 1.0),
            (np.zeros(3), np.zeros(3), 2.0, 1.0),
        ],
    )
    def test_invalid_path_vectors_are_rejected(
        self, start, end, start_time, end_time
    ):
        with pytest.raises(ValueError):
            PathVector(start, end, start_time, end_time)


class TestFastSpectralMath:
    def test_fast_path_respects_error_budget_and_rejects_unsafe_input(self):
        amplitudes = np.array(
            [148.0e-6, 12.0e-6, -4.0e-6, 2.0e-6],
            dtype=np.float64,
        )
        frequencies = np.array([0.0, 3100.0, 6200.0, 9300.0])
        phases = np.array([[0.0, 0.7, 2.1, 5.4]], dtype=np.float64)
        duration = 1.0e-3
        bound = spectral_approximation_error_bound(
            amplitudes, frequencies, duration
        )
        maximum_error = 0.0
        for time in np.linspace(0.0, duration, 201):
            exact, _, _ = evaluate_spectra_reference(
                time,
                0,
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
            approximate, _, _ = evaluate_spectra_fast(
                time,
                0,
                phases.astype(np.float32),
                phases.astype(np.float32),
                phases.astype(np.float32),
                amplitudes.astype(np.float32),
                frequencies.astype(np.float32),
                amplitudes.astype(np.float32),
                frequencies.astype(np.float32),
                amplitudes.astype(np.float32),
                frequencies.astype(np.float32),
            )
            maximum_error = max(maximum_error, abs(exact - approximate))
        assert maximum_error <= bound
        assert bound <= RESOLUTION * 0.25
        unsafe_bound = spectral_approximation_error_bound(
            np.array([1.0e-3]),
            np.array([1.0e12]),
            1.0,
        )
        assert unsafe_bound > RESOLUTION * 0.25

        remaining_error = RESOLUTION * 0.25 - bound
        curvature = np.sum(
            np.abs(amplitudes) * (2.0 * np.pi * frequencies) ** 2
        )
        time_step = np.sqrt(8.0 * remaining_error / curvature)
        point_count = int(np.ceil(duration / time_step)) + 1
        offsets = np.array([0, point_count], dtype=np.int64)
        phases32 = phases.astype(np.float32)
        amplitudes32 = amplitudes.astype(np.float32)
        frequencies32 = frequencies.astype(np.float32)
        table = build_spectral_tables(
            np.array([duration]),
            offsets,
            np.zeros(1, dtype=np.int64),
            np.array([point_count], dtype=np.int64),
            phases32,
            phases32,
            phases32,
            amplitudes32,
            frequencies32,
            amplitudes32,
            frequencies32,
            amplitudes32,
            frequencies32,
        )
        maximum_table_error = 0.0
        for fraction in np.linspace(0.0, 1.0, 201):
            exact, _, _ = evaluate_spectra_reference(
                fraction * duration,
                0,
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
            interpolated, _, _ = evaluate_spectral_table(
                fraction,
                0,
                offsets,
                table,
                np.zeros(1, dtype=np.int64),
                np.array([point_count], dtype=np.int64),
            )
            maximum_table_error = max(
                maximum_table_error, abs(exact - interpolated)
            )
        assert maximum_table_error <= RESOLUTION * 0.25

        sample_start = int(np.floor(0.25 * (point_count - 1)))
        sample_stop = int(np.floor(0.75 * (point_count - 1))) + 2
        window_offsets = np.array([0, sample_stop - sample_start])
        window_table = build_spectral_tables(
            np.array([duration]),
            window_offsets,
            np.array([sample_start], dtype=np.int64),
            np.array([point_count], dtype=np.int64),
            phases32,
            phases32,
            phases32,
            amplitudes32,
            frequencies32,
            amplitudes32,
            frequencies32,
            amplitudes32,
            frequencies32,
        )
        for fraction in np.linspace(0.25, 0.75, 101):
            complete_value = evaluate_spectral_table(
                fraction,
                0,
                offsets,
                table,
                np.zeros(1, dtype=np.int64),
                np.array([point_count], dtype=np.int64),
            )
            window_value = evaluate_spectral_table(
                fraction,
                0,
                window_offsets,
                window_table,
                np.array([sample_start], dtype=np.int64),
                np.array([point_count], dtype=np.int64),
            )
            np.testing.assert_array_equal(window_value, complete_value)


def prepare_vector(vector, melt_pool):
    vector.set_melt_pool_properties(melt_pool)
    return vector


class TestComputeMeltMask:
    def test_memory_budget_priority_and_ordered_streaming(
        self,
        constant_melt_pool,
        path_vector,
        monkeypatch,
    ):
        mebibyte = 1024**2
        monkeypatch.setattr(
            resources,
            "read_cgroup_remaining_bytes",
            lambda: None,
        )
        monkeypatch.setattr(
            resources,
            "available_memory_bytes",
            lambda: 100 * mebibyte,
        )
        monkeypatch.setenv("RAPTOR_MEMORY_LIMIT_MB", "40")
        assert resources.resolve_memory_budget(20) == (20 * mebibyte, "api")
        assert resources.resolve_memory_budget(None) == (
            40 * mebibyte,
            "environment",
        )
        monkeypatch.delenv("RAPTOR_MEMORY_LIMIT_MB")
        assert resources.resolve_memory_budget(None) == (
            80 * mebibyte,
            "automatic",
        )

        second_vector = PathVector(
            np.array([0.0, 50.0e-6, 0.0]),
            np.array([1.0e-3, 50.0e-6, 0.0]),
            0.0,
            1.0e-3,
        )
        second_vector.set_coordinate_frame()
        vectors = [
            prepare_vector(path_vector, constant_melt_pool),
            prepare_vector(second_vector, constant_melt_pool),
        ]
        grid = Grid(
            RESOLUTION,
            bound_box=np.array(
                [
                    [0.0, -100.0e-6, -20.0e-6],
                    [50.0e-6, 150.0e-6, 20.0e-6],
                ]
            ),
        )
        active, _, tile_size, offsets, indices = build_spatial_index(
            grid.origin,
            grid.shape,
            grid.resolution,
            vectors,
        )
        spectral_plan = prepare_spectral_history_plan(
            grid.resolution,
            constant_melt_pool,
            active,
        )
        fixed_bytes = 100
        spectral_offsets_bytes = (
            spectral_plan.point_counts.size + 1
        ) * np.dtype(np.int64).itemsize
        largest_vector_bytes = (
            int(np.max(spectral_plan.point_counts))
            * 3
            * np.dtype(np.float32).itemsize
        )
        streaming_minimum = (
            fixed_bytes + spectral_offsets_bytes + largest_vector_bytes
        )
        execution_plan = resources.plan_memory_execution(
            streaming_minimum,
            "test",
            fixed_bytes,
            spectral_plan,
            0,
        )
        assert execution_plan.mode == "streamed"
        assert execution_plan.batches == [(0, 1), (1, 2)]
        with pytest.raises(MemoryError, match="memory_limit_mb"):
            resources.plan_memory_execution(
                streaming_minimum - 1,
                "test",
                fixed_bytes,
                spectral_plan,
                0,
            )

        resident = compute_melt_mask_grid(
            grid.origin,
            grid.shape,
            grid.resolution,
            constant_melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=offsets,
            candidate_indices=indices,
        )

        original_planner = core.plan_memory_execution

        def force_two_batches(*args):
            plan = original_planner(*args)
            plan.mode = "streamed"
            plan.batches = [(0, 1), (1, 2)]
            return plan

        monkeypatch.setattr(
            core,
            "plan_memory_execution",
            force_two_batches,
        )
        diagnostics = {}
        streamed = compute_melt_mask_grid(
            grid.origin,
            grid.shape,
            grid.resolution,
            constant_melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=offsets,
            candidate_indices=indices,
            diagnostics=diagnostics,
        )
        np.testing.assert_array_equal(streamed, resident)
        assert diagnostics["execution_mode"] == "streamed"
        assert diagnostics["spectral_table_batches"] == 2

    def test_optimized_grid_kernel_and_orientation_contract(
        self, constant_melt_pool, path_vector
    ):
        prepare_vector(path_vector, constant_melt_pool)
        grid = Grid(
            RESOLUTION,
            bound_box=np.array([[0.0, 0.0, 0.0], [20.0e-6, 200.0e-6, 10.0e-6]]),
        )
        active, shape, tile_size, offsets, indices = build_spatial_index(
            grid.origin, grid.shape, grid.resolution, [path_vector]
        )
        actual = compute_melt_mask_grid(
            grid.origin,
            grid.shape,
            grid.resolution,
            constant_melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=offsets,
            candidate_indices=indices,
        ).reshape(grid.shape)
        expected = np.zeros(grid.shape, dtype=np.int8)
        half_width = float(np.float32(WIDTH)) / 2.0
        height = float(np.float32(HEIGHT))
        depth = float(np.float32(DEPTH))
        for voxel_y in range(grid.shape[1]):
            y = grid.origin[1] + voxel_y * grid.resolution
            for voxel_z in range(grid.shape[2]):
                z = grid.origin[2] + voxel_z * grid.resolution
                melted, boundary = classify_horizontal_melt_and_boundary(
                    y * y,
                    (y / half_width) ** 2,
                    half_width,
                    z,
                    height,
                    depth,
                    2.0,
                    2.0,
                    RESOLUTION,
                )
                expected[:, voxel_y, voxel_z] = 2 if boundary else int(melted)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.int8

        sloped_vector = PathVector(
            np.array([0.0, 0.0, 0.0]),
            np.array([20.0e-6, 0.0, 10.0e-6]),
            0.0,
            1.0e-3,
        )
        sloped_vector.set_coordinate_frame()
        prepare_vector(sloped_vector, constant_melt_pool)
        active, shape, tile_size, offsets, indices = build_spatial_index(
            grid.origin, grid.shape, grid.resolution, [sloped_vector]
        )
        with pytest.raises(ValueError, match="horizontal path vectors"):
            compute_melt_mask_grid(
                grid.origin,
                grid.shape,
                grid.resolution,
                constant_melt_pool,
                active,
                tile_size=tile_size,
                candidate_offsets=offsets,
                candidate_indices=indices,
            )

    def test_tile_size_changes_index_only(
        self,
        constant_melt_pool,
        path_vector,
    ):
        prepare_vector(path_vector, constant_melt_pool)
        grid = Grid(
            RESOLUTION,
            bound_box=np.array(
                [[0.0, -100.0e-6, -10.0e-6], [50.0e-6, 100.0e-6, 10.0e-6]]
            ),
        )
        reference = None
        for requested_tile_size in (1, 7, 128):
            active, _, tile_size, offsets, indices = build_spatial_index(
                grid.origin,
                grid.shape,
                grid.resolution,
                [path_vector],
                tile_size=requested_tile_size,
            )
            actual = compute_melt_mask_grid(
                grid.origin,
                grid.shape,
                grid.resolution,
                constant_melt_pool,
                active,
                tile_size=tile_size,
                candidate_offsets=offsets,
                candidate_indices=indices,
            )
            if reference is None:
                reference = actual
            else:
                np.testing.assert_array_equal(actual, reference)

    def test_exact_spatial_index_is_complete_ordered_and_equivalent(
        self,
        constant_melt_pool,
    ):
        grid = Grid(
            RESOLUTION,
            bound_box=np.array(
                [[0.0, 0.0, -10.0e-6], [1.0e-3, 1.0e-3, 10.0e-6]]
            ),
        )
        vectors = [
            PathVector(
                np.array([-100.0e-6, 0.0, 0.0]),
                np.array([1.1e-3, 1.0e-3, 0.0]),
                0.0,
                1.0e-3,
            ),
            PathVector(
                np.array([-100.0e-6, 1.0e-3, 0.0]),
                np.array([1.1e-3, 0.0, 0.0]),
                0.0,
                1.0e-3,
            ),
            PathVector(
                np.array([-100.0e-6, 0.5e-3, 0.0]),
                np.array([1.1e-3, 0.5e-3, 0.0]),
                0.0,
                1.0e-3,
            ),
            PathVector(
                np.array([0.5e-3, -100.0e-6, 0.0]),
                np.array([0.5e-3, 1.1e-3, 0.0]),
                0.0,
                1.0e-3,
            ),
        ]
        for vector in vectors:
            vector.set_coordinate_frame()
            prepare_vector(vector, constant_melt_pool)

        active, shape, tile_size, offsets, indices = build_spatial_index(
            grid.origin,
            grid.shape,
            grid.resolution,
            vectors,
            tile_size=10,
        )
        for tile_index in range(offsets.size - 1):
            tile_indices = indices[
                offsets[tile_index] : offsets[tile_index + 1]
            ]
            assert np.all(np.diff(tile_indices) > 0)

        tile_count_y = (shape[1] + tile_size - 1) // tile_size
        for voxel_x in range(shape[0]):
            x = grid.origin[0] + voxel_x * grid.resolution
            for voxel_y in range(shape[1]):
                y = grid.origin[1] + voxel_y * grid.resolution
                tile_index = (
                    voxel_x // tile_size
                ) * tile_count_y + voxel_y // tile_size
                tile_indices = indices[
                    offsets[tile_index] : offsets[tile_index + 1]
                ]
                for vector_index, vector in enumerate(active):
                    delta_x = x - vector.centroid[0]
                    delta_y = y - vector.centroid[1]
                    transverse = delta_x * vector.e0[0] + delta_y * vector.e0[1]
                    longitudinal = (
                        delta_x * vector.e1[0] + delta_y * vector.e1[1]
                    )
                    if (
                        transverse * transverse <= vector.L0_sqr
                        and longitudinal * longitudinal <= vector.L1_sqr
                    ):
                        assert vector_index in tile_indices

        exact = compute_melt_mask_grid(
            grid.origin,
            grid.shape,
            grid.resolution,
            constant_melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=offsets,
            candidate_indices=indices,
        )
        tile_count = offsets.size - 1
        exhaustive_offsets = np.arange(tile_count + 1, dtype=np.int64) * len(
            active
        )
        exhaustive_indices = np.tile(
            np.arange(len(active), dtype=np.int32),
            tile_count,
        )
        exhaustive = compute_melt_mask_grid(
            grid.origin,
            grid.shape,
            grid.resolution,
            constant_melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=exhaustive_offsets,
            candidate_indices=exhaustive_indices,
        )
        np.testing.assert_array_equal(exact, exhaustive)
        assert indices.size < exhaustive_indices.size

    def test_spectral_error_control_changes_table_density(
        self,
        path_vector,
    ):
        width = np.array([[WIDTH, 0.0, 0.0], [10.0e-6, 1000.0, 0.2]])
        depth = np.array([[DEPTH, 0.0, 0.0], [5.0e-6, 1000.0, 0.3]])
        height = np.array([[HEIGHT, 0.0, 0.0], [5.0e-6, 1000.0, 0.4]])
        melt_pool = MeltPool(
            width,
            depth,
            height,
            WIDTH,
            DEPTH,
            HEIGHT,
            2.0,
            2.0,
            2.0,
            False,
        )
        prepare_vector(path_vector, melt_pool)
        grid = Grid(
            RESOLUTION,
            bound_box=np.array(
                [[0.0, -20.0e-6, 0.0], [20.0e-6, 20.0e-6, 10.0e-6]]
            ),
        )
        active, _, tile_size, offsets, indices = build_spatial_index(
            grid.origin,
            grid.shape,
            grid.resolution,
            [path_vector],
        )

        diagnostics = {}
        compute_melt_mask_grid(
            grid.origin,
            grid.shape,
            grid.resolution,
            melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=offsets,
            candidate_indices=indices,
            spectral_error_fraction=0.5,
            diagnostics=diagnostics,
        )
        loose_points = diagnostics["full_spectral_table_points"]

        compute_melt_mask_grid(
            grid.origin,
            grid.shape,
            grid.resolution,
            melt_pool,
            active,
            tile_size=tile_size,
            candidate_offsets=offsets,
            candidate_indices=indices,
            spectral_error_fraction=0.125,
            diagnostics=diagnostics,
        )
        assert diagnostics["full_spectral_table_points"] > loose_points

    def test_spatial_index_retains_vector_whose_melt_envelope_reaches_grid(
        self, constant_melt_pool
    ):
        near = PathVector(
            np.array([0.0, 120.0e-6, 0.0]),
            np.array([1.0e-3, 120.0e-6, 0.0]),
            0.0,
            1.0,
        )
        far = PathVector(
            np.array([0.0, 1.0e-3, 0.0]),
            np.array([1.0e-3, 1.0e-3, 0.0]),
            0.0,
            1.0,
        )
        for vector in (near, far):
            vector.set_coordinate_frame()
            vector.set_melt_pool_properties(constant_melt_pool)
        active, *_ = build_spatial_index(
            np.zeros(3),
            (101, 21, 21),
            RESOLUTION,
            [near, far],
        )
        assert near in active
        assert far not in active

    def test_supplied_dimension_phases_remain_independent(self):
        width = np.array([[WIDTH, 0.0, 0.1], [1.0e-6, 1.0, 0.2]])
        depth = np.array([[DEPTH, 0.0, 0.3]])
        height = np.array([[HEIGHT, 0.0, 0.4], [1.0e-6, 2.0, 0.5]])
        melt_pool = MeltPool(
            width,
            depth,
            height,
            WIDTH,
            DEPTH,
            HEIGHT,
            2.0,
            2.0,
            2.0,
            False,
        )
        vector = PathVector(np.zeros(3), np.ones(3), 0.0, 1.0)
        vector.set_coordinate_frame()
        vector.set_melt_pool_properties(melt_pool)
        np.testing.assert_array_equal(vector.width_phases, width[:, 2])
        np.testing.assert_array_equal(vector.depth_phases, depth[:, 2])
        np.testing.assert_array_equal(vector.height_phases, height[:, 2])
