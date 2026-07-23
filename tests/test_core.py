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

from raptor.core import (
    compute_distance_to_boundary,
    compute_melt_mask,
    compute_melt_mask_implicit,
)
from raptor.structures import MeltPool, PathVector


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


def distance(y, z, height_shape=2.0, depth_shape=2.0):
    return compute_distance_to_boundary(
        y,
        z,
        WIDTH,
        HEIGHT,
        DEPTH,
        height_shape,
        depth_shape,
        RESOLUTION,
    )


class TestDistanceToBoundary:
    def test_center_is_inside(self):
        result = distance(0.0, 0.0)
        assert np.isfinite(result)
        assert result == -min(WIDTH / 2.0, HEIGHT, DEPTH)

    @pytest.mark.parametrize(
        ("y", "z"),
        [
            (WIDTH / 2.0, 0.0),
            (-WIDTH / 2.0, 0.0),
            (0.0, HEIGHT),
            (0.0, -DEPTH),
        ],
    )
    def test_axis_boundaries_have_zero_distance(self, y, z):
        assert distance(y, z) == pytest.approx(0.0, abs=1.0e-15)

    @pytest.mark.parametrize(
        ("y", "z"),
        [(WIDTH / 4.0, 0.0), (0.0, HEIGHT / 2.0), (0.0, -DEPTH / 2.0)],
    )
    def test_interior_points_are_negative(self, y, z):
        assert distance(y, z) < 0.0

    @pytest.mark.parametrize(
        ("y", "z"),
        [(WIDTH, 0.0), (0.0, 2.0 * HEIGHT), (0.0, -2.0 * DEPTH)],
    )
    def test_exterior_points_are_positive(self, y, z):
        assert distance(y, z) > 0.0

    def test_width_symmetry(self):
        positive = distance(40.0e-6, 20.0e-6)
        negative = distance(-40.0e-6, 20.0e-6)
        assert positive == pytest.approx(negative)

    def test_positive_and_negative_z_use_different_dimensions(self):
        assert distance(0.0, HEIGHT) == pytest.approx(0.0, abs=1.0e-15)
        assert distance(0.0, -DEPTH) == pytest.approx(0.0, abs=1.0e-15)

    def test_parabolic_shape(self):
        y = 0.8 * WIDTH / 2.0
        z = 0.5 * HEIGHT
        assert distance(y, z, height_shape=1.0) > 0.0

    def test_elliptical_shape(self):
        y = 0.8 * WIDTH / 2.0
        z = 0.5 * HEIGHT
        assert distance(y, z, height_shape=2.0) < 0.0

    def test_bell_shape(self):
        y = 0.5 * WIDTH / 2.0
        z = 0.64 * HEIGHT
        assert distance(y, z, height_shape=0.5) > 0.0


def prepare_vector(vector, melt_pool):
    vector.set_melt_pool_properties(melt_pool)
    return vector


def unpack_kernel_arguments(voxels, melt_pool, vectors):
    return (
        voxels,
        RESOLUTION,
        np.zeros(voxels.shape[0], dtype=np.int8),
        np.array([v.start_point for v in vectors]),
        np.array([v.end_point for v in vectors]),
        np.array([v.e0 for v in vectors]),
        np.array([v.e1 for v in vectors]),
        np.array([v.e2 for v in vectors]),
        np.array([v.L0 for v in vectors]),
        np.array([v.L1 for v in vectors]),
        np.array([v.L2 for v in vectors]),
        np.array([v.start_time for v in vectors]),
        np.array([v.end_time for v in vectors]),
        np.array([v.AABB for v in vectors]),
        np.array([v.phases for v in vectors]),
        np.array([v.centroid for v in vectors]),
        np.array([v.distance for v in vectors]),
        melt_pool.width_oscillations[:, 0],
        melt_pool.width_oscillations[:, 1],
        melt_pool.depth_oscillations[:, 0],
        melt_pool.depth_oscillations[:, 1],
        melt_pool.height_oscillations[:, 0],
        melt_pool.height_oscillations[:, 1],
        melt_pool.height_shape_factor,
        melt_pool.depth_shape_factor,
    )


class TestComputeMeltMask:
    def test_center_boundary_and_outside_labels(self, constant_melt_pool, path_vector):
        prepare_vector(path_vector, constant_melt_pool)
        voxels = np.array(
            [
                [0.5e-3, 0.0, 0.0],
                [0.5e-3, WIDTH / 2.0, 0.0],
                [0.5e-3, 2.0 * WIDTH, 0.0],
            ],
            dtype=np.float64,
        )

        result = compute_melt_mask(
            voxels, RESOLUTION, constant_melt_pool, [path_vector]
        )

        np.testing.assert_array_equal(result, np.array([1, 2, 0], dtype=np.int8))

    def test_output_shape_and_dtype(self, constant_melt_pool, path_vector):
        prepare_vector(path_vector, constant_melt_pool)
        voxels = np.zeros((7, 3), dtype=np.float64)
        result = compute_melt_mask(
            voxels, RESOLUTION, constant_melt_pool, [path_vector]
        )
        assert result.shape == (7,)
        assert result.dtype == np.int8

    def test_aabb_culls_distant_voxel(self, constant_melt_pool, path_vector):
        prepare_vector(path_vector, constant_melt_pool)
        voxel = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
        result = compute_melt_mask(voxel, RESOLUTION, constant_melt_pool, [path_vector])
        assert result[0] == 0

    def test_overlapping_boundaries_are_intersections(
        self, constant_melt_pool, path_vector
    ):
        duplicate = PathVector(
            path_vector.start_point.copy(),
            path_vector.end_point.copy(),
            path_vector.start_time,
            path_vector.end_time,
        )
        duplicate.set_coordinate_frame()
        vectors = [path_vector, duplicate]
        for vector in vectors:
            prepare_vector(vector, constant_melt_pool)
        voxel = np.array([[0.5e-3, WIDTH / 2.0, 0.0]], dtype=np.float64)

        result = compute_melt_mask(voxel, RESOLUTION, constant_melt_pool, vectors)

        assert result[0] == 3

    def test_zero_length_vector(self, constant_melt_pool):
        point = np.array([0.5e-3, 0.0, 0.0], dtype=np.float64)
        vector = PathVector(point.copy(), point.copy(), 0.0, 0.0)
        vector.set_coordinate_frame()
        prepare_vector(vector, constant_melt_pool)

        result = compute_melt_mask(
            point.reshape(1, 3), RESOLUTION, constant_melt_pool, [vector]
        )

        assert result[0] == 1

    def test_implicit_kernel_matches_wrapper(self, constant_melt_pool, path_vector):
        prepare_vector(path_vector, constant_melt_pool)
        voxels = np.array(
            [[0.25e-3, 0.0, 0.0], [0.75e-3, WIDTH / 2.0, 0.0]],
            dtype=np.float64,
        )
        expected = compute_melt_mask(
            voxels, RESOLUTION, constant_melt_pool, [path_vector]
        )

        actual = compute_melt_mask_implicit(
            *unpack_kernel_arguments(voxels, constant_melt_pool, [path_vector])
        )

        np.testing.assert_array_equal(actual, expected)
