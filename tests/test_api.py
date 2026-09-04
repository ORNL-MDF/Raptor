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
"""
Test suite for raptor.api module.

This module tests the public API, including grid and path generation,
spectral components, melt pools, porosity, and VTK output.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import vtk
from pathlib import Path

# Import the module under test
from raptor.api import (
    create_grid,
    create_path_vectors,
    compute_required_modes,
    compute_spectral_components,
    create_melt_pool,
    compute_porosity,
    compute_phase_histogram,
    write_vtk,
    compute_morphology,
    write_morphology,
)
from raptor.structures import Grid, PathVector, MeltPool


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_bound_box():
    """Fixture providing a sample bounding box for testing."""
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]]
    )  # min point  # max point


@pytest.fixture
def sample_voxel_resolution():
    """Fixture providing a sample voxel resolution."""
    return 0.01


@pytest.fixture
def sample_path_vectors():
    """Fixture providing sample path vectors."""
    start_point = np.array([0.0, 0.0, 0.0])
    end_point = np.array([1.0, 1.0, 0.0])
    start_time = 0.0
    end_time = 1.0
    path_vector = PathVector(
        start_point=start_point,
        end_point=end_point,
        start_time=start_time,
        end_time=end_time,
    )
    return [path_vector]


@pytest.fixture
def sample_process_parameters():
    """Fixture providing sample process parameters."""
    return {
        "power": 200.0,
        "scan_speed": 1.0,
        "hatch_spacing": 0.1,
        "layer_height": 0.05,
        "rotation": 67.0,
        "scan_extension": 0.1,
        "extra_layers": 0,
    }


@pytest.fixture
def sample_time_series_data():
    """Fixture providing sample time series data for melt pool."""
    t = np.arange(100, dtype=np.float64) / 100.0
    values = 0.0001 + 0.00002 * np.sin(2 * np.pi * 5 * t)
    return np.column_stack([t, values])


@pytest.fixture
def sample_melt_pool_dict(sample_time_series_data):
    """Fixture providing a sample melt pool dictionary."""
    return {
        "width": (sample_time_series_data, 3, 1.0, 2.0),
        "depth": (sample_time_series_data, 3, 1.0, 2.0),
        "height": (sample_time_series_data, 3, 1.0, 2.0),
    }


@pytest.fixture
def sample_morphology_fields():
    """Fixture providing sample morphology fields."""
    return ["area", "centroid"]


@pytest.fixture
def temp_output_dir():
    """Fixture providing a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_simulation():
    resolution = 10.0e-6
    bound_box = np.array([[0.0, 0.0, 0.0], [20.0e-6, 20.0e-6, 10.0e-6]])
    grid = create_grid(resolution, bound_box=bound_box)

    vector = PathVector(
        np.array([0.0, 10.0e-6, 0.0], dtype=np.float64),
        np.array([20.0e-6, 10.0e-6, 0.0], dtype=np.float64),
        0.0,
        1.0,
    )
    vector.set_coordinate_frame()

    width = np.array([[20.0e-6, 0.0, 0.0]], dtype=np.float64)
    depth = np.array([[10.0e-6, 0.0, 0.0]], dtype=np.float64)
    height = np.array([[10.0e-6, 0.0, 0.0]], dtype=np.float64)
    melt_pool = MeltPool(
        width,
        depth,
        height,
        20.0e-6,
        10.0e-6,
        10.0e-6,
        2.0,
        2.0,
        2.0,
        False,
    )
    return grid, [vector], melt_pool


# =============================================================================
# Tests for create_grid
# =============================================================================


class TestCreateGrid:
    """Test cases for the create_grid function."""

    def test_create_grid_with_bound_box(
        self, sample_voxel_resolution, sample_bound_box
    ):
        """Test grid creation with a bounding box."""
        grid = create_grid(sample_voxel_resolution, bound_box=sample_bound_box)

        assert isinstance(grid, Grid)
        assert grid.resolution == sample_voxel_resolution
        assert grid.origin.shape == (3,)
        assert grid.shape[0] * grid.shape[1] * grid.shape[2] > 0
        assert grid.voxels.shape == (
            grid.shape[0] * grid.shape[1] * grid.shape[2],
            3,
        )

    def test_create_grid_with_path_vectors(
        self, sample_voxel_resolution, sample_path_vectors
    ):
        """Test grid creation with path vectors."""
        grid = create_grid(
            sample_voxel_resolution, path_vectors=sample_path_vectors
        )

        assert isinstance(grid, Grid)
        assert grid.resolution == sample_voxel_resolution
        assert np.all(grid.origin == np.array([0.0, 0.0, 0.0]))

    def test_grid_coordinates_are_lazy(
        self, sample_voxel_resolution, sample_bound_box
    ):
        grid = create_grid(sample_voxel_resolution, bound_box=sample_bound_box)
        assert grid._voxels is None
        _ = grid.voxels
        assert grid._voxels is not None

    @pytest.mark.parametrize("resolution", [-0.01, 0.0, np.nan, np.inf])
    def test_create_grid_invalid_resolution(self, resolution, sample_bound_box):
        """Test grid creation with invalid voxel resolution."""
        with pytest.raises(ValueError, match="finite and positive"):
            create_grid(resolution, bound_box=sample_bound_box)

    def test_create_grid_invalid_bound_box(self, sample_voxel_resolution):
        """Test grid creation with invalid bounding box."""
        invalid_bound_box = np.array([[0, 0, 0], [1, -1, 1]])
        with pytest.raises(ValueError):
            create_grid(sample_voxel_resolution, bound_box=invalid_bound_box)

    def test_create_grid_invalid_path_vectors(self, sample_voxel_resolution):
        """Test grid creation with invalid path vectors."""
        invalid_path_vectors = [123, "invalid", None]
        with pytest.raises(ValueError):
            create_grid(
                sample_voxel_resolution, path_vectors=invalid_path_vectors
            )

    def test_create_grid_rejects_empty_path_list(self, sample_voxel_resolution):
        with pytest.raises(ValueError, match="at least one"):
            create_grid(sample_voxel_resolution, path_vectors=[])


# =============================================================================
# Tests for create_path_vectors
# =============================================================================


class TestCreatePathVectors:
    """Test cases for the create_path_vectors function."""

    def test_generated_paths_respect_nonzero_minimum_z(
        self, sample_process_parameters
    ):
        bound_box = np.array([[0.0, 0.0, 2.0], [1.0, 1.0, 2.1]])
        path_vectors = create_path_vectors(
            bound_box,
            **sample_process_parameters,
        )

        assert path_vectors
        assert path_vectors[0].start_point[2] == pytest.approx(2.0)
        assert min(vector.start_point[2] for vector in path_vectors) >= 2.0

    def test_create_path_vectors_multiple_layers(
        self, sample_bound_box, sample_process_parameters
    ):
        """Test path vector generation for multiple layers."""
        params = sample_process_parameters.copy()
        params["extra_layers"] = 3

        path_vectors = create_path_vectors(sample_bound_box, **params)

        base_params = sample_process_parameters.copy()
        base_params["extra_layers"] = 0
        base_vectors = create_path_vectors(sample_bound_box, **base_params)
        assert len(path_vectors) > len(base_vectors)
        assert max(v.start_point[2] for v in path_vectors) > max(
            v.start_point[2] for v in base_vectors
        )

    def test_generated_vector_timeline_is_contiguous(
        self, sample_bound_box, sample_process_parameters
    ):
        path_vectors = create_path_vectors(
            sample_bound_box, **sample_process_parameters
        )

        for previous, current in zip(path_vectors, path_vectors[1:]):
            assert current.start_time == pytest.approx(previous.end_time)
        assert path_vectors[-1].end_time == pytest.approx(
            sum(vector.duration for vector in path_vectors)
        )

    def test_create_path_vectors_rotation(
        self, sample_bound_box, sample_process_parameters
    ):
        """Test path vector generation with rotation."""
        params = sample_process_parameters.copy()
        params["rotation"] = 90.0
        vectors = create_path_vectors(sample_bound_box, **params)
        points_per_layer = len(
            np.arange(
                sample_bound_box[0, 1] - params["scan_extension"],
                sample_bound_box[1, 1] + params["scan_extension"],
                params["hatch_spacing"],
            )
        )
        np.testing.assert_allclose(vectors[0].e1, [1.0, 0.0, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(
            vectors[points_per_layer].e1, [0.0, 1.0, 0.0], atol=1.0e-12
        )

    def test_create_path_vectors_hatch_spacing(
        self, sample_bound_box, sample_process_parameters
    ):
        """Test path vector generation with different hatch spacings."""
        fine = sample_process_parameters.copy()
        coarse = sample_process_parameters.copy()
        fine["hatch_spacing"] = 0.05
        coarse["hatch_spacing"] = 0.2
        assert len(create_path_vectors(sample_bound_box, **fine)) > len(
            create_path_vectors(sample_bound_box, **coarse)
        )


# =============================================================================
# Tests for compute_spectral_components
# =============================================================================


class TestComputeSpectralComponents:
    """Test cases for the compute_spectral_components function."""

    def test_compute_spectral_components_preserves_standard_deviation(
        self, sample_time_series_data
    ):
        """Test correct amplitude when the signal frequency is retained."""
        spectral_array = compute_spectral_components(sample_time_series_data, 6)
        time = sample_time_series_data[:, 0]
        reconstructed = np.zeros_like(time)

        for amplitude, frequency, phase in spectral_array:
            reconstructed += amplitude * np.cos(
                2.0 * np.pi * frequency * time + phase
            )

        assert reconstructed.mean() == pytest.approx(
            sample_time_series_data[:, 1].mean()
        )
        assert reconstructed.std() == pytest.approx(
            sample_time_series_data[:, 1].std()
        )

    def test_explicit_mode_count_retains_low_frequency_prefix(self):
        """Explicit mode counts preserve the original truncation semantics."""
        sampling_frequency = 10_000.0
        time = np.arange(1000) / sampling_frequency
        values = 1.0 + 0.2 * np.sin(2.0 * np.pi * 2000.0 * time)
        spectral_array = compute_spectral_components(
            np.column_stack([time, values]), 2
        )

        assert spectral_array[1, 1] == pytest.approx(10.0)

    def test_tolerance_selects_minimum_modes_and_meets_rmse(self):
        sampling_frequency = 1000.0
        time = np.arange(1000) / sampling_frequency
        values = (
            2.0
            + 0.30 * np.cos(2.0 * np.pi * 20.0 * time + 0.2)
            + 0.04 * np.cos(2.0 * np.pi * 170.0 * time - 0.4)
        )
        data = np.column_stack([time, values])

        spectral_array = compute_spectral_components(data, tolerance=0.03)
        reconstructed = sum(
            amplitude * np.cos(2.0 * np.pi * frequency * time + phase)
            for amplitude, frequency, phase in spectral_array
        )

        assert spectral_array.shape == (2, 3)
        assert spectral_array[1, 1] == pytest.approx(20.0)
        assert np.sqrt(np.mean((values - reconstructed) ** 2)) <= 0.03
        assert compute_required_modes(data, 0.03) == 2

    def test_mode_cap_retains_highest_energy_tolerance_modes(self):
        sampling_frequency = 1000.0
        time = np.arange(1000) / sampling_frequency
        values = (
            1.0
            + 0.1 * np.cos(2.0 * np.pi * 20.0 * time)
            + 0.9 * np.cos(2.0 * np.pi * 170.0 * time)
        )
        data = np.column_stack([time, values])

        with pytest.warns(RuntimeWarning, match="tolerance will not be met"):
            spectral_array = compute_spectral_components(
                data, n_modes=2, tolerance=0.01
            )

        assert spectral_array.shape == (2, 3)
        assert spectral_array[1, 1] == pytest.approx(170.0)

    def test_tolerance_modes_are_returned_in_frequency_order(self):
        sampling_frequency = 1000.0
        time = np.arange(1000) / sampling_frequency
        values = (
            1.0
            + 0.1 * np.cos(2.0 * np.pi * 20.0 * time)
            + 0.9 * np.cos(2.0 * np.pi * 170.0 * time)
        )

        spectral_array = compute_spectral_components(
            np.column_stack([time, values]), tolerance=0.01
        )

        assert spectral_array[:, 1].tolist() == pytest.approx(
            [0.0, 20.0, 170.0]
        )

    def test_tolerance_zero_reconstructs_even_length_signal(self):
        time = 0.25 + np.arange(100) / 100.0
        values = 3.0 + 0.2 * np.cos(2.0 * np.pi * 5.0 * time)
        values += 0.1 * np.cos(2.0 * np.pi * 50.0 * time)
        spectral_array = compute_spectral_components(
            np.column_stack([time, values]), tolerance=0.0
        )
        reconstructed = sum(
            amplitude * np.cos(2.0 * np.pi * frequency * time + phase)
            for amplitude, frequency, phase in spectral_array
        )
        np.testing.assert_allclose(reconstructed, values, atol=1.0e-13)

    def test_compute_spectral_components_preserves_standard_deviation(
        self, sample_time_series_data
    ):
        """Test correct amplitude when the signal frequency is retained."""
        spectral_array = compute_spectral_components(sample_time_series_data, 6)
        time = sample_time_series_data[:, 0]
        reconstructed = np.zeros_like(time)

        for amplitude, frequency, phase in spectral_array:
            reconstructed += amplitude * np.cos(2.0 * np.pi * frequency * time + phase)

        assert reconstructed.mean() == pytest.approx(
            sample_time_series_data[:, 1].mean()
        )
        assert reconstructed.std() == pytest.approx(sample_time_series_data[:, 1].std())

    def test_explicit_mode_count_retains_low_frequency_prefix(self):
        """Explicit mode counts preserve the original truncation semantics."""
        sampling_frequency = 10_000.0
        time = np.arange(1000) / sampling_frequency
        values = 1.0 + 0.2 * np.sin(2.0 * np.pi * 2000.0 * time)
        spectral_array = compute_spectral_components(np.column_stack([time, values]), 2)

        assert spectral_array[1, 1] == pytest.approx(10.0)

    def test_tolerance_selects_minimum_modes_and_meets_rmse(self):
        sampling_frequency = 1000.0
        time = np.arange(1000) / sampling_frequency
        values = (
            2.0
            + 0.30 * np.cos(2.0 * np.pi * 20.0 * time + 0.2)
            + 0.04 * np.cos(2.0 * np.pi * 170.0 * time - 0.4)
        )
        data = np.column_stack([time, values])

        spectral_array = compute_spectral_components(data, tolerance=0.03)
        reconstructed = sum(
            amplitude * np.cos(2.0 * np.pi * frequency * time + phase)
            for amplitude, frequency, phase in spectral_array
        )

        assert spectral_array.shape == (2, 3)
        assert spectral_array[1, 1] == pytest.approx(20.0)
        assert np.sqrt(np.mean((values - reconstructed) ** 2)) <= 0.03
        assert compute_required_modes(data, 0.03) == 2

    def test_mode_cap_retains_highest_energy_tolerance_modes(self):
        sampling_frequency = 1000.0
        time = np.arange(1000) / sampling_frequency
        values = (
            1.0
            + 0.1 * np.cos(2.0 * np.pi * 20.0 * time)
            + 0.9 * np.cos(2.0 * np.pi * 170.0 * time)
        )
        data = np.column_stack([time, values])

        with pytest.warns(RuntimeWarning, match="tolerance will not be met"):
            spectral_array = compute_spectral_components(
                data, n_modes=2, tolerance=0.01
            )

        assert spectral_array.shape == (2, 3)
        assert spectral_array[1, 1] == pytest.approx(170.0)

    def test_tolerance_modes_are_returned_in_frequency_order(self):
        sampling_frequency = 1000.0
        time = np.arange(1000) / sampling_frequency
        values = (
            1.0
            + 0.1 * np.cos(2.0 * np.pi * 20.0 * time)
            + 0.9 * np.cos(2.0 * np.pi * 170.0 * time)
        )

        spectral_array = compute_spectral_components(
            np.column_stack([time, values]), tolerance=0.01
        )

        assert spectral_array[:, 1].tolist() == pytest.approx([0.0, 20.0, 170.0])

    def test_tolerance_zero_reconstructs_even_length_signal(self):
        time = 0.25 + np.arange(100) / 100.0
        values = 3.0 + 0.2 * np.cos(2.0 * np.pi * 5.0 * time)
        values += 0.1 * np.cos(2.0 * np.pi * 50.0 * time)
        spectral_array = compute_spectral_components(
            np.column_stack([time, values]), tolerance=0.0
        )
        reconstructed = sum(
            amplitude * np.cos(2.0 * np.pi * frequency * time + phase)
            for amplitude, frequency, phase in spectral_array
        )
        np.testing.assert_allclose(reconstructed, values, atol=1.0e-13)

    def test_compute_spectral_components_invalid_input(self):
        """Test spectral component computation with invalid input."""
        with pytest.raises(ValueError, match="shape"):
            compute_spectral_components(np.array([[0.0, 1.0]]), 1)
        with pytest.raises(ValueError, match="shape"):
            compute_spectral_components(np.ones((4, 1)), 2)
        with pytest.raises(ValueError, match="either n_modes"):
            compute_spectral_components(np.ones((4, 2)))

    @pytest.mark.parametrize("n_modes", [0, -1, 1.5])
    def test_compute_spectral_components_rejects_invalid_mode_counts(
        self, sample_time_series_data, n_modes
    ):
        with pytest.raises(ValueError, match="positive integer"):
            compute_spectral_components(
                sample_time_series_data, n_modes=n_modes
            )

    def test_tolerance_rejects_mode_count_above_available_bins(
        self, sample_time_series_data
    ):
        with pytest.raises(ValueError, match="cannot exceed"):
            compute_spectral_components(
                sample_time_series_data,
                n_modes=sample_time_series_data.shape[0],
                tolerance=0.0,
            )


# =============================================================================
# Tests for create_melt_pool
# =============================================================================


class TestCreateMeltPool:
    """Test cases for the create_melt_pool function."""

    def test_create_melt_pool_scales_all_spectral_modes(
        self, sample_time_series_data
    ):
        scale_factor = 2.0
        expected = compute_spectral_components(sample_time_series_data, 3)
        melt_pool = create_melt_pool(
            {
                "width": (sample_time_series_data, 3, scale_factor, 2.0),
                "depth": (sample_time_series_data, 3, 1.0, 2.0),
                "height": (sample_time_series_data, 3, 1.0, 2.0),
            },
            enable_random_phases=False,
        )

        np.testing.assert_allclose(
            melt_pool.width_oscillations[:, 0],
            scale_factor * expected[:, 0],
        )

    def test_create_melt_pool_rejects_spectral_array_input(
        self, sample_time_series_data
    ):
        """Raw (n, 3) spectral arrays are no longer an accepted input."""
        spectral_array = np.ones((3, 3))
        melt_pool_data = {
            "width": (spectral_array, 3, 1.0, 2.0),
            "depth": (sample_time_series_data, 3, 1.0, 2.0),
            "height": (sample_time_series_data, 3, 1.0, 2.0),
        }

        with pytest.raises(ValueError, match="shape \\(n, 2\\)"):
            create_melt_pool(melt_pool_data, enable_random_phases=False)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("data", np.nan, "finite values"),
            ("scale", 0.0, "scale must be finite and positive"),
            ("shape", 0.0, "shape factor must be finite and positive"),
        ],
    )
    def test_create_melt_pool_rejects_invalid_physical_inputs(
        self, sample_time_series_data, field, value, message
    ):
        data = sample_time_series_data.copy()
        scale = 1.0
        shape = 2.0
        if field == "data":
            data[1, 1] = value
        elif field == "scale":
            scale = value
        else:
            shape = value
        melt_pool_data = {
            key: (data, 3, scale, shape) for key in ("width", "depth", "height")
        }

        with pytest.raises(ValueError, match=message):
            create_melt_pool(
                melt_pool_data,
                enable_random_phases=False,
            )

    def test_create_melt_pool_preserves_independent_mode_counts(
        self, sample_time_series_data
    ):
        """Each dimension retains only its independently selected modes."""
        melt_pool = create_melt_pool(
            {
                "width": (sample_time_series_data, 1, 1.0, 2.0),
                "depth": (sample_time_series_data, 3, 1.0, 2.0),
                "height": (sample_time_series_data, 1, 1.0, 2.0),
            },
            enable_random_phases=False,
        )
        assert melt_pool.width_oscillations.shape == (1, 3)
        assert melt_pool.depth_oscillations.shape == (3, 3)
        assert melt_pool.height_oscillations.shape == (1, 3)

    def test_create_melt_pool_scaling(self, sample_time_series_data):
        """Test that scaling is correctly applied."""
        scale_factor = 2.0
        melt_pool_dict = {
            "width": (sample_time_series_data, 3, scale_factor, 2.0),
            "depth": (sample_time_series_data, 3, 1.0, 2.0),
            "height": (sample_time_series_data, 3, 1.0, 2.0),
        }

        melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=False)

        assert melt_pool.width_mean == pytest.approx(
            scale_factor * sample_time_series_data[:, 1].mean()
        )
        assert melt_pool.depth_mean == pytest.approx(
            sample_time_series_data[:, 1].mean()
        )

    def test_create_melt_pool_allows_fixed_and_tolerance_mode_counts(
        self, sample_time_series_data
    ):
        melt_pool = create_melt_pool(
            {
                "width": (sample_time_series_data, 2, 1.0, 2.0),
                "depth": (sample_time_series_data, None, 1.0, 2.0),
                "height": (sample_time_series_data, None, 1.0, 2.0),
            },
            enable_random_phases=False,
            tolerance=1.0e-7,
        )
        assert (
            melt_pool.width_oscillations.shape
            == melt_pool.depth_oscillations.shape
        )
        assert (
            melt_pool.depth_oscillations.shape
            == melt_pool.height_oscillations.shape
        )

    def test_create_melt_pool_rejects_unsupported_width_exponent(
        self, sample_time_series_data
    ):
        data = {
            "width": (sample_time_series_data, 3, 1.0, 1.0),
            "depth": (sample_time_series_data, 3, 1.0, 2.0),
            "height": (sample_time_series_data, 3, 1.0, 2.0),
        }
        with pytest.raises(ValueError, match="width shape factor"):
            create_melt_pool(data, enable_random_phases=False)

    def test_create_melt_pool_invalid_data_shape(self):
        """Test melt pool creation with invalid data shape."""
        invalid = np.ones((5, 4))
        data = {
            key: (invalid, 2, 1.0, 2.0) for key in ("width", "depth", "height")
        }
        with pytest.raises(ValueError, match="shape \\(n, 2\\)"):
            create_melt_pool(data, enable_random_phases=False)


# =============================================================================
# Tests for compute_porosity
# =============================================================================


class TestComputePorosity:
    """Test cases for the compute_porosity function."""

    def test_compute_porosity_contract(self, minimal_simulation):
        grid, vectors, melt_pool = minimal_simulation
        result = compute_porosity(grid, vectors, melt_pool)
        assert np.any(result != 0)
        assert result.shape == grid.shape
        assert result.dtype == np.int8

        explicit_tile = compute_porosity(
            grid,
            vectors,
            melt_pool,
            tile_width=grid.resolution,
        )
        np.testing.assert_array_equal(explicit_tile, result)

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"tile_width": 0.0}, "tile_width"),
            ({"spectral_error_fraction": 0.0}, "spectral_error_fraction"),
            ({"spectral_error_fraction": 1.01}, "spectral_error_fraction"),
            (
                {"memory_limit_mb": 0},
                "memory_limit_mb",
            ),
        ],
    )
    def test_compute_porosity_validates_performance_controls(
        self,
        minimal_simulation,
        options,
        message,
    ):
        grid, vectors, melt_pool = minimal_simulation
        with pytest.raises(ValueError, match=message):
            compute_porosity(
                grid,
                vectors,
                melt_pool,
                **options,
            )

    def test_compute_porosity_random_seed_is_reproducible(
        self, minimal_simulation
    ):
        grid, vectors, melt_pool = minimal_simulation
        melt_pool.enable_random_phases = True
        first = compute_porosity(
            grid,
            vectors,
            melt_pool,
            random_seed=42,
        )
        second = compute_porosity(
            grid,
            vectors,
            melt_pool,
            random_seed=42,
        )
        np.testing.assert_array_equal(first, second)

    def test_compute_porosity_rejects_nonpositive_dimension_history(
        self, minimal_simulation
    ):
        grid, vectors, _ = minimal_simulation
        oscillating_width = np.array(
            [[100.0e-6, 0.0, 0.0], [200.0e-6, 1.0, 0.0]]
        )
        constant = np.array([[20.0e-6, 0.0, 0.0]])
        melt_pool = MeltPool(
            oscillating_width,
            constant,
            constant,
            300.0e-6,
            20.0e-6,
            20.0e-6,
            2.0,
            2.0,
            2.0,
            False,
        )

        with pytest.raises(ValueError, match="must remain positive"):
            compute_porosity(
                grid,
                vectors,
                melt_pool,
            )

    def test_compute_phase_histogram(self):
        phases = np.array(
            [[[0, 1, 1], [2, 3, 3]], [[1, 1, 2], [3, 3, 3]]],
            dtype=np.int8,
        )
        assert compute_phase_histogram(phases) == {
            0: 1,
            1: 4,
            2: 2,
            3: 5,
        }


# =============================================================================
# Tests for write_vtk
# =============================================================================


class TestWriteVtk:
    """Test cases for the write_vtk function."""

    def test_write_vtk_round_trip_is_physical_and_compatible(
        self, temp_output_dir
    ):
        from xml.etree import ElementTree

        origin = np.array([1.0, 2.0, 3.0])
        resolution = 0.25
        porosity = np.zeros((2, 3, 4), dtype=np.int8)
        porosity[1, 2, 3] = 3
        output_path = temp_output_dir / "mapped.vti"

        write_vtk(origin, resolution, porosity, output_path)

        root = ElementTree.parse(output_path).getroot()
        assert root.attrib["compressor"] == "vtkLZ4DataCompressor"
        data_array = root.find(".//DataArray")
        assert data_array is not None
        assert data_array.attrib["format"] == "binary"
        assert root.find("AppendedData") is None

        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(output_path))
        reader.Update()
        output = reader.GetOutput()
        assert output.GetDimensions() == (4, 3, 2)
        assert (
            output.GetPointData().GetScalars().GetDataType()
            == vtk.VTK_SIGNED_CHAR
        )
        expected_direction = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )
        actual_direction = np.array(
            [
                [
                    output.GetDirectionMatrix().GetElement(row, column)
                    for column in range(3)
                ]
                for row in range(3)
            ]
        )
        np.testing.assert_array_equal(actual_direction, expected_direction)
        point_id = output.ComputePointId((3, 2, 1))
        assert output.GetPointData().GetScalars().GetTuple1(point_id) == 3
        np.testing.assert_allclose(
            output.GetPoint(point_id),
            origin + resolution * np.array([1.0, 2.0, 3.0]),
        )

    def test_write_vtk_reports_writer_failure(self, temp_output_dir):
        output_path = temp_output_dir / "missing" / "result.vti"
        with pytest.raises(OSError, match="Unable to write"):
            write_vtk(
                np.zeros(3),
                1.0,
                np.ones((2, 2, 2), dtype=np.int8),
                output_path,
            )


# =============================================================================
# Tests for compute_morphology
# =============================================================================


class TestComputeMorphology:
    """Test cases for the compute_morphology function."""

    def test_compute_morphology_no_pores(self):
        porosity = np.ones((10, 10, 10), dtype=np.uint8)
        properties = compute_morphology(porosity, 0.01, ["area"])
        assert properties["area"].size == 0

    def test_dense_morphology_fallback(self):
        porosity = np.ones((8, 8, 8), dtype=np.uint8)
        porosity[2:5, 2:5, 2:5] = 0
        fields = ["area", "bbox"]
        properties = compute_morphology(porosity, 1.0, fields)
        assert set(properties) == {
            "area",
            "bbox-0",
            "bbox-1",
            "bbox-2",
            "bbox-3",
            "bbox-4",
            "bbox-5",
        }
        np.testing.assert_array_equal(properties["area"], [27.0])

    def test_sparse_morphology_matches_dense_reference(self):
        """Sparse labeling retains the established 26-neighbor semantics."""
        from skimage import measure
        from skimage.morphology import remove_small_objects

        porosity = np.ones((40, 40, 40), dtype=np.uint8)
        porosity[2:5, 3:6, 4:7] = 0
        porosity[10, 10, 10] = 0
        porosity[11, 11, 11] = 0
        porosity[38, 20, 20] = 0
        porosity[39, 21, 21] = 0
        porosity[30, 30, 30] = 0  # Removed by the min-size filter.
        resolution = 0.25
        fields = [
            "label",
            "area",
            "centroid",
            "equivalent_diameter_area",
        ]

        actual = compute_morphology(porosity, resolution, fields)
        dense_mask = remove_small_objects(
            porosity == 0, min_size=2, connectivity=3
        )
        expected = measure.regionprops_table(
            measure.label(dense_mask, connectivity=3),
            spacing=resolution,
            properties=fields,
        )

        assert actual.keys() == expected.keys()
        for field in actual:
            np.testing.assert_allclose(actual[field], expected[field])


# =============================================================================
# Tests for write_morphology
# =============================================================================


class TestWriteMorphology:
    """Test cases for the write_morphology function."""

    def test_write_morphology_empty_properties(self, temp_output_dir):
        """Test morphology writing with empty properties."""
        output_path = temp_output_dir / "empty.csv"
        result = write_morphology({"area": np.array([])}, output_path)
        assert result is None
        assert not output_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestApiIntegration:
    """Integration tests combining multiple API functions."""

    def test_full_workflow(
        self,
        sample_voxel_resolution,
        sample_bound_box,
        sample_process_parameters,
        sample_melt_pool_dict,
        sample_morphology_fields,
        temp_output_dir,
    ):
        """Test complete workflow including morphology analysis."""
        grid = create_grid(sample_voxel_resolution, bound_box=sample_bound_box)
        path_vectors = create_path_vectors(
            sample_bound_box, **sample_process_parameters
        )
        melt_pool = create_melt_pool(
            sample_melt_pool_dict, enable_random_phases=False
        )
        porosity = compute_porosity(grid, path_vectors, melt_pool)
        vtk_output_path = temp_output_dir / "full_workflow.vti"
        write_vtk(grid.origin, grid.resolution, porosity, vtk_output_path)
        morphology = compute_morphology(
            porosity, sample_voxel_resolution, sample_morphology_fields
        )
        morphology_output_path = temp_output_dir / "morphology.csv"
        write_morphology(morphology, str(morphology_output_path))
        assert vtk_output_path.exists()
        assert morphology_output_path.exists()
