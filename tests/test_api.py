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

This module contains unit tests for all public API functions in the raptor.api module,
including grid creation, path vector generation, spectral component computation,
melt pool creation, porosity computation, and VTK output generation.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

# Import the module under test
from raptor.api import (
    create_grid,
    create_path_vectors,
    compute_spectral_components,
    create_melt_pool,
    compute_porosity,
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
    return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]])  # min point  # max point


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
    t = np.linspace(0, 1, 100)
    values = 0.0001 + 0.00002 * np.sin(2 * np.pi * 5 * t)
    return np.column_stack([t, values])


@pytest.fixture
def sample_spectral_components():
    """Fixture providing sample spectral components."""
    # Format: [amplitude, frequency, phase]
    return np.array(
        [
            [0.0001, 0.0, 0.0],  # mode 0 (mean)
            [0.00002, 5.0, 0.0],  # mode 1
            [0.00001, 10.0, np.pi / 2],  # mode 2
        ],
        dtype=np.float64,
    )


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
        assert grid.voxels.shape == (grid.shape[0] * grid.shape[1] * grid.shape[2], 3)

    def test_create_grid_with_path_vectors(
        self, sample_voxel_resolution, sample_path_vectors
    ):
        """Test grid creation with path vectors."""
        grid = create_grid(sample_voxel_resolution, path_vectors=sample_path_vectors)

        assert isinstance(grid, Grid)
        assert grid.resolution == sample_voxel_resolution
        assert np.all(grid.origin == np.array([0.0, 0.0, 0.0]))

    def test_create_grid_invalid_resolution(self, bound_box=sample_bound_box):
        """Test grid creation with invalid voxel resolution."""
        with pytest.raises(ValueError):
            create_grid(-0.01, bound_box=bound_box)

    def test_create_grid_invalid_bound_box(self, sample_voxel_resolution):
        """Test grid creation with invalid bounding box."""
        invalid_bound_box = np.array([[0, 0, 0], [1, -1, 1]])
        with pytest.raises(ValueError):
            create_grid(sample_voxel_resolution, bound_box=invalid_bound_box)

    def test_create_grid_invalid_path_vectors(self, sample_voxel_resolution):
        """Test grid creation with invalid path vectors."""
        invalid_path_vectors = [123, "invalid", None]
        with pytest.raises(ValueError):
            create_grid(sample_voxel_resolution, path_vectors=invalid_path_vectors)


# =============================================================================
# Tests for create_path_vectors
# =============================================================================


class TestCreatePathVectors:
    """Test cases for the create_path_vectors function."""

    def test_create_path_vectors_basic(
        self, sample_bound_box, sample_process_parameters
    ):
        """Test basic path vector generation."""
        path_vectors = create_path_vectors(
            sample_bound_box, **sample_process_parameters
        )

        assert isinstance(path_vectors, list)
        assert len(path_vectors) > 0
        assert all(isinstance(pv, PathVector) for pv in path_vectors)

    def test_create_path_vectors_single_layer(
        self, sample_bound_box, sample_process_parameters
    ):
        """Test path vector generation for a single layer."""
        params = sample_process_parameters.copy()
        params["extra_layers"] = 0

        path_vectors = create_path_vectors(sample_bound_box, **params)

        # TODO: Verify single layer generation
        assert len(path_vectors) > 0

    def test_create_path_vectors_multiple_layers(
        self, sample_bound_box, sample_process_parameters
    ):
        """Test path vector generation for multiple layers."""
        params = sample_process_parameters.copy()
        params["extra_layers"] = 3

        path_vectors = create_path_vectors(sample_bound_box, **params)

        # TODO: Verify multiple layer generation
        assert len(path_vectors) > 0

    def test_create_path_vectors_rotation(
        self, sample_bound_box, sample_process_parameters
    ):
        """Test path vector generation with rotation."""
        # TODO: Test different rotation angles
        pass

    def test_create_path_vectors_hatch_spacing(
        self, sample_bound_box, sample_process_parameters
    ):
        """Test path vector generation with different hatch spacings."""
        # TODO: Test effect of hatch spacing on vector count
        pass

    def test_create_path_vectors_invalid_parameters(self, sample_bound_box):
        """Test path vector generation with invalid parameters."""
        # TODO: Test with negative or invalid values
        pass


# =============================================================================
# Tests for compute_spectral_components
# =============================================================================


class TestComputeSpectralComponents:
    """Test cases for the compute_spectral_components function."""

    def test_compute_spectral_components_basic(self, sample_time_series_data):
        """Test basic spectral component computation."""
        n_modes = 3
        spectral_array = compute_spectral_components(sample_time_series_data, n_modes)

        assert isinstance(spectral_array, np.ndarray)
        assert spectral_array.shape == (n_modes, 3)
        assert spectral_array.dtype == np.float64

    def test_compute_spectral_components_single_mode(self, sample_time_series_data):
        """Test spectral component computation with single mode."""
        n_modes = 1
        spectral_array = compute_spectral_components(sample_time_series_data, n_modes)

        assert spectral_array.shape == (1, 3)
        assert spectral_array[0, 1] == 0  # frequency should be 0
        assert spectral_array[0, 2] == 0  # phase should be 0

    def test_compute_spectral_components_multiple_modes(self, sample_time_series_data):
        """Test spectral component computation with multiple modes."""
        for n_modes in [2, 5, 10]:
            spectral_array = compute_spectral_components(
                sample_time_series_data, n_modes
            )
            assert spectral_array.shape == (n_modes, 3)

    def test_compute_spectral_components_mean_value(self, sample_time_series_data):
        """Test that mode 0 matches the mean of input data."""
        spectral_array = compute_spectral_components(sample_time_series_data, 3)
        expected_mean = sample_time_series_data[:, 1].mean()

        np.testing.assert_allclose(spectral_array[0, 0], expected_mean)

    def test_compute_spectral_components_invalid_input(self):
        """Test spectral component computation with invalid input."""
        # TODO: Test with malformed input data
        pass


# =============================================================================
# Tests for create_melt_pool
# =============================================================================


class TestCreateMeltPool:
    """Test cases for the create_melt_pool function."""

    def test_create_melt_pool_basic(self, sample_melt_pool_dict):
        """Test basic melt pool creation."""
        melt_pool = create_melt_pool(sample_melt_pool_dict, enable_random_phases=False)

        assert isinstance(melt_pool, MeltPool)
        assert melt_pool.enable_random_phases == False

    def test_create_melt_pool_random_phases(self, sample_melt_pool_dict):
        """Test melt pool creation with random phases enabled."""
        melt_pool = create_melt_pool(sample_melt_pool_dict, enable_random_phases=True)

        assert melt_pool.enable_random_phases == True

    def test_create_melt_pool_spectral_input(self, sample_spectral_components):
        """Test melt pool creation with spectral component input."""
        melt_pool_dict = {
            "width": (sample_spectral_components, 3, 1.0, 2.0),
            "depth": (sample_spectral_components, 3, 1.0, 2.0),
            "height": (sample_spectral_components, 3, 1.0, 2.0),
        }

        melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=False)

        assert isinstance(melt_pool, MeltPool)

    def test_create_melt_pool_mode_padding(self):
        """Test that melt pool correctly pads modes to match maximum."""
        # TODO: Create inputs with different mode counts
        pass

    def test_create_melt_pool_scaling(self, sample_time_series_data):
        """Test that scaling is correctly applied."""
        scale_factor = 2.0
        melt_pool_dict = {
            "width": (sample_time_series_data, 3, scale_factor, 2.0),
            "depth": (sample_time_series_data, 3, 1.0, 2.0),
            "height": (sample_time_series_data, 3, 1.0, 2.0),
        }

        melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=False)

        # TODO: Verify scaling is applied correctly
        assert isinstance(melt_pool, MeltPool)

    def test_create_melt_pool_shape_factors(self, sample_melt_pool_dict):
        """Test that shape factors are correctly set."""
        melt_pool = create_melt_pool(sample_melt_pool_dict, enable_random_phases=False)

        assert melt_pool.depth_shape_factor == 2.0
        assert melt_pool.height_shape_factor == 2.0

    def test_create_melt_pool_invalid_data_shape(self):
        """Test melt pool creation with invalid data shape."""
        # TODO: Test with data that is not Nx2 or Nx3
        pass


# =============================================================================
# Tests for compute_porosity
# =============================================================================


class TestComputePorosity:
    """Test cases for the compute_porosity function."""

    def test_compute_porosity_basic(self):
        """Test basic porosity computation."""
        # TODO: Create minimal grid, path vectors, and melt pool
        pass

    def test_compute_porosity_with_warmup(self):
        """Test porosity computation with JIT warmup enabled."""
        # TODO: Test with jit_warmup=True
        pass

    def test_compute_porosity_without_warmup(self):
        """Test porosity computation with JIT warmup disabled."""
        # TODO: Test with jit_warmup=False
        pass

    def test_compute_porosity_output_shape(self):
        """Test that output porosity field has correct shape."""
        # TODO: Verify output shape matches grid shape
        pass

    def test_compute_porosity_output_dtype(self):
        """Test that output porosity field has correct dtype."""
        # TODO: Verify output dtype is int8
        pass

    def test_compute_porosity_single_vector(self):
        """Test porosity computation with a single path vector."""
        # TODO: Test minimal case
        pass


# =============================================================================
# Tests for write_vtk
# =============================================================================


class TestWriteVtk:
    """Test cases for the write_vtk function."""

    def test_write_vtk_basic(self, temp_output_dir):
        """Test basic VTK file writing."""
        origin = np.array([0.0, 0.0, 0.0])
        voxel_resolution = 0.01
        porosity = np.zeros((10, 10, 10), dtype=np.int8)
        porosity[5, 5, 5] = 1

        output_path = temp_output_dir / "test_output.vti"

        write_vtk(origin, voxel_resolution, porosity, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_write_vtk_file_creation(self, temp_output_dir):
        """Test that VTK file is created at specified path."""
        origin = np.array([0.0, 0.0, 0.0])
        porosity = np.ones((5, 5, 5), dtype=np.int8)
        output_path = temp_output_dir / "porosity.vti"

        write_vtk(origin, 0.01, porosity, str(output_path))

        assert output_path.exists()

    def test_write_vtk_different_origins(self, temp_output_dir):
        """Test VTK writing with different origin points."""
        # TODO: Test with various origin coordinates
        pass

    def test_write_vtk_different_resolutions(self, temp_output_dir):
        """Test VTK writing with different voxel resolutions."""
        # TODO: Test with various resolutions
        pass

    def test_write_vtk_large_array(self, temp_output_dir):
        """Test VTK writing with large porosity array."""
        # TODO: Test with larger array sizes
        pass

    def test_write_vtk_invalid_path(self):
        """Test VTK writing with invalid output path."""
        # TODO: Test with invalid file path
        pass


# =============================================================================
# Tests for compute_morphology
# =============================================================================


class TestComputeMorphology:
    """Test cases for the compute_morphology function."""

    def test_compute_morphology_basic(self):
        """Test basic morphology computation."""
        porosity = np.zeros((20, 20, 20), dtype=np.uint8)
        porosity[5:8, 5:8, 5:8] = 1  # Add a pore
        porosity[15:18, 15:18, 15:18] = 1  # Add another pore

        morphology_fields = ["area", "centroid"]
        properties = compute_morphology(porosity, 0.01, morphology_fields)

        assert isinstance(properties, (dict, np.ndarray))
        # TODO: Add more specific assertions

    def test_compute_morphology_single_pore(self):
        """Test morphology computation with a single pore."""
        # TODO: Create porosity with single isolated pore
        pass

    def test_compute_morphology_multiple_pores(self):
        """Test morphology computation with multiple pores."""
        # TODO: Create porosity with multiple isolated pores
        pass

    def test_compute_morphology_no_pores(self):
        """Test morphology computation with no pores."""
        porosity = np.zeros((10, 10, 10), dtype=np.uint8)

        properties = compute_morphology(porosity, 0.01, ["area"])

        # TODO: Verify behavior with no pores
        assert isinstance(properties, (dict, np.ndarray))

    def test_compute_morphology_all_fields(self):
        """Test morphology computation with all available fields."""
        # TODO: Test with comprehensive list of morphology fields
        pass


# =============================================================================
# Tests for write_morphology
# =============================================================================


class TestWriteMorphology:
    """Test cases for the write_morphology function."""

    def test_write_morphology_basic(self, temp_output_dir):
        """Test basic morphology file writing."""
        properties = {
            "area": np.array([1.0, 2.0, 3.0]),
            "centroid-0": np.array([0.5, 1.5, 2.5]),
            "centroid-1": np.array([0.5, 1.5, 2.5]),
            "centroid-2": np.array([0.5, 1.5, 2.5]),
        }

        output_path = temp_output_dir / "morphology.csv"

        write_morphology(properties, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_write_morphology_file_format(self, temp_output_dir):
        """Test that morphology file has correct CSV format."""
        # TODO: Parse and verify CSV structure
        pass

    def test_write_morphology_empty_properties(self, temp_output_dir):
        """Test morphology writing with empty properties."""
        # TODO: Test edge case with no defects
        pass

    def test_write_morphology_column_headers(self, temp_output_dir):
        """Test that column headers match property keys."""
        # TODO: Verify CSV headers
        pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestApiIntegration:
    """Integration tests combining multiple API functions."""

    def test_full_workflow_basic(
        self,
        sample_voxel_resolution,
        sample_bound_box,
        sample_process_parameters,
        sample_melt_pool_dict,
        temp_output_dir,
    ):
        """Test complete workflow from grid creation to VTK output."""
        grid = create_grid(sample_voxel_resolution, bound_box=sample_bound_box)
        path_vectors = create_path_vectors(
            sample_bound_box, **sample_process_parameters
        )
        melt_pool = create_melt_pool(sample_melt_pool_dict, enable_random_phases=False)
        porosity = compute_porosity(grid, path_vectors, melt_pool, jit_warmup=True)
        output_path = temp_output_dir / "full_workflow.vti"
        write_vtk(grid.origin, grid.resolution, porosity, str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_full_workflow_with_morphology(
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
        melt_pool = create_melt_pool(sample_melt_pool_dict, enable_random_phases=False)
        porosity = compute_porosity(grid, path_vectors, melt_pool, jit_warmup=True)
        morphology = compute_morphology(
            porosity, sample_voxel_resolution, sample_morphology_fields
        )
        morphology_output_path = temp_output_dir / "morphology.csv"
        write_morphology(morphology, str(morphology_output_path))
        assert morphology_output_path.exists()
        assert morphology_output_path.stat().st_size > 0
