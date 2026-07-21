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
import pandas as pd
import tempfile
import vtk
import xml.etree.ElementTree as ET
from pathlib import Path

# Import the module under test
from raptor.api import (
    create_grid,
    create_path_vectors,
    compute_required_modes,
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
    t = np.arange(100, dtype=np.float64) / 100.0
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

        assert len(path_vectors) > 0
        assert min(vector.start_point[2] for vector in path_vectors) == 0.0

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

    def test_compute_spectral_components_mean_value(
        self, sample_time_series_data, sample_voxel_resolution
    ):
        """Test that mode 0 matches the mean of input data."""
        spectral_array = compute_spectral_components(
            sample_time_series_data, n_modes=None, tolerance=sample_voxel_resolution
        )
        expected_mean = sample_time_series_data[:, 1].mean()

        np.testing.assert_allclose(spectral_array[0, 0], expected_mean)

        # assert spectral_array[1, 1] == pytest.approx(2000.0)

    def test_compute_spectral_components_invalid_input(self):
        """Test spectral component computation with invalid input."""
        with pytest.raises(ValueError, match="shape"):
            compute_spectral_components(np.array([[0.0, 1.0]]), 1)
        with pytest.raises(ValueError, match="shape"):
            compute_spectral_components(np.ones((4, 1)), 2)
        with pytest.raises(ValueError, match="exactly one"):
            compute_spectral_components(np.ones((4, 2)))


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
        one_mode = np.array([[1.0e-4, 0.0, 0.0]])
        three_modes = np.array(
            [[1.0e-4, 0.0, 0.0], [1.0e-6, 1.0, 0.0], [1.0e-6, 2.0, 0.0]]
        )
        melt_pool = create_melt_pool(
            {
                "width": (one_mode, 1, 1.0, 2.0),
                "depth": (three_modes, 3, 1.0, 2.0),
                "height": (one_mode, 1, 1.0, 2.0),
            },
            enable_random_phases=False,
        )
        assert melt_pool.width_oscillations.shape == (3, 3)
        assert melt_pool.depth_oscillations.shape == (3, 3)
        assert melt_pool.height_oscillations.shape == (3, 3)
        np.testing.assert_array_equal(melt_pool.width_oscillations[1:], 0.0)

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
        assert melt_pool.width_oscillations.shape == melt_pool.depth_oscillations.shape
        assert melt_pool.depth_oscillations.shape == melt_pool.height_oscillations.shape

    def test_create_melt_pool_shape_factors(self, sample_melt_pool_dict):
        """Test that shape factors are correctly set."""
        melt_pool = create_melt_pool(sample_melt_pool_dict, enable_random_phases=False)

        assert melt_pool.depth_shape_factor == 2.0
        assert melt_pool.height_shape_factor == 2.0

    def test_create_melt_pool_invalid_data_shape(self):
        """Test melt pool creation with invalid data shape."""
        invalid = np.ones((5, 4))
        data = {key: (invalid, 2, 1.0, 2.0) for key in ("width", "depth", "height")}
        with pytest.raises(ValueError, match="Unsupported data shape"):
            create_melt_pool(data, enable_random_phases=False)


# =============================================================================
# Tests for compute_porosity
# =============================================================================


class TestComputePorosity:
    """Test cases for the compute_porosity function."""

    def test_compute_porosity_basic(self, minimal_simulation):
        """Test basic porosity computation."""
        grid, vectors, melt_pool = minimal_simulation
        result = compute_porosity(grid, vectors, melt_pool, jit_warmup=False)
        assert np.any(result != 0)

    def test_compute_porosity_with_warmup(self, minimal_simulation, capsys):
        """Test porosity computation with JIT warmup enabled."""
        grid, vectors, melt_pool = minimal_simulation
        compute_porosity(grid, vectors, melt_pool, jit_warmup=True)
        assert "JIT warmup complete" in capsys.readouterr().out

    def test_compute_porosity_without_warmup(self, minimal_simulation, capsys):
        """Test porosity computation with JIT warmup disabled."""
        grid, vectors, melt_pool = minimal_simulation
        compute_porosity(grid, vectors, melt_pool, jit_warmup=False)
        assert "JIT Warmup" not in capsys.readouterr().out

    def test_compute_porosity_output_shape(self, minimal_simulation):
        """Test that output porosity field has correct shape."""
        grid, vectors, melt_pool = minimal_simulation
        result = compute_porosity(grid, vectors, melt_pool, jit_warmup=False)
        assert result.shape == grid.shape

    def test_compute_porosity_output_dtype(self, minimal_simulation):
        """Test that output porosity field has correct dtype."""
        grid, vectors, melt_pool = minimal_simulation
        result = compute_porosity(grid, vectors, melt_pool, jit_warmup=False)
        assert result.dtype == np.int8

    def test_compute_porosity_single_vector(self, minimal_simulation):
        """Test porosity computation with a single path vector."""
        grid, vectors, melt_pool = minimal_simulation
        result = compute_porosity(grid, vectors[:1], melt_pool, jit_warmup=False)
        assert result.shape == grid.shape


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
        root = ET.parse(output_path).getroot()
        data_array = root.find("./ImageData/Piece/PointData/DataArray")
        assert data_array is not None
        assert data_array.attrib["format"] == "binary"
        assert root.find("AppendedData") is None

    def test_write_vtk_file_creation(self, temp_output_dir):
        """Test that VTK file is created at specified path."""
        origin = np.array([0.0, 0.0, 0.0])
        porosity = np.ones((5, 5, 5), dtype=np.int8)
        output_path = temp_output_dir / "porosity.vti"

        write_vtk(origin, 0.01, porosity, str(output_path))

        assert output_path.exists()

    def test_write_vtk_different_origins(self, temp_output_dir):
        """Test VTK writing with different origin points."""
        origin = np.array([1.0, -2.0, 3.0])
        output_path = temp_output_dir / "origin.vti"
        write_vtk(origin, 0.25, np.ones((2, 3, 4), dtype=np.int8), output_path)
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(output_path))
        reader.Update()
        np.testing.assert_allclose(reader.GetOutput().GetOrigin(), origin)

    def test_write_vtk_different_resolutions(self, temp_output_dir):
        """Test VTK writing with different voxel resolutions."""
        output_path = temp_output_dir / "spacing.vti"
        write_vtk(np.zeros(3), 2.5e-6, np.ones((2, 2, 2), dtype=np.int8), output_path)
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(output_path))
        reader.Update()
        np.testing.assert_allclose(reader.GetOutput().GetSpacing(), [2.5e-6] * 3)


# =============================================================================
# Tests for compute_morphology
# =============================================================================


class TestComputeMorphology:
    """Test cases for the compute_morphology function."""

    def test_compute_morphology_basic(self):
        """Test basic morphology computation."""
        porosity = np.ones((20, 20, 20), dtype=np.uint8)
        porosity[5:8, 5:8, 5:8] = 0
        porosity[15:18, 15:18, 15:18] = 0

        morphology_fields = ["area", "centroid"]
        properties = compute_morphology(porosity, 0.01, morphology_fields)

        assert len(properties["area"]) == 2
        assert properties["centroid-0"].shape == (2,)

    def test_compute_morphology_single_pore(self):
        """Test morphology computation with a single pore."""
        porosity = np.ones((8, 8, 8), dtype=np.uint8)
        porosity[2:4, 2:4, 2:4] = 0
        properties = compute_morphology(porosity, 1.0, ["area"])
        np.testing.assert_array_equal(properties["area"], [8.0])

    def test_compute_morphology_multiple_pores(self):
        """Test morphology computation with multiple pores."""
        porosity = np.ones((10, 10, 10), dtype=np.uint8)
        porosity[1:3, 1:3, 1:3] = 0
        porosity[7:9, 7:9, 7:9] = 0
        properties = compute_morphology(porosity, 1.0, ["area"])
        np.testing.assert_array_equal(np.sort(properties["area"]), [8.0, 8.0])

    def test_compute_morphology_no_pores(self):
        """Test morphology computation with no pores."""
        porosity = np.ones((10, 10, 10), dtype=np.uint8)

        properties = compute_morphology(porosity, 0.01, ["area"])

        assert properties["area"].size == 0

    def test_compute_morphology_all_fields(self):
        """Test morphology computation with all available fields."""
        porosity = np.ones((8, 8, 8), dtype=np.uint8)
        porosity[2:5, 2:5, 2:5] = 0
        fields = ["area", "centroid", "equivalent_diameter_area"]
        properties = compute_morphology(porosity, 1.0, fields)
        assert set(properties) == {
            "area",
            "centroid-0",
            "centroid-1",
            "centroid-2",
            "equivalent_diameter_area",
        }


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
        properties = {"area": np.array([1.0, 2.0]), "label": np.array([3, 4])}
        output_path = temp_output_dir / "format.csv"
        write_morphology(properties, output_path)
        frame = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(frame, pd.DataFrame(properties))

    def test_write_morphology_empty_properties(self, temp_output_dir):
        """Test morphology writing with empty properties."""
        output_path = temp_output_dir / "empty.csv"
        result = write_morphology({"area": np.array([])}, output_path)
        assert result is None
        assert not output_path.exists()

    def test_write_morphology_column_headers(self, temp_output_dir):
        """Test that column headers match property keys."""
        output_path = temp_output_dir / "headers.csv"
        write_morphology(
            {"area": np.array([1.0]), "equivalent_diameter": np.array([2.0])},
            output_path,
        )
        assert list(pd.read_csv(output_path).columns) == [
            "area",
            "equivalent_diameter",
        ]


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
