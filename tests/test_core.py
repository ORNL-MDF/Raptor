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
Test suite for raptor.core module.

This module contains unit tests for the core computation functions including
geometric containment testing, melt mask computation, and the main parallelized
implicit melt mask calculation.
"""

import pytest
import numpy as np
from typing import List
from unittest.mock import Mock, MagicMock

# Import the module under test
from raptor.core import (
    is_inside,
    compute_melt_mask,
    compute_melt_mask_implicit
)
from raptor.structures import MeltPool, PathVector


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_melt_pool():
    """Fixture providing a sample MeltPool object."""
    width_osc = np.array([
        [0.0001, 0.0, 0.0],
        [0.00002, 5.0, 0.0],
        [0.00001, 10.0, 0.0]
    ], dtype=np.float64)
    
    depth_osc = np.array([
        [0.00008, 0.0, 0.0],
        [0.00001, 5.0, 0.0],
        [0.00001, 10.0, 0.0]
    ], dtype=np.float64)
    
    height_osc = np.array([
        [0.00006, 0.0, 0.0],
        [0.00001, 5.0, 0.0],
        [0.00001, 10.0, 0.0]
    ], dtype=np.float64)
    
    return MeltPool(
        width_oscillations=width_osc,
        depth_oscillations=depth_osc,
        height_oscillations=height_osc,
        width_max=0.00012,
        depth_max=0.0001,
        height_max=0.00008,
        width_shape_factor=2.0,
        height_shape_factor=2.0,
        depth_shape_factor=2.0,
        enable_random_phases=False
    )


@pytest.fixture
def sample_path_vector():
    """Fixture providing a sample PathVector object."""
    start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    end = np.array([0.001, 0.0, 0.0], dtype=np.float64)
    
    path_vec = PathVector(
        start_point=start,
        end_point=end,
        start_time=0.0,
        end_time=0.001
    )
    path_vec.set_coordinate_frame()
    
    return path_vec


@pytest.fixture
def sample_path_vectors(sample_path_vector):
    """Fixture providing a list of sample PathVector objects."""
    vectors = []
    
    # Create multiple parallel vectors
    for i in range(3):
        start = np.array([0.0, i * 0.0001, 0.0], dtype=np.float64)
        end = np.array([0.001, i * 0.0001, 0.0], dtype=np.float64)
        
        vec = PathVector(
            start_point=start,
            end_point=end,
            start_time=i * 0.001,
            end_time=(i + 1) * 0.001
        )
        vec.set_coordinate_frame()
        vectors.append(vec)
    
    return vectors


@pytest.fixture
def sample_voxels():
    """Fixture providing a sample voxel array."""
    # Create a small grid of voxels
    x = np.linspace(0, 0.001, 10)
    y = np.linspace(0, 0.0005, 5)
    z = np.linspace(0, 0.0002, 3)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    voxels = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        zz.ravel()
    ])
    
    return voxels.astype(np.float64)


@pytest.fixture
def minimal_voxels():
    """Fixture providing a minimal voxel array for testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [0.0005, 0.0, 0.0],
        [0.001, 0.0, 0.0]
    ], dtype=np.float64)


# =============================================================================
# Tests for is_inside function
# =============================================================================

class TestIsInside:
    """Test cases for the is_inside geometric containment function."""
    
    def test_is_inside_center_point(self):
        """Test that the center point (0,0) is always inside."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        result = is_inside(0.0, 0.0, width, height, depth, 2.0, 2.0)
        
        assert result == True
    
    def test_is_inside_ellipse_shape(self):
        """Test containment with ellipse shape (n=2)."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        # Point on the boundary (should be inside)
        y = width / 2.0
        z = 0.0
        result = is_inside(y, z, width, height, depth, 2.0, 2.0)
        assert result == True
        
        # Point clearly outside
        y = width
        z = height
        result = is_inside(y, z, width, height, depth, 2.0, 2.0)
        assert result == False
    
    def test_is_inside_parabola_shape(self):
        """Test containment with parabola shape (n=1)."""
        # TODO: Test with n=1
        pass
    
    def test_is_inside_bell_shape(self):
        """Test containment with bell-like shape (n=0.5)."""
        # TODO: Test with n=0.5
        pass
    
    def test_is_inside_positive_z(self):
        """Test containment for points with positive z (using height)."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        # Point with positive z
        y = 0.0
        z = height / 2.0  # Should be inside
        result = is_inside(y, z, width, height, depth, 2.0, 2.0)
        assert result == True
    
    def test_is_inside_negative_z(self):
        """Test containment for points with negative z (using depth)."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        # Point with negative z
        y = 0.0
        z = -depth / 2.0  # Should be inside
        result = is_inside(y, z, width, height, depth, 2.0, 2.0)
        assert result == True
    
    def test_is_inside_different_shape_factors(self):
        """Test containment with different shape factors for height and depth."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        # Different shape factors
        result = is_inside(0.00005, 0.00003, width, height, depth, 2.0, 10.0)
        # TODO: Verify different shape factor behavior
    
    def test_is_inside_boundary_cases(self):
        """Test points exactly on the boundary."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        # Point at width boundary (y = width/2, z = 0)
        result = is_inside(width / 2.0, 0.0, width, height, depth, 2.0, 2.0)
        assert result == True
        
        # Point at height boundary (y = 0, z = height)
        result = is_inside(0.0, height, width, height, depth, 2.0, 2.0)
        assert result == True
    
    def test_is_inside_outside_points(self):
        """Test points clearly outside the shape."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        # Point far outside
        result = is_inside(width * 2, height * 2, width, height, depth, 2.0, 2.0)
        assert result == False
    
    def test_is_inside_zero_dimensions(self):
        """Test with zero or very small dimensions."""
        # TODO: Test edge case with zero width/height/depth
        pass
    
    def test_is_inside_symmetry(self):
        """Test symmetry about y-axis."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        # Test symmetry for positive and negative y
        result_pos = is_inside(0.00005, 0.00002, width, height, depth, 2.0, 2.0)
        result_neg = is_inside(-0.00005, 0.00002, width, height, depth, 2.0, 2.0)
        
        assert result_pos == result_neg


# =============================================================================
# Tests for compute_melt_mask function
# =============================================================================

class TestComputeMeltMask:
    """Test cases for the compute_melt_mask wrapper function."""
    
    def test_compute_melt_mask_basic(self, minimal_voxels, sample_melt_pool, sample_path_vectors):
        """Test basic melt mask computation."""
        # Set melt pool properties for path vectors
        for vec in sample_path_vectors:
            vec.set_melt_pool_properties(sample_melt_pool)
        
        result = compute_melt_mask(minimal_voxels, sample_melt_pool, sample_path_vectors)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.bool_
        assert result.shape[0] == minimal_voxels.shape[0]
    
    def test_compute_melt_mask_unpacking(self, minimal_voxels, sample_melt_pool, sample_path_vectors):
        """Test that MeltPool and PathVector objects are correctly unpacked."""
        for vec in sample_path_vectors:
            vec.set_melt_pool_properties(sample_melt_pool)
        
        result = compute_melt_mask(minimal_voxels, sample_melt_pool, sample_path_vectors)
        
        # TODO: Verify correct unpacking of properties
        assert isinstance(result, np.ndarray)
    
    def test_compute_melt_mask_single_voxel(self, sample_melt_pool, sample_path_vectors):
        """Test with a single voxel."""
        single_voxel = np.array([[0.0005, 0.0, 0.0]], dtype=np.float64)
        
        for vec in sample_path_vectors:
            vec.set_melt_pool_properties(sample_melt_pool)
        
        result = compute_melt_mask(single_voxel, sample_melt_pool, sample_path_vectors)
        
        assert result.shape[0] == 1
        assert isinstance(result[0], (bool, np.bool_))
    
    def test_compute_melt_mask_single_path_vector(self, minimal_voxels, sample_melt_pool, sample_path_vector):
        """Test with a single path vector."""
        sample_path_vector.set_melt_pool_properties(sample_melt_pool)
        
        result = compute_melt_mask(minimal_voxels, sample_melt_pool, [sample_path_vector])
        
        assert result.shape[0] == minimal_voxels.shape[0]
    
    def test_compute_melt_mask_output_shape(self, sample_voxels, sample_melt_pool, sample_path_vectors):
        """Test that output shape matches input voxel count."""
        for vec in sample_path_vectors:
            vec.set_melt_pool_properties(sample_melt_pool)
        
        result = compute_melt_mask(sample_voxels, sample_melt_pool, sample_path_vectors)
        
        assert result.shape[0] == sample_voxels.shape[0]
    
    def test_compute_melt_mask_dtype(self, minimal_voxels, sample_melt_pool, sample_path_vectors):
        """Test that output has correct boolean dtype."""
        for vec in sample_path_vectors:
            vec.set_melt_pool_properties(sample_melt_pool)
        
        result = compute_melt_mask(minimal_voxels, sample_melt_pool, sample_path_vectors)
        
        assert result.dtype == np.bool_


# =============================================================================
# Tests for compute_melt_mask_implicit function
# =============================================================================

class TestComputeMeltMaskImplicit:
    """Test cases for the compute_melt_mask_implicit parallelized function."""
    
    def test_compute_melt_mask_implicit_basic(self, minimal_voxels, sample_melt_pool, sample_path_vectors):
        """Test basic implicit melt mask computation."""
        # Set up melt pool properties
        for vec in sample_path_vectors:
            vec.set_melt_pool_properties(sample_melt_pool)
        
        # Prepare inputs
        n_voxels = minimal_voxels.shape[0]
        melt_mask = np.zeros(n_voxels, dtype=np.bool_)
        
        start_points = np.array([v.start_point for v in sample_path_vectors])
        end_points = np.array([v.end_point for v in sample_path_vectors])
        e0 = np.array([v.e0 for v in sample_path_vectors])
        e1 = np.array([v.e1 for v in sample_path_vectors])
        e2 = np.array([v.e2 for v in sample_path_vectors])
        L0 = np.array([v.L0 for v in sample_path_vectors])
        L1 = np.array([v.L1 for v in sample_path_vectors])
        L2 = np.array([v.L2 for v in sample_path_vectors])
        start_times = np.array([v.start_time for v in sample_path_vectors])
        end_times = np.array([v.end_time for v in sample_path_vectors])
        AABB = np.array([v.AABB for v in sample_path_vectors])
        phases = np.array([v.phases for v in sample_path_vectors])
        centroids = np.array([v.centroid for v in sample_path_vectors])
        distances = np.array([v.distance for v in sample_path_vectors])
        
        width_amp = sample_melt_pool.width_oscillations[:, 0]
        width_freq = sample_melt_pool.width_oscillations[:, 1]
        depth_amp = sample_melt_pool.depth_oscillations[:, 0]
        depth_freq = sample_melt_pool.depth_oscillations[:, 1]
        height_amp = sample_melt_pool.height_oscillations[:, 0]
        height_freq = sample_melt_pool.height_oscillations[:, 1]
        
        result = compute_melt_mask_implicit(
            minimal_voxels, melt_mask,
            start_points, end_points,
            e0, e1, e2,
            L0, L1, L2,
            start_times, end_times,
            AABB, phases, centroids, distances,
            width_amp, width_freq,
            depth_amp, depth_freq,
            height_amp, height_freq,
            sample_melt_pool.height_shape_factor,
            sample_melt_pool.depth_shape_factor
        )
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.bool_
        assert result.shape[0] == n_voxels
    
    def test_compute_melt_mask_implicit_aabb_culling(self):
        """Test that AABB (axis-aligned bounding box) culling works correctly."""
        # TODO: Create test where voxels are outside AABB
        pass
    
    def test_compute_melt_mask_implicit_obb_culling(self):
        """Test that OBB (oriented bounding box) culling works correctly."""
        # TODO: Create test where voxels pass AABB but fail OBB test
        pass
    
    def test_compute_melt_mask_implicit_time_fraction(self):
        """Test time fraction calculation along path vector."""
        # TODO: Test time interpolation along scan path
        pass
    
    def test_compute_melt_mask_implicit_coordinate_transform(self):
        """Test transformation to local coordinate system."""
        # TODO: Verify local coordinate calculation
        pass
    
    def test_compute_melt_mask_implicit_oscillation_computation(self):
        """Test that oscillations are correctly computed."""
        # TODO: Verify width, depth, height oscillation calculations
        pass
    
    def test_compute_melt_mask_implicit_zero_distance_vector(self):
        """Test handling of path vectors with zero distance."""
        # TODO: Test edge case with zero-length vectors
        pass
    
    def test_compute_melt_mask_implicit_boundary_voxels(self):
        """Test voxels exactly on path boundaries."""
        # TODO: Test boundary conditions
        pass


# =============================================================================
# Integration Tests
# =============================================================================

class TestCoreIntegration:
    """Integration tests combining multiple core functions."""
    
    def test_is_inside_with_compute_melt_mask(self, minimal_voxels, sample_melt_pool, sample_path_vectors):
        """Test integration of is_inside with compute_melt_mask."""
        for vec in sample_path_vectors:
            vec.set_melt_pool_properties(sample_melt_pool)
        
        result = compute_melt_mask(minimal_voxels, sample_melt_pool, sample_path_vectors)
        
        # Verify result is consistent
        assert isinstance(result, np.ndarray)


# =============================================================================
# Geometric Tests
# =============================================================================

class TestGeometricAccuracy:
    """Tests for geometric accuracy of computations."""
    
    def test_lamé_curve_accuracy(self):
        """Test accuracy of Lamé curve implementation."""
        # TODO: Compare with analytical solutions
        pass
    
    def test_coordinate_transformation_accuracy(self):
        """Test accuracy of coordinate transformations."""
        # TODO: Verify orthogonality and normalization
        pass
    
    def test_distance_calculation_accuracy(self):
        """Test accuracy of distance calculations."""
        # TODO: Compare with known distances
        pass
    
    def test_time_interpolation_accuracy(self):
        """Test accuracy of time interpolation along paths."""
        # TODO: Verify linear interpolation
        pass
    
    def test_bounding_box_accuracy(self):
        """Test that bounding boxes correctly contain geometry."""
        # TODO: Verify AABB and OBB correctness
        pass


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestCoreEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_is_inside_very_small_dimensions(self):
        """Test is_inside with very small dimensions."""
        result = is_inside(0.0, 0.0, 1e-10, 1e-10, 1e-10, 2.0, 2.0)
        # TODO: Verify behavior with tiny dimensions
        pass
    
    def test_compute_melt_mask_single_mode(self):
        """Test with single mode (no oscillations)."""
        # TODO: Test with n_modes = 1
        pass


# =============================================================================
# Shape Factor Tests
# =============================================================================

class TestShapeFactors:
    """Detailed tests for different shape factors."""
    
    def test_ellipse_shape_k2(self):
        """Test elliptical cross-section (k=2)."""
        width = 0.0002
        height = 0.0001
        depth = 0.00008
        
        # Test points on ellipse boundary
        # For ellipse: (y/a)^2 + (z/b)^k = 1
        # k=2
        a = width / 2.0
        b = height
        
        # Point on boundary
        y = a * np.cos(np.pi / 4)
        z = b * np.sin(np.pi / 4)
        
        result = is_inside(y, z, width, height, depth, 2.0, 2.0)
        assert result == True
    
    def test_parabola_shape_k1(self):
        """Test parabolic cross-section (k=1)."""
        # TODO: Test with k=1 for parabolic shape
        pass

    def test_bell_shape_k05(self):
        """Test bell-like cross-section (k=0.5)."""
        # TODO: Test with k=0.5 for bell-like shape
        pass
    
    def test_mixed_shape_factors(self):
        """Test with different shape factors for top and bottom."""
        # TODO: Test height_shape_factor != depth_shape_factor
        pass


# =============================================================================
# Oscillation Tests
# =============================================================================

class TestOscillations:
    """Tests for melt pool oscillation effects."""
    
    def test_oscillation_amplitude_effect(self):
        """Test effect of oscillation amplitudes on melt mask."""
        # TODO: Vary amplitudes and check melt mask changes
        pass
    
    def test_oscillation_frequency_effect(self):
        """Test effect of oscillation frequencies on melt mask."""
        # TODO: Vary frequencies and check melt mask changes
        pass
    
    def test_oscillation_phase_effect(self):
        """Test effect of phase shifts on melt mask."""
        # TODO: Vary phases and check melt mask changes
        pass
    
    def test_multiple_modes_superposition(self):
        """Test that multiple modes combine correctly."""
        # TODO: Verify superposition of multiple oscillation modes
        pass
    
    def test_random_phases(self):
        """Test melt pool with random phases enabled."""
        # TODO: Test random phase generation and effects
        pass


# =============================================================================
# Coordinate Frame Tests
# =============================================================================

class TestCoordinateFrames:
    """Tests for coordinate frame transformations."""
    
    def test_local_to_global_transformation(self):
        """Test transformation from local to global coordinates."""
        # TODO: Verify coordinate transformations
        pass
    
    def test_orthogonal_basis_vectors(self):
        """Test that e0, e1, e2 form an orthogonal basis."""
        # TODO: Verify orthogonality
        pass
    
    def test_coordinate_frame_along_x_axis(self):
        """Test coordinate frame for vector along X axis."""
        # TODO: Test aligned with X
        pass
    
    def test_coordinate_frame_arbitrary_direction(self):
        """Test coordinate frame for arbitrary vector direction."""
        # TODO: Test arbitrary orientation
        pass
