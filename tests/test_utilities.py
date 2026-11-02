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
import numpy as np
import pytest

from raptor.structures import PathVector
from raptor.utilities import (
    DEFAULT_POWDER_CONSOLIDATION,
    MeltPoolFilter,
    ScanPathBuilder,
    apply_powder_consolidation,
    reconstruct_spectral_signal,
)


def make_filter(*, voxel_resolution=5.0e-6, random_seed=None):
    melt_pool_filter = MeltPoolFilter(
        mu=148.0e-6,
        sigma=40.0e-6,
        scan_speed=1.7,
        confidence=0.95,
        ci_relative_width=0.30,
        voxel_resolution=voxel_resolution,
        correlation_tolerance=0.20,
        random_seed=random_seed,
    )
    melt_pool_filter.add_effect("melt_pool", [300.0e-6, None, 1.0])
    return melt_pool_filter


def test_melt_pool_filter_recovers_requested_statistics():
    melt_pool_filter = make_filter()
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(
        1.0, melt_pool_filter.n_points, melt_pool_filter.t
    )

    assert np.isfinite(width_data).all()
    np.testing.assert_allclose(
        width_data[:, 1].mean(), melt_pool_filter.mu, atol=1e-15
    )
    np.testing.assert_allclose(
        width_data[:, 1].std(), melt_pool_filter.sigma, atol=1e-15
    )
    assert melt_pool_filter.planned_variance_ci["precision_satisfied"]


def test_melt_pool_filter_normalizes_covariance_between_effects():
    melt_pool_filter = make_filter()
    melt_pool_filter.add_effect("overlapping", [300.0e-6, None, 1.0])
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(
        1.0, melt_pool_filter.n_points, melt_pool_filter.t
    )

    np.testing.assert_allclose(
        width_data[:, 1].mean(), melt_pool_filter.mu, atol=1e-15
    )
    np.testing.assert_allclose(
        width_data[:, 1].std(), melt_pool_filter.sigma, atol=1e-15
    )


def test_melt_pool_filter_seed_reproduces_filtered_time_series():
    first_filter = make_filter(random_seed=42)
    second_filter = make_filter(random_seed=42)
    first_filter.initialize()
    second_filter.initialize()

    first = first_filter.generate_fluctuations(
        1.0, first_filter.n_points, first_filter.t
    )
    second = second_filter.generate_fluctuations(
        1.0, second_filter.n_points, second_filter.t
    )

    np.testing.assert_array_equal(first, second)


def test_duration_is_the_minimum_accepted_sample_count():
    melt_pool_filter = make_filter()
    melt_pool_filter.initialize()
    model_correlation = melt_pool_filter._model_autocorrelation()

    previous_n = melt_pool_filter.n_points - 1
    previous_effective_n = melt_pool_filter._effective_sample_size(
        previous_n, model_correlation
    )
    previous_ci = melt_pool_filter._variance_ci(
        melt_pool_filter.sigma**2, previous_effective_n
    )

    minimum_periods = max(
        4, int(np.ceil(1.0 / melt_pool_filter.correlation_tolerance))
    )
    minimum_n = (
        int(
            np.ceil(
                minimum_periods
                * (300.0e-6 / melt_pool_filter.scan_speed)
                * melt_pool_filter.fs
            )
        )
        + 1
    )
    if previous_n >= minimum_n:
        assert not previous_ci["precision_satisfied"]


def test_reconstruct_spectral_signal():
    time = np.arange(100) / 100.0
    components = np.array([[2.0, 0.0, 0.0], [0.5, 4.0, np.pi / 3.0]])
    expected = 2.0 + 0.5 * np.cos(2.0 * np.pi * 4.0 * time + np.pi / 3.0)
    np.testing.assert_allclose(
        reconstruct_spectral_signal(time, components), expected, atol=1e-14
    )


# =============================================================================
# Powder consolidation: ScanPathBuilder and apply_powder_consolidation
# =============================================================================
BOX = np.array([[0.0, 0.0, 0.0], [200.0e-6, 200.0e-6, 200.0e-6]])
LAYER = 30.0e-6


def build_vectors(powder_consolidation=None, extra_layers=0):
    kwargs = {}
    if powder_consolidation is not None:
        kwargs["powder_consolidation"] = powder_consolidation
    builder = ScanPathBuilder(
        BOX,
        370.0,
        1.3,
        70.0e-6,
        LAYER,
        67.0,
        100.0e-6,
        extra_layers,
        **kwargs,
    )
    builder.generate_layers()
    return builder, builder.process_vectors()


def layer_heights(vectors):
    return np.unique(np.round([v.start_point[2] for v in vectors], 12))


def test_default_consolidation_is_one_half():
    builder, default_vectors = build_vectors()
    _, explicit_vectors = build_vectors(powder_consolidation=0.5)

    assert builder.powder_consolidation == DEFAULT_POWDER_CONSOLIDATION == 0.5
    assert builder.layer_pitch == pytest.approx(2.0 * LAYER)
    for a, b in zip(default_vectors, explicit_vectors, strict=True):
        np.testing.assert_array_equal(a.start_point, b.start_point)
        np.testing.assert_array_equal(a.end_point, b.end_point)


def test_unit_consolidation_keeps_nominal_layer_height():
    builder, vectors = build_vectors(powder_consolidation=1.0)

    assert builder.layer_pitch == LAYER
    heights = layer_heights(vectors)
    np.testing.assert_allclose(heights, np.arange(heights.size) * LAYER)


def test_consolidation_sets_layer_pitch_to_height_over_phi():
    builder, vectors = build_vectors(powder_consolidation=0.5)
    _, nominal = build_vectors(powder_consolidation=1.0)

    assert builder.layer_pitch == pytest.approx(2.0 * LAYER)
    heights = layer_heights(vectors)
    np.testing.assert_allclose(np.diff(heights), 2.0 * LAYER)
    assert heights.max() >= BOX[1, 2]
    # In-plane geometry of each layer is unchanged.
    nominal_xy = {
        tuple(np.round(np.r_[v.start_point[:2], v.end_point[:2]], 12))
        for v in nominal
    }
    for v in vectors:
        xy = tuple(np.round(np.r_[v.start_point[:2], v.end_point[:2]], 12))
        assert xy in nominal_xy


@pytest.mark.parametrize("phi", [0.0, -0.5, 1.5, np.nan, np.inf, True])
def test_invalid_powder_consolidation_is_rejected(phi):
    with pytest.raises(ValueError, match="powder_consolidation"):
        build_vectors(powder_consolidation=phi)


def make_vector(z0, z1, x=0.0):
    vector = PathVector(
        np.array([x, 0.0, z0]), np.array([x + 1.0e-4, 0.0, z1]), 0.0, 1.0
    )
    vector.set_coordinate_frame()
    return vector


def test_apply_powder_consolidation_identity_at_one():
    vectors = [make_vector(1.0e-3, 1.0e-3), make_vector(1.03e-3, 1.03e-3)]
    out = apply_powder_consolidation(vectors, 1.0)
    for a, b in zip(vectors, out):
        np.testing.assert_allclose(a.start_point, b.start_point)
        np.testing.assert_allclose(a.end_point, b.end_point)
        assert b is not a


def test_apply_powder_consolidation_stretches_about_reference():
    vectors = [make_vector(1.0e-3, 1.0e-3), make_vector(1.03e-3, 1.03e-3)]
    out = apply_powder_consolidation(vectors, 0.5)
    np.testing.assert_allclose(
        [v.start_point[2] for v in out], [1.0e-3, 1.06e-3]
    )
    np.testing.assert_allclose(
        [v.start_point[:2] for v in out], [v.start_point[:2] for v in vectors]
    )
    assert out[1].start_time == vectors[1].start_time
    assert hasattr(out[1], "e0")  # coordinate frame set

    shifted = apply_powder_consolidation(vectors, 0.5, reference_z=0.0)
    np.testing.assert_allclose(
        [v.start_point[2] for v in shifted], [2.0e-3, 2.06e-3]
    )
    # Input vectors are not modified.
    assert vectors[1].start_point[2] == pytest.approx(1.03e-3)
