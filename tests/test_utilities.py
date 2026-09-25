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

from raptor.utilities import (
    MeltPoolFilter,
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
