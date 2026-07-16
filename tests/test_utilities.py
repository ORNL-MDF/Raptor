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

from raptor.utilities import MeltPoolFilter


def test_melt_pool_filter_recovers_requested_statistics():
    mean = 148.0e-6
    standard_deviation = 40.0e-6
    scan_speed = 1.7
    melt_pool_length = 300.0e-6
    time_scale = melt_pool_length / scan_speed
    sampling_frequency = 340_000.0

    melt_pool_filter = MeltPoolFilter(
        mean,
        standard_deviation,
        scan_speed,
        [sampling_frequency, 30.0 * time_scale],
    )
    melt_pool_filter.add_effect("melt_pool", [melt_pool_length, None, 1.0])
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(1.0)

    assert np.isfinite(width_data).all()
    np.testing.assert_allclose(width_data[:, 1].mean(), mean, atol=1.0e-15)
    np.testing.assert_allclose(width_data[:, 1].std(), standard_deviation, atol=1.0e-15)


def test_melt_pool_filter_remains_finite_at_high_sampling_frequency():
    melt_pool_filter = MeltPoolFilter(
        148.0e-6,
        40.0e-6,
        1.7,
        [2_500_000.0, 0.002],
    )
    melt_pool_filter.add_effect("melt_pool", [300.0e-6, None, 1.0])
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(1.0)

    assert np.isfinite(width_data).all()
