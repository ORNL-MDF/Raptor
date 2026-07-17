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
    resolution = 5.0e-6
    
    variance_significance = 0.05
    ci_tolerance_window = 0.01


    melt_pool_filter = MeltPoolFilter(
        mean,
        standard_deviation,
        scan_speed,
        variance_significance,
        ci_tolerance_window,
        resolution
    )
    melt_pool_filter.add_effect("melt_pool", [melt_pool_length, None, 1.0])
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(1.0, melt_pool_filter.n_points, melt_pool_filter.t)

    assert np.isfinite(width_data).all()
    assert np.all((width_data[:,1].std()**2 >= standard_deviation**2 * (1 - ci_tolerance_window)) & (width_data[:,1].std()**2 <= standard_deviation**2 * (1 + ci_tolerance_window)))