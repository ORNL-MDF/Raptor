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

from raptor.core import compute_distance_to_boundary


def test_melt_pool_center_is_inside():
    width = 148.0e-6
    height = 59.2e-6
    depth = 118.4e-6

    distance = compute_distance_to_boundary(
        0.0,
        0.0,
        width,
        height,
        depth,
        1.0,
        1.0,
        5.0e-6,
    )

    assert np.isfinite(distance)
    assert distance == -min(width / 2.0, height, depth)


def test_near_center_points_remain_finite():
    dimensions = (148.0e-6, 59.2e-6, 118.4e-6)

    for y, z in ((1.0e-12, 0.0), (0.0, 1.0e-12), (0.0, -1.0e-12)):
        distance = compute_distance_to_boundary(
            y,
            z,
            *dimensions,
            1.0,
            1.0,
            5.0e-6,
        )

        assert np.isfinite(distance)
        assert distance < 0.0
