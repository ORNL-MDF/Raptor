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
from dataclasses import dataclass
import numpy as np
from numba.experimental import jitclass
from numba.typed import List as numbaList
from typing import List, Optional
from numba import int32, int64, float64, boolean

# Define jitclass specifications
melt_pool_spec = [
    ("width_oscillations", float64[:, :]),
    ("depth_oscillations", float64[:, :]),
    ("height_oscillations", float64[:, :]),
    ("width_shape_factor", float64),
    ("height_shape_factor", float64),
    ("depth_shape_factor", float64),
    ("width_max", float64),
    ("depth_max", float64),
    ("height_max", float64),
    ("enable_random_phases", boolean),
    ("width_mean", float64),
    ("depth_mean", float64),
    ("height_mean", float64),
    ("n_modes", int64),
]

path_vector_spec = [
    ("start_point", float64[:]),
    ("end_point", float64[:]),
    ("start_time", float64),
    ("end_time", float64),
    ("duration", float64),
    ("distance", float64[:]),
    ("AABB", float64[:]),
    ("e0", float64[:]),  # Width direction -> global X
    ("e1", float64[:]),  # Scan direction -> global Y
    ("e2", float64[:]),  # Depth direction -> global Z
    ("L0", float64),  # OBB half-length along e0
    ("L1", float64),  # OBB half-length along e1
    ("L2", float64),  # OBB half-length along e2
    ("phases", float64[:]),
    ("centroid", float64[:]),
]


@jitclass(melt_pool_spec)
class MeltPool:
    """
    Represents the melt pool with oscillation properties.

    Attributes:
        width_mean: mean (mode number 0) of the oscillations
    """

    def __init__(
        self,
        width_oscillations: float64[:, :],
        depth_oscillations: float64[:, :],
        height_oscillations: float64[:, :],
        width_max: float64,
        depth_max: float64,
        height_max: float64,
        width_shape_factor: float64,
        height_shape_factor: float64,
        depth_shape_factor: float64,
        enable_random_phases: boolean,
    ):
        self.width_oscillations = width_oscillations
        self.depth_oscillations = depth_oscillations
        self.height_oscillations = height_oscillations

        self.width_max = width_max
        self.depth_max = depth_max
        self.height_max = height_max

        self.height_shape_factor = height_shape_factor
        self.depth_shape_factor = depth_shape_factor

        self.enable_random_phases = enable_random_phases

        self.width_mean = self.width_oscillations[0, 0]
        self.depth_mean = self.depth_oscillations[0, 0]
        self.height_mean = self.height_oscillations[0, 0]


@jitclass(path_vector_spec)
class PathVector:
    """
    Represents a scan vector with a melt pool dependent bounding box
    """

    def __init__(
        self,
        start_point: float64[:],
        end_point: float64[:],
        start_time: float64,
        end_time: float64,
    ):
        self.start_point = start_point
        self.end_point = end_point
        self.start_time = start_time
        self.end_time = end_time

    def set_coordinate_frame(self) -> None:
        self.distance = self.end_point - self.start_point
        self.centroid = (self.end_point + self.start_point) / 2.0

        self.duration = self.end_time - self.start_time

        # Calculate local coordinate frame
        dx, dy = self.distance[0], self.distance[1]
        Lxy = np.hypot(dx, dy)
        if Lxy < 1e-12:
            self.e0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            self.e1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            self.e0 = np.array([-dy / Lxy, dx / Lxy, 0.0], dtype=np.float64)
            self.e1 = np.array([dx / Lxy, dy / Lxy, 0.0], dtype=np.float64)
        self.e2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def set_melt_pool_properties(self, melt_pool: MeltPool) -> None:
        """
        Sets melt pool dependent properties on the vector.
        """
        # Setting cosine expansion phases for the vector.
        if melt_pool.enable_random_phases:
            size = melt_pool.width_oscillations.shape[0] - 1
            random_phase = np.random.uniform(0.0, 2.0 * np.pi, size)
            zero_phase = np.array([0.0], dtype=np.float64)
            self.phases = np.hstack((zero_phase, random_phase)).astype(np.float64)
        else:
            self.phases = melt_pool.width_oscillations[:, 2].astype(np.float64)

        # Setting bounding box properties for the vector.
        # 1. --- Calculate Axis-Aligned Bounding Box (AABB) ---
        width_max, depth_max, height_max = (
            melt_pool.width_max,
            melt_pool.depth_max,
            melt_pool.height_max,
        )
        p_min = np.minimum(self.start_point, self.end_point)
        p_max = np.maximum(self.start_point, self.end_point)
        pad_xy = width_max / 2.0
        self.AABB = np.array(
            [
                p_min[0] - pad_xy,  # x-min
                p_max[0] + pad_xy,  # x-max
                p_min[1] - pad_xy,  # y-min
                p_max[1] + pad_xy,  # y-max
                p_min[2] - depth_max,  # z-min
                p_max[2] + height_max,  # z-max
            ],
            dtype=np.float64,
        )

        # 2. --- Calculate Oriented Bounding Box (OBB) half-lengths ---
        self.L0 = width_max / 2.0
        self.L1 = np.hypot(p_max[0] - p_min[0], p_max[1] - p_min[1]) / 2.0
        self.L2 = max(height_max, depth_max)


class Grid:
    """
    Represents the discrete voxel grid for the simulation domain.

    The grid boundaries can be defined either by a fixed bounding box or
    be automatically generated from a list of scan path vectors.
    """

    def __init__(
        self,
        voxel_resolution: float,
        bound_box: Optional[np.ndarray] = None,
        path_vectors: Optional[List[PathVector]] = None,
    ):
        self.resolution = voxel_resolution

        if bound_box is not None:
            # Option 1: Grid is constructed from a user-defined bounding box.
            gx0, gy0, gz0 = bound_box[0]
            gx1, gy1, gz1 = bound_box[1]

        elif path_vectors is not None:
            # Option 2: Grid is constructed from boundaries of path vectors.
            all_points = np.vstack(
                [p.start_point for p in path_vectors]
                + [p.end_point for p in path_vectors]
            )
            xmin, ymin, zmin = all_points.min(axis=0)
            xmax, ymax, zmax = all_points.max(axis=0)

            gx0, gy0, gz0 = xmin, ymin, zmin
            gx1, gy1, gz1 = xmax, ymax, zmax
        else:
            raise ValueError(
                "Grid construction failed: "
                "You must provide either a 'bound_box' "
                "or a non-empty list of 'path_vectors'."
            )

        self.origin = np.array([gx0, gy0, gz0])

        xg = np.arange(gx0, gx1 + self.resolution / 2.0, self.resolution)
        yg = np.arange(gy0, gy1 + self.resolution / 2.0, self.resolution)
        zg = np.arange(gz0, gz1 + self.resolution / 2.0, self.resolution)

        self.shape = (len(xg), len(yg), len(zg))
        self.n_voxels = self.shape[0] * self.shape[1] * self.shape[2]

        X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
        self.voxels = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.copy()
