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
from typing import List, Optional


class MeltPool:
    """
    Represents the melt pool with oscillation properties.

    Attributes:
        width_mean: mean (mode number 0) of the oscillations
    """

    def __init__(
        self,
        width_oscillations: np.ndarray,
        depth_oscillations: np.ndarray,
        height_oscillations: np.ndarray,
        width_max: float,
        depth_max: float,
        height_max: float,
        width_shape_factor: float,
        height_shape_factor: float,
        depth_shape_factor: float,
        enable_random_phases: bool,
    ):
        self.width_oscillations = width_oscillations
        self.depth_oscillations = depth_oscillations
        self.height_oscillations = height_oscillations

        self.width_max = width_max
        self.depth_max = depth_max
        self.height_max = height_max

        self.width_shape_factor = width_shape_factor
        self.height_shape_factor = height_shape_factor
        self.depth_shape_factor = depth_shape_factor

        self.enable_random_phases = enable_random_phases

        self.width_mean = self.width_oscillations[0, 0]
        self.depth_mean = self.depth_oscillations[0, 0]
        self.height_mean = self.height_oscillations[0, 0]
        self.n_modes = max(
            len(self.width_oscillations),
            len(self.depth_oscillations),
            len(self.height_oscillations),
        )


class PathVector:
    """
    Represents a scan vector with a melt pool dependent bounding box
    """

    def __init__(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        start_time: float,
        end_time: float,
    ):
        start_point = np.asarray(start_point, dtype=np.float64)
        end_point = np.asarray(end_point, dtype=np.float64)
        if start_point.shape != (3,) or end_point.shape != (3,):
            raise ValueError("Path-vector points must have shape (3,).")
        if (
            not np.isfinite(start_point).all()
            or not np.isfinite(end_point).all()
        ):
            raise ValueError("Path-vector points must contain finite values.")
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("Path-vector times must be finite.")
        if end_time < start_time:
            raise ValueError(
                "Path-vector end_time must be greater than or equal to "
                "start_time."
            )
        self.start_point = start_point
        self.end_point = end_point
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def set_coordinate_frame(self) -> None:
        self.distance = self.end_point - self.start_point
        self.distance = np.sign(self.distance) * np.maximum(
            np.abs(self.distance), 1e-12
        )
        self.centroid = (self.end_point + self.start_point) / 2.0

        self.duration = self.end_time - self.start_time
        self.distance_sqr = (
            self.distance[0] * self.distance[0]
            + self.distance[1] * self.distance[1]
            + self.distance[2] * self.distance[2]
        )
        self.inv_distance_sqr = (
            1.0 / self.distance_sqr if self.distance_sqr > 1e-24 else 0.0
        )

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

    def set_melt_pool_properties(
        self,
        melt_pool: MeltPool,
        rng=None,
        common_phases: Optional[np.ndarray] = None,
    ) -> None:
        """Attach dimension phases and a conservative melt-pool envelope.

        ``common_phases`` allows callers to generate many phase rows in one
        vectorized operation. Each spectral dimension retains its own mode
        count and receives a copy of the corresponding prefix.
        """
        if melt_pool.enable_random_phases:
            max_modes = max(
                melt_pool.width_oscillations.shape[0],
                melt_pool.depth_oscillations.shape[0],
                melt_pool.height_oscillations.shape[0],
            )
            if common_phases is None:
                if rng is None:
                    rng = np.random
                common_phases = np.empty(max_modes, dtype=np.float64)
                common_phases[0] = 0.0
                common_phases[1:] = rng.uniform(
                    0.0,
                    2.0 * np.pi,
                    max_modes - 1,
                )
            self.width_phases = common_phases[
                : melt_pool.width_oscillations.shape[0]
            ].copy()
            self.depth_phases = common_phases[
                : melt_pool.depth_oscillations.shape[0]
            ].copy()
            self.height_phases = common_phases[
                : melt_pool.height_oscillations.shape[0]
            ].copy()
        else:
            self.width_phases = melt_pool.width_oscillations[:, 2].astype(
                np.float64
            )
            self.depth_phases = melt_pool.depth_oscillations[:, 2].astype(
                np.float64
            )
            self.height_phases = melt_pool.height_oscillations[:, 2].astype(
                np.float64
            )
        # Backward-compatible alias for consumers that used the original field.
        self.phases = self.width_phases

        # Axis-aligned bounds conservatively cull whole path vectors.
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

        # Oriented bounds support exact in-plane candidate tests.
        self.L0 = width_max / 2.0
        self.L1 = np.hypot(p_max[0] - p_min[0], p_max[1] - p_min[1]) / 2.0
        self.L2 = max(height_max, depth_max)
        self.L0_sqr = self.L0 * self.L0
        self.L1_sqr = self.L1 * self.L1


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
        if not np.isfinite(voxel_resolution) or voxel_resolution <= 0.0:
            raise ValueError("Voxel resolution must be finite and positive.")
        self.resolution = float(voxel_resolution)

        if bound_box is not None:
            # Option 1: Grid is constructed from a user-defined bounding box.
            bound_box = np.asarray(bound_box, dtype=np.float64)
            if bound_box.shape != (2, 3):
                raise ValueError(
                    "Bounding box must be of shape (2, 3) "
                    "representing [[x0, y0, z0], [x1, y1, z1]]."
                )
            if not np.isfinite(bound_box).all():
                raise ValueError("Bounding box must contain finite values.")
            if np.any(bound_box[1] <= bound_box[0]):
                raise ValueError(
                    "Invalid bounding box: "
                    "Maximum corner must be greater than minimum corner."
                )
            gx0, gy0, gz0 = bound_box[0]
            gx1, gy1, gz1 = bound_box[1]

        elif path_vectors is not None:
            # Option 2: Grid is constructed from boundaries of path vectors.
            if not path_vectors:
                raise ValueError(
                    "'path_vectors' must contain at least one PathVector."
                )
            for pv in path_vectors:
                if not isinstance(pv, PathVector):
                    raise ValueError(
                        "All elements in 'path_vectors' must be "
                        "of type PathVector."
                    )
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
        self._voxels = None

    @property
    def voxels(self) -> np.ndarray:
        """Materialize coordinates lazily for direct access."""
        if self._voxels is None:
            xg = self.origin[0] + np.arange(self.shape[0]) * self.resolution
            yg = self.origin[1] + np.arange(self.shape[1]) * self.resolution
            zg = self.origin[2] + np.arange(self.shape[2]) * self.resolution
            X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
            self._voxels = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        return self._voxels
