from dataclasses import dataclass
import numpy as np
from numba.experimental import jitclass
from numba.typed import List as numbaList
from typing import List, Optional
from numba import int32, int64, float64, boolean

# Define jitclass specifications
bezier_spec = [
    ("n_points", int64),
    ("weights", float64[:, :]),
    ("polygon", float64[:, :]),
]

melt_pool_spec = [
    ("width_oscillations", float64[:, :]),
    ("depth_oscillations", float64[:, :]),
    ("height_oscillations", float64[:, :]),
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


@jitclass(bezier_spec)
class Bezier:
    def __init__(self, n_points: int64):
        self.n_points = n_points
        n_polygon_pts = n_points + (n_points - 1)

        self.polygon = np.empty((n_polygon_pts, 2), dtype=np.float64)

        t_p = np.linspace(0, 1, n_points)
        weights = np.empty((n_points, 4), dtype=np.float64)
        omt = 1.0 - t_p
        tsq = t_p * t_p
        omtsq = omt * omt
        weights[:, 0] = omtsq * omt
        weights[:, 1] = 3.0 * t_p * omtsq
        weights[:, 2] = 3.0 * tsq * omt
        weights[:, 3] = tsq * t_p
        self.weights = weights

    def update(self, width: float64, depth: float64, height: float64) -> None:
        height_control = 4.0 / 3.0 * height
        depth_control = 4.0 / 3.0 * depth

        # Top control points
        p_top = np.array(
            [
                [-width / 2.0, 0.0],
                [-width / 4.0, height_control],
                [width / 4.0, height_control],
                [width / 2.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Bottom control points
        p_bottom = np.array(
            [
                [width / 2.0, 0.0],
                [width / 4.0, -depth_control],
                [-width / 4.0, -depth_control],
                [-width / 2.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Update top half of the polygon
        for i in range(self.n_points):
            # dot product for the X coordinate
            self.polygon[i, 0] = (
                self.weights[i, 0] * p_top[0, 0]
                + self.weights[i, 1] * p_top[1, 0]
                + self.weights[i, 2] * p_top[2, 0]
                + self.weights[i, 3] * p_top[3, 0]
            )
            # dot product for the Y coordinate
            self.polygon[i, 1] = (
                self.weights[i, 0] * p_top[0, 1]
                + self.weights[i, 1] * p_top[1, 1]
                + self.weights[i, 2] * p_top[2, 1]
                + self.weights[i, 3] * p_top[3, 1]
            )

        # Update bottom half of the polygon
        for i in range(1, self.n_points):
            idx = self.n_points + i - 1
            # dot product for the X coordinate
            self.polygon[idx, 0] = (
                self.weights[i, 0] * p_bottom[0, 0]
                + self.weights[i, 1] * p_bottom[1, 0]
                + self.weights[i, 2] * p_bottom[2, 0]
                + self.weights[i, 3] * p_bottom[3, 0]
            )
            # dot product for the Y coordinate
            self.polygon[idx, 1] = (
                self.weights[i, 0] * p_bottom[0, 1]
                + self.weights[i, 1] * p_bottom[1, 1]
                + self.weights[i, 2] * p_bottom[2, 1]
                + self.weights[i, 3] * p_bottom[3, 1]
            )

    def point_in_polygon(self, x: float64, y: float64) -> bool:
        n_vertices = self.polygon.shape[0]
        px0 = self.polygon[n_vertices - 1, 0]
        py0 = self.polygon[n_vertices - 1, 1]

        is_inside = False

        for i in range(n_vertices):
            px1, py1 = self.polygon[i, 0], self.polygon[i, 1]
            crosses_y = (py1 > y) != (py0 > y)
            left_of_edge = (x - px1) * (py0 - py1) < (px0 - px1) * (y - py1)
            is_inside += crosses_y and (left_of_edge != (py1 > py0))
            px0, py0 = px1, py1

        return (is_inside % 2) == 1


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
        enable_random_phases: boolean,
    ):
        self.width_oscillations = width_oscillations
        self.depth_oscillations = depth_oscillations
        self.height_oscillations = height_oscillations

        self.width_max = width_max
        self.depth_max = width_max
        self.height_max = width_max

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

    def set_phases(self, melt_pool: MeltPool) -> None:
        if melt_pool.enable_random_phases:
            size = melt_pool.width_oscillations.shape[0] - 1
            random_phase = np.random.uniform(0.0, 2.0 * np.pi, size)
            zero_phase = np.array([0.0], dtype=np.float64)
            self.phases = np.hstack((zero_phase, random_phase)).astype(np.float64)
        else:
            self.phases = melt_pool.width_oscillations[:, 2].astype(np.float64)

    def set_bound_box(self, melt_pool: MeltPool) -> None:
        """
        Calculates and sets the OBB half-lengths and the AABB for this vector
        based on the physical properties of a given MeltPool.
        """
        # Unpack max dimensions from the melt pool for clarity
        width_max = melt_pool.width_max
        depth_max = melt_pool.depth_max
        height_max = melt_pool.height_max

        # --- Calculate Oriented Bounding Box (OBB) half-lengths ---
        self.L0 = width_max / 2.0
        self.L1 = np.hypot(self.distance[0], self.distance[1]) / 2.0
        self.L2 = max(height_max, depth_max) / 2.0

        # --- Calculate Axis-Aligned Bounding Box (AABB) ---
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

    def set_melt_pool_properties(self, melt_pool: MeltPool) -> None:
        """
        Orchestrator to set melt pool dependent properties on the vector.
        """
        self.set_phases(melt_pool)
        self.set_bound_box(melt_pool)


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
