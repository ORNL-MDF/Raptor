import numpy as np
from numba import njit, prange
from typing import List, Tuple
from .structures import MeltPool, PathVector, Bezier


def compute_melt_mask(
    voxels: np.ndarray,
    melt_pool: MeltPool,
    path_vectors: List[PathVector],
    bezier: Bezier,
):
    """
    Unpacks jitclasses into arrays to pass to compute_melt_mask_implicit().
    """
    n_voxels = voxels.shape[0]
    melt_mask = np.zeros(n_voxels, dtype=np.bool_)

    # --- Unpack MeltPool object ---
    width_amplitudes = melt_pool.width_oscillations[:, 0]
    width_frequencies = melt_pool.width_oscillations[:, 1]

    depth_amplitudes = melt_pool.depth_oscillations[:, 0]
    depth_frequencies = melt_pool.depth_oscillations[:, 1]

    height_amplitudes = melt_pool.height_oscillations[:, 0]
    height_frequencies = melt_pool.height_oscillations[:, 1]

    # --- Unpack the list of PathVector objects ---
    start_points = np.array([p.start_point for p in path_vectors])
    end_points = np.array([p.end_point for p in path_vectors])
    distances = np.array([p.distance for p in path_vectors])

    e0 = np.array([p.e0 for p in path_vectors])
    e1 = np.array([p.e1 for p in path_vectors])
    e2 = np.array([p.e2 for p in path_vectors])

    L0 = np.array([p.L0 for p in path_vectors])
    L1 = np.array([p.L1 for p in path_vectors])
    L2 = np.array([p.L2 for p in path_vectors])

    start_times = np.array([p.start_time for p in path_vectors])
    end_times = np.array([p.end_time for p in path_vectors])

    AABB = np.array([p.AABB for p in path_vectors])
    centroids = np.array([p.centroid for p in path_vectors])
    phases = np.array([p.phases for p in path_vectors])

    return compute_melt_mask_implicit(
        voxels,
        melt_mask,
        start_points,
        end_points,
        e0,
        e1,
        e2,
        L0,
        L1,
        L2,
        start_times,
        end_times,
        AABB,
        phases,
        centroids,
        distances,
        width_amplitudes,
        width_frequencies,
        depth_amplitudes,
        depth_frequencies,
        height_amplitudes,
        height_frequencies,
        bezier,
    )


@njit(parallel=True, fastmath=True)
def compute_melt_mask_implicit(
    voxels: np.ndarray,
    melt_mask: np.ndarray,
    start_points: np.ndarray,
    end_points: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    L0: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    AABB: np.ndarray,
    phases: np.ndarray,
    centroids: np.ndarray,
    distances: np.ndarray,
    width_amplitudes: np.ndarray,
    width_frequencies: np.ndarray,
    depth_amplitudes: np.ndarray,
    depth_frequencies: np.ndarray,
    height_amplitudes: np.ndarray,
    height_frequencies: np.ndarray,
    bezier: Bezier,
) -> np.ndarray:
    """
    Implicit compute melt mask function.
    """

    n_voxels = voxels.shape[0]
    n_vectors = start_points.shape[0]
    n_modes = width_amplitudes.shape[0]

    for i in prange(n_voxels):
        vx, vy, vz = voxels[i, 0], voxels[i, 1], voxels[i, 2]
        is_voxel_melted = False

        for j in range(n_vectors):
            # 1. Check if voxel is in axis-aligned bounding box (AABB)
            if not (
                AABB[j, 0] <= vx <= AABB[j, 1]
                and AABB[j, 2] <= vy <= AABB[j, 3]
                and AABB[j, 4] <= vz <= AABB[j, 5]
            ):
                continue

            # --- Create voxel position vector ONCE for reuse ---
            voxel_pos = np.array([vx, vy, vz])

            # 2. Check if voxel is in oriented bounding box (OBB)
            vec_centroid_to_voxel = voxel_pos - centroids[j, :]

            # Project distance onto scan axis (e1)
            dist_sq_scan = np.dot(vec_centroid_to_voxel, e1[j, :]) ** 2
            if dist_sq_scan > L1[j] ** 2:
                continue

            # Project distance onto width axis (e0)
            dist_sq_width = np.dot(vec_centroid_to_voxel, e0[j, :]) ** 2
            if dist_sq_width > L0[j] ** 2:
                continue

            distance = distances[j, :]

            # 3. Project voxel onto path vector
            vec_start_to_voxel = voxel_pos - start_points[j, :]
            dist_sqr = np.dot(distance, distance)

            time_fraction = 0.0
            time_fraction = (
                np.dot(vec_start_to_voxel, distance) / dist_sqr
                if dist_sqr > 1e-12
                else 0.0
            )
            time_fraction = max(0.0, min(1.0, time_fraction))

            time = start_times[j] + time_fraction * (end_times[j] - start_times[j])

            # Get vector from projected point on path to voxel
            projected_point = start_points[j, :] + time_fraction * distance
            vec_path_to_voxel = voxel_pos - projected_point

            # Project onto local frame for 2D polygon check
            local_y = np.dot(vec_path_to_voxel, e0[j, :])  # Projection on width axis
            local_z = np.dot(vec_path_to_voxel, e2[j, :])  # Projection on depth axis

            # 4. Compute dynamic melt pool dimensions
            width, depth, height = 0.0, 0.0, 0.0
            phase = phases[j, :]
            two_pi_t = 2.0 * np.pi * time

            for k in range(n_modes):
                phase_k = phase[k]
                width += width_amplitudes[k] * np.cos(
                    two_pi_t * width_frequencies[k] + phase_k
                )

                depth += depth_amplitudes[k] * np.cos(
                    two_pi_t * depth_frequencies[k] + phase_k
                )

                height += height_amplitudes[k] * np.cos(
                    two_pi_t * height_frequencies[k] + phase_k
                )

            # 5. Point in polygon check
            bezier.update(width, depth, height)
            if bezier.point_in_polygon(local_y, local_z):
                is_voxel_melted = True
                break

        melt_mask[i] = is_voxel_melted
    return melt_mask
