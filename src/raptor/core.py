import numpy as np
from numba import njit, prange
from typing import List, Tuple
from .structures import MeltPool, PathVector


@njit(inline='always',fastmath=True)
def is_inside(local_y: float, local_z: float, width:float, height:float, shape_factor:int) -> float:
    """
    Checks if a voxel at (local_y,local_z) is inside the (half) melt pool cross section.
    Based on the superellipsoid equation:
    (y/a)^2 + (z/h)^(1/shape_factor) = 1
    z = h * (1 - (y/a)^2)^(shape_factor)
    h = height or depth, depending on if local_z is positive or negative.
    """
    a = width / 2.0
    b =  np.maximum(1 - (local_y / a)**2, 0)
    return np.abs(local_z) <= np.abs(height * np.power(b, shape_factor))

def compute_melt_mask(
    voxels: np.ndarray,
    melt_pool: MeltPool,
    path_vectors: List[PathVector]
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

    height_shape_factor = melt_pool.height_shape_factor
    depth_shape_factor = melt_pool.depth_shape_factor

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
        height_shape_factor,
        depth_shape_factor
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
    height_shape_factor: np.float64,
    depth_shape_factor: np.float64
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
            if (
                vx < AABB[j, 0]
                or vx > AABB[j, 1]
                or vy < AABB[j, 2]
                or vy > AABB[j, 3]
                or vz < AABB[j, 4]
                or vz > AABB[j, 5]
            ):
                continue
            
        
            vec_cx = vx - centroids[j, 0]
            vec_cy = vy - centroids[j, 1]
            vec_cz = vz - centroids[j, 2]

            dot_e0 = vec_cx * e0[j, 0] + vec_cy * e0[j, 1] + vec_cz * e0[j, 2]
            if (dot_e0 * dot_e0) > (L0[j] * L0[j]):
                continue

            dot_e1 = vec_cx * e1[j, 0] + vec_cy * e1[j, 1] + vec_cz * e1[j, 2]
            if (dot_e1 * dot_e1) > (L1[j] * L1[j]):
                continue


            dist_sqr = (
                distances[j, 0] * distances[j, 0]
                + distances[j, 1] * distances[j, 1]
                + distances[j, 2] * distances[j, 2]
            )

            time_fraction = 0.0
            if dist_sqr > 1e-24:
                vec_sx = vx - start_points[j, 0]
                vec_sy = vy - start_points[j, 1]
                vec_sz = vz - start_points[j, 2]

                dot_dist = (
                    vec_sx * distances[j, 0]
                    + vec_sy * distances[j, 1]
                    + vec_sz * distances[j, 2]
                )

                time_fraction = dot_dist / dist_sqr

            time_fraction = max(0.0, min(1.0, time_fraction))
            time = start_times[j] + time_fraction * (end_times[j] - start_times[j])

            vec_path_x = vx - (start_points[j, 0] + time_fraction * distances[j, 0])
            vec_path_y = vy - (start_points[j, 1] + time_fraction * distances[j, 1])
            vec_path_z = vz - (start_points[j, 2] + time_fraction * distances[j, 2])

            local_y = (
                vec_path_x * e0[j, 0] + vec_path_y * e0[j, 1] + vec_path_z * e0[j, 2]
            )
            local_z = (
                vec_path_x * e2[j, 0] + vec_path_y * e2[j, 1] + vec_path_z * e2[j, 2]
            )

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

            if local_z > 0:
                is_voxel_melted = is_inside(local_y,local_z,width,height,height_shape_factor)
                if is_voxel_melted:
                    break
            elif local_z < 0:
                is_voxel_melted = is_inside(local_y,local_z,width,depth,depth_shape_factor)
                if is_voxel_melted:
                    break
            else:
                is_voxel_melted = np.abs(local_y) < width/2.0
                if is_voxel_melted:
                    break

        melt_mask[i] = is_voxel_melted

    return melt_mask
