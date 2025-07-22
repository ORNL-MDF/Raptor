import numpy as np
from numba import njit, prange
from typing import List, Tuple
from .structures import MeltPool, PathVector


def local_frame_2d(dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a local 2D frame based on a direction vector (dx, dy).

    Args:
        dx: X-component of the segment direction.
        dy: Y-component of the segment direction.

    Returns:
        Tuple (e_width, e_depth_dir):
        e_width: Unit vector for width direction in XY plane.
        e_scan: Unit vector for local scan direction in XY plane.
        e_depth_dir: Unit vector for depth direction (global Z-axis).
    """

    Lxy = np.hypot(dx, dy)
    e_width = (
        np.array([1.0, 0.0, 0.0])
        if Lxy < 1e-12
        else np.array([-dy / Lxy, dx / Lxy, 0.0])
    )
    e_scan = (
        np.array([0.0, 1.0, 0.0])
        if Lxy < 1e-12
        else np.array([dx / Lxy, dy / Lxy, 0.0])
    )
    e_depth_dir = np.array([0.0, 0.0, 1.0])
    return e_width, e_scan, e_depth_dir


@njit(inline="always", fastmath=True)
def bezier_vertices(
    W_d: float,
    Dh_d: float,
    Dm_d: float,
    n_pts_h: int,
    B_m_c: np.ndarray,
    poly_out: np.ndarray,
):
    """
    Calculates vertices of a 2D Bezier polygon for melt pool cross-section.
    """

    s = 4.0 / 3.0
    Ph_c = s * Dh_d
    Pm_c = s * Dm_d

    CPt = np.empty((4, 2), dtype=np.float64)
    CPt[0, 0] = -W_d / 2.0
    CPt[0, 1] = 0.0
    CPt[1, 0] = -W_d / 4.0
    CPt[1, 1] = Ph_c
    CPt[2, 0] = W_d / 4.0
    CPt[2, 1] = Ph_c
    CPt[3, 0] = W_d / 2.0
    CPt[3, 1] = 0.0

    CPb = np.empty((4, 2), dtype=np.float64)
    CPb[0, :] = CPt[3, :]
    CPb[1, 0] = W_d / 4.0
    CPb[1, 1] = -Pm_c
    CPb[2, 0] = -W_d / 4.0
    CPb[2, 1] = -Pm_c
    CPb[3, :] = CPt[0, :]

    for i in range(n_pts_h):
        top_x = (
            B_m_c[i, 0] * CPt[0, 0]
            + B_m_c[i, 1] * CPt[1, 0]
            + B_m_c[i, 2] * CPt[2, 0]
            + B_m_c[i, 3] * CPt[3, 0]
        )
        top_y = (
            B_m_c[i, 0] * CPt[0, 1]
            + B_m_c[i, 1] * CPt[1, 1]
            + B_m_c[i, 2] * CPt[2, 1]
            + B_m_c[i, 3] * CPt[3, 1]
        )
        poly_out[i, 0] = top_x
        poly_out[i, 1] = top_y

    for i in range(1, n_pts_h):
        bot_x = (
            B_m_c[i, 0] * CPb[0, 0]
            + B_m_c[i, 1] * CPb[1, 0]
            + B_m_c[i, 2] * CPb[2, 0]
            + B_m_c[i, 3] * CPb[3, 0]
        )
        bot_y = (
            B_m_c[i, 0] * CPb[0, 1]
            + B_m_c[i, 1] * CPb[1, 1]
            + B_m_c[i, 2] * CPb[2, 1]
            + B_m_c[i, 3] * CPb[3, 1]
        )
        poly_out[n_pts_h + i - 1, 0] = bot_x
        poly_out[n_pts_h + i - 1, 1] = bot_y


@njit(inline="always", fastmath=True)
def point_in_poly(xt: float, yt: float, poly_v: np.ndarray) -> bool:
    """
    Checks if a point is inside a polygon.
    """

    num_vertices = poly_v.shape[0]
    is_inside_int = 0
    px_prev, py_prev = poly_v[num_vertices - 1]

    for i in range(num_vertices):
        px_curr, py_curr = poly_v[i]

        crosses_y_level = (py_curr > yt) != (py_prev > yt)

        is_left_of_edge = (xt - px_curr) * (py_prev - py_curr) < (px_prev - px_curr) * (
            yt - py_curr
        )

        is_inside_int += crosses_y_level and (is_left_of_edge != (py_curr > py_prev))

        px_prev, py_prev = px_curr, py_curr

    return (is_inside_int % 2) == 1


def compute_melt_mask(
    vox_g: np.ndarray,
    Bmc_s_g: np.ndarray,
    n_pts_b_h_g: int,
    meltpool: MeltPool,
    active_vectors: List[PathVector],
    spatter_centroid: np.ndarray,
    spatter_r: float
):
    """
    Unpacks jitclasses into arrays to pass to compute_melt_mask_implicit().
    """
    # construct array for melt mask
    n_vox = vox_g.shape[0]
    melt_mask = np.zeros(n_vox, dtype=np.bool_)

    # unpack meltpool properties
    osc_W, osc_Dm, osc_Dh = (
        meltpool.osc_info_W,
        meltpool.osc_info_Dm,
        meltpool.osc_info_Dh,
    )
    osc_w_amp = osc_W[:, 0]
    osc_w_freq = osc_W[:, 1]

    osc_dm_amp = osc_Dm[:, 0]
    osc_dm_freq = osc_Dm[:, 1]

    osc_dh_amp = osc_Dh[:, 0]
    osc_dh_freq = osc_Dh[:, 1]
    max_modes = meltpool.max_modes

    # construct arrays for segment properties
    n_seg = len(active_vectors)
    ss_g = np.zeros(shape=(n_seg, 3))
    se_g = np.zeros(shape=(n_seg, 3))
    ew_g = np.zeros(shape=(n_seg, 3))
    es_g = np.zeros(shape=(n_seg, 3))
    ed_g = np.zeros(shape=(n_seg, 3))
    sst_g = np.zeros(n_seg)
    set_g = np.zeros(n_seg)
    s_infl_aabb_g = np.zeros(shape=(n_seg, 6))
    seg_rand_phs_g = np.zeros(shape=(n_seg, max_modes))
    seg_centroids = np.zeros(shape=(n_seg, 3))
    seg_lx = np.zeros(n_seg)
    seg_ly = np.zeros(n_seg)

    # unpack segment properties
    for j in range(n_seg):
        vec = active_vectors[j]
        ss_g[j, :] = vec.start_coord
        se_g[j, :] = vec.end_coord
        ew_g[j, :] = vec.ew
        es_g[j, :] = vec.es
        ed_g[j, :] = vec.ed
        sst_g[j] = vec.start_t
        set_g[j] = vec.end_t
        s_infl_aabb_g[j, :] = vec.aabb
        seg_rand_phs_g[j, :] = vec.ph
        seg_centroids[j, :] = vec.centroid
        seg_lx[j] = vec.lx
        seg_ly[j] = vec.ly

    n_poly_pts = n_pts_b_h_g + (n_pts_b_h_g - 1)

    return compute_melt_mask_implicit(
        n_vox,
        vox_g,
        melt_mask,
        n_pts_b_h_g,
        n_seg,
        ss_g,
        se_g,
        ew_g,
        es_g,
        ed_g,
        sst_g,
        set_g,
        s_infl_aabb_g,
        seg_rand_phs_g,
        seg_centroids,
        seg_lx,
        seg_ly,
        Bmc_s_g,
        n_poly_pts,
        osc_w_amp,
        osc_w_freq,
        osc_dm_amp,
        osc_dm_freq,
        osc_dh_amp,
        osc_dh_freq,
        spatter_centroid,
        spatter_r
    )


@njit(parallel=True, fastmath=True)
def compute_melt_mask_implicit(
    n_vox: int,
    vox_g: np.ndarray,
    melt_mask: np.ndarray,
    n_pts_b_h_g: int,
    n_seg: int,
    ss_g: np.ndarray,
    se_g: np.ndarray,
    ew_g: np.ndarray,
    es_g: np.ndarray,
    ed_g: np.ndarray,
    sst_g: np.ndarray,
    set_g: np.ndarray,
    s_infl_aabb_g: np.ndarray,
    seg_rand_phs_g: np.ndarray,
    seg_centroids: np.ndarray,
    seg_lx: np.ndarray,
    seg_ly: np.ndarray,
    Bmc_s_g: np.ndarray,
    n_poly_pts: int,
    osc_w_amp: np.ndarray,
    osc_w_freq: np.ndarray,
    osc_dm_amp: np.ndarray,
    osc_dm_freq: np.ndarray,
    osc_dh_amp: np.ndarray,
    osc_dh_freq: np.ndarray,
    spatter_centroid: np.ndarray,
    spatter_r: float
) -> np.ndarray:
    """
    Implicit compute melt mask function optimized for parallelization.
    """

    centroid = spatter_centroid
    print("centroid", centroid)
    radius = spatter_r

    for i_v in prange(n_vox):
        vx, vy, vz = vox_g[i_v, 0], vox_g[i_v, 1], vox_g[i_v, 2]
        is_voxel_melted = False

        poly_s = np.empty((n_poly_pts, 2), dtype=np.float64)

        for j_s in range(n_seg):
            # 1. Coarse AABB check
            if (
                vx < s_infl_aabb_g[j_s, 0]
                or vx > s_infl_aabb_g[j_s, 1]
                or vy < s_infl_aabb_g[j_s, 2]
                or vy > s_infl_aabb_g[j_s, 3]
                or vz < s_infl_aabb_g[j_s, 4]
                or vz > s_infl_aabb_g[j_s, 5]
            ):
                continue

            # 2. Fine OBB check
            vref_x = vx - seg_centroids[j_s, 0]
            vref_y = vy - seg_centroids[j_s, 1]
            vref_z = vz - seg_centroids[j_s, 2]

            dot_es = (
                vref_x * es_g[j_s, 0] + vref_y * es_g[j_s, 1] + vref_z * es_g[j_s, 2]
            )
            if dot_es * dot_es > seg_ly[j_s] * seg_ly[j_s]:
                continue

            dot_ew = (
                vref_x * ew_g[j_s, 0] + vref_y * ew_g[j_s, 1] + vref_z * ew_g[j_s, 2]
            )
            if dot_ew * dot_ew > seg_lx[j_s] * seg_lx[j_s]:
                continue

            # 3. Project voxel onto segment line
            p0x, p0y, p0z = ss_g[j_s, 0], ss_g[j_s, 1], ss_g[j_s, 2]
            p1x, p1y, p1z = se_g[j_s, 0], se_g[j_s, 1], se_g[j_s, 2]
            sdx, sdy, sdz = p1x - p0x, p1y - p0y, p1z - p0z
            len_sq = sdx * sdx + sdy * sdy + sdz * sdz
            v0x, v0y, v0z = vx - p0x, vy - p0y, vz - p0z
            dotp = v0x * sdx + v0y * sdy + v0z * sdz
            tu = dotp / len_sq if len_sq > 1e-24 else 0.0
            tu_clamped = max(0.0, min(1.0, tu))
            t_osc = sst_g[j_s] + tu_clamped * (set_g[j_s] - sst_g[j_s])

            # 4. Compute melt pool dimensions
            Wd = 0
            Dmd = 0
            Dhd = 0

            phase = seg_rand_phs_g[j_s]
            two_pi_t = 2 * np.pi * t_osc

            for k_n in range(phase.shape[0]):
                Wd += osc_w_amp[k_n] * np.cos(two_pi_t * osc_w_freq[k_n] + phase[k_n])
                Dmd += osc_dm_amp[k_n] * np.cos(
                    two_pi_t * osc_dm_freq[k_n] + phase[k_n]
                )
                Dhd += osc_dh_amp[k_n] * np.cos(
                    two_pi_t * osc_dh_freq[k_n] + phase[k_n]
                )

            # 5. Final point in polygon check
            dpx = vx - (p0x + tu * sdx)
            dpy = vy - (p0y + tu * sdy)
            dpz = vz - (p0z + tu * sdz)
            xl = dpx * ew_g[j_s, 0] + dpy * ew_g[j_s, 1] + dpz * ew_g[j_s, 2]
            yl = dpx * ed_g[j_s, 0] + dpy * ed_g[j_s, 1] + dpz * ed_g[j_s, 2]

            segment_center = ss_g[j_s, :] + tu_clamped * (se_g[j_s, :] - ss_g[j_s, :])

            # Overlap particle, set melt pool dimensions to zero
            distanceSqr = (
                (segment_center[0] - centroid[0]) ** 2
                + (segment_center[1] - centroid[1]) ** 2
                + (segment_center[2] - centroid[2]) ** 2
            )
            if distanceSqr < radius * radius:
                Wd = 1e-10
                Dhd = 1e-10
                Dmd = 1e-10

            bezier_vertices(Wd, Dhd, Dmd, n_pts_b_h_g, Bmc_s_g, poly_s)

            if point_in_poly(xl, yl, poly_s):
                is_voxel_melted = True
                break

        # spatter particle is solid
        distanceSqr = (
            (vx - centroid[0]) ** 2 + (vy - centroid[1]) ** 2 + (vz - centroid[2]) ** 2
        )
        if distanceSqr <= radius * radius:
            is_voxel_melted = True

        melt_mask[i_v] = is_voxel_melted

    return melt_mask
