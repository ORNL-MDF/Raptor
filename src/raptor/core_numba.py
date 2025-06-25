import numpy as np
from numba import jit, njit, prange
from typing import List, Tuple
from .structures import MeltPool, PathVector
import time


def local_frame_2d(dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a local 2D frame based on a direction vector (dx, dy).

    Args:
        dx: X-component of the segment direction.
        dy: Y-component of the segment direction.

    Returns:
        Tuple (e_width, e_depth_dir):
        e_width: Unit vector for width direction in XY plane.
        e_depth_dir: Unit vector for depth direction (global Z-axis).
    """
    Lxy = np.hypot(dx, dy)
    if Lxy < 1e-12:
        e_width = np.array([1.0, 0.0, 0.0])
        e_scan = np.array([0.0, 1.0, 0.0])
        e_depth_dir = np.array([0.0, 0.0, 1.0])
    else:
        e_width = np.array([-dy / Lxy, dx / Lxy, 0.0])
        e_scan = np.array([dx / Lxy, dy / Lxy, 0.0])
        e_depth_dir = np.array([0.0, 0.0, 1.0])
    return e_width, e_scan, e_depth_dir


@njit
def bezier_vertices(
    W_d: float, Dh_d: float, Dm_d: float, n_pts_h: int, B_m_c: np.ndarray
) -> np.ndarray:
    """
    Calculates vertices of a 2D Bezier polygon for melt pool cross-section.

    Args:
        W_d: Current width of the melt pool.
        Dh_d: Current hump depth/height.
        Dm_d: Current melt depth.
        n_pts_h: Number of points per half Bezier curve (must be >= 2).
        B_m_c: Bezier basis function values (Bernstein polynomials).

    Returns:
        A NumPy array of (x,y) vertices for the polygon.
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

    top_s = np.empty((n_pts_h, 2), dtype=np.float64)
    bot_s = np.empty((n_pts_h, 2), dtype=np.float64)

    for i in range(n_pts_h):
        top_s[i, 0] = (
            B_m_c[i, 0] * CPt[0, 0]
            + B_m_c[i, 1] * CPt[1, 0]
            + B_m_c[i, 2] * CPt[2, 0]
            + B_m_c[i, 3] * CPt[3, 0]
        )
        top_s[i, 1] = (
            B_m_c[i, 0] * CPt[0, 1]
            + B_m_c[i, 1] * CPt[1, 1]
            + B_m_c[i, 2] * CPt[2, 1]
            + B_m_c[i, 3] * CPt[3, 1]
        )
        bot_s[i, 0] = (
            B_m_c[i, 0] * CPb[0, 0]
            + B_m_c[i, 1] * CPb[1, 0]
            + B_m_c[i, 2] * CPb[2, 0]
            + B_m_c[i, 3] * CPb[3, 0]
        )
        bot_s[i, 1] = (
            B_m_c[i, 0] * CPb[0, 1]
            + B_m_c[i, 1] * CPb[1, 1]
            + B_m_c[i, 2] * CPb[2, 1]
            + B_m_c[i, 3] * CPb[3, 1]
        )

    n_poly = n_pts_h + (n_pts_h - 1)
    poly = np.empty((n_poly, 2), dtype=np.float64)

    poly[:n_pts_h] = top_s
    poly[n_pts_h:] = bot_s[1:]
    return poly


@njit
def point_in_poly(xt: float, yt: float, poly_v: np.ndarray) -> bool:
    """
    Checks if a point is inside a polygon using a fully branchless (predicated)
    algorithm suitable for high-throughput GPU execution.
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


@njit(inline="always")
def _cosine_expansion(
    t_osc: np.ndarray, osc_a: np.ndarray, osc_f: np.ndarray, osc_p: np.ndarray
) -> float:
    dim_terms = 0
    for k_n in range(osc_a.shape[0]):
        dim_terms += osc_a[k_n] * np.cos(2 * np.pi * osc_f[k_n] * t_osc + osc_p[k_n])
    return dim_terms


@njit(parallel=True, fastmath=True)
def compute_melt_mask(
    vox_g: np.ndarray,
    Bmc_s_g: np.ndarray,
    n_pts_b_h_g: int,
    meltpool: MeltPool,
    active_vectors: List[PathVector],
) -> np.ndarray:
    """
    Computes a boolean mask indicating which voxels are melted by any scan segment.
    """
    # construct array for melt mask
    n_vox = vox_g.shape[0]
    melt_mask_g = np.zeros(n_vox, dtype=np.bool_)

    # unpack meltpool properties
    osc_W, osc_Dm, osc_Dh = (
        meltpool.osc_info_W,
        meltpool.osc_info_Dm,
        meltpool.osc_info_Dh,
    )
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
    seg_slsq_g = np.zeros(n_seg)
    seg_centroids = np.zeros(shape=(n_seg, 3))
    seg_lx = np.zeros(n_seg)
    seg_ly = np.zeros(n_seg)
    seg_lz = np.zeros(n_seg)

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
        seg_slsq_g[j] = vec.slsq
        seg_centroids[j, :] = vec.centroid
        seg_lx[j] = vec.lx
        seg_ly[j] = vec.ly
        seg_lz[j] = vec.lz

    for i_v in prange(n_vox):
        vx, vy, vz = vox_g[i_v, 0], vox_g[i_v, 1], vox_g[i_v, 2]
        is_voxel_melted = False

        for j_s in range(n_seg):
            # coarse aabb check
            if (
                vx < s_infl_aabb_g[j_s, 0]
                or vx > s_infl_aabb_g[j_s, 1]
                or vy < s_infl_aabb_g[j_s, 2]
                or vy > s_infl_aabb_g[j_s, 3]
                or vz < s_infl_aabb_g[j_s, 4]
                or vz > s_infl_aabb_g[j_s, 5]
            ):
                continue
            # fine obb check
            vref = vox_g[i_v] - seg_centroids[j_s]
            ew_dir = ew_g[j_s] * (
                vref[0] * ew_g[j_s] + vref[1] * ew_g[j_s, 1] + vref[2] * ew_g[j_s, 2]
            )
            proj_ew_vref = (ew_dir[0] ** 2 + ew_dir[1] ** 2 + ew_dir[2] ** 2) ** 0.5
            es_dir = es_g[j_s] * (
                vref[0] * es_g[j_s, 0] + vref[1] * es_g[j_s, 1] + vref[2] * es_g[j_s, 2]
            )
            proj_es_vref = (es_dir[0] ** 2 + es_dir[1] ** 2 + es_dir[2] ** 2) ** 0.5
            ed_dir = ed_g[j_s] * (
                vref[0] * ed_g[j_s, 0] + vref[1] * ed_g[j_s, 1] + vref[2] * ed_g[j_s, 2]
            )
            proj_ed_vref = (ed_dir[0] ** 2 + ed_dir[1] ** 2 + ed_dir[2] ** 2) ** 0.5
            if not (
                proj_ew_vref < seg_lx[j_s]
                and proj_es_vref < seg_ly[j_s]
                and proj_ed_vref < seg_lz[j_s]
            ):
                continue

            p0x, p0y, p0z = ss_g[j_s, 0], ss_g[j_s, 1], ss_g[j_s, 2]
            p1x, p1y, p1z = se_g[j_s, 0], se_g[j_s, 1], se_g[j_s, 2]

            sdx, sdy, sdz = p1x - p0x, p1y - p0y, p1z - p0z
            len_sq = sdx * sdx + sdy * sdy + sdz * sdz

            v0x, v0y, v0z = vx - p0x, vy - p0y, vz - p0z
            dotp = v0x * sdx + v0y * sdy + v0z * sdz

            tu = dotp / len_sq if len_sq > 1e-24 else 0.0

            tu_clamped = max(0.0, min(1.0, tu))

            t_osc = sst_g[j_s] + tu_clamped * (set_g[j_s] - sst_g[j_s])

            Wd = _cosine_expansion(t_osc, osc_W[:, 0], osc_W[:, 1], seg_rand_phs_g[j_s])
            Dmd = _cosine_expansion(
                t_osc, osc_Dm[:, 0], osc_Dm[:, 1], seg_rand_phs_g[j_s]
            )
            Dhd = _cosine_expansion(
                t_osc, osc_Dh[:, 0], osc_Dh[:, 1], seg_rand_phs_g[j_s]
            )

            cpx = p0x + tu_clamped * sdx
            cpy = p0y + tu_clamped * sdy
            cpz = p0z + tu_clamped * sdz
            dist_sq_to_seg = (vx - cpx) ** 2 + (vy - cpy) ** 2 + (vz - cpz) ** 2

            Lcap_sq = (Wd / 2.0) ** 2
            if dist_sq_to_seg >= Lcap_sq:
                continue

            ptlx, ptly, ptlz = p0x + tu * sdx, p0y + tu * sdy, p0z + tu * sdz
            dpx, dpy, dpz = vx - ptlx, vy - ptly, vz - ptlz
            ewx, ewy, ewz = ew_g[j_s, 0], ew_g[j_s, 1], ew_g[j_s, 2]
            edx, edy, edz = ed_g[j_s, 0], ed_g[j_s, 1], ed_g[j_s, 2]
            xl = dpx * ewx + dpy * ewy + dpz * ewz
            yl = dpx * edx + dpy * edy + dpz * edz

            poly_s = bezier_vertices(Wd, Dhd, Dmd, n_pts_b_h_g, Bmc_s_g)

            if point_in_poly(xl, yl, poly_s):
                is_voxel_melted = True
                break

        melt_mask_g[i_v] = is_voxel_melted

    return melt_mask_g
