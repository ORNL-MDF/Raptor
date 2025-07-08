import time
from typing import List, Tuple, Optional
from numba.typed import List as numbaList
import numpy as np
import vtk
from vtk.util import numpy_support
from skimage import measure
from skimage.morphology import remove_small_objects


from .structures import MeltPool
from .io import read_scan_path
from .core_numba import compute_melt_mask, local_frame_2d


def compute_spectral_components(mp_data, nmodes):
    # mp_data assumed to be np.ndarray(N,2)
    # where N is the number of timesteps in the input data
    dt = mp_data[1, 0] - mp_data[0, 0]
    mode0 = mp_data[:, 1].mean()
    fft_res = np.fft.fft(mp_data[:, 1])
    F = np.zeros_like(fft_res)
    for i in range(1, nmodes):
        F[i] = fft_res[i]
        F[len(fft_res) - i] = fft_res[len(fft_res) - i]

    freqs = np.float64(1 / (dt * len(fft_res))) * np.arange(nmodes, dtype=np.float64)
    exp_phases = np.float64(np.angle(F[:nmodes]))
    amps = np.float64(np.abs(F[:nmodes]) / len(fft_res))
    if nmodes==1:
        spectral_array = np.array([[mode0, 0 , 0]])
    else:
        spectral_array = np.vstack(
            [
                np.array([mode0, 0, 0]),
                np.vstack([amps[1:], freqs[1:], exp_phases[1:]]).transpose(),
            ]
        )
    return np.float64(spectral_array)


def construct_meltpool(mp_full_data: dict, en_rand_ph: bool) -> dict:
    # passing in the data read from file and consolidating the spectral components
    osc_dict = {}
    max_modes = 0
    for key in mp_full_data.keys():
        mp_data, scale, nmodes = mp_full_data[key]
        max_modes = np.max([nmodes, max_modes])
        if mp_data.shape[1] == 2:
            # measurement sequence
            max_ratio = mp_data[:, 1].max() / mp_data[:, 1].mean()
            spectral_components = compute_spectral_components(mp_data, nmodes)
            spectral_components[:, 0] *= scale
            osc_dict[key] = (spectral_components, max_ratio)
        elif mp_data.shape[1] == 3:
            # spectral array
            # compute max ratio from sum of amplitudes/mode0 amplitude
            # assuming that the number of modes are small enough s.t. perfect
            # constructive interference doesn't yield a very large max_ratio
            As = mp_data[:, 0]
            max_ratio = np.sum(As) / As[0]
            osc_dict[key] = (mp_data, max_ratio)

    # pad the spectral components
    for key in osc_dict.keys():
        if osc_dict[key][0].shape[0] < max_modes:
            pad = np.zeros(
                shape=(max_modes - osc_dict[key][0].shape[0], osc_dict[key][0].shape[1])
            )
            osc_dict[key] = (np.vstack([osc_dict[key][0], pad]), osc_dict[key][1])
    # unpack meltpool arguments
    osc_W, max_rW = osc_dict["width"]
    osc_Dm, max_rDm = osc_dict["depth"]
    osc_Dh, max_rDh = osc_dict["hump"]
    mp = MeltPool(osc_W, osc_Dm, osc_Dh, max_rW, max_rDm, max_rDh, en_rand_ph)
    return mp


def compute_porosity(
    scan_file_paths: List[str],
    layer_height: float,
    voxel_res: float,
    n_bezier_pts_half: int,
    meltpool: MeltPool,
    boundBox: Optional[np.ndarray] = None,
) -> None:
    """
    Main computation: computes porosity field.
    """
    all_vectors = numbaList()

    time_offset = 0.0
    start_L = 0
    end_L = len(scan_file_paths)
    rve_bb_infl_factor = 0.1
    if boundBox is not None:
        bBx0, bBy0, bBz0 = boundBox[0]
        bBx1, bBy1, bBz1 = boundBox[1]
        bbx0_infl, bBy0_infl, bBz0_infl = boundBox[0] * (1 - rve_bb_infl_factor)
        bbx1_infl, bBy1_infl, bBz1_infl = boundBox[1] * (1 + rve_bb_infl_factor)
        start_L = int(np.max([np.floor(bBz0 / layer_height) - 1, start_L]))
        end_L = int(np.min([bBz1 / layer_height, end_L]))
    print(f"Starting at L{start_L}, ending at L{end_L-1}")
    for layer_idx, sfp in enumerate(scan_file_paths[start_L:end_L]):
        z_offset = layer_idx * layer_height
        print(f"Reading Layer {start_L+layer_idx}: {sfp}")
        active_vectors = read_scan_path(sfp)
        if not active_vectors:
            print(f"Warning: No segments in L{start_L+layer_idx} ({sfp}). Skipping.")
            continue
        max_t_layer = active_vectors[-1].start_t if active_vectors else 0.0
        for vec in active_vectors:
            vec.start_coord[2] += z_offset
            vec.end_coord[2] += z_offset
            vec.start_t += time_offset
            vec.end_t += time_offset

            vec_bb = np.array(
                [
                    [vec.start_coord[0], vec.start_coord[1]],
                    [vec.end_coord[0], vec.end_coord[1]],
                ]
            )
            if boundBox is not None:
                # check if path is inside the inflated rve bounding box
                if (
                    (np.max(vec_bb[:, 0]) < np.min([bbx0_infl, bbx1_infl]))
                    or (np.min(vec_bb[:, 0]) > np.max([bbx0_infl, bbx1_infl]))
                    or (np.max(vec_bb[:, 1]) < np.min([bBy0_infl, bBy1_infl]))
                    or (np.min(vec_bb[:, 1]) > np.max([bBy0_infl, bBy1_infl]))
                ):
                    continue
                else:
                    all_vectors.append(vec)
            else:
                all_vectors.append(vec)
        if active_vectors:
            time_offset += max_t_layer

    if not all_vectors:
        raise ValueError("No segments found in any scan path files.")

    print(f"Total active (exposure) vectors: {len(all_vectors)}")

    meltpool.max_modes = int(
        np.max(
            [
                meltpool.osc_info_W.shape[0],
                meltpool.osc_info_Dm.shape[0],
                meltpool.osc_info_Dh.shape[0],
            ]
        )
    )

    for j in range(len(all_vectors)):
        if meltpool.enable_rand_phs:
            # ensure 0 phase shift for the zeroth mode (mean)
            all_vectors[j].ph = np.float64(
                np.hstack(
                    [
                        0,
                        np.random.uniform(
                            low=0, high=2 * np.pi, size=meltpool.max_modes - 1
                        ),
                    ]
                )
            )
        else:
            all_vectors[j].ph = meltpool.osc_info_W[:, 2].astype(np.float64) # experimental phases

    t_p = np.linspace(0, 1, n_bezier_pts_half)  # Bezier t parameter
    Bmcs_np = np.empty((n_bezier_pts_half, 4))  # Bezier basis coeffs
    omt = 1.0 - t_p
    tsq = t_p * t_p
    omtsq = omt * omt
    Bmcs_np[:, 0] = omtsq * omt
    Bmcs_np[:, 1] = 3.0 * t_p * omtsq
    Bmcs_np[:, 2] = 3.0 * tsq * omt
    Bmcs_np[:, 3] = tsq * t_p

    # compute oscillation
    maxW = meltpool.base_W * meltpool.max_rW
    maxDm = meltpool.base_Dm * meltpool.max_rDm
    maxDh = meltpool.base_Dh * meltpool.max_rDh
    minf_aabb = 0.05
    maxW = max(maxW, meltpool.base_W * minf_aabb)
    maxDm = max(maxDm, meltpool.base_Dm * minf_aabb)
    maxDh = max(0.0, maxDh)  # Hump cannot be negative
    if meltpool.base_Dh > 1e-9:
        maxDh = max(maxDh, meltpool.base_Dh * minf_aabb)
    else:
        maxDh = 0.0

    max_r_xy = maxW / 2.0  # Max melt radius in XY for AABB

    for j in range(len(all_vectors)):
        dx, dy = all_vectors[j].end_coord[:2] - all_vectors[j].start_coord[:2]
        all_vectors[j].ew, all_vectors[j].es, all_vectors[j].ed = local_frame_2d(dx, dy)
        p0x, p0y, p0z = all_vectors[j].start_coord
        p1x, p1y, p1z = all_vectors[j].end_coord

        smx, sMx = min(p0x, p1x), max(p0x, p1x)
        smy, sMy = min(p0y, p1y), max(p0y, p1y)
        smz, sMz = min(p0z, p1z), max(p0z, p1z)

        # segment OBB
        all_vectors[j].centroid = np.array(
            [(smx + sMx) / 2, (smy + sMy) / 2, (smz + sMz) / 2]
        )
        all_vectors[j].lx = maxW / 2
        all_vectors[j].ly = np.hypot(sMx - smx, sMy - smy) / 2
        all_vectors[j].lz = (maxDh + maxDm) / 2

        # segment AABB
        all_vectors[j].aabb = np.array(
            [
                smx - max_r_xy,
                sMx + max_r_xy,
                smy - max_r_xy,
                sMy + max_r_xy,
                smz - maxDm,
                sMz + maxDh,
            ],
            dtype=np.float64,
        )

    if boundBox is None:
        all_pts_np = np.vstack(
            (
                [vec.start_coord for vec in all_vectors],
                [vec.end_coord for vec in all_vectors],
            )
        )
        xmin, ymin, zmin = all_pts_np.min(axis=0)
        xmax, ymax, zmax = all_pts_np.max(axis=0)

        bufxy = maxW * 2.0
        gx0, gx1 = xmin - bufxy, xmax + bufxy
        gy0, gy1 = ymin - bufxy, ymax + bufxy
        gz0, gz1 = zmin - 1.5 * maxDm, zmax + 1.5 * maxDh
    else:
        xmin, ymin, zmin = boundBox[0]
        xmax, ymax, zmax = boundBox[1]
        gx0, gy0, gz0 = boundBox[0]
        gx1, gy1, gz1 = boundBox[1]

    xg = np.arange(gx0, gx1 + voxel_res / 2.0, voxel_res)
    yg = np.arange(gy0, gy1 + voxel_res / 2.0, voxel_res)
    zg = np.arange(gz0, gz1 + voxel_res / 2.0, voxel_res)

    nx, ny, nz = len(xg), len(yg), len(zg)
    total_vox = nx * ny * nz
    if total_vox == 0:
        msg = f"Warning: Voxel grid empty ({nx}x{ny}x{nz}). No VTK."
        print(msg)
        return

    print(
        f"Grid: {nx}x{ny}x{nz} = {total_vox} vox. "
        f"Z-scan:[{zmin:.2e},{zmax:.2e}], Z-grid:[{gz0:.2e},{gz1:.2e}]"
    )

    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
    vox_np = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.copy()

    print("Running Numba-accelerated melt-mask calculation...")
    t0 = time.time()
    melted = compute_melt_mask(
        vox_np, Bmcs_np, n_bezier_pts_half, meltpool, all_vectors
    )

    t_elapsed = time.time() - t0
    n_melted = melted.sum()
    print(
        f" -> Melt-mask computation: {t_elapsed:.1f}s. "
        f"Melted {n_melted}/{total_vox} voxels. "
        f"{total_vox-n_melted} unmelted voxels."
    )

    porosity = (~melted).astype(np.int8).reshape((nx, ny, nz), order="C")
    return np.array([xg[0], yg[0], zg[0]]), porosity


def write_vtk(
    origin: np.ndarray, voxel_res: float, porosity: np.ndarray, vtk_output_path: str
) -> None:
    """
    Generates porosity VTK.
    """

    imageData = vtk.vtkImageData()

    nx, ny, nz = porosity.shape

    imageData.SetDimensions(nx, ny, nz)
    imageData.SetOrigin(origin[0], origin[1], origin[2])
    imageData.SetSpacing(voxel_res, voxel_res, voxel_res)

    porosity_vtk_order = np.transpose(porosity, (2, 1, 0))

    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=porosity_vtk_order.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    vtk_data_array.SetName("porosity")
    imageData.GetPointData().SetScalars(vtk_data_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vtk_output_path)
    writer.SetInputData(imageData)
    writer.SetDataModeToBinary()

    writer.Write()
    del porosity
    del porosity_vtk_order

    print(f"VTK porosity map written to: {vtk_output_path}")

def compute_morphology(
        porosity : np.ndarray,
        voxel_res: float,
        morph_fields: List[str]
) -> np.ndarray:
    """
    Extracts pores, computes morphology features.
    """
    labeled_defects = measure.label(porosity,connectivity=3)
    minsize=2
    filtered_defects = remove_small_objects(labeled_defects,minsize)
    return measure.regionprops_table(filtered_defects,spacing=voxel_res,properties=morph_fields)

def write_morphology(props: dict,
                     morphology_output_path: str
)-> None:
    """
    Writes morphology output as a .csv.
    Note that props must be list of RegionProperties from skimage.measure.
    """
    columns = ','.join([key for key in props.keys()])
    morph_arr = np.vstack([props[key] for key in props.keys()]).transpose()
    np.savetxt(
        morphology_output_path,
        morph_arr,
        header=columns,
        delimiter=',',
        comments=''
    )
    print(f"Morphology features of {morph_arr.shape[0]} defects written to: {morphology_output_path}")

