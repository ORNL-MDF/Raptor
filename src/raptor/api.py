import time
from typing import List, Tuple, Optional
from numba.typed import List as numbaList
import numpy as np
import vtk
from vtk.util import numpy_support
from skimage import measure
from skimage.morphology import remove_small_objects

from .utilities import ScanPathBuilder
from .structures import MeltPool, PathVector, Grid, Bezier
from .io import read_scan_path
from .core import compute_melt_mask


def create_grid(
    voxel_resolution: float,
    *,
    path_vectors: Optional[List[PathVector]] = None,
    bound_box: Optional[np.ndarray] = None,
) -> Grid:

    return Grid(
        voxel_resolution=voxel_resolution,
        path_vectors=path_vectors,
        bound_box=bound_box,
    )


def create_path_vectors(
    bound_box: np.ndarray,
    power: float,
    scan_speed: float,
    hatch_spacing: float,
    layer_height: float,
    rotation: float,
    scan_extension: float,
    extra_layers: int,
) -> List[PathVector]:

    scan_path_builder = ScanPathBuilder(
        bound_box,
        power,
        scan_speed,
        hatch_spacing,
        layer_height,
        rotation,
        scan_extension,
        extra_layers,
    )

    scan_path_builder.generate_layers()
    scan_path_builder.construct_vectors()
    return scan_path_builder.process_vectors()


def compute_spectral_components(melt_pool_data, n_modes) -> np.ndarray:

    dt = melt_pool_data[1, 0] - melt_pool_data[0, 0]
    mode0 = melt_pool_data[:, 1].mean()
    fft_resolution = np.fft.fft(melt_pool_data[:, 1])
    F = np.zeros_like(fft_resolution)
    n_fft = len(fft_resolution)

    for i in range(1, n_modes):
        F[i] = fft_resolution[i]
        F[n_fft - i] = fft_resolution[n_fft - i]

    frequencies = np.float64(1 / (dt * n_fft)) * np.arange(n_modes, dtype=np.float64)
    phases = np.float64(np.angle(F[:n_modes]))
    amplitudes = np.float64(np.abs(F[:n_modes]) / n_fft)

    if n_modes == 1:
        spectral_array = np.array([[mode0, 0, 0]])
    else:
        spectral_array = np.vstack(
            [
                np.array([mode0, 0, 0]),
                np.vstack([amplitudes[1:], frequencies[1:], phases[1:]]).transpose(),
            ]
        )
    return np.float64(spectral_array)


def create_melt_pool(melt_pool_dict: dict, enable_random_phases: bool) -> MeltPool:

    processed_components = {}
    max_modes = 0

    # 1. Determine the maximum number of modes required.
    for _, _, n_modes in melt_pool_dict.values():
        max_modes = max(max_modes, n_modes)

    # 2. Process each component into its spectral format
    for key, (data, scale, n_modes) in melt_pool_dict.items():
        # Option A: Input data is a raw time-series [time, value]
        if data.shape[1] == 2:
            max_dimension = data[:, 1].max()
            spectral_array = compute_spectral_components(data, n_modes)
            spectral_array[0, 0] *= scale

        # Option B: Input data is a spectral array [amplitude, frequency, phase]
        if data.shape[1] == 3:
            amplitudes = amplitudes[:, 0]
            max_dimension = np.sum(amplitudes)
            spectral_array = data.copy()

        # Pad the array with zeros if it has fewer modes than the max.
        current_modes = spectral_array.shape[0]
        if current_modes < max_modes:
            pad_width = spectral_array.shape[1]
            pad_array = np.zeros(shape=(max_modes - current_modes, pad_width))
            spectral_array = np.vstack([spectral_array, pad_array])

        # Store the final, padded spectral array and the calculated max ratio.
        processed_components[key] = (spectral_array, max_dimension)

    # 3. Create the MeltPool object
    width_oscillations, width_max = processed_components["width"]
    depth_oscillations, depth_max = processed_components["depth"]
    height_oscillations, height_max = processed_components["height"]

    melt_pool = MeltPool(
        width_oscillations,
        depth_oscillations,
        height_oscillations,
        width_max,
        depth_max,
        height_max,
        enable_random_phases,
    )

    return melt_pool


"""
def create_scan_path_vectors(
    scan_file_paths: List[str],
    layer_height,
    bound_box: Optional[np.ndarray] = None
) -> List[PathVector]:
    all_vectors = numbaList()

    time_offset = 0.0
    start_L = 0
    end_L = len(scan_file_paths)
    rve_bb_infl_factor = 0.1
    if bound_box is not None:
        bBx0, bBy0, bBz0 = bound_box[0]
        bBx1, bBy1, bBz1 = bound_box[1]
        bbx0_infl, bBy0_infl, bBz0_infl = bound_box[0] * (1 - rve_bb_infl_factor)
        bbx1_infl, bBy1_infl, bBz1_infl = bound_box[1] * (1 + rve_bb_infl_factor)
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
            if bound_box is not None:
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
    return all_vectors
"""


def compute_porosity(
    grid: Grid,
    path_vectors: List[PathVector],
    melt_pool: MeltPool,
    bezier: Bezier
) -> None:
    """
    Main computation: computes porosity field.
    """

    print("JIT Warmup...")
    t_start_warmup = time.time()

    # Warm up the vecto property assignment.
    if path_vectors:
        path_vectors[0].set_melt_pool_properties(melt_pool)

    # Warm up the main, parallelized compute kernel.
    if grid.n_voxels > 0 and path_vectors:
        _ = compute_melt_mask(grid.voxels[0:1], melt_pool, path_vectors[0:1], bezier)
    
    print(f" -> JIT warmup complete ({time.time() - t_start_warmup:.8f}s).")

    # --- Main Computation ---

    # 1. Assign physics-based properties (AABB, phases) to all path vectors.
    print("Preparing path vectors for simulation...")
    t0_setup = time.time()
    for vector in path_vectors:
        vector.set_melt_pool_properties(melt_pool)
    print(f" -> Vector preparation complete ({time.time() - t0_setup:.8f}s).")

    print("Running melt-mask calculation...")
    t0_run = time.time()
    melted_mask_flat = compute_melt_mask(grid.voxels, melt_pool, path_vectors, bezier)
    t_elapsed = time.time() - t0_run
    
    n_melted = melted_mask_flat.sum()
    print(
        f" -> Melt-mask computation complete ({t_elapsed:.8f}s). "
        f"Melted {n_melted} of {grid.n_voxels} voxels."
    )

    # 3. Reshape the flat mask into the final 3D porosity field.
    porosity_field = (~melted_mask_flat).astype(np.int8).reshape(grid.shape, order="C")

    return porosity_field


def write_vtk(
    origin: np.ndarray,
    voxel_resolution: float,
    porosity: np.ndarray,
    vtk_output_path: str,
) -> None:
    """
    Generates porosity VTK.
    """

    imageData = vtk.vtkImageData()

    nx, ny, nz = porosity.shape

    imageData.SetDimensions(nx, ny, nz)
    imageData.SetOrigin(origin[0], origin[1], origin[2])
    imageData.SetSpacing(voxel_resolution, voxel_resolution, voxel_resolution)

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
    porosity: np.ndarray, voxel_resolution: float, morphology_fields: List[str]
) -> np.ndarray:
    """
    Extracts pores, computes morphology features.
    """
    labeled_defects = measure.label(porosity, connectivity=3)
    minsize = 2
    filtered_defects = remove_small_objects(labeled_defects, minsize)

    return measure.regionproperties_table(
        filtered_defects, spacing=voxel_resolution, properties=morphology_fields
    )


def write_morphology(properties: dict, morphology_output_path: str) -> None:
    """
    Writes morphology output as a .csv.
    """
    columns = ",".join([key for key in properties.keys()])

    morphology = np.vstack([properties[key] for key in properties.keys()]).transpose()

    np.savetxt(
        morphology_output_path, morphology, header=columns, delimiter=",", comments=""
    )

    print(
        f"Morphology features of {morphology.shape[0]} "
        f"defects written to: {morphology_output_path}"
    )
