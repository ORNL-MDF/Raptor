import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import vtk
from vtk.util import numpy_support
from skimage import measure
from skimage.morphology import remove_small_objects

from .utilities import ScanPathBuilder
from .structures import MeltPool, PathVector, Grid
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
    return scan_path_builder.process_vectors()


def compute_spectral_components(melt_pool_data: np.ndarray, n_modes: int) -> np.ndarray:

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


def create_melt_pool(
    melt_pool_dict: Dict[str, Any], melt_pool_height_shape_factor: float, melt_pool_depth_shape_factor: float, enable_random_phases: bool
) -> MeltPool:

    processed_components: Dict[str, Tuple[np.ndarray, float]] = {}
    max_modes = 0

    # 1. Determine the maximum number of modes required.
    for _, _, n_modes in melt_pool_dict.values():
        max_modes = max(max_modes, n_modes)

    # 2. Process each component into its spectral format
    for key, (data, scale, n_modes) in melt_pool_dict.items():
        # Option A: Input data is a raw time-series [time, value]
        if data.shape[1] == 2:
            spectral_array = compute_spectral_components(data, n_modes)
            spectral_array[:,0] *= scale

        # Option B: Input data is a spectral array [amplitude, frequency, phase]
        elif data.shape[1] == 3:
            spectral_array = data.copy()

        else:
            raise ValueError(
                f"Unsupported data shape: {data.shape}.  Must be [time, value] or [amplitude, frequency, phase]"
            )

        # Pad the array with zeros if it has fewer modes than the max.
        current_modes = spectral_array.shape[0]
        if current_modes < max_modes:
            pad_array = np.zeros(
                shape=(max_modes - current_modes, spectral_array.shape[1]),
                dtype=np.float64,
            )
            spectral_array = np.vstack([spectral_array, pad_array])


        processed_components[key] = spectral_array

    # 3. Create the MeltPool object
    width_oscillations = processed_components["width"]
    depth_oscillations = processed_components["depth"]
    height_oscillations = processed_components["height"]

    melt_pool = MeltPool(
        width_oscillations,
        depth_oscillations,
        height_oscillations,
        width_oscillations[:,0].sum(axis=0),
        depth_oscillations[:,0].sum(axis=0),
        height_oscillations[:,0].sum(axis=0),
        melt_pool_height_shape_factor,
        melt_pool_depth_shape_factor,
        enable_random_phases,
    )

    return melt_pool


def compute_porosity(
    grid: Grid, path_vectors: List[PathVector], melt_pool: MeltPool
) -> None:
    """
    Main computation: computes porosity field.
    """

    print("JIT Warmup...")
    t_start_warmup = time.time()

    # Warm up the vector property assignment.
    if path_vectors:
        path_vectors[0].set_melt_pool_properties(melt_pool)

    # Warm up the main, parallelized compute kernel.
    if grid.n_voxels > 0 and path_vectors:
        _ = compute_melt_mask(grid.voxels[0:1], melt_pool, path_vectors[0:1])

    print(f" -> JIT warmup complete ({time.time() - t_start_warmup:.8f}s).")

    print(f"Preparing {len(path_vectors)} path vectors for simulation...")
    t0_setup = time.time()
    for vector in path_vectors:
        vector.set_melt_pool_properties(melt_pool)
    print(f" -> Vector preparation complete ({time.time() - t0_setup:.8f}s).")

    print("Running melt-mask calculation...")
    t0_run = time.time()
    melted_mask_flat = compute_melt_mask(grid.voxels, melt_pool, path_vectors)
    t_elapsed = time.time() - t0_run

    n_melted = melted_mask_flat.sum()
    print(
        f" -> Melt-mask computation complete ({t_elapsed:.8f}s). "
        f"Melted {n_melted} of {grid.n_voxels} voxels."
    )

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
