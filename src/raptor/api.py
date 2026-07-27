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
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from .utilities import ScanPathBuilder
from .structures import MeltPool, PathVector, Grid
from .io import read_scan_path
from .core import (
    build_spatial_index,
    compute_melt_mask_grid,
)
from .morphology import (
    collect_zero_indices,
    count_phase_codes,
    label_sparse_defects,
)
from .resources import validate_memory_limit
from .spectral import (
    DEFAULT_SPECTRAL_ERROR_FRACTION,
    validate_spectral_error_fraction,
)


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


def compute_required_modes(data: np.ndarray, reconstruction_rmse: float) -> int:
    """Return the minimum number of real Fourier modes for an RMSE target."""
    return compute_spectral_components(
        data, tolerance=reconstruction_rmse
    ).shape[0]


def compute_spectral_components(
    melt_pool_data: np.ndarray,
    n_modes: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> np.ndarray:
    """Convert a uniformly sampled real signal to a sparse cosine expansion.

    ``n_modes`` includes the mean (DC) mode.  By itself, it retains the
    corresponding low-frequency prefix.  When ``tolerance`` is used, the
    smallest set of Fourier bins whose discarded energy satisfies the requested
    reconstruction RMSE is retained.  If both are provided, ``n_modes`` caps
    that set while preserving its highest-energy bins; the requested tolerance
    cannot be met when the cap is smaller than the required set.
    """
    data = np.asarray(melt_pool_data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < 2:
        raise ValueError("melt_pool_data must have shape (n, 2) with n >= 2.")
    if not np.isfinite(data).all():
        raise ValueError("melt_pool_data must contain only finite values.")
    if n_modes is None and tolerance is None:
        raise ValueError("Provide either n_modes, tolerance, or both.")
    if n_modes is not None and (
        not isinstance(n_modes, (int, np.integer)) or n_modes < 1
    ):
        raise ValueError("n_modes must be a positive integer.")

    time_values = data[:, 0]
    signal = data[:, 1]
    time_steps = np.diff(time_values)
    dt = time_steps[0]
    if dt <= 0.0 or not np.allclose(time_steps, dt, rtol=1.0e-7, atol=0.0):
        raise ValueError(
            "Time samples must be strictly increasing and uniform."
        )

    n_samples = signal.size
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n_samples, d=dt)
    candidate_bins = np.arange(1, fft_values.size)
    if n_modes is not None and n_modes > fft_values.size:
        raise ValueError(
            f"n_modes cannot exceed {fft_values.size} for this time series."
        )

    # Parseval energy represented by each positive-frequency cosine.  Interior
    # rFFT bins represent a conjugate pair; the Nyquist bin does not.
    energy_weights = np.full(fft_values.size, 2.0)
    energy_weights[0] = 1.0
    if n_samples % 2 == 0:
        energy_weights[-1] = 1.0
    if tolerance is not None:
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be a finite, non-negative value.")
        candidate_energy = (
            energy_weights[candidate_bins]
            * np.abs(fft_values[candidate_bins]) ** 2
        )
        energy_indices = np.argsort(candidate_energy)[::-1]
        energy_order = candidate_bins[energy_indices]
        allowed_energy = (tolerance * n_samples) ** 2
        ordered_energy = candidate_energy[energy_indices]
        required_energy = ordered_energy.sum() - allowed_energy
        retained_count = (
            0
            if required_energy <= 0.0
            else np.searchsorted(
                np.cumsum(ordered_energy), required_energy, side="left"
            )
            + 1
        )
        selected_bins = energy_order[:retained_count]
        if n_modes is not None and selected_bins.size > n_modes - 1:
            warnings.warn(
                f"Requested tolerance requires {selected_bins.size + 1} modes; "
                f"limiting the result to {n_modes} modes, so the tolerance "
                "will not be met.",
                RuntimeWarning,
                stacklevel=2,
            )
            selected_bins = selected_bins[: n_modes - 1]
    else:
        selected_bins = candidate_bins[: n_modes - 1]

    # Return modes in frequency order after any energy-ranked selection and cap.
    selected_bins = np.sort(selected_bins)

    amplitudes = energy_weights[selected_bins] * np.abs(
        fft_values[selected_bins]
    )
    amplitudes /= n_samples
    phases = np.angle(fft_values[selected_bins])
    # FFT phases are relative to sample zero; account for an absolute time axis.
    phases -= 2.0 * np.pi * frequencies[selected_bins] * time_values[0]

    spectral_array = np.empty((selected_bins.size + 1, 3), dtype=np.float64)
    spectral_array[0] = (signal.mean(), 0.0, 0.0)
    spectral_array[1:, 0] = amplitudes
    spectral_array[1:, 1] = frequencies[selected_bins]
    spectral_array[1:, 2] = phases
    return spectral_array


def create_melt_pool(
    melt_pool_dict: Dict[str, Any],
    enable_random_phases: bool,
    tolerance: Optional[float] = None,
) -> MeltPool:

    processed_components: Dict[str, np.ndarray] = {}
    required_dimensions = ("width", "depth", "height")
    missing_dimensions = [
        dimension
        for dimension in required_dimensions
        if dimension not in melt_pool_dict
    ]
    if missing_dimensions:
        raise ValueError(
            "melt_pool_dict is missing required dimensions: "
            + ", ".join(missing_dimensions)
        )

    for key in required_dimensions:
        try:
            data, n_modes, scale, shape_factor = melt_pool_dict[key]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key} must be a (data, n_modes, scale, shape_factor) tuple."
            ) from exc

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{key} scale must be finite and positive.")
        if not np.isfinite(shape_factor) or shape_factor <= 0.0:
            raise ValueError(f"{key} shape factor must be finite and positive.")
        if key == "width" and shape_factor != 2.0:
            raise ValueError(
                "The width shape factor must be 2.0; the optimized "
                "transverse exponent is fixed."
            )

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 1:
            raise ValueError(
                f"Unsupported data shape for {key}: {data.shape}. "
                "Must be [time, value] or "
                "[amplitude, frequency, phase]."
            )
        if not np.isfinite(data).all():
            raise ValueError(f"{key} data must contain only finite values.")

        # Option A: Input data is a raw time-series [time, value]
        if data.shape[1] == 2:
            component_tolerance = tolerance
            if tolerance is not None and scale != 0.0:
                component_tolerance = tolerance / abs(scale)
            spectral_array = compute_spectral_components(
                data,
                n_modes=n_modes,
                tolerance=component_tolerance,
            )
            spectral_array[:, 0] *= scale

        # Option B: Input data is a spectral array [amplitude, frequency, phase]
        elif data.shape[1] == 3:
            if data[0, 1] != 0.0:
                raise ValueError(
                    f"{key} spectral data must begin with a zero-frequency "
                    "DC component."
                )
            spectral_array = data.copy()
            negative_amplitudes = spectral_array[:, 0] < 0.0
            spectral_array[negative_amplitudes, 0] *= -1.0
            spectral_array[negative_amplitudes, 2] += np.pi
            spectral_array[:, 2] = np.remainder(
                spectral_array[:, 2], 2.0 * np.pi
            )
            dc_dimension = spectral_array[0, 0] * np.cos(spectral_array[0, 2])
            if dc_dimension <= 0.0:
                raise ValueError(
                    f"{key} spectral data must have a positive DC dimension."
                )
            spectral_array[0] = (dc_dimension, 0.0, 0.0)
            spectral_array[:, 0] *= scale

        else:
            raise ValueError(
                f"Unsupported data shape for {key}: {data.shape}. "
                f"Must be [time, value] or [amplitude, frequency, phase]"
            )

        processed_components[key] = np.asarray(spectral_array, dtype=np.float64)

    # 3. Create the MeltPool object
    width_oscillations = processed_components["width"]
    depth_oscillations = processed_components["depth"]
    height_oscillations = processed_components["height"]

    # 4. Unpack shape factors
    width_shape_factor = melt_pool_dict["width"][-1]
    depth_shape_factor = melt_pool_dict["depth"][-1]
    height_shape_factor = melt_pool_dict["height"][-1]

    melt_pool = MeltPool(
        width_oscillations,
        depth_oscillations,
        height_oscillations,
        np.abs(width_oscillations[:, 0]).sum(axis=0),
        np.abs(depth_oscillations[:, 0]).sum(axis=0),
        np.abs(height_oscillations[:, 0]).sum(axis=0),
        width_shape_factor,
        height_shape_factor,
        depth_shape_factor,
        enable_random_phases,
    )

    return melt_pool


def compute_porosity(
    grid: Grid,
    path_vectors: List[PathVector],
    melt_pool: MeltPool,
    random_seed: Optional[int] = None,
    *,
    tile_width: Optional[float] = None,
    spectral_error_fraction: float = DEFAULT_SPECTRAL_ERROR_FRACTION,
    memory_limit_mb: Optional[int] = None,
) -> np.ndarray:
    """Compute the porosity phase field.

    ``tile_width`` is a performance-only spatial-index control in metres.
    ``None`` selects the automatic default. ``spectral_error_fraction`` is
    the maximum spectral and interpolation error as a fraction of one voxel.
    ``memory_limit_mb`` is the per-process core-computation memory budget in
    units of 1024**2 bytes. ``None`` uses ``RAPTOR_MEMORY_LIMIT_MB`` when set,
    otherwise 80 percent of currently available memory. A smaller budget
    streams ordered spectral-table batches without reducing accuracy.
    """
    if tile_width is not None and (
        not np.isfinite(tile_width) or tile_width <= 0.0
    ):
        raise ValueError("tile_width must be finite and positive or None.")
    validate_spectral_error_fraction(spectral_error_fraction)
    validate_memory_limit(memory_limit_mb)

    requested_tile_size = None
    if tile_width is not None:
        requested_tile_size = max(
            1,
            int(round(tile_width / grid.resolution)),
        )

    print(f"Preparing {len(path_vectors)} path vectors for simulation...")
    t0_setup = time.time()
    phase_rng = np.random.RandomState(random_seed)
    for vector in path_vectors:
        vector.set_melt_pool_properties(melt_pool, phase_rng)
    print(f" -> Vector preparation complete ({time.time() - t0_setup:.8f}s).")

    print("Running melt-mask calculation...")
    t0_run = time.time()
    (
        active_vectors,
        voxel_shape,
        tile_size,
        candidate_offsets,
        candidate_indices,
    ) = build_spatial_index(
        grid.origin,
        grid.shape,
        grid.resolution,
        path_vectors,
        tile_size=requested_tile_size,
    )
    candidate_counts = np.diff(candidate_offsets)
    print(
        " -> Spatial index: "
        f"{len(active_vectors)}/{len(path_vectors)} vectors, "
        f"{len(candidate_counts)} tiles, "
        f"{candidate_counts.mean():.2f} mean candidates/tile, "
        f"width={tile_size} voxels "
        f"({tile_size * grid.resolution * 1.0e6:.3f} µm)."
    )
    melted_mask_flat = compute_melt_mask_grid(
        grid.origin,
        grid.shape,
        grid.resolution,
        melt_pool,
        active_vectors,
        tile_size=tile_size,
        candidate_offsets=candidate_offsets,
        candidate_indices=candidate_indices,
        spectral_error_fraction=spectral_error_fraction,
        memory_limit_mb=memory_limit_mb,
        report=True,
    )
    t_elapsed = time.time() - t0_run

    phase_counts = count_phase_codes(melted_mask_flat)
    n_melted = int(phase_counts[1:].sum())
    print(
        f" -> Melt-mask computation complete ({t_elapsed:.8f}s). "
        f"Melted {n_melted} of {grid.n_voxels} voxels."
    )

    porosity_field = melted_mask_flat.reshape(grid.shape, order="C")

    return porosity_field


def compute_phase_histogram(porosity: np.ndarray) -> Dict[int, int]:
    """Count Raptor phase codes 0 through 3 in one parallel pass."""
    counts = count_phase_codes(porosity.reshape(-1))
    return {phase: int(counts[phase]) for phase in range(4)}


def write_vtk(
    origin: np.ndarray,
    voxel_resolution: float,
    porosity: np.ndarray,
    vtk_output_path: str,
) -> None:
    """
    Generates porosity VTK.
    """
    import vtk
    from vtk.util import numpy_support

    origin = np.asarray(origin, dtype=np.float64)
    porosity = np.asarray(porosity)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be a finite array with shape (3,).")
    if not np.isfinite(voxel_resolution) or voxel_resolution <= 0.0:
        raise ValueError("voxel_resolution must be finite and positive.")
    if porosity.ndim != 3:
        raise ValueError("porosity must be a three-dimensional array.")

    imageData = vtk.vtkImageData()
    nx, ny, nz = porosity.shape

    # Raptor computes with z contiguous for cache-efficient column traversal.
    # Expose that buffer to VTK without a gigabyte-scale Fortran-order copy by
    # mapping VTK's x/y/z index axes to Raptor's z/y/x memory axes.
    imageData.SetDimensions(nz, ny, nx)
    imageData.SetOrigin(origin[0], origin[1], origin[2])
    imageData.SetSpacing(voxel_resolution, voxel_resolution, voxel_resolution)
    direction = vtk.vtkMatrix3x3()
    direction.DeepCopy((0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0))
    imageData.SetDirectionMatrix(direction)

    porosity_buffer = np.ascontiguousarray(porosity, dtype=np.int8).reshape(-1)
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=porosity_buffer,
        deep=False,
        array_type=vtk.VTK_SIGNED_CHAR,
    )
    vtk_data_array.SetName("Phase")
    imageData.GetPointData().SetScalars(vtk_data_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vtk_output_path)
    writer.SetInputData(imageData)
    writer.SetCompressorTypeToLZ4()
    # Inline base64 remains readable by older ParaView/VTK releases for very
    # large arrays. VTK 9.1 cannot reliably parse billion-element appended
    # payloads written by newer VTK versions, even when the payload is encoded.
    writer.SetBlockSize(1 << 20)
    writer.SetDataModeToBinary()

    write_status = writer.Write()
    error_code = writer.GetErrorCode()
    if (
        write_status != 1
        or error_code != vtk.vtkErrorCode.NoError
        or not Path(vtk_output_path).is_file()
    ):
        error_name = vtk.vtkErrorCode.GetStringFromErrorCode(error_code)
        raise OSError(
            f"Unable to write VTK file: {vtk_output_path} ({error_name})"
        )

    print(f"VTK phase map written to: {vtk_output_path}")


def compute_morphology(
    porosity: np.ndarray, voxel_resolution: float, morphology_fields: List[str]
) -> Dict[str, np.ndarray]:
    """
    Extracts pores, computes morphology features.
    """
    porosity = np.asarray(porosity)
    if porosity.ndim != 3:
        raise ValueError("porosity must be a three-dimensional array.")
    if not np.isfinite(voxel_resolution) or voxel_resolution <= 0.0:
        raise ValueError("voxel_resolution must be finite and positive.")
    if not morphology_fields:
        raise ValueError("morphology_fields must contain at least one field.")

    supported_sparse_fields = {
        "area",
        "centroid",
        "equivalent_diameter_area",
        "label",
    }
    n_defects = int(count_phase_codes(porosity.reshape(-1))[0])
    print(f"Identifying connected defects...")
    print(
        f" -> Found {n_defects} defect voxels. "
        f"Computing morphology features..."
    )
    min_size = 2
    sparse_limit = min(int(porosity.size * 0.01), 10_000_000)
    if n_defects <= sparse_limit and set(morphology_fields).issubset(
        supported_sparse_fields
    ):
        flat_indices = collect_zero_indices(
            porosity.reshape(-1), int(n_defects)
        )
        point_labels, n_components = label_sparse_defects(
            flat_indices, porosity.shape, min_size
        )
        valid = point_labels >= 0
        labels = point_labels[valid]
        component_counts = np.bincount(labels, minlength=n_components).astype(
            np.float64
        )
        component_volume = component_counts * voxel_resolution**3
        properties = {}
        for field in morphology_fields:
            if field == "label":
                properties["label"] = np.arange(
                    1, n_components + 1, dtype=np.int64
                )
            elif field == "area":
                properties["area"] = component_volume
            elif field == "equivalent_diameter_area":
                properties["equivalent_diameter_area"] = (
                    6.0 * component_volume / np.pi
                ) ** (1.0 / 3.0)
            elif field == "centroid":
                valid_indices = flat_indices[valid]
                yz_size = porosity.shape[1] * porosity.shape[2]
                x = valid_indices // yz_size
                remainder = valid_indices - x * yz_size
                y = remainder // porosity.shape[2]
                z = remainder - y * porosity.shape[2]
                for axis, coordinates in enumerate((x, y, z)):
                    coordinate_sum = np.bincount(
                        labels,
                        weights=coordinates,
                        minlength=n_components,
                    )
                    properties[f"centroid-{axis}"] = (
                        coordinate_sum / component_counts * voxel_resolution
                    )
        return properties

    from skimage import measure
    from skimage.morphology import remove_small_objects

    defect_structure = porosity == 0
    filtered_defects = remove_small_objects(
        defect_structure, min_size=min_size, connectivity=3
    )
    labeled_defects = measure.label(filtered_defects, connectivity=3)

    return measure.regionprops_table(
        labeled_defects, spacing=voxel_resolution, properties=morphology_fields
    )


def write_morphology(properties: dict, morphology_output_path: str) -> None:
    """
    Writes morphology output as a .csv.
    """
    import pandas as pd

    morphology_df = pd.DataFrame(properties, index=None)
    if len(morphology_df) == 0:
        print(
            f"Either no defects were found or all defects were single-voxel. "
            f"No morphology features to write."
        )
        return None
    else:
        morphology_df.to_csv(morphology_output_path, index=False)
        print(
            f"Morphology features of {len(morphology_df)} "
            f"defects written to: {morphology_output_path}"
        )


def visualize(vtk_output_path: str) -> None:
    """
    Visualize the porosity field in its native metre coordinate system.
    """
    import pyvista as pv
    from matplotlib.colors import ListedColormap

    rve = pv.read(vtk_output_path)
    outline = rve.outline()
    pore_rve = rve.threshold([-0.5, 0.5], scalars="Phase")
    render_pore_structure = pore_rve.n_points > 0

    annotations = (
        {
            0.5: "Pore",
            1.5: "Melted",
            2.5: "Boundary",
            3.5: "Intersection",
        }
        if render_pore_structure
        else {
            1.5: "Melted",
            2.5: "Boundary",
            3.5: "Intersection",
        }
    )
    n_colors = 4 if render_pore_structure else 3
    phase_cmap = (
        ListedColormap(
            [
                (1.0, 0.0, 0.0),
                (0.7, 0.7, 0.7),
                (0.2, 0.2, 0.2),
                (1.0, 1.0, 0.0),
            ],
            name="phase_cmap",
            N=n_colors,
        )
        if render_pore_structure
        else ListedColormap(
            [
                (0.7, 0.7, 0.7),
                (0.2, 0.2, 0.2),
                (1.0, 1.0, 0.0),
            ],
            name="phase_cmap",
            N=n_colors,
        )
    )

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

    if render_pore_structure:
        pl.subplot(0, 1)
        pore_rve_clip_actor = pl.add_mesh(
            pore_rve.clip(normal=(1, 0, 0), origin=(rve.bounds[1], 0, 0)),
            scalars="Phase",
            cmap=ListedColormap(
                [
                    (1.0, 0.0, 0.0),
                ],
                name="phase_cmap_pore",
                N=1,
            ),
            interpolate_before_map=False,
            lighting=False,
            opacity=1.0,
            scalar_bar_args={
                "n_labels": 0,
            },
        )
        pl.add_mesh(outline, color="black", line_width=1)

        label_args = {
            "font_size": 12,
            "color": "black",
            "font_family": "arial",
            "fmt": "%.0e",
        }

        pl.show_grid(
            xtitle="X (m)",
            ytitle="Y (m)",
            ztitle="Z (m)",
            grid=False,
            location="outer",
            **label_args,
        )

        pl.add_axes()

    pl.subplot(0, 0)
    rve_clipped = rve.clip(normal=(1, 0, 0), origin=(rve.bounds[1], 0, 0))

    clip_actor = pl.add_mesh(
        rve_clipped,
        scalars="Phase",
        cmap=phase_cmap,
        clim=(0, 4) if render_pore_structure else (1, 4),
        categories=True,
        n_colors=n_colors,
        annotations=annotations,
        interpolate_before_map=False,
        lighting=False,
        opacity=1.0,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Phase", "n_labels": 0},
    )
    pl.add_mesh(outline, color="black", line_width=1)

    label_args = {
        "font_size": 12,
        "color": "black",
        "font_family": "arial",
        "fmt": "%.0e",
    }

    pl.show_grid(
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
        grid=False,
        location="outer",
        **label_args,
    )

    pl.add_axes()

    def update_clip(normal, origin):
        new_clipped = rve.clip(normal=normal, origin=origin)
        clip_actor.mapper.SetInputData(new_clipped)
        (
            pore_rve_clip_actor.mapper.SetInputData(
                new_clipped.threshold([-0.5, 0.5], scalars="Phase")
            )
            if render_pore_structure
            else None
        )

    pl.add_plane_widget(
        update_clip,
        normal=(1, 0, 0),
        origin=rve.center,
        bounds=rve.bounds,
        color="blue",
        outline_translation=False,
    )

    pl.link_views()  # Link the two views for synchronized interaction

    pl.show()
