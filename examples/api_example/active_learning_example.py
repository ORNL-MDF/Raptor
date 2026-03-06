import numpy as np
from pathlib import Path
from raptor.io import read_data
from raptor.api import (
    create_grid,
    create_melt_pool,
    compute_porosity,
    compute_morphology,
)
from raptor.utilities import ScanPathBuilder


def compute_mean_and_std(defects: np.ndarray):
    if len(defects) == 0:
        y = 0.0
        yerr = np.nan
    elif len(defects) == 1:
        y = defects[0]
        yerr = defects[0]
    else:
        y = np.mean(defects)
        yerr = np.std(defects, ddof=1) / np.sqrt(len(defects))

    return y, yerr


def defects_model(
    hatch_spacing_m: float,
    layer_thickness_m: float,
    query_volume_mm3: float = 1.0,
    voxel_resolution_m: float = 5e-6,
    metric_names: list[str] = ["equivalent_diameter_area"],
) -> np.ndarray:

    # Create the 3D voxel grid for the RVE
    rve_min_point = np.array([0.0, 0.0, 0.0])
    rve_max_point = np.array([5e-4, 5e-4, 5e-4])
    rve_bounding_box = np.array([rve_min_point, rve_max_point])

    grid = create_grid(voxel_resolution=voxel_resolution_m, bound_box=rve_bounding_box)
    rve_volume_mm3 = (1e3 * (rve_max_point - rve_min_point)).prod()
    num_rves = int(np.ceil(query_volume_mm3 / rve_volume_mm3))

    # Configure the statistical melt pool model
    SCRIPT_DIR = Path(__file__).resolve().parent
    melt_pool_data_path = (
        SCRIPT_DIR / ".." / "data" / "meltPoolData" / "ULI_v1700_theta0_widths.txt"
    )
    base_width_data = read_data(melt_pool_data_path)
    melt_pool_dict = {
        "width": (
            base_width_data,
            50,
            1.0,
            2,
        ),
        "depth": (
            base_width_data,
            50,
            0.8,
            1,
        ),
        "height": (
            base_width_data,
            50,
            0.4,
            1,
        ),
    }
    melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

    # Build the laser paths
    laser_power_watts = 370.0
    laser_velocity_m_per_s = 1.7
    layer_rotation_angle_deg = 67.0
    scan_extension_distance_m = max(rve_max_point - rve_min_point)
    num_extra_layers = 0

    scan_path_builder = ScanPathBuilder(
        rve_bounding_box,
        laser_power_watts,
        laser_velocity_m_per_s,
        hatch_spacing_m,
        layer_thickness_m,
        layer_rotation_angle_deg,
        scan_extension_distance_m,
        num_extra_layers,
    )
    scan_path_builder.generate_layers()
    path_vectors = scan_path_builder.process_vectors()

    # Run Raptor and calculate defect metrics
    defect_metrics = []
    for i in range(num_rves):
        porosity = compute_porosity(
            grid, path_vectors, melt_pool, (i == 0)  # jit warmup
        )
        metrics = compute_morphology(porosity, grid.resolution, metric_names)
        defect_metrics.append(metrics)

    combined_metrics = {}
    for name in metric_names:
        arrays_to_concat = [m[name] for m in defect_metrics if name in m]

        if arrays_to_concat:
            combined_metrics[name] = np.concatenate(arrays_to_concat)
        else:
            combined_metrics[name] = np.array([])

    return combined_metrics


# --- Run example ---
if __name__ == "__main__":

    hatch_spacing_m = np.array([60, 80, 100, 120, 140, 160, 180, 200]) * 1e-6
    layer_thickness_m = np.array([50.0]) * 1e-6

    x0, x1 = np.broadcast_arrays(hatch_spacing_m, layer_thickness_m)

    y = np.zeros(x0.shape)
    yerr = np.zeros(x0.shape)

    for i in range(x0.size):
        defects = defects_model(x0.flat[i], x1.flat[i])

        target_metric = defects["equivalent_diameter_area"]
        y.flat[i], yerr.flat[i] = compute_mean_and_std(target_metric)

    print("Mean Defects:", y)
    print("Standard Errors:", yerr)

    import matplotlib.pyplot as plt

    plt.errorbar(x0, y, yerr=yerr)
    plt.show()
