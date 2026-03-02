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

# --- GLOBAL SETTINGS ---
query_volume_mm3 = 1.0
voxel_resolution_m = 5.0e-6

# --- RVE (Representative Volume Element) Dimensions ---
rve_min_point = np.array([0.0, 0.0, 0.0])
rve_max_point = np.array([5.0e-4, 5.0e-4, 5.0e-4])
rve_bounding_box = np.array([rve_min_point, rve_max_point])

# --- Laser & Scan Path Machine Parameters ---
laser_power_watts = 370.0  # Laser power in Watts
laser_velocity_m_s = 1.7  # Laser scan speed in meters per second
powder_layer_thickness_m = 50.0e-6  # 50 microns layer height
layer_rotation_angle_deg = 67.0  # Rotation of the scan between layers
scan_extension_distance_m = max(rve_max_point - rve_min_point)
num_extra_layers = 0

# --- Melt Pool Morphology Parameters ---
SCRIPT_DIR = Path(__file__).resolve().parent
melt_pool_data_path = (
    SCRIPT_DIR / ".." / "data" / "meltPoolData" / "ULI_v1700_theta0_widths.txt"
)
base_width_data = read_data(melt_pool_data_path)
melt_pool_modes = 50

# Shape identifiers defined by Raptor: 1 = Parabola, 2 = Ellipse
melt_pool_width_scale = 1.0
melt_pool_depth_scale = 0.8
melt_pool_height_scale = 0.4

width_shape_type = 2
depth_shape_type = 1
height_shape_type = 1

# Create the 3D voxel grid for the RVE
grid = create_grid(voxel_resolution=voxel_resolution_m, bound_box=rve_bounding_box)

# Configure the dictionary that defines the 3D melt pool volume
melt_pool_dict = {
    "width": (
        base_width_data,
        melt_pool_modes,
        melt_pool_width_scale,
        width_shape_type,
    ),
    "depth": (
        base_width_data,
        melt_pool_modes,
        melt_pool_depth_scale,
        depth_shape_type,
    ),
    "height": (
        base_width_data,
        melt_pool_modes,
        melt_pool_height_scale,
        height_shape_type,
    ),
}

# Generate the moving melt pool with realistic random spatial fluctuations
melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

rve_volume_mm3 = (1e3 * (rve_max_point - rve_min_point)).prod()
print(rve_volume_mm3)
num_rves = int(np.ceil(query_volume_mm3 / rve_volume_mm3))
print(f"  -> Simulating {num_rves} RVEs to fill {query_volume_mm3} mm^3...")


# --- Function to evaluate hatch spacing ----
def evaluate_hatch_spacing(hatch_spacing_m: float, requires_jit_warmup: bool) -> float:
    # Build the laser paths using our machine parameters
    scan_path_builder = ScanPathBuilder(
        rve_bounding_box,
        laser_power_watts,
        laser_velocity_m_s,
        hatch_spacing_m,
        powder_layer_thickness_m,
        layer_rotation_angle_deg,
        scan_extension_distance_m,
        num_extra_layers,
    )
    scan_path_builder.generate_layers()
    path_vectors = scan_path_builder.process_vectors()

    defect_metrics = []
    for i in range(num_rves):
        jit_warmup = (i == 0) and requires_jit_warmup

        porosity = compute_porosity(
            grid, path_vectors, melt_pool, jit_warmup=jit_warmup
        )

        metrics = compute_morphology(
            porosity, grid.resolution, ["equivalent_diameter_area"]
        )

        defect_metrics.append(metrics["equivalent_diameter_area"])

    defect_metrics = np.concatenate(defect_metrics)

    if len(defect_metrics) == 0:
        defect_metrics = [0, 0]

    return np.array(defect_metrics)


# --- Run example ---
# Hatch spacings to sample
x = np.array([80, 100, 120, 140, 160, 180, 200]) * 1e-6

mean = []
std = []
for i, xi in enumerate(x):
    output = evaluate_hatch_spacing(xi, (i == 0))
    mean.append(np.mean(output))
    std.append(np.std(output))

# Plot example result
import matplotlib.pyplot as plt

plt.errorbar(x, mean, yerr=std)
plt.xlabel("Hatch spacing ")
plt.ylabel("Equivalent diameter area (microns square)")
plt.minorticks_on()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
