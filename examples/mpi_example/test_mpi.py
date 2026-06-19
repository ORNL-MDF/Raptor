# test_mpi.py
import numpy as np
from mpi4py import MPI
import h5py
from raptor.api import (
    create_path_vectors,
    create_melt_pool,
    create_grid,
    compute_porosity,
    write_vtk,
    compute_morphology,
    write_morphology,
    visualize,
)
from raptor.utilities import ScanPathBuilder, MeltPoolFilter
import joblib
import pickle

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
INITIAL_BOUNDS = ((0.5, 1.5), (100, 300))  # [velocity (m/s), power (W)]
UNIT_BOUNDS = ((0.0, 1.0), (0.0, 1.0))

# scaling helper functions
def x_to_unit(X):
    X = np.asarray(X, dtype=float)
    lo = np.array([b[0] for b in INITIAL_BOUNDS])
    hi = np.array([b[1] for b in INITIAL_BOUNDS])
    return (X - lo) / (hi - lo + 1e-12)


def x_from_unit(U):
    U = np.asarray(U, dtype=float)
    lo = np.array([b[0] for b in INITIAL_BOUNDS])
    hi = np.array([b[1] for b in INITIAL_BOUNDS])
    return U * (hi - lo) + lo


def load_meltpool_surrogates(joblib_path):
    # Loading pretrained gaussian process surrogates for melt pool dimensions and variability.
    meltpool_surrogates = joblib.load(joblib_path)
    return meltpool_surrogates

def load_meltpool_data(pickle_path):
    with open(pickle_path, "rb") as f:
        meltpool_data = pickle.load(f)
    return meltpool_data

def run_raptor(power, velocity, hatch_spacing, layer_height, features, 
               edgelength=5.0e-4):    
    
    # 1. Create voxel grid for the representative volume element (RVE)
    min_point = np.array([0.0, 0.0, 0.0])
    edgelength = np.atleast_1d(edgelength)
    if len(edgelength) == 1:
        max_point = np.array([edgelength[0], edgelength[0], edgelength[0]])
    elif len(edgelength) == 3:
        max_point = np.array([edgelength[0], edgelength[1], edgelength[2]])
    else:
        raise ValueError("edgelength must be a scalar or a list/array of three values.")
    bound_box = np.array([min_point, max_point])
    voxel_resolution = 5.0e-6

    grid = create_grid(voxel_resolution, bound_box=bound_box)

    # 2. Create path vectors through the representative volume element (RVE)
    power = power
    velocity = velocity
    hatch_spacing = hatch_spacing
    layer_height = layer_height
    rotation = 67
    scan_extension = max(max_point - min_point)
    extra_layers = 10

    scan_path_builder = ScanPathBuilder(
        bound_box,
        power,
        velocity,
        hatch_spacing,
        layer_height,
        rotation,
        scan_extension,
        extra_layers,
    )

    scan_path_builder.generate_layers()
    path_vectors = scan_path_builder.process_vectors()

    # 3. Create melt pools from convolution filter
    mu_w = features["width_mean"]
    sig_w = features["width_std"]
    mu_d = features["depth_mean"]
    sig_d = features["depth_std"]
    mu_h = features["height_mean"]
    sig_h = features["height_std"]

    frequency = 250000

    duration = 0.08

    # Instantiate object
    mp_w_filter = MeltPoolFilter(mu_w, sig_w, velocity, [frequency, duration])
    mp_d_filter = MeltPoolFilter(mu_d, sig_d, velocity, [frequency, duration])
    mp_h_filter = MeltPoolFilter(mu_h, sig_h, velocity, [frequency, duration])

    # Define physical scales
    mp_w_filter.add_effect("melt_pool", [500e-6, None, 1])
    mp_d_filter.add_effect("melt_pool", [500e-6, None, 1])
    mp_h_filter.add_effect("melt_pool", [500e-6, None, 1])

    # Generate stochastic melt pool
    mp_w_filter.initialize()
    width_data = mp_w_filter.generate_fluctuations(1)

    mp_d_filter.initialize()
    depth_data = mp_d_filter.generate_fluctuations(1)

    mp_h_filter.initialize()
    height_data = mp_h_filter.generate_fluctuations(1)
    
    n_modes = 50

    # scale melt pool data by constant factor
    width_scale = 1.0
    depth_scale = 1.0
    height_scale = 1.0

    # assign shape to melt pool and cap (1 = parabola, 2 = ellipse)
    width_shape = 2  # placeholder
    height_shape = 1
    depth_shape = 1

    melt_pool_dict = {
        "width": (width_data, n_modes, width_scale, width_shape),
        "depth": (depth_data, n_modes, depth_scale, depth_shape),
        "height": (height_data, n_modes, height_scale, height_shape),
    }

    melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

    # 4. Compute porosity using conic section / superellipse curves for melt pool mask
    porosity = compute_porosity(grid, path_vectors, melt_pool, jit_warmup=True)

    if np.any(porosity>0):
        morphology = compute_morphology(porosity, grid.resolution, morphology_fields=['coords'])
        return np.concatenate(morphology['coords']) if len(morphology['coords']) > 0 else np.array([])
    else:
        return np.array([])
    
def load_sve_cases(nruns=5):
    gp_models = load_meltpool_surrogates("/Users/vamsi/Desktop/phd/mcml/projects/ornl/summer2026/mpi_acceleration/pvhl/meltpool_model/meltpool_gp_bundle.joblib")
    gp_models = gp_models["gp_dict"]
    meltpool_data_y = load_meltpool_data("/Users/vamsi/Desktop/phd/mcml/projects/ornl/summer2026/mpi_acceleration/pvhl/meltpool_model/meltpool_data_y.pkl")
    # p_range = [100, 150, 200, 250, 300]
    # v_range = [0.5, 0.75, 1.0, 1.25, 1.5]
    # h_range = np.arange(60e-6, 180e-6, 5e-6)
    # l_range = np.arange(20e-6, 50e-6, 5e-6)
    p_range = [150, 200]
    v_range = [1.0, 1.25]
    h_range = np.arange(100e-6, 180e-6, 10e-6)
    l_range = np.arange(20e-6, 50e-6, 10e-6)
    cases = []
    for p in p_range:
        for v in v_range:
            features = {}
            for dim in gp_models.keys():
                x_unit = x_to_unit([p, v]).reshape(1, -1)
                pred = gp_models[dim].predict(x_unit) * np.std(meltpool_data_y[dim] + 1e-12) + np.mean(meltpool_data_y[dim])
                features[dim] = pred[0]
            for h in h_range:
                for l in l_range:
                     for n in range(nruns):
                         cases.append({"power": p, "velocity": v, "hatch_spacing": h, "layer_height": l, "features": features,"run": n})
    return cases


def run_sve(case):

    if rank == 0:
        print(f"Running case: {case}")
    
    porosity = run_raptor(case["power"], case["velocity"], case["hatch_spacing"], case["layer_height"], case["features"])
   
    if rank == 0:
        print(f"Completed case: {case}")
    
    return porosity



if __name__ == "__main__":
    cases = load_sve_cases()
    results = []
    my_cases = cases[rank::size]  # Distribute cases among processes
    for case in my_cases:
        sve_defect_coords = run_raptor(case["power"], case["velocity"], case["hatch_spacing"], case["layer_height"], case["features"])
        results.append((case, sve_defect_coords)) # store case, volume fraction, and run index for each case
    # Gather results at root process
    all_results = comm.gather(results, root=0)
    if rank == 0:
        # Gather list of results from all processes
        all_results = [item for sublist in all_results for item in sublist]  # Flatten list of results
        
        output_filename = "sweep_ensemble_results.h5"
        print(f"Saving results to {output_filename}...")
        with h5py.File(output_filename, "w") as f:
            for i, (case, defect_coords) in enumerate(all_results):
                p_val = case["power"]
                v_val = case["velocity"]
                h_val = case["hatch_spacing"]
                l_val = case["layer_height"]
                run_idx = case["run"]

                group_path = f"power_{p_val:.1e}/velocity_{v_val:.1e}/hatch_spacing_{h_val:.1e}/layer_height_{l_val:.1e}/run_{run_idx}"

                group = f.create_group(group_path)

                group.attrs["power"] = p_val
                group.attrs["velocity"] = v_val
                group.attrs["hatch_spacing"] = h_val
                group.attrs["layer_height"] = l_val
                group.attrs["run"] = run_idx

                group.create_dataset("defect_coords", data=defect_coords)
        print("Results saved successfully.")
                
