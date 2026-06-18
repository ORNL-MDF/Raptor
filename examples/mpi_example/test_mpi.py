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

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def run_raptor(width, depth, sigma=0, edgelength=5.0e-4):

    
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
    power = 370
    velocity = 1.7
    hatch_spacing = 140e-6
    layer_height = 30e-6
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
    mean = width
    std_dev = sigma if sigma > 0 else 0.01e-6
    frequency = 250000

    duration = 0.08

    # Instantiate object
    mp_w_filter = MeltPoolFilter(width, std_dev, velocity, [frequency, duration])
    mp_d_filter = MeltPoolFilter(depth, std_dev, velocity, [frequency, duration])

    # Define physical scales
    mp_w_filter.add_effect("melt_pool", [500e-6, None, 1])
    mp_d_filter.add_effect("melt_pool", [500e-6, None, 1])

    # Generate stochastic melt pool
    mp_w_filter.initialize()
    width_data = mp_w_filter.generate_fluctuations(1)

    mp_d_filter.initialize()
    depth_data = mp_d_filter.generate_fluctuations(1)
    
    n_modes = 50

    # scale melt pool data by constant factor
    width_scale = 1.0
    depth_scale = 1.0
    height_scale = 0.4

    # assign shape to melt pool and cap (1 = parabola, 2 = ellipse)
    width_shape = 2  # placeholder
    height_shape = 1
    depth_shape = 1

    melt_pool_dict = {
        "width": (width_data, n_modes, width_scale, width_shape),
        "depth": (depth_data, n_modes, depth_scale, depth_shape),
        "height": (width_data, n_modes, height_scale, height_shape),
    }

    melt_pool = create_melt_pool(melt_pool_dict, enable_random_phases=True)

    # 4. Compute porosity using conic section / superellipse curves for melt pool mask
    porosity = compute_porosity(grid, path_vectors, melt_pool, jit_warmup=True)

    if np.any(porosity>0):
        morphology = compute_morphology(porosity, grid.resolution, morphology_fields=['coords'])
        return np.concatenate(morphology['coords']) if len(morphology['coords']) > 0 else np.array([])
    else:
        return np.array([])
    

def run_sve(case):

    if rank == 0:
        print(f"Running case: {case}")
    
    porosity = run_raptor(case["width"], case["depth"])
   
    if rank == 0:
        print(f"Completed case: {case}")
    
    return porosity

def load_sve_cases(nruns=2):
    wbounds = [100e-6, 200e-6]
    dbounds = [100e-6, 200e-6]
    cases = []
    for w in np.linspace(wbounds[0], wbounds[1], 5):
        for d in np.linspace(dbounds[0], dbounds[1], 5):
            for n in range(nruns):
                cases.append({"width": w, "depth": d, "run": n})
    return cases

if __name__ == "__main__":
    cases = load_sve_cases()
    results = []
    my_cases = cases[rank::size]  # Distribute cases among processes
    for case in my_cases:
        w,d = case["width"], case["depth"]
        sve_defect_coords = run_raptor(w,d)
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
                w_val = case["width"]
                d_val = case["depth"]
                run_idx = case["run"]

                group_path = f"width_{w_val:.1e}/depth_{d_val:.1e}/run_{run_idx}"

                group = f.create_group(group_path)

                group.attrs["width"] = w_val
                group.attrs["depth"] = d_val
                group.attrs["run"] = run_idx

                group.create_dataset("defect_coords", data=defect_coords)
        print("Results saved successfully.")
                
