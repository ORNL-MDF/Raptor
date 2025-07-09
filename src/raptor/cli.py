import argparse
import os
import sys
import traceback
import yaml
import numpy as np

from .api import (
    compute_spectral_components,
    construct_meltpool,
    compute_porosity,
    write_vtk,
    compute_morphology,
    write_morphology,
)
from .io import read_data
from .structures import MeltPool


def main() -> int:
    """
    Parses args, loads config, runs porosity prediction, processes pore data.
    """
    parser = argparse.ArgumentParser(description="LPBF Porosity Predictor")
    parser.add_argument("config_file", type=str, help="Path to YAML config file.")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.config_file):
            print(f"Error: Config file not found: {args.config_file}")
            return 1
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
        config_dir = os.path.dirname(os.path.abspath(args.config_file))
        print(f"Loaded config: {args.config_file}")
    except Exception as e:
        print(f"Error loading/parsing YAML '{args.config_file}': {e}")
        return 1

    try:
        # read scan path files
        scan_paths = config.get("scan_paths", {})

        # read simulation parameter dictionary
        parameters = config.get("parameters", {})
        layer_height = parameters["layer_height"]
        n_bezier_pts_half = int(parameters["bezier_points_per_half"])
        voxel_resolution = parameters["voxel_resolution"]
        en_rand_ph = parameters["enable_random_segment_phase"]

        # read melt pool dictionary (time series or spectral components)
        melt_pool_dict = config.get("melt_pool_data", {})
        melt_pool_data = {}
        for key in melt_pool_dict:
            datatype = melt_pool_dict[key]["type"]
            if datatype == "time_series":
                try:
                    filepath = melt_pool_dict[key]["file_name"]
                    nmodes = int(melt_pool_dict[key]["nmodes"])
                    scale = melt_pool_dict[key]["scale"]
                    melt_pool_data[key] = (read_data(filepath), scale, nmodes)
                except:
                    print(
                        "Error reading the specified {} {} data format.".format(
                            key, datatype
                        )
                    )
            elif datatype == "spectral_components":
                try:
                    filepath = melt_pool_dict[key]["file_name"]
                    nmodes = int(melt_pool_dict[key]["nmodes"])
                    scale = melt_pool_dict[key]["scale"]
                    print("Scaling the zeroth mode (mean) of {} spectral array.")
                    spec_array = read_data(filepath)
                    melt_pool_data[key] = (spec_array, scale, nmodes)
                except:
                    print(
                        "Error reading the specified {} {} data format.".format(
                            key, datatype
                        )
                    )

        # read representative volume element (RVE) dictionary
        rve = config.get("rve", {})
        try:
            bounding_box = np.array([rve["min_point"], rve["max_point"]])
        except:
            print("Warning: 'rve' was not found, defaulting to none")
            bounding_box = None

        # read output dictionary
        output = config.get("output", {})
        vtk_dict = output.get("vtk", {})
        morphology_dict = output.get("morphology", {})
        vtk_file_name = vtk_dict.get("file_name", None)
        morphology_file_name = morphology_dict.get("file_name", None)
        morphology_fields = morphology_dict.get("fields", None)

        print(vtk_file_name, morphology_file_name, morphology_fields)

    except KeyError as e:
        print(f"Error: Missing key '{e}' in YAML config '{args.config_file}'.")
        return 1

    # convert relative file paths to absolute file paths
    scan_path_files = [
        os.path.join(config_dir, path) if not os.path.isabs(path) else path
        for path in scan_paths
    ]
    vtk_file = (
        os.path.join(config_dir, vtk_file_name)
        if not os.path.isabs(vtk_file_name)
        else vtk_file_name
    )
    morpholopy_file = (
        os.path.join(config_dir, morphology_file_name)
        if not os.path.isabs(morphology_file_name)
        else morphology_file_name
    )

    # report summary of input parameters for simulation
    print("\n--- Simulation Parameters ---")
    param_summary = {
        "Scan Paths": scan_path_files,
        "Random Phases": f"{en_rand_ph}",
        "Layer Height": f"{layer_height:.2e} m",
        "VTK Output": vtk_file,
        "Morphology Output": morpholopy_file,
        "Voxel Res": f"{voxel_resolution:.2e} m",
        "Bezier Pts/Half": n_bezier_pts_half,
    }

    for k, v_item in param_summary.items():
        if isinstance(v_item, list):
            print(f"  {k}:")
            [print(f"    - {p}") for p in v_item]
        else:
            print(f"  {k}: {v_item}")

    print("  Melt pool data:")
    for key in melt_pool_data:
        print(f"    {key} datatype: " + melt_pool_dict[key]["type"])
        print(f"    {key} path : " + melt_pool_dict[key]["file_name"])
        print(f"    {key} scaling : {melt_pool_data[key][1]}")
        print(f"    {key} modes : {melt_pool_data[key][2]}")

    if bounding_box is not None:
        print("  RVE data:")
        print(
            f"    xmin,ymin,zmin = {bounding_box[0][0]} m,{bounding_box[0][1]} m,{bounding_box[0][2]} m"
        )
        print(
            f"    xmax,ymax,zmax = {bounding_box[1][0]} m,{bounding_box[1][1]} m,{bounding_box[1][2]} m"
        )

    try:

        # construct melt pools
        melt_pool = construct_meltpool(melt_pool_data, en_rand_ph)

        # compute porosity
        centroid, porosity = compute_porosity(
            scan_path_files,
            layer_height,
            voxel_resolution,
            n_bezier_pts_half,
            melt_pool,
            bounding_box,
        )

        # write VTK (optional)
        if vtk_dict:
            write_vtk(centroid, voxel_resolution, porosity, vtk_file)

        # write morphology metrics (optional)
        if morphology_fields:
            write_morphology
            (
                compute_morphology(porosity, voxel_resolution, morphology_fields),
                morpholopy_file,
            )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    except ValueError as ve:
        print(f"Error: {ve}")
        return 1

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def run():
    """Entry point for the console script."""
    main()
    sys.exit()
