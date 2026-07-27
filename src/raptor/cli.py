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
import argparse
import os
import yaml
import numpy as np
from pathlib import Path

from .api import (
    create_grid,
    create_melt_pool,
    compute_porosity,
    write_vtk,
    compute_morphology,
    write_morphology,
)
from .io import read_data, read_scan_path


def main() -> int:
    """
    Parses args, loads config, runs porosity prediction, processes pore data.
    """
    parser = argparse.ArgumentParser(description="LPBF Porosity Predictor")
    parser.add_argument(
        "config_file", type=str, help="Path to YAML config file."
    )
    args = parser.parse_args()

    try:
        if not os.path.exists(args.config_file):
            print(f"Error: Config file not found: {args.config_file}")
            return 1
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("The YAML document must contain a mapping.")
        config_dir = os.path.dirname(os.path.abspath(args.config_file))
        print(f"Loaded config: {args.config_file}")
    except Exception as e:
        print(f"Error loading/parsing YAML '{args.config_file}': {e}")
        return 1

    try:
        # read scan path files
        scan_paths = config.get("scan_paths", [])
        if not isinstance(scan_paths, list) or not scan_paths:
            raise ValueError("'scan_paths' must be a non-empty list.")

        # read simulation parameter dictionary
        scan_pattern_parameters = config.get("parameters", {})
        if not isinstance(scan_pattern_parameters, dict):
            raise ValueError("'parameters' must be a mapping.")
        layer_height = scan_pattern_parameters["layer_height"]
        voxel_resolution = scan_pattern_parameters["voxel_resolution"]
        enable_random_phases = scan_pattern_parameters[
            "enable_random_segment_phase"
        ]
        porosity_options = {
            key: scan_pattern_parameters[key]
            for key in (
                "random_seed",
                "tile_width",
                "spectral_error_fraction",
                "memory_limit_mb",
            )
            if key in scan_pattern_parameters
        }

        # read melt pool dictionary (time series or spectral components)
        melt_pool_dict = config.get("melt_pool_data", {})
        if not isinstance(melt_pool_dict, dict):
            raise ValueError("'melt_pool_data' must be a mapping.")
        melt_pool_data = {}
        try:
            for key, dimension_config in melt_pool_dict.items():
                datatype = dimension_config["type"]
                if datatype not in {"time_series", "spectral_components"}:
                    raise ValueError(
                        f"Unsupported {key} melt-pool data type: {datatype}"
                    )
                filepath = Path(dimension_config["file_name"])
                if not filepath.is_absolute():
                    filepath = Path(config_dir) / filepath
                scale = dimension_config["scale"]
                nmodes = int(dimension_config["nmodes"])
                shape_factor = (
                    2 if key == "width" else dimension_config["shape"]
                )
                melt_pool_data[key] = (
                    read_data(filepath),
                    nmodes,
                    scale,
                    shape_factor,
                )
        except (KeyError, TypeError, ValueError, OSError) as error:
            print(f"Error reading melt-pool data: {error}")
            return 1

        # read representative volume element (RVE) dictionary
        rve = config.get("rve", {})
        if rve:
            bounding_box = np.array([rve["min_point"], rve["max_point"]])
        else:
            print("Warning: 'rve' was not found, defaulting to none")
            bounding_box = None

        # read output dictionary
        output = config.get("output", {})
        if not isinstance(output, dict):
            raise ValueError("'output' must be a mapping.")
        vtk_dict = output.get("vtk", {})
        morphology_dict = output.get("morphology", {})
        vtk_file_name = vtk_dict.get("file_name", None)
        morphology_file_name = morphology_dict.get("file_name", None)
        morphology_fields = morphology_dict.get("fields", None)

    except KeyError as e:
        print(f"Error: Missing key '{e}' in YAML config '{args.config_file}'.")
        return 1
    except (TypeError, ValueError) as error:
        print(f"Error: Invalid configuration: {error}")
        return 1

    # convert relative file paths to absolute file paths
    scan_path_files = [
        os.path.join(config_dir, path) if not os.path.isabs(path) else path
        for path in scan_paths
    ]
    vtk_file = None
    if vtk_file_name:
        vtk_file = (
            os.path.join(config_dir, vtk_file_name)
            if not os.path.isabs(vtk_file_name)
            else vtk_file_name
        )
    morphology_file = None
    if morphology_file_name:
        morphology_file = (
            os.path.join(config_dir, morphology_file_name)
            if not os.path.isabs(morphology_file_name)
            else morphology_file_name
        )

    # report summary of input parameters for simulation
    print("\n--- Simulation Parameters ---")
    param_summary = {
        "Scan Paths": scan_path_files,
        "Random Phases": f"{enable_random_phases}",
        "Layer Height": f"{layer_height:.2e} m",
        "VTK Output": vtk_file,
        "Morphology Output": morphology_file,
        "Voxel Res": f"{voxel_resolution:.2e} m",
    }

    for k, v_item in param_summary.items():
        if isinstance(v_item, list):
            print(f"  {k}:")
            for path in v_item:
                print(f"    - {path}")
        else:
            print(f"  {k}: {v_item}")

    print("  Melt pool data:")
    for key in melt_pool_data:
        print(f"    {key} datatype: " + melt_pool_dict[key]["type"])
        print(f"    {key} path : " + melt_pool_dict[key]["file_name"])
        print(f"    {key} scaling : {melt_pool_data[key][2]}")
        print(f"    {key} modes : {melt_pool_data[key][1]}")

    if bounding_box is not None:
        print("  RVE data:")
        minimum = ",".join(f"{value} m" for value in bounding_box[0])
        maximum = ",".join(f"{value} m" for value in bounding_box[1])
        print(f"    xmin,ymin,zmin = {minimum}")
        print(f"    xmax,ymax,zmax = {maximum}")

    try:

        # compute scan vectors
        all_vectors = []
        for scan_path_file in scan_path_files:
            scan_vectors = read_scan_path(scan_path_file)
            for vector in scan_vectors:
                vector.set_coordinate_frame()
                all_vectors.append(vector)
        if not all_vectors:
            raise ValueError("No line-raster scan vectors were found.")

        melt_pool = create_melt_pool(melt_pool_data, enable_random_phases)

        # instantiate voxel grid
        grid = (
            create_grid(voxel_resolution, bound_box=bounding_box)
            if bounding_box is not None
            else create_grid(voxel_resolution, path_vectors=all_vectors)
        )

        # compute porosity
        porosity = compute_porosity(
            grid,
            all_vectors,
            melt_pool,
            **porosity_options,
        )

        # write VTK (optional)
        if vtk_dict and vtk_file:
            write_vtk(grid.origin, grid.resolution, porosity, vtk_file)

        # write morphology metrics (optional)
        if morphology_fields and morphology_file:
            defect_morphologies = compute_morphology(
                porosity, voxel_resolution, morphology_fields
            )
            write_morphology(defect_morphologies, morphology_file)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    except (MemoryError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def run():
    """Entry point for the console script."""
    raise SystemExit(main())
