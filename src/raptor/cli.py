import argparse
import os
import sys
import traceback
import yaml
import numpy as np

from .api import compute_porosity_vtk, compute_spectral_components, construct_meltpool
from .io import read_data
from .structures import MeltPool


def main() -> int:
    """
    Parses args, loads config, runs porosity prediction. Returns exit code.
    """
    parser = argparse.ArgumentParser(description="LPBF Porosity Predictor")
    parser.add_argument("config_file", type=str, help="Path to YAML config file.")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.config_file):
            print(f"Error: Config file not found: {args.config_file}")
            return 1
        with open(args.config_file, "r") as f:
            cfg = yaml.safe_load(f)
        cfg_dir = os.path.dirname(os.path.abspath(args.config_file))
        print(f"Loaded config: {args.config_file}")
    except Exception as e:
        print(f"Error loading/parsing YAML '{args.config_file}': {e}")
        return 1

    try:
        sfp_rel = cfg["scan_paths"]
        vtk_rel = cfg["output_vtk_filename"]
        p = cfg.get("parameters", {})
        lh_val = p["layer_height"]
        n_pts_bh = int(p["bezier_points_per_half"])
        d_res = p["voxel_resolution"]
        en_rand_ph = p["enable_random_segment_phase"]

        mp = cfg["melt_pool_data"]
        mp_full_data = {}
        for key in mp:
            datatype = mp[key]["type"]
            if datatype == "time_series":
                try:
                    filepath = mp[key]["file_name"]
                    nmodes = int(mp[key]["nmodes"])
                    scale = mp[key]["scale"]
                    mp_full_data[key] = (scale * read_data(filepath), scale, nmodes)
                except:
                    print("Error reading the specified {} {} data format.".format(key,datatype))
            elif datatype == "spectral_components":
                try:
                    filepath = mp[key]["file_name"]
                    nmodes = int(mp[key]["nmodes"])
                    scale = mp[key]["scale"]
                    print("Scaling the zeroth mode (mean) of {} spectral array.")
                    spec_array = read_data(filepath)
                    spec_array[0,0] *= scale
                    mp_full_data[key] = (spec_array, scale, nmodes)
                except:
                    print("Error reading the specified {} {} data format.".format(key,datatype))

        rve = cfg.get("rve", {})
        try:
            p0 = rve["boundBox"]["p0"]
            p1 = rve["boundBox"]["p1"]
            boundingBox = np.array([p0, p1])
        except:
            print("Warning: 'boundBox' was not found, defaulting to none")
            boundingBox = None
        """
        rve = cfg["rve"]
        if rve["boundBox"] is None:
            boundingBox = None
        else:
            p0 = rve["boundBox"]["p0"]
            p1 = rve["boundBox"]["p1"]
            boundingBox = np.array([p0,p1])
        """
    except KeyError as e:
        print(f"Error: Missing key '{e}' in YAML config '{args.config_file}'.")
        return 1

    if not isinstance(sfp_rel, list) or not sfp_rel:
        print("Error: 'scan_paths' must be a non-empty list.")
        return 1
    if not isinstance(vtk_rel, str) or not vtk_rel:
        print("Error: 'output_vtk_filename' must be a string.")
        return 1
    if not isinstance(lh_val, (int, float)) or lh_val <= 0:
        print("Error: 'layer_height' must be a positive number.")
        return 1
    if not isinstance(n_pts_bh, int) or n_pts_bh < 2:
        print("Error: 'bezier_points_per_half' must be int >= 2.")
        return 1

    sfp_abs = [
        os.path.join(cfg_dir, pth) if not os.path.isabs(pth) else pth for pth in sfp_rel
    ]
    vtk_abs = os.path.join(cfg_dir, vtk_rel) if not os.path.isabs(vtk_rel) else vtk_rel

    print("\n--- Simulation Parameters ---")
    param_summary = {
        "Scan Paths": sfp_abs,
        "  Random Phases": f"{en_rand_ph}",
        "Layer Height": f"{lh_val:.2e} m",
        "VTK Output": vtk_abs,
        "Voxel Res": f"{d_res:.2e} m",
        "Bezier Pts/Half": n_pts_bh,
    }
    for k, v_item in param_summary.items():
        if isinstance(v_item, list):
            print(f"  {k}:")
            [print(f"    - {p}") for p in v_item]
        else:
            print(f"  {k}: {v_item}")

    print("  Melt pool data:")
    for key in mp_full_data:
        print(f"    {key} datatype: "+mp[key]["type"])
        print(f"    {key} path : " + mp[key]["file_name"])
        print(f"    {key} scaling : {mp_full_data[key][1]}")
        print(f"    {key} modes : {mp_full_data[key][2]}")


    if boundingBox is not None:
        print("  RVE data:")
        print(f"    xmin,ymin,zmin = {p0[0]} m,{p0[1]} m,{p0[2]} m")
        print(f"    xmax,ymax,zmax = {p1[0]} m,{p1[1]} m,{p1[2]} m")

    try:

        mp = construct_meltpool(mp_full_data, en_rand_ph)

        compute_porosity_vtk(
            sfp_abs, lh_val, vtk_abs, d_res, n_pts_bh, mp, boundBox=boundingBox
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
