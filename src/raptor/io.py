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
import os
import numpy as np
from typing import List, Tuple

from .structures import PathVector


def read_data(fname: str) -> np.ndarray:
    """
    Reads data from a .txt or .csv file.
    Two types of input data structures are supported:
        1. Melt pool timeseries -- T x 2 array of time, measurement
        2. Spectral component array -- N x 3 array of amplitudes,
           frequencies, and phases indexed by mode number.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Melt pool measurement file not found: {fname}"
        )
    with open(fname, "r") as input_file:
        first_data_line = next(
            (
                line
                for line in input_file
                if line.strip() and not line.lstrip().startswith("#")
            ),
            "",
        )
    delimiter = "," if "," in first_data_line else None
    # Preserve timestamp precision. Casting finely spaced absolute times to
    # float32 can make an otherwise uniform series appear non-uniform to FFT
    # consumers such as ``compute_spectral_components``.
    return np.loadtxt(fname, delimiter=delimiter, dtype=np.float64)


def read_scan_path(fname: str) -> List[PathVector]:
    """
    Reads scan path data from a file.
    """
    path_vector_mode: List[int] = []
    path_vector_position: List[np.ndarray] = []
    path_vector_parameter: List[float] = []
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Scan path file not found: {fname}")

    with open(fname, "r") as f:
        try:
            next(f)
        except StopIteration:
            return []

        for line_number, line in enumerate(f, 2):
            # reads each scan vector in layer
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Invalid scan-path row {line_number}: expected at least "
                    "six columns."
                )
            try:
                m_str, x_str, y_str, z_str, p_str, pr_str = parts[:6]
                mode_value = float(m_str)
                mode = int(mode_value)
                position = np.array([float(x_str), float(y_str), float(z_str)])
                power = float(p_str)
                parameter = float(pr_str)
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric value on scan-path row {line_number}."
                ) from error
            if mode_value != mode or mode not in (0, 1):
                raise ValueError(
                    f"Invalid mode on scan-path row {line_number}; "
                    "expected 0 or 1."
                )
            if (
                not np.isfinite(position).all()
                or not np.isfinite(power)
                or not np.isfinite(parameter)
            ):
                raise ValueError(
                    f"Non-finite value on scan-path row {line_number}."
                )
            if (mode == 0 and parameter <= 0.0) or (
                mode == 1 and parameter < 0.0
            ):
                raise ValueError(
                    f"Invalid parameter on scan-path row {line_number}."
                )
            path_vector_mode.append(mode)
            path_vector_position.append(position)
            path_vector_parameter.append(parameter)

    path_vector_time: List[float] = []
    start_t: List[float] = []
    end_t: List[float] = []
    start_pos: List[np.ndarray] = []
    end_pos: List[np.ndarray] = []

    if not path_vector_mode:
        return []

    if path_vector_mode[0] == 1:
        path_vector_time.append(path_vector_parameter[0])
    else:
        path_vector_time.append(0.0)
    for i in range(1, len(path_vector_mode)):
        i_prev = i - 1
        if path_vector_mode[i] == 1:
            dt = path_vector_parameter[i]
        else:
            dist = np.linalg.norm(
                path_vector_position[i] - path_vector_position[i_prev]
            )
            dt = (
                dist / path_vector_parameter[i]
                if path_vector_parameter[i] > 1e-12
                else 0.0
            )
        path_vector_time.append(path_vector_time[i_prev] + dt)

    for i in range(len(path_vector_time)):
        if path_vector_mode[i] == 0:
            if i == 0:
                print("exposure on first segment. skipping.")
                continue
            start_pos.append(path_vector_position[i - 1].copy())
            end_pos.append(path_vector_position[i].copy())
            start_t.append(path_vector_time[i - 1])
            end_t.append(path_vector_time[i])

    path_vectors = []
    for sc, ec, st, et in zip(start_pos, end_pos, start_t, end_t):
        vec = PathVector(sc, ec, st, et)
        path_vectors.append(vec)

    return path_vectors
