import os
import numpy as np
from typing import List, Tuple

from .structures import PathVector


def read_data(fname: str) -> np.ndarray:
    """
    Reads data from a .txt or .csv file.
    Two types of input data structures are supported:
        1. Melt pool timeseries -- T x 2 array of time, measurement
        2. Spectral component array -- N x 3 array of amplitudes, frequencies, and phases,
                                       indexed by modenumber (low to high frequency)
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Melt pool measurement file not found: {fname}")
    return np.loadtxt(fname, delimiter=",", dtype=np.float32)


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

        for scan_vec_num, line in enumerate(f, 1):
            # reads each scan vector in layer
            parts = line.split()
            if len(parts) < 6:
                continue
            m_str, x_str, y_str, z_str, p_str, pr_str = parts[:6]
            mode = int(float(m_str))
            position = np.array([float(x_str), float(y_str), float(z_str)])
            parameter = float(pr_str)
            path_vector_mode.append(mode)
            path_vector_position.append(position)
            path_vector_parameter.append(parameter)

    path_vector_time: List[float] = []
    start_t: List[float] = []
    end_t: List[float] = []
    start_pos: List[np.ndarray] = []
    end_pos: List[np.ndarray] = []

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
