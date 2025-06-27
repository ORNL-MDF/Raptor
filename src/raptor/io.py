import os
import numpy as np
from typing import List, Tuple

from .structures import PathSegment, PathVector


def read_data(fname: str) -> np.ndarray:
    """
    Reads data from a .txt or .csv file.
    Two types of input data structures are supported:
        1. Melt pool timeseries -- T x 2 array of time, measurement
        2. Spectral component array -- N x 3 array of amplitudes, frequences, and phases,
                                       indexed by modenumber (low to high frequency)
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Melt pool measurement file not found: {fname}")
    return np.loadtxt(fname, delimiter=",", dtype=np.float32)


def read_scan_path(fname: str) -> Tuple[List[PathSegment], List[PathVector]]:
    """
    Reads scan path data from a file.
    """
    segments: List[PathSegment] = []
    segment_mode: List[int] = []
    segment_position: List[np.ndarray] = []
    segment_parameter: List[float] = []
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Scan path file not found: {fname}")

    with open(fname, "r") as f:
        try:
            next(f)  # Skip header line
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
            segment_mode.append(mode)
            segment_position.append(position)
            segment_parameter.append(parameter)
            try:
                segment = PathSegment(
                    mode=np.int32(m_str),
                    position=np.array(
                        [float(x_str), float(y_str), float(z_str)], dtype=np.float64
                    ),
                    power=float(p_str),
                    parameter=float(pr_str),
                )
                segments.append(segment)
            except ValueError:
                continue

    segment_time: List[float] = []
    start_t: List[float] = []
    end_t: List[float] = []
    start_pos: List[np.ndarray] = []
    end_pos: List[np.ndarray] = []

    if segment_mode[0] == 1:
        segment_time.append(segment_parameter[0])
    else:
        segment_time.append(0.0)
    for i in range(1, len(segment_mode)):
        i_prev = i - 1
        if segment_mode[i] == 1:
            dt = segment_parameter[i]
        else:
            dist = np.linalg.norm(segment_position[i] - segment_position[i_prev])
            dt = dist / segment_parameter[i] if segment_parameter[i] > 1e-12 else 0.0
        segment_time.append(segment_time[i_prev] + dt)

    for i in range(len(segment_time)):
        if segment_mode[i] == 0:
            if i == 0:
                print("exposure on first segment. skipping.")
                continue
            start_pos.append(segment_position[i - 1].copy())
            end_pos.append(segment_position[i].copy())
            start_t.append(segment_time[i - 1])
            end_t.append(segment_time[i])

    active_vectors = []
    for sc, ec, st, et in zip(start_pos, end_pos, start_t, end_t):
        vec = PathVector(sc, ec, st, et)
        active_vectors.append(vec)

    return active_vectors
