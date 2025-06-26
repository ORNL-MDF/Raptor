from dataclasses import dataclass
import numpy as np
from numba.experimental import jitclass
from numba import int32, int64, float64, boolean

# PathSegment Class
pathsegment_spec = [
    ("mode", int32),
    ("position", float64[:]),
    ("power", float64),
    ("parameter", float64),
    ("time", float64),
    ("start_t", float64),
    ("end_t", float64),
    ("s", float64[:]),
    ("e", float64[:]),
    ("aabb", float64[:]),
    ("ew", float64[:]),
    ("ed", float64[:]),
    ("ph", float64[:]),
]


@jitclass(pathsegment_spec)
class PathSegment:
    """
    Represents a segment of a scan path.

    Attributes:
        mode: Segment mode (0 for exposure/mark, 1 for time delay).
        position: Endpoint of the segment (x, y, z) as a NumPy array.
        power: Laser power for the segment.
        parameter: Mode-dependent parameter (speed for mode 0, time for mode 1).
        time: Accumulated time at the end of this segment.
    """

    def __init__(self, mode, position, power, parameter):
        self.mode = mode
        self.position = position
        self.power = power
        self.parameter = parameter
        self.time = 0.0


# PathVector Class
pathvec_spec = [
    ("start_coord", float64[:]),
    ("end_coord", float64[:]),
    ("start_t", float64),
    ("end_t", float64),
    ("dt", float64),
    ("slsq", float64),
    ("aabb", float64[:]),
    ("ew", float64[:]),
    ("es", float64[:]),
    ("ed", float64[:]),
    ("ph", float64[:]),
    ("centroid", float64[:]),
    ("lx", float64),
    ("ly", float64),
    ("lz", float64),
]


@jitclass(pathvec_spec)
class PathVector:
    """
    Represents a scan vector.

    Attributes:
        start_coord: starting coordinate of the vector
        end_coord: ending coordinate of the vector
        start_t: (global) start time
        end_t: (global) end time
        dt: duration of scan
        slsq: segment length squared
    """

    def __init__(self, start_coord, end_coord, start_t, end_t):
        self.start_coord = start_coord
        self.end_coord = end_coord
        self.start_t = start_t
        self.end_t = end_t

        self.dt = self.start_t - self.end_t
        diff = self.end_coord - self.start_coord
        self.slsq = np.sqrt(np.sum(diff**2))


# MeltPool Class
meltpool_spec = [
    ("osc_info_W", float64[:, :]),
    ("osc_info_Dm", float64[:, :]),
    ("osc_info_Dh", float64[:, :]),
    ("max_rW", float64),
    ("max_rDm", float64),
    ("max_rDh", float64),
    ("enable_rand_phs", boolean),
    ("base_W", float64),
    ("base_Dm", float64),
    ("base_Dh", float64),
    ("max_modes", int64),
]


@jitclass(meltpool_spec)
class MeltPool:
    """
    Represents the melt pool (with oscillation properties)

    Attributes:
        base_W: mean (modenumber 0) of the oscillations
    """

    def __init__(
        self,
        osc_info_W,
        osc_info_Dm,
        osc_info_Dh,
        max_rW,
        max_rDm,
        max_rDh,
        enable_rand_phs,
    ):
        self.osc_info_W = osc_info_W
        self.osc_info_Dm = osc_info_Dm
        self.osc_info_Dh = osc_info_Dh
        self.max_rW = max_rW
        self.max_rDm = max_rDm
        self.max_rDh = max_rDh
        self.enable_rand_phs = enable_rand_phs
        self.base_W = self.osc_info_W[0, 0]
        self.base_Dm = self.osc_info_Dm[0, 0]
        self.base_Dh = self.osc_info_Dh[0, 0]
