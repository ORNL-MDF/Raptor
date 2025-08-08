from dataclasses import dataclass
import numpy as np
from numba.experimental import jitclass
from numba.typed import List as numbaList
from typing import List
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


# ScanStrategyBuilder
class ScanStrategyBuilder:
    """
    Handles scan strategy generation from process parameters.
    """

    def __init__(
        self,
        rvedims: np.ndarray,
        power: float,
        velocity: float,
        hatch: float,
        layer_thickness: float,
        rotation: float,
        overhang_hatch: float,
        additional_layers: int,
        output_name: str,
    ):
        self.output_name = output_name
        self.rvedims = rvedims
        self.power, self.velocity = power, velocity
        self.hatch = hatch
        self.layer_thickness = layer_thickness
        self.rotation = np.deg2rad(rotation)
        self.overhang_hatch = overhang_hatch
        self.rot_center = np.array(
            [
                self.rvedims[0] / 2,
                self.rvedims[1] / 2,
            ]
        )  # defining the rotation center as the center of the rve.
        self.additional_layers = additional_layers
        self.nlayers = np.int16(
            (self.rvedims[2] // self.layer_thickness + 1) + self.additional_layers
        )
        self.layers = {}

    def generate_0th_layer(self):
        """
        Generates the zeroth layer (global z=0) aligned nominally with [1,0,0].
        """
        xmin = (
            -self.overhang_hatch
        )  # starting from overhang_hatch outside the rve dimensions
        xmax = self.rvedims[1] + self.overhang_hatch
        ymin = -self.overhang_hatch
        ymax = self.rvedims[1] + self.overhang_hatch
        ylocs = np.arange(ymin, ymax, self.hatch)
        starts = np.vstack(
            [np.ones_like(ylocs) * xmin, np.arange(ymin, ymax, self.hatch)]
        ).transpose()
        ends = np.vstack(
            [np.ones_like(ylocs) * xmax, np.arange(ymin, ymax, self.hatch)]
        ).transpose()
        self.layers[0] = [starts, ends]

    def rotation_mat(self, k):
        """
        Returns the 2D rotation matrix given by self.rotation*k, k>=1.
        """
        return np.array(
            [
                [np.cos(k * self.rotation), -np.sin(k * self.rotation)],
                [np.sin(k * self.rotation), np.cos(k * self.rotation)],
            ]
        )

    def generate_kth_layer(self, k):
        """
        Generates layers rotated by self.rotation*k, k>=1
        """
        rot_mat = self.rotation_mat(k)
        # apply rotations relative to rotation center origin
        l_k_starts = np.array(
            [
                np.matmul(rot_mat, s - self.rot_center) + self.rot_center
                for s in self.layers[0][0]
            ]
        )
        l_k_ends = np.array(
            [
                np.matmul(rot_mat, e - self.rot_center) + self.rot_center
                for e in self.layers[0][1]
            ]
        )
        self.layers[k] = [l_k_starts, l_k_ends]

    def generate_layers(self):
        """
        Generates layers starting from the 0th by successive applications of rotation matrix.
        """
        self.generate_0th_layer()
        for k in range(1, self.nlayers + 1):
            self.generate_kth_layer(k)

    def construct_vectors(self) -> list[PathVector]:
        """
        Constructs PathVector objects based on current layer dictionary.
        The list of PathVectors is associated with a key in a new dictionary.
        """
        if not self.layers.keys():
            print("No layers generated. Aborting.")
            return
        else:
            self.pathvec_layers = {}
            for l_key in self.layers.keys():
                l_start, l_end = self.layers[l_key]
                se_pairs = [
                    np.vstack(
                        [
                            np.hstack([1, s, l_key * self.layer_thickness, 0, 0]),
                            np.hstack(
                                [
                                    0,
                                    e,
                                    l_key * self.layer_thickness,
                                    self.power,
                                    self.velocity,
                                ]
                            ),
                        ]
                    )
                    for s, e in zip(l_start, l_end)
                ]
                allpaths = np.vstack([np.vstack(se) for se in se_pairs])
                segment_mode: List[int] = []
                segment_position: List[np.ndarray] = []
                segment_parameter: List[float] = []
                for path in allpaths:
                    m, x, y, z, _, p = path
                    position = np.array([x, y, z])
                    segment_mode.append(m)
                    segment_position.append(position)
                    segment_parameter.append(p)

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
                        dist = np.linalg.norm(
                            segment_position[i] - segment_position[i_prev]
                        )
                        dt = (
                            dist / segment_parameter[i]
                            if segment_parameter[i] > 1e-12
                            else 0.0
                        )
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
                self.pathvec_layers[l_key] = active_vectors

    def process_vectors(self) -> List[PathVector]:
        """
        Downselects vectors and computes local time offsets.
        Note that the bounding box is defined implicitly by specifying rvedims at instantiation.
        """
        self.rveBoundBox = np.array(
            [
                [0.0e-6, 0.0e-6, self.layer_thickness],
                [
                    self.rvedims[0],
                    self.rvedims[1],
                    self.layer_thickness + self.rvedims[2],
                ],
            ]
        )
        time_offset = 0.0
        start_L = 0
        end_L = len(self.layers.keys())
        rve_bb_infl_factor = 0.1
        bbx0_infl, bBy0_infl, bBz0_infl = self.rveBoundBox[0] * (1 - rve_bb_infl_factor)
        bbx1_infl, bBy1_infl, bBz1_infl = self.rveBoundBox[1] * (1 + rve_bb_infl_factor)
        start_L = int(np.max([np.floor(bBz0_infl / self.layer_thickness) - 1, start_L]))
        end_L = int(np.min([bBz1_infl / self.layer_thickness, end_L]))
        all_vectors = numbaList()
        print(f"Starting at L{start_L}, ending at L{end_L-1}")
        for layerkey in range(start_L, end_L):
            z_offset = layerkey * self.layer_thickness
            print(f"Computing layer {layerkey}.")
            layer_vectors = self.pathvec_layers[layerkey]
            if not layer_vectors:
                print(f"Warning: No segments in L{layerkey}). Skipping.")
                continue
            max_t_layer = layer_vectors[-1].start_t if layer_vectors else 0.0
            for vec in layer_vectors:
                vec.start_coord[2] += z_offset
                vec.end_coord[2] += z_offset
                vec.start_t += time_offset
                vec.end_t += time_offset

                vec_bb = np.array(
                    [
                        [vec.start_coord[0], vec.start_coord[1]],
                        [vec.end_coord[0], vec.end_coord[1]],
                    ]
                )
                # check if path is inside the inflated rve bounding box
                if (
                    (np.max(vec_bb[:, 0]) < np.min([bbx0_infl, bbx1_infl]))
                    or (np.min(vec_bb[:, 0]) > np.max([bbx0_infl, bbx1_infl]))
                    or (np.max(vec_bb[:, 1]) < np.min([bBy0_infl, bBy1_infl]))
                    or (np.min(vec_bb[:, 1]) > np.max([bBy0_infl, bBy1_infl]))
                ):
                    continue
                else:
                    all_vectors.append(vec)
            if layer_vectors:
                time_offset += max_t_layer

        if not all_vectors:
            raise ValueError("No PathVectors found in the scan strategy.")

        print(f"Total active (exposure) vectors: {len(all_vectors)}")
        return all_vectors

    def write_layers(self):
        """
        Writes scan paths as files.
        """
        if len(self.layers.keys()) == 0:
            print("No layers found to write.")
            return
        else:
            for l_key in self.layers.keys():
                l_start, l_end = self.layers[l_key]
                se_pairs = [
                    np.vstack(
                        [
                            np.hstack([1, s, l_key * self.layer_thickness, 0, 0]),
                            np.hstack(
                                [
                                    0,
                                    e,
                                    l_key * self.layer_thickness,
                                    self.power,
                                    self.velocity,
                                ]
                            ),
                        ]
                    )
                    for s, e in zip(l_start, l_end)
                ]
                allpaths = np.vstack([np.vstack(se) for se in se_pairs])
                header_str = "Mode X(m) Y(m) Z(m) Power(W) tParam"
                filename = self.output_name + "{}".format(l_key)
                np.savetxt(
                    filename,
                    allpaths,
                    fmt="%.6f",
                    delimiter=" ",
                    header=header_str,
                    comments="",
                )
                print(
                    "Wrote file " + filename + " for layer {}.".format(l_key), end="\n"
                )
