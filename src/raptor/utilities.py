import numpy as np
from typing import List, Tuple
from .structures import PathVector


class ScanPathBuilder:
    """
    Handles scan strategy generation from process parameters using explicit boundaries.
    """

    def __init__(
        self,
        bound_box: np.ndarray,
        power: float,
        scan_speed: float,
        hatch_spacing: float,
        layer_height: float,
        rotation: float,
        scan_extension: float,
        extra_layers: int,
    ):
        """
        Initializes the builder with geometric and process parameters.

        Args:
            min_point: The [x, y, z] minimum corner of the part volume.
            max_point: The [x, y, z] maximum corner of the part volume.
            power: Laser power in Watts.
            scan_speed: Scan speed in m/s.
            hatch_spacing: Distance between adjacent scan vectors.
            layer_height: Thickness of each layer.
            rotation: Inter-layer rotation angle in degrees.
            scan_extension: Extra length to add to scan vectors beyond the part boundary.
            extra_layers: Extra layers to generate above the defined part volume.
        """
        self.min_point = bound_box[0]
        self.max_point = bound_box[1]

        self.power = power
        self.scan_speed = scan_speed
        self.hatch_spacing = hatch_spacing
        self.layer_height = layer_height
        self.rotation = np.deg2rad(rotation)
        self.scan_extension = scan_extension
        self.extra_layers = extra_layers

        self.dimensions = self.max_point - self.min_point

        self.center_of_rotation = (self.min_point[:2] + self.max_point[:2]) / 2.0
        self.nlayers = np.int16(
            (self.dimensions[2] // self.layer_height + 1) + self.extra_layers
        )

        self.layers = {}
        self.path_vector_layers = {}

    def generate_layers(self):
        """
        Generates all layers by rotating the base layer.
        """

        # 1. Generate the base layer aligned nominally with [1,0,0]
        xmin = self.min_point[0] - self.scan_extension
        xmax = self.max_point[0] + self.scan_extension

        ymin = self.min_point[1] - self.scan_extension
        ymax = self.max_point[1] + self.scan_extension

        ys = np.arange(ymin, ymax, self.hatch_spacing)
        starts = np.vstack([np.ones_like(ys) * xmin, ys]).transpose()
        ends = np.vstack([np.ones_like(ys) * xmax, ys]).transpose()
        self.layers[0] = [starts, ends]

        # 2. Generate the kth layer by rotating the base layer
        for k in range(1, self.nlayers + 1):
            angle = k * self.rotation
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

            starts = np.array(
                [
                    np.matmul(rotation_matrix, s - self.center_of_rotation)
                    + self.center_of_rotation
                    for s in self.layers[0][0]
                ]
            )
            ends = np.array(
                [
                    np.matmul(rotation_matrix, e - self.center_of_rotation)
                    + self.center_of_rotation
                    for e in self.layers[0][1]
                ]
            )
            self.layers[k] = [starts, ends]
    
    def process_vectors(self):
        """
        Creates and processes PathVector objects from the generated layers.
        """
        if not self.layers.keys():
            print("No layers generated. Aborting.")
            return
        time_offset = 0.0
        rve_bound_box = np.array([self.min_point, self.max_point])
        # constructing vectors.
        for layer_key,(layer_start,layer_end) in self.layers.items():
            if layer_start.size==0:
                self.path_vector_layers[layer_key] = []
                continue
            active_vectors = []
            layer_time = time_offset
            for start_xy,end_xy in zip(layer_start,layer_end):
                # defining start, end points and start, end times
                vector_start = np.array([start_xy[0],start_xy[1],layer_key * self.layer_height])
                vector_end = np.array([end_xy[0],end_xy[1],layer_key * self.layer_height])
                vector_length = np.linalg.norm(vector_end - vector_start)
                scan_duration = vector_length / self.scan_speed if self.scan_speed > 1e-12 else 0.0
                if layer_key >=1 and not active_vectors:
                    start_time = self.path_vector_layers[layer_key-1][-1].start_time
                else:
                    start_time = layer_time
                end_time = start_time + scan_duration
                # PathVector object instantiation
                path_vector = PathVector(vector_start,vector_end,start_time,end_time)
                active_vectors.append(path_vector)
                layer_time = end_time
            self.path_vector_layers[layer_key] = active_vectors
            time_offset = active_vectors[-1].end_time if active_vectors else 0.0

        # condensing and returning all vectors
        all_vectors = []
        for layer_key,layer_vectors in self.path_vector_layers.items():
            for vec in layer_vectors:
                # not currently filtering
                vec.set_coordinate_frame()
                all_vectors.append(vec)
        return all_vectors

    def write_layers(self, ouput_name):
        """
        Writes the generated raw scan paths to text files.
        """

        for l_key, (l_start, l_end) in self.layers.items():
            if l_start.size == 0:
                continue

            se_pairs = [
                np.vstack(
                    [
                        np.hstack([1, s, l_key * self.layer_height, 0, 0]),
                        np.hstack(
                            [
                                0,
                                e,
                                l_key * self.layer_height,
                                self.power,
                                self.scan_speed,
                            ]
                        ),
                    ]
                )
                for s, e in zip(l_start, l_end)
            ]

            allpaths = np.vstack(se_pairs)
            header_str = "Mode X(m) Y(m) Z(m) Power(W) tParam"
            filename = f"{output_name}_layer_{l_key}.txt"

            np.savetxt(
                filename,
                allpaths,
                fmt="%.6f",
                delimiter=" ",
                header=header_str,
                comments="",
            )
            print(f"Wrote file {filename}")
