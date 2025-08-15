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
        span_speed: float,
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
            span_speed: Scan speed in m/s.
            hatch_spacing: Distance between adjacent scan vectors.
            layer_height: Thickness of each layer.
            rotation: Inter-layer rotation angle in degrees.
            scan_extension: Extra length to add to scan vectors beyond the part boundary.
            extra_layers: Extra layers to generate above the defined part volume.
        """
        self.min_point = bound_box[0]
        self.max_point = bound_box[1]

        self.power = power
        self.span_speed = span_speed
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
            rotation_maxtrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

            starts = np.array(
                [
                    np.matmul(rotation_maxtrix, s - self.center_of_rotation)
                    + self.center_of_rotation
                    for s in self.layers[0][0]
                ]
            )
            ends = np.array(
                [
                    np.matmul(rotation_maxtrix, e - self.center_of_rotation)
                    + self.center_of_rotation
                    for e in self.layers[0][1]
                ]
            )
            self.layers[k] = [starts, ends]

    def construct_vectors(self):
        """
        Construct timed PathVector objects from the generated raw paths.
        """
        for l_key, (l_start, l_end) in self.layers.items():
            if l_start.size == 0:
                self.path_vector_layers[l_key] = []
                continue

            active_vectors = []
            layer_time = 0.0
            for start_xy, end_xy in zip(l_start, l_end):
                start_point = np.array([start_xy[0], start_xy[1], 0.0])
                end_point = np.array([end_xy[0], end_xy[1], 0.0])

                dist = np.linalg.norm(end_point - start_point)
                duration = dist / self.span_speed if self.span_speed > 1e-12 else 0.0

                start_time = layer_time
                end_time = start_time + duration

                path_vector = PathVector(start_point, end_point, start_time, end_time)
                active_vectors.append(path_vector)
                layer_time = end_time

            self.path_vector_layers[l_key] = active_vectors

    def process_vectors(self) -> List[PathVector]:
        """
        Filter vectors to keep those inside the RVE and computes global time offsets.
        """
        if not self.path_vector_layers:
            raise ValueError(
                "No path_vector constructed. Call construct_vectors() first."
            )

        rve_bound_box = np.array([self.min_point, self.max_point])
        time_offset = 0.0
        all_vectors = []

        print(f"Processing {len(self.path_vector_layers)} layers...")
        for layer_key, layer_vectors in self.path_vector_layers.items():
            if not layer_vectors:
                continue

            max_t_layer = layer_vectors[-1].end_time if layer_vectors else 0.0

            for vec in layer_vectors:
                vec.start_point[2] += layer_key * self.layer_height
                vec.end_point[2] += layer_key * self.layer_height
                vec.start_time += time_offset
                vec.end_time += time_offset

                mid_point = (vec.start_point + vec.end_point) / 2.0
                if np.any(mid_point < rve_bound_box[0]) or np.any(
                    mid_point > rve_bound_box[1]
                ):
                    continue

                all_vectors.append(vec)

            if layer_vectors:
                time_offset += max_t_layer

        if not all_vectors:
            raise ValueError("No path vectors found inside the specified RVE.")

        print(f"Total active (exposure) vectors: {len(all_vectors)}")
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
                                self.span_speed,
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
