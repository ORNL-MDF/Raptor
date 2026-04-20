import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.use("agg")

import numpy as np
from intersect_sdk import (
    INTERSECT_JSON_VALUE,
    HierarchyConfig,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
    default_intersect_lifecycle_loop,
)

from dial_dataclass import (
    DialInputPredictions,
    DialInputSingleOtherStrategy,
    DialWorkflowCreationParamsClient,
    DialWorkflowDatasetUpdate,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------
INITIAL_BOUNDS = ((0.5, 1.5), (100, 300))  # [velocity (m/s), power (W)]
UNIT_BOUNDS = ((0.0, 1.0), (0.0, 1.0))
NUM_DIMS = len(INITIAL_BOUNDS)
MESHGRID_SIZE = 101
INITIAL_MESHGRIDS = np.meshgrid(
    *[
        np.linspace(dim_bounds[0], dim_bounds[1], MESHGRID_SIZE)
        for dim_bounds in INITIAL_BOUNDS
    ],
    indexing="ij",
)
INITIAL_POINTS_TO_PREDICT = np.hstack([mg.reshape(-1, 1) for mg in INITIAL_MESHGRIDS])


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def x_to_unit(X):
    X = np.asarray(X, dtype=float)
    lo = np.array([b[0] for b in INITIAL_BOUNDS])
    hi = np.array([b[1] for b in INITIAL_BOUNDS])
    return (X - lo) / (hi - lo + 1e-12)


def x_from_unit(U):
    U = np.asarray(U, dtype=float)
    lo = np.array([b[0] for b in INITIAL_BOUNDS])
    hi = np.array([b[1] for b in INITIAL_BOUNDS])
    return U * (hi - lo) + lo


def melt_pool_dataset(convert_to_m_per_s: float = 1e-3, convert_to_m: float = 1e-6):
    json_files = [
        "parameters_A.json",
        "parameters_B.json",
        "parameters_C.json",
        "parameters_D.json",
        "parameters_E.json",
    ]

    all_keys = set()

    for json_file in json_files:
        try:
            with open(json_file, "r") as file:
                data = json.load(file)
                all_keys.update(data.keys())
        except FileNotFoundError:
            pass

    data_dict: dict[str, list] = {key: [] for key in all_keys}

    for json_file in json_files:
        try:
            with open(json_file, "r") as file:
                data = json.load(file)
                for key in all_keys:
                    data_dict[key].extend(
                        data.get(key, [None] * len(next(iter(data.values()))))
                    )
        except FileNotFoundError:
            continue

    df = pd.DataFrame(data_dict)

    df["Velocity"] = df["Velocity"] * convert_to_m_per_s

    power = pd.concat([df["Power"], df["Power"]], ignore_index=True)
    velocity = pd.concat([df["Velocity"], df["Velocity"]], ignore_index=True)

    depth = pd.concat([df["right_depth"], df["left_depth"]], ignore_index=True)
    width = pd.concat([df["right_width"], df["left_width"]], ignore_index=True)
    height = pd.concat([df["right_height"], df["left_height"]], ignore_index=True)

    depth = depth * convert_to_m
    width = width * convert_to_m
    height = height * convert_to_m

    combined_df = pd.DataFrame(
        {
            "Velocity": velocity,
            "Power": power,
            "depth": depth,
            "width": width,
            "height": height,
        }
    )

    grouped = (
        combined_df.groupby(["Velocity", "Power"])
        .agg(
            depth_mean=("depth", "mean"),
            depth_std=("depth", lambda x: x.std(ddof=0)),
            width_mean=("width", "mean"),
            width_std=("width", lambda x: x.std(ddof=0)),
            height_mean=("height", "mean"),
            height_std=("height", lambda x: x.std(ddof=0)),
        )
        .reset_index()
    )

    dataset_x = np.stack([grouped["Velocity"], grouped["Power"]], axis=1)

    dataset_y_dict = {
        "depth_mean": grouped["depth_mean"].values,
        "depth_std": grouped["depth_std"].values,
        "width_mean": grouped["width_mean"].values,
        "width_std": grouped["width_std"].values,
        "height_mean": grouped["height_mean"].values,
        "height_std": grouped["height_std"].values,
    }

    return dataset_x, dataset_y_dict


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def graph(mean_grid, variance, dataset_x, feature_name):
    plt.clf()
    plt.contourf(
        INITIAL_MESHGRIDS[0],
        INITIAL_MESHGRIDS[1],
        mean_grid,
        extend="both",
    )
    cbar = plt.colorbar()
    cbar.set_label(f"{feature_name} (m)")

    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Power (W)")

    X_train = np.array(dataset_x)
    plt.scatter(
        X_train[:, 0], X_train[:, 1], facecolor="none", color="black", marker="o"
    )

    plt.savefig(f"function_value_{feature_name}.png")


# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------
class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination
        self.workflow_id = ""

        dataset_x, dataset_y_dict = melt_pool_dataset()

        self.dataset_x = dataset_x.tolist()
        self.dataset_x_unit = x_to_unit(self.dataset_x).tolist()
        self.dataset_y_dict = dataset_y_dict

        self.bounds_unit = UNIT_BOUNDS

        self.features = [
            "depth_mean",
            "depth_std",
            "width_mean",
            "width_std",
            "height_mean",
            "height_std",
        ]
        self.feature_idx = 0
        self.surrogate_results: dict[str, tuple[Any, Any]] = {}

    def assemble_message(
        self, operation: str, **kwargs: Any
    ) -> IntersectClientCallback:
        current_feature = self.features[self.feature_idx]
        current_y = self.dataset_y_dict[current_feature].tolist()

        # nondimensionalized lengthscale, on the normalized x data
        length_scale = .5
        # prior variance of the kernel
        prior_variance = 2.0
        # standard deviation of output error yerr
        yerr = 1.e-1

        if operation == "initialize_workflow":
            payload = DialWorkflowCreationParamsClient(
                dataset_x=self.dataset_x_unit,
                dataset_y=current_y,
                dim_x=2,
                bounds=self.bounds_unit,
                kernel="matern",
                length_per_dimension=True,
                y_is_good=False,
                backend="sklearn",
                kernel_args={"length_scale": length_scale, "length_scale_bounds": "fixed",
                             "constant_value": prior_variance, "constant_value_bounds": "fixed",
                             "noise_level": yerr**2, "noise_level_bounds": "fixed", # noise level is noise variance
                             },
                backend_args=None,
                seed=-1,
                preprocess_standardize=True,
            )
        elif operation == "update_workflow_with_data":
            payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                **kwargs,
            )
        elif operation == "get_next_point":
            payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy="upper_confidence_bound",
                strategy_args={"exploit": 0., "explore": 1},
            )
        elif operation == "get_surrogate_values":
            points_to_predict_unit = x_to_unit(INITIAL_POINTS_TO_PREDICT).tolist()
            payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=points_to_predict_unit,
            )

        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.service_destination,
                    operation=f"dial.{operation}",
                    payload=payload,
                )
            ]
        )

    def __call__(
        self,
        _source: str,
        operation: str,
        has_error: bool,
        payload: INTERSECT_JSON_VALUE,
    ) -> IntersectClientCallback:

        if has_error:
            print("============ERROR==============", file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            print(file=sys.stderr)
            raise Exception

        if operation == "dial.initialize_workflow":
            self.workflow_id = payload
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.update_workflow_with_data":
            pass

        if operation == "dial.get_surrogate_values":
            data = payload['data']
            mean_grid = np.array(data[0]).reshape((MESHGRID_SIZE,) * NUM_DIMS)
            variance = np.array(data[1]).reshape((MESHGRID_SIZE,) * NUM_DIMS)

            current_feature = self.features[self.feature_idx]
            self.surrogate_results[current_feature] = (mean_grid, variance)

            graph(mean_grid, variance, self.dataset_x, current_feature)

            self.feature_idx += 1

            if self.feature_idx < len(self.features):
                return self.assemble_message("initialize_workflow")
            else:
                save_data = {}
                for feature, (mean_grid, _) in self.surrogate_results.items():
                    save_data[f"{feature}"] = mean_grid

                save_data["velocity"] = np.linspace(
                    INITIAL_BOUNDS[0][0], INITIAL_BOUNDS[0][1], MESHGRID_SIZE
                )
                save_data["power"] = np.linspace(
                    INITIAL_BOUNDS[1][0], INITIAL_BOUNDS[1][1], MESHGRID_SIZE
                )

                np.savez("melt_pool_surrogates.npz", **save_data)
                print("Saved surrogates to 'melt_pool_surrogates.npz'")

                raise Exception("DONE!")

        if operation == "dial.get_next_point":
            pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated client")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    with Path(args.config).open("rb") as f:
        from_config_file = json.load(f)

    active_learning = ActiveLearningOrchestrator(
        service_destination=HierarchyConfig(
            **from_config_file["intersect-hierarchy"]
        ).hierarchy_string(".")
    )

    config = IntersectClientConfig(
        initial_message_event_config=active_learning.assemble_message(
            "initialize_workflow"
        ),
        **from_config_file["intersect"],
    )

    client = IntersectClient(
        config=config,
        user_callback=active_learning,
    )

    default_intersect_lifecycle_loop(
        client,
    )
