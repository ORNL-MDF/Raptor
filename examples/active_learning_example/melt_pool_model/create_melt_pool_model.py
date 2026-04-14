import argparse
import json
import logging
import os
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

# from scipy.stats import qmc
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
INITIAL_BOUNDS = [[500, 1500], [100, 300]]  # [velocity, power]
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
NUM_ITERATIONS = 35


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def melt_pool_dataset():
    # List of JSON file names
    json_files = [
        "parameters_A.json",
        "parameters_B.json",
        "parameters_C.json",
        "parameters_D.json",
        "parameters_E.json",
    ]

    all_keys = set()

    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)
            all_keys.update(data.keys())

    data_dict = {key: [] for key in all_keys}
    file_labels = []

    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)
            for key in all_keys:
                data_dict[key].extend(data.get(key, [None] * len(data_dict[key])))
            file_labels.extend(
                [json_file.split("_")[1].split(".")[0]]
                * len(data.get(list(data.keys())[0], []))
            )

    df = pd.DataFrame(data_dict)
    df["Location"] = file_labels

    power = pd.concat([df["Power"], df["Power"]], ignore_index=True)
    velocity = pd.concat([df["Velocity"], df["Velocity"]], ignore_index=True)

    dataset_x = np.stack([velocity, power], axis=1)
    dataset_y = pd.concat([df["right_depth"], df["left_depth"]], ignore_index=True)

    return dataset_x, dataset_y


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def graph(mean_grid, variance, dataset_x, dataset_y):
    plt.clf()
    plt.contourf(
        INITIAL_MESHGRIDS[0],
        INITIAL_MESHGRIDS[1],
        mean_grid,
        extend="both",
    )
    cbar = plt.colorbar()
    cbar.set_label("Melt pool depth (um)")

    plt.xlabel("Velocity (um/s)")
    plt.ylabel("Power (W)")

    X_train = np.array(dataset_x)
    plt.scatter(
        X_train[:, 0], X_train[:, 1], facecolor="none", color="black", marker="o"
    )

    plt.savefig("function_value.png")

    plt.clf()
    plt.contourf(
        INITIAL_MESHGRIDS[0],
        INITIAL_MESHGRIDS[1],
        variance,
        extend="both",
    )
    cbar = plt.colorbar()
    cbar.set_label("Melt pool depth variance (um)")

    plt.xlabel("Velocity (um/s)")
    plt.ylabel("Power (W)")

    X_train = np.array(dataset_x)
    plt.scatter(
        X_train[:, 0], X_train[:, 1], facecolor="none", color="black", marker="o"
    )

    plt.savefig("function_variance.png")


# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------
class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination
        self.workflow_id = ""

        dataset_x, dataset_y = melt_pool_dataset()

        self.dataset_x = dataset_x
        self.dataset_y: list[float] = dataset_y

    def assemble_message(
        self, operation: str, **kwargs: Any
    ) -> IntersectClientCallback:
        if operation == "initialize_workflow":
            payload = DialWorkflowCreationParamsClient(
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=INITIAL_BOUNDS,
                kernel="matern",
                length_per_dimension=False,
                y_is_good=False,
                backend="sklearn",
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
                strategy="expected_improvement",
            )
        elif operation == "get_surrogate_values":
            payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=INITIAL_POINTS_TO_PREDICT,
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

        print("in call")

        if has_error:
            print("============ERROR==============", file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            print(file=sys.stderr)
            raise Exception

        if operation == "dial.initialize_workflow":
            self.workflow_id: str = payload
            return self.assemble_message("get_surrogate_values")

        if operation == "dial.update_workflow_with_data":
            # This should go unused for surrogate-only workflows
            pass

        if operation == "dial.get_surrogate_values":
            self.mean_grid = np.array(payload[0]).reshape((MESHGRID_SIZE,) * NUM_DIMS)
            self.variance = np.array(payload[1]).reshape((MESHGRID_SIZE,) * NUM_DIMS)
            graph(self.mean_grid, self.variance, self.dataset_x, self.dataset_y)
            raise Exception("DONE!")

        if operation == "dial.get_next_point":
            # This should go unused for surrogate-only workflows
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

    try:
        with Path(args.config).open("rb") as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical("unable to load config file: %s", str(e))
        sys.exit(1)

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
