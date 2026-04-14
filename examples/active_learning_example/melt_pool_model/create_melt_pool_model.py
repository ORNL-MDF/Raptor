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


# AM SPECIFIC FUNCTIONS ------------------------------------------------------------------------------------------------------
def am_dataset_coleman():
    # List of JSON file names
    json_files = [
        "parameters_A.json",
        "parameters_B.json",
        "parameters_C.json",
        "parameters_D.json",
        "parameters_E.json",
    ]

    # Initialize a set to store all keys
    all_keys = set()

    # Loop through the JSON files to determine all keys
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)
            all_keys.update(data.keys())

    # Initialize a dictionary to store lists of data
    data_dict = {key: [] for key in all_keys}
    file_labels = []

    # Loop through the JSON files to collect data and labels
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)
            for key in all_keys:
                data_dict[key].extend(data.get(key, [None] * len(data_dict[key])))
            file_labels.extend(
                [json_file.split("_")[1].split(".")[0]]
                * len(data.get(list(data.keys())[0], []))
            )

    # Create a DataFrame using the collected data and labels
    df = pd.DataFrame(data_dict)
    df["Location"] = file_labels

    df["average_depth"] = (df["right_depth"] + df["left_depth"]) / 2.0

    dataset_x = [df["Power"], df["Velocity"]]
    dataset_x = np.transpose(dataset_x)
    print(dataset_x)
    dataset_y = df["average_depth"]
    print(dataset_y)

    return dataset_x, dataset_y


def graph(mean_grid, variance, dataset_x, dataset_y):
    if NUM_DIMS == 2:
        plt.clf()
        plt.contourf(
            INITIAL_MESHGRIDS[0],
            INITIAL_MESHGRIDS[1],
            mean_grid,
            extend="both",
        )
        cbar = plt.colorbar()
        cbar.set_label("Melt pool depth (um)")
        plt.xlabel("Power (W)")
        plt.ylabel("Velocity (um/s)")
        # add black dots for data points and a red marker for the recommendation:
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
        plt.xlabel("Power (W)")
        plt.ylabel("Velocity (um/s)")
        # add black dots for data points and a red marker for the recommendation:
        X_train = np.array(dataset_x)
        plt.scatter(
            X_train[:, 0], X_train[:, 1], facecolor="none", color="black", marker="o"
        )

        plt.savefig("function_variance.png")

        # Line plot
        plt.clf()
        plt.plot(INITIAL_MESHGRIDS[0][:, 0], mean_grid[:, 0], "b-")
        upper_variance_bound = mean_grid[:, 0] + variance[:, 0]
        lower_variance_bound = mean_grid[:, 0] - variance[:, 0]

        print(upper_variance_bound.shape)

        plt.fill_between(
            INITIAL_MESHGRIDS[0][:, 0],
            upper_variance_bound,
            lower_variance_bound,
            alpha=0.9,
        )

        tolerance = 1e-5
        for idx, val in enumerate(dataset_x):
            power = dataset_x[idx][0]
            velocity = dataset_x[idx][1]
            if np.abs(velocity - 500) < tolerance:
                plt.plot(power, dataset_y[idx], "r.")

        plt.savefig("lineout.png")

    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        message = (
            "Number of dimensions is not equal to two -\nBayesian Optimization plot is not available.\n"
            "Add plotting to the graph(self) function in\nautomated_client.py to generate a custom plot."
        )
        ax.text(0.5, 0.5, message, fontsize=18, ha="center", va="center", wrap=True)
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig("graph.png")


# MANUAL INPUTS ------------------------------------------------------------------------------------------------------

INITIAL_BOUNDS = [[100, 300], [500, 1500]]
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

# ORCHESTRATOR ------------------------------------------------------------------------------------------------------


class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination

        # This value gets populated from the return value of initializing the workflow
        self.workflow_id = ""

        dataset_x, dataset_y = am_dataset_coleman()

        self.dataset_x = dataset_x
        self.dataset_y: list[float] = dataset_y

    # create a message to send to the server
    def assemble_message(
        self, operation: str, **kwargs: Any
    ) -> IntersectClientCallback:
        if operation == "initialize_workflow":
            payload = DialWorkflowCreationParamsClient(
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=INITIAL_BOUNDS,
                kernel="rbf",
                length_per_dimension=False,  # allow the matern to use separate length scales for the two parameters
                y_is_good=False,  # we wish to minimize y (the error)
                backend="sklearn",  # "sklearn" or "gpax"
                seed=-1,  # Use seed = -1 for random results
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
            print("operation is get_surrogate_values")
            print(INITIAL_POINTS_TO_PREDICT)
            print(self.workflow_id)
            payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=INITIAL_POINTS_TO_PREDICT,
            )
        else:
            err_msg = f"Invalid operation {operation}"
            raise Exception(err_msg)  # noqa: TRY002
        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.service_destination,
                    operation=f"dial.{operation}",
                    payload=payload,
                )
            ]
        )

    # The callback function.  This is called whenever the server responds to our message.
    # This could instead be implemented by defining a callback method (and passing it later), but here we chose to directly make the object callable.
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
            raise Exception  # noqa: TRY002 (break INTERSECT loop)
        if operation == "dial.initialize_workflow":
            self.workflow_id: str = payload
            return self.assemble_message("get_surrogate_values")
        if operation == "dial.update_workflow_with_data":
            # This should go unused for surrogate-only workflows
            pass
        if (
            operation == "dial.get_surrogate_values"
        ):  # if we receive a grid of surrogate values, record it for graphing, then ask for the next recommended point
            self.mean_grid = np.array(payload[0]).reshape((MESHGRID_SIZE,) * NUM_DIMS)
            self.variance = np.array(payload[1]).reshape((MESHGRID_SIZE,) * NUM_DIMS)
            graph(self.mean_grid, self.variance, self.dataset_x, self.dataset_y)
            raise Exception("DONE!")
        if operation == "dial.get_next_point":
            # This should go unused for surrogate-only workflows
            pass

        err_msg = f"Unknown operation received: {operation}"
        raise Exception(err_msg)  # noqa: TRY002 (INTERSECT interaction mechanism)


# MAIN ------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # In production, everything in this dictionary should come from a configuration file, command line arguments, or environment variables.
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

    # use the orchestator to create the client
    client = IntersectClient(
        config=config,
        user_callback=active_learning,  # the callback (here we use a callable object, as discussed above)
    )

    # This will run the send message -> wait for response -> callback -> repeat cycle until we have 25 points (and then raise an Exception)
    default_intersect_lifecycle_loop(
        client,
    )
