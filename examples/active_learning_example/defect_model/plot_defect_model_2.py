import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scalers import SCALER_REGISTRY, InputScaler

matplotlib.use("Agg")


# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------
# Critical flaw size (meters)
D_CRIT_LIST = [10e-6, 20e-6, 40e-6]


def main(filepath):
    # Load Data
    data = np.load(filepath, allow_pickle=True)
    mean_grid = data["mean_grid"]
    variance_grid = data["variance_grid"]
    dataset_x = data["dataset_x"]
    dataset_y = data["dataset_y"]
    dataset_yerr = data["dataset_yerr"]
    bounds = data["bounds"]
    n_grids = data["n_grids"]
    laser_power = float(data["laser_power"])
    laser_velocity = float(data["laser_velocity"])

    def to_mm(v):
        return v * 1e3

    def to_um(v):
        return v * 1e6

    Xg_1d = [
        np.linspace(*bound, n_grid) for bound, n_grid in zip(bounds, n_grids)
    ]
    Xg, Yg = np.meshgrid(*Xg_1d)

    bounds = np.asarray(bounds)
    train_x = np.asarray(dataset_x)
    train_y = np.asarray(dataset_y).reshape(-1)
    train_yerr = np.asarray(dataset_yerr).reshape(-1)

    try:
        mean_grid = np.asarray(mean_grid).reshape(Xg.shape)
        std_grid = np.sqrt(np.asarray(variance_grid)).reshape(Xg.shape)
    except ValueError as e:
        print(f"could not reshape mean and std grid arrays {e}")
        raise

    print(to_mm(np.hstack((train_x, train_y.reshape((-1, 1))))))

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    CS = ax.contour(
        to_mm(Xg),
        to_mm(Yg),
        to_um(mean_grid),
        [5] + [to_um(crit) for crit in D_CRIT_LIST] + [80, 120, 170],
        linewidths=2,
    )

    ax.scatter(
        to_mm(train_x[:, 0]),
        to_mm(train_x[:, 1]),
        to_um(train_y),
        color="black",
        marker="x",
        zorder=2,
        alpha=0.6,
        label="Training data (maximum pore size estimated by CVAR)",
    )
    ax.scatter(
        to_mm(train_x[:, 0]),
        to_mm(train_x[:, 1]),
        to_um(train_yerr),
        color="blue",
        marker="o",
        zorder=1,
        alpha=0.6,
        label="Uncertainty of training data",
    )

    ax.set_xlim(*to_mm(bounds[0]))
    ax.set_ylim(*to_mm(bounds[1]))
    ax.set_xlabel("Hatch Spacing (mm)")
    ax.set_ylabel("Layer Height (mm)")
    ax.set_title(f"Process map for P={laser_power}W, V={laser_velocity}m/s")
    ax.minorticks_on()

    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.clabel(CS, fontsize=10)
    # ax.colorbar()

    plt.tight_layout()
    output_filename = "process_map_2.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"Successfully generated {output_filename}")

    input_scaler = InputScaler(bounds=bounds)
    try:
        scaler = data["scaler"].item()
        print(scaler)
    except Exception as e:
        print(f"did not find scaler, skipping next plot: {e}")
        exit(1)

    # fit a new sable model
    from sable import ScaledRBFModel, DiscretizedSurrogateModel
    import logging

    logging.basicConfig(level=logging.INFO)

    fm = ScaledRBFModel(
        x_dimension=2,
        x_range=(0.0, 1.0),
        sigma_range=(1e-2, 0.5),
        gamma=0.3,
    )
    model = DiscretizedSurrogateModel(
        featuremodel=fm,
        n_features=20000,
        prior_std=2.0,
        p=1.0,
    )

    model.fit(input_scaler.to_unit(train_x), *scaler.scale(train_y, train_yerr))

    ## plot the new sable model
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    XYg = np.stack((Xg.reshape(-1), Yg.reshape(-1)), axis=1)
    plot_mean = False
    if plot_mean:
        mean_new, std_new = model.predict(input_scaler.to_unit(XYg))
        mean_new, std_new = scaler.unscale(mean_new, std_new)
        mean_new = np.asarray(mean_new).reshape(Xg.shape)
        CS = ax.contour(
            to_mm(Xg),
            to_mm(Yg),
            to_um(mean_new),
            [5] + [crit * 1e6 for crit in D_CRIT_LIST] + [80, 120, 170],
            linewidths=2,
        )
    else:
        n_samples = 20
        y_sample = model.sample_posterior(input_scaler.to_unit(XYg), n_samples)
        for y in y_sample.T:
            y = np.asarray(scaler.unscale(y, 0.0)[0])
            CS = ax.contour(
                to_mm(Xg),
                to_mm(Yg),
                to_um(y.reshape(Xg.shape)),
                [30],
                linewidths=2,
                alpha=0.3,
            )

    ax.scatter(
        to_mm(train_x[:, 0]),
        to_mm(train_x[:, 1]),
        to_um(train_y),
        color="black",
        marker="x",
        zorder=2,
        alpha=0.5,
        label="Training data (maximum pore size estimated by CVAR)",
    )

    ax.set_xlim(*to_mm(bounds[0]))
    ax.set_ylim(*to_mm(bounds[1]))
    ax.set_xlabel("Hatch Spacing (mm)")
    ax.set_ylabel("Layer Height (mm)")
    ax.set_title(f"Process map for P={laser_power}W, V={laser_velocity}m/s")
    ax.minorticks_on()

    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.clabel(CS, fontsize=10)
    # ax.colorbar()

    plt.tight_layout()
    output_filename = "process_map_2_new.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"Successfully generated {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 2-input process map from saved surrogate data."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="defect_model_surrogate_2.npz",
        help="Path to the saved .npz surrogate file.",
    )
    args = parser.parse_args()
    main(args.file)
