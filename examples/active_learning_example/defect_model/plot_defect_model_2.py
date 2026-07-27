import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------
# Critical flaw size (meters)
D_CRIT_LIST = [10e-6, 20e-6, 40e-6]


def main(filepath):
    # Load Data
    data = np.load(filepath)
    mean_grid = data["mean_grid"]
    variance_grid = data["variance_grid"]
    dataset_x = data["dataset_x"]
    dataset_y = data["dataset_y"]
    dataset_yerr = data["dataset_yerr"]
    bounds = data["bounds"]
    n_grids = data["n_grids"]
    laser_power = float(data["laser_power"])
    laser_velocity = float(data["laser_velocity"])

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    Xg_mm_1d = [
        np.linspace(*bound, n_grid) * 1e3 for bound, n_grid in zip(bounds, n_grids)
    ]
    Xg_mm, Yg_mm = np.meshgrid(*Xg_mm_1d)

    train_x_mm = np.asarray(dataset_x) * 1e3
    train_y_um = np.asarray(dataset_y).reshape(-1) * 1e6
    train_yerr_um = np.asarray(dataset_yerr).reshape(-1) * 1e6

    try:
        mean_um = np.asarray(mean_grid).reshape(Xg_mm.shape) * 1e6
        std_um = np.sqrt(np.asarray(variance_grid)).reshape(Xg_mm.shape) * 1e6
    except ValueError as e:
        print(f"could not reshape mean and std grid arrays {e}")
        raise

    print(np.hstack((train_x_mm, train_y_um.reshape((-1, 1)))))

    CS = ax.contour(
        Xg_mm,
        Yg_mm,
        mean_um,
        [5] + [crit * 1e6 for crit in D_CRIT_LIST] + [80, 120, 170],
        linewidth=2,
    )

    ax.scatter(
        train_x_mm[:, 0],
        train_x_mm[:, 1],
        train_y_um,
        color="black",
        marker="x",
        zorder=2,
        alpha=0.6,
        label="Training data (maximum pore size estimated by CVAR)",
    )
    ax.scatter(
        train_x_mm[:, 0],
        train_x_mm[:, 1],
        5 * train_yerr_um,
        color="blue",
        marker="o",
        zorder=1,
        alpha=0.6,
        label="Uncertainty of training data",
    )

    ax.set_xlim(np.min(Xg_mm), np.max(Xg_mm))
    ax.set_ylim(np.min(Yg_mm), np.max(Yg_mm))
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
