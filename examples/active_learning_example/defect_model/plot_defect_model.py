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
    laser_power = float(data["laser_power"])
    laser_velocity = float(data["laser_velocity"])

    meshgrid_size = len(mean_grid)

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    Xg_mm = np.linspace(bounds[0][0], bounds[0][1], meshgrid_size) * 1e3
    train_x_mm = np.array(dataset_x).flatten() * 1e3

    actual_train_y_um = dataset_y * 1e6
    actual_train_yerr_um = dataset_yerr * 1e6
    actual_mean_um = mean_grid.flatten() * 1e6
    std_um = np.sqrt(variance_grid.flatten()) * 1e6

    ax.plot(
        Xg_mm,
        actual_mean_um,
        color="royalblue",
        linewidth=2,
        label="GP Mean Prediction",
    )

    ax.fill_between(
        Xg_mm,
        np.maximum(0, actual_mean_um - 2 * std_um),
        actual_mean_um + 2 * std_um,
        color="royalblue",
        alpha=0.15,
        label="95% Conf. Interval",
    )

    ax.errorbar(
        train_x_mm,
        actual_train_y_um,
        yerr=2 * actual_train_yerr_um,
        color="black",
        ls="none",
        alpha=0.6,
        label="Training errors 2σ",
    )

    ax.scatter(
        train_x_mm,
        actual_train_y_um,
        color="black",
        marker="x",
        s=15,
        alpha=0.6,
        label="Training data",
    )

    # colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(D_CRIT_LIST)))

    for i, d_crit in enumerate(D_CRIT_LIST):
        d_um = d_crit * 1e6

        if np.max(actual_mean_um) > d_um:
            hs_limit = np.interp(d_um, actual_mean_um, Xg_mm)

            ax.hlines(
                d_um,
                np.min(Xg_mm),
                hs_limit,
                color="k",
                linestyle="-",
                linewidth=1,
                alpha=1,
            )

            ax.vlines(hs_limit, 0, d_um, colors="k", linestyles="-", linewidth=1)
            ax.plot(hs_limit, d_um, "o", color="k", markersize=4)

            ax.annotate(
                r"$h_{opt}$" + f"= {hs_limit:.3f} mm",
                xy=(np.min(Xg_mm), d_um),
                xytext=(3, 3),
                textcoords="offset points",
                color="k",
                fontsize=8,
            )

    ax.set_xlim(np.min(Xg_mm), np.max(Xg_mm))
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(bottom=0, top=np.max(actual_mean_um) * 1.5)
    ax.set_xlabel("Hatch Spacing (mm)")
    ax.set_ylabel("Maximum Pore Diameter ($\mu$m)")
    ax.set_title(f"P={laser_power}W, V={laser_velocity}m/s")
    ax.minorticks_on()

    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_filename = "process_map.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"Successfully generated {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot process map from saved surrogate data."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="defect_model_surrogate.npz",
        help="Path to the saved .npz surrogate file.",
    )
    args = parser.parse_args()
    main(args.file)
