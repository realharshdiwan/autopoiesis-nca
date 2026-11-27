import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.autolayout": True,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "lines.linewidth": 2.0,
        }
    )


def rgba_to_rgb_composite(rgba: np.ndarray, bg_color: tuple = (0.96, 0.96, 0.98)) -> np.ndarray:
    """
    Composites RGBA image [4, H, W] onto a background color for high-contrast visualization.
    """
    rgba = np.clip(rgba, 0.0, 1.0)
    rgb = np.transpose(rgba[:3], (1, 2, 0))  # [H, W, 3]
    alpha = np.transpose(rgba[3:4], (1, 2, 0))  # [H, W, 1]
    bg = np.array(bg_color).reshape((1, 1, 3))
    composite = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(composite, 0.0, 1.0)


def plot_fig1_regeneration(trajectories_path: str, out_dir: str):
    data = np.load(trajectories_path, allow_pickle=True)
    m3_snapshots = data["M3: Regen-NCA_snapshots"].item()
    m4_snapshots = data["M4: Severed-NCA_snapshots"].item()

    timesteps = [96, 104, 128, 160]
    titles = [
        r"$t = 96$ (Lesion)",
        r"$t = 104$ ($\Delta t = 8$)",
        r"$t = 128$ ($\Delta t = 32$)",
        r"$t = 160$ (Healed / Stalled)",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(8.5, 3.4))

    for col, (t, title) in enumerate(zip(timesteps, titles)):
        # Top row: M3: Regen-NCA (Intact signaling)
        ax_top = axes[0, col]
        img_m3 = rgba_to_rgb_composite(m3_snapshots[t])
        ax_top.imshow(img_m3)
        ax_top.set_title(title, fontsize=9.5, fontweight="bold")
        ax_top.axis("off")

        # Bottom row: M4: Severed-NCA (Severed signaling)
        ax_bot = axes[1, col]
        img_m4 = rgba_to_rgb_composite(m4_snapshots[t])
        ax_bot.imshow(img_m4)
        ax_bot.axis("off")

    # Add row labels
    axes[0, 0].text(
        -0.10,
        0.5,
        "M3: Regen-NCA\n(Intact)",
        transform=axes[0, 0].transAxes,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="right",
        color="#0d5c75",
    )
    axes[1, 0].text(
        -0.10,
        0.5,
        "M4: Severed-NCA\n(Severed)",
        transform=axes[1, 0].transAxes,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="right",
        color="#a83232",
    )

    plt.tight_layout(pad=0.6)
    pdf_path = os.path.join(out_dir, "fig1_regeneration.pdf")
    png_path = os.path.join(out_dir, "fig1_regeneration.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 1: {pdf_path}")


def plot_fig2_te_phase(trajectories_path: str, noise_sweep_path: str, out_dir: str):
    data = np.load(trajectories_path, allow_pickle=True)
    df_noise = pd.read_csv(noise_sweep_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.0))

    # Color palette
    colors = {
        "M1: Grow-NCA": "#7f7f7f",
        "M2: Persist-NCA": "#1f77b4",
        "M3: Regen-NCA": "#2ca02c",
        "M4: Severed-NCA": "#d62728",
        "M5: Noise-NCA": "#ff7f0e",
    }
    linestyles = {
        "M1: Grow-NCA": ":",
        "M2: Persist-NCA": "--",
        "M3: Regen-NCA": "-",
        "M4: Severed-NCA": "-.",
        "M5: Noise-NCA": "-",
    }

    # Panel (a): Rolling Transfer Entropy
    variants = ["M1: Grow-NCA", "M2: Persist-NCA", "M3: Regen-NCA", "M4: Severed-NCA", "M5: Noise-NCA"]
    for var in variants:
        key = f"{var}_rolling_te"
        if key in data:
            te_data = data[key]
            steps = [pt[0] for pt in te_data]
            vals = [pt[1] for pt in te_data]
            ax1.plot(
                steps,
                vals,
                label=var.replace("M", "M"),
                color=colors.get(var, "#333333"),
                linestyle=linestyles.get(var, "-"),
                linewidth=2.0 if "Regen" in var or "Severed" in var else 1.4,
            )

    ax1.axvline(x=96, color="#444444", linestyle="--", alpha=0.75, linewidth=1.2)
    ax1.text(98, ax1.get_ylim()[1] * 0.88 if ax1.get_ylim()[1] > 0 else 0.5, "Lesion $\mathcal{E}_r$ ($t=96$)", fontsize=8.5, color="#444444", style="italic")
    ax1.set_xlabel("Discrete Rollout Step $t$")
    ax1.set_ylabel(r"Transfer Entropy $\text{TE}_{\text{int} \to \text{bnd}}$ (nats)")
    ax1.set_title(r"$\mathbf{(a)}$ Interior-to-Membrane Information Flow", fontsize=10.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="#ffffff", framealpha=0.9, loc="upper right", fontsize=7.5)

    # Panel (b): Steady-state membrane fidelity vs continuous noise
    metric_col = "membrane_fidelity" if "membrane_fidelity" in df_noise.columns else "rho_rec"
    for var in variants:
        sub = df_noise[df_noise["variant"] == var]
        if len(sub) > 0:
            grouped = sub.groupby("sigma")[metric_col].agg(["mean", "std"]).reset_index()
            sigmas = grouped["sigma"].values
            means = grouped["mean"].values
            stds = grouped["std"].fillna(0.0).values

            means_clamped = np.clip(means, 0.0, 1.05)
            lower_clamped = np.clip(means - stds, 0.0, 1.05)
            upper_clamped = np.clip(means + stds, 0.0, 1.05)

            ax2.plot(
                sigmas,
                means_clamped,
                label=var,
                color=colors.get(var, "#333333"),
                linestyle=linestyles.get(var, "-"),
                marker="o",
                markersize=4.0,
                linewidth=1.8,
            )
            ax2.fill_between(sigmas, lower_clamped, upper_clamped, color=colors.get(var, "#333333"), alpha=0.15)

    ax2.set_xlabel(r"Boundary Noise Intensity $\sigma$")
    ax2.set_ylabel(r"Membrane Fidelity ($1 - \mathcal{L}_{\text{L2}}^{\text{bnd}}$)")
    ax2.set_title(r"$\mathbf{(b)}$ Membrane Fidelity vs. Boundary Stress", fontsize=10.5)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, facecolor="#ffffff", framealpha=0.9, loc="lower left", fontsize=7.5)

    plt.tight_layout(pad=0.6)
    pdf_path = os.path.join(out_dir, "fig2_te_phase.pdf")
    png_path = os.path.join(out_dir, "fig2_te_phase.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 2: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default="paper")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()

    traj_path = os.path.join(args.results_dir, "trajectories.npz")
    noise_path = os.path.join(args.results_dir, "noise_sweep.csv")

    if not os.path.exists(traj_path) or not os.path.exists(noise_path):
        print(f"Error: Required result files not found in {args.results_dir}.")
        return

    plot_fig1_regeneration(traj_path, args.out_dir)
    plot_fig2_te_phase(traj_path, noise_path, args.out_dir)


if __name__ == "__main__":
    main()
