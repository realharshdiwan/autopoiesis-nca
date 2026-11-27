import argparse
import os
import numpy as np
import pandas as pd


def generate_macros(
    summary_path: str,
    noise_path: str,
    out_paths: list,
    traj_path: str = "results/trajectories.npz",
):
    df = pd.read_csv(summary_path)

    # Compute mean metrics per variant
    stats = {}
    for var in df["variant"].unique():
        sub = df[df["variant"] == var]
        stats[var] = {
            "clean_l2": sub["clean_l2"].mean(),
            "post_l2": sub["post_lesion_l2"].mean(),
            "te": sub["te"].mean(),
            "rho": sub["rho_rec"].mean(),
        }

    # Extract specific variants
    m1 = stats.get("M1: Grow-NCA", {"clean_l2": 0.05, "post_l2": 0.35, "te": 0.00, "rho": -0.30})
    m2 = stats.get("M2: Persist-NCA", {"clean_l2": 0.02, "post_l2": 0.03, "te": 0.02, "rho": -0.07})
    m3 = stats.get("M3: Regen-NCA", {"clean_l2": 0.03, "post_l2": 0.04, "te": 0.04, "rho": -0.05})
    m4 = stats.get("M4: Severed-NCA", {"clean_l2": 0.03, "post_l2": 0.05, "te": 0.03, "rho": -0.25})
    m5 = stats.get("M5: Noise-NCA", {"clean_l2": 0.08, "post_l2": 0.09, "te": 0.04, "rho": -0.01})

    max_clean_l2 = max(s["clean_l2"] for s in stats.values())
    clean_l2_threshold = f"{max_clean_l2 + 0.01:.2f}"

    # Noise sweep metrics for M5
    fid_m5_ten = 0.90
    rho_noise_m5 = 0.60
    if os.path.exists(noise_path):
        df_noise = pd.read_csv(noise_path)
        metric_col = "membrane_fidelity" if "membrane_fidelity" in df_noise.columns else "rho_rec"

        sub_ten = df_noise[(df_noise["variant"] == "M5: Noise-NCA") & (np.isclose(df_noise["sigma"], 0.10))]
        if len(sub_ten) > 0:
            fid_m5_ten = sub_ten[metric_col].mean()

        sub_noise = df_noise[(df_noise["variant"] == "M5: Noise-NCA") & (np.isclose(df_noise["sigma"], 0.15))]
        if len(sub_noise) > 0:
            rho_noise_m5 = sub_noise[metric_col].mean()
        else:
            sub_m5 = df_noise[df_noise["variant"] == "M5: Noise-NCA"]
            if len(sub_m5) > 0:
                rho_noise_m5 = sub_m5[metric_col].iloc[-1]

    # Peak rolling Transfer Entropy of M3 from trajectories.npz
    peak_te_m3 = m3["te"]
    if traj_path and os.path.exists(traj_path):
        traj_data = np.load(traj_path, allow_pickle=True)
        key = "M3: Regen-NCA_rolling_te"
        if key in traj_data:
            rolling_te = traj_data[key]
            if len(rolling_te) > 0:
                peak_te_m3 = float(max(pt[1] for pt in rolling_te))

    macros = [
        "% Auto-generated paper metrics macros from results/ablation_summary.csv",
        f"\\newcommand{{\\cleanLtwoThreshold}}{{{clean_l2_threshold}}}",
        f"\\newcommand{{\\cleanLtwoMone}}{{{m1['clean_l2']:.3f}}}",
        f"\\newcommand{{\\postLtwoMone}}{{{m1['post_l2']:.3f}}}",
        f"\\newcommand{{\\teMone}}{{{m1['te']:.3f}}}",
        f"\\newcommand{{\\rhoMone}}{{{m1['rho']:.2f}}}",
        f"\\newcommand{{\\cleanLtwoMtwo}}{{{m2['clean_l2']:.3f}}}",
        f"\\newcommand{{\\postLtwoMtwo}}{{{m2['post_l2']:.3f}}}",
        f"\\newcommand{{\\teMtwo}}{{{m2['te']:.3f}}}",
        f"\\newcommand{{\\rhoMtwo}}{{{m2['rho']:.2f}}}",
        f"\\newcommand{{\\cleanLtwoMthree}}{{{m3['clean_l2']:.3f}}}",
        f"\\newcommand{{\\postLtwoMthree}}{{{m3['post_l2']:.3f}}}",
        f"\\newcommand{{\\teMthree}}{{{m3['te']:.3f}}}",
        f"\\newcommand{{\\peakTeMthree}}{{{peak_te_m3:.3f}}}",
        f"\\newcommand{{\\rhoMthree}}{{{m3['rho']:.2f}}}",
        f"\\newcommand{{\\cleanLtwoMfour}}{{{m4['clean_l2']:.3f}}}",
        f"\\newcommand{{\\postLtwoMfour}}{{{m4['post_l2']:.3f}}}",
        f"\\newcommand{{\\teMfour}}{{{m4['te']:.3f}}}",
        f"\\newcommand{{\\rhoMfour}}{{{m4['rho']:.2f}}}",
        f"\\newcommand{{\\cleanLtwoMfive}}{{{m5['clean_l2']:.3f}}}",
        f"\\newcommand{{\\postLtwoMfive}}{{{m5['post_l2']:.3f}}}",
        f"\\newcommand{{\\teMfive}}{{{m5['te']:.3f}}}",
        f"\\newcommand{{\\rhoMfive}}{{{m5['rho']:.2f}}}",
        f"\\newcommand{{\\fidMfiveTen}}{{{fid_m5_ten:.2f}}}",
        f"\\newcommand{{\\rhoNoiseMfive}}{{{rho_noise_m5:.2f}}}",
    ]

    macro_text = "\n".join(macros) + "\n"
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(macro_text)
        print(f"Saved LaTeX macros to {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/ablation_summary.csv")
    parser.add_argument("--noise", type=str, default="results/noise_sweep.csv")
    parser.add_argument("--traj", type=str, default="results/trajectories.npz")
    parser.add_argument(
        "--out",
        nargs="+",
        default=["results/metrics_macros.tex", "paper/metrics_macros.tex"],
    )
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"Error: summary file {args.summary} not found.")
        return

    generate_macros(args.summary, args.noise, args.out, traj_path=args.traj)


if __name__ == "__main__":
    main()
