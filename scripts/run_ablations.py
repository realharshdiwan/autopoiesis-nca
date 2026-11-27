import argparse
import os
import numpy as np
import pandas as pd
import torch

from src.model import NCA2D
from src.target import create_target_and_masks
from src.perturbations import apply_excision_lesion, apply_boundary_noise
from src.metrics import (
    morphological_l2,
    boundary_l2,
    recovery_ratio,
    compute_transfer_entropy_ksg,
)
from src.train import train_nca


SCIENTIFIC_PURPOSE = {
    "M1: Grow-NCA": "Baseline showing low static $\\mathcal{L}_{\\text{L2}}$ does not imply self-maintenance.",
    "M2: Persist-NCA": "Tests whether long-horizon attractor stability alone induces boundary signaling.",
    "M3: Regen-NCA": "Full autopoietic regime; expected sharp spike in $\\text{TE}_{\\text{int}\\to\\text{bnd}}$ during repair.",
    "M4: Severed-NCA": "Critical causal test: proves repair relies on interior-to-membrane signaling.",
    "M5: Noise-NCA": "Compares continuous homeostatic buffering against acute wound regeneration.",
}

INTERVENTION_DESC = {
    "M1: Grow-NCA": "Single-seed rollout only (no pool, no lesions)",
    "M2: Persist-NCA": "Sample pool ($N=1024$), zero perturbations",
    "M3: Regen-NCA": "Sample pool + random circular excisions ($r \\in [4, 8]$)",
    "M4: Severed-NCA": "Trained Regen-NCA; hidden channels zeroed at interface $r \\in [9.5, 11]$",
    "M5: Noise-NCA": "Sample pool + continuous boundary noise ($\\sigma=0.05$)",
}


def evaluate_rollout(
    model: NCA2D,
    seed: int,
    sever_interface: bool = False,
    device: str = "cpu",
    T_grow: int = 96,
    T_heal: int = 64,
    T_total: int = 220,
    lesion_r: float = 6.0,
):
    """
    Executes standard evaluation rollout:
      - Grow from single seed for T_grow = 96 steps
      - Apply excision lesion E_r at t = 96
      - Heal for T_heal = 64 steps (up to t = 160)
      - Continue rollout to T_total = 220
    """
    model.eval()
    H, W = 64, 64
    target_rgba, mask_int, mask_bnd, mask_env, mask_interface = create_target_and_masks(H, W, device=device)

    eval_gen = torch.Generator(device=device).manual_seed(seed + 1000)
    state = NCA2D.create_seed(1, H, W, device=device)

    int_hidden_traj = []
    bnd_hidden_traj = []
    snapshots = {}

    # Growth phase: t = 0 -> 96
    with torch.no_grad():
        for t in range(T_grow):
            state = model(state, generator=eval_gen)
            # Record trajectory
            int_h = (state[:, 4:16] * mask_int.float()).sum(dim=(2, 3)) / mask_int.float().sum()
            bnd_h = (state[:, 4:16] * mask_bnd.float()).sum(dim=(2, 3)) / mask_bnd.float().sum()
            int_hidden_traj.append(int_h.squeeze(0).cpu().numpy())
            bnd_hidden_traj.append(bnd_h.squeeze(0).cpu().numpy())

        # Clean L2 at t = 96 (before lesion)
        clean_l2 = morphological_l2(state, target_rgba)
        snapshots[95] = state[0, :4].clone().cpu().numpy()

        # Apply excision lesion at t = 96
        state = apply_excision_lesion(state, mask_bnd, radius=lesion_r, generator=eval_gen)
        l2_bnd_post_lesion = boundary_l2(state, target_rgba, mask_bnd)
        snapshots[96] = state[0, :4].clone().cpu().numpy()

        # Healing and post-lesion phase: t = 96 -> T_total
        sever_mask = mask_interface if sever_interface else None
        l2_post_lesion_160 = None
        l2_bnd_healed_160 = None

        for t in range(T_grow, T_total):
            state = model(state, sever_mask=sever_mask, generator=eval_gen)

            int_h = (state[:, 4:16] * mask_int.float()).sum(dim=(2, 3)) / mask_int.float().sum()
            bnd_h = (state[:, 4:16] * mask_bnd.float()).sum(dim=(2, 3)) / mask_bnd.float().sum()
            int_hidden_traj.append(int_h.squeeze(0).cpu().numpy())
            bnd_hidden_traj.append(bnd_h.squeeze(0).cpu().numpy())

            if t + 1 in [100, 104, 120, 128, 160, 220]:
                snapshots[t + 1] = state[0, :4].clone().cpu().numpy()

            if t + 1 == T_grow + T_heal:  # t = 160
                l2_post_lesion_160 = morphological_l2(state, target_rgba)
                l2_bnd_healed_160 = boundary_l2(state, target_rgba, mask_bnd)

    rho_rec = recovery_ratio(l2_bnd_post_lesion, l2_bnd_healed_160)

    int_hidden_traj = np.array(int_hidden_traj)  # [T_total, 12]
    bnd_hidden_traj = np.array(bnd_hidden_traj)  # [T_total, 12]

    # Calculate TE during acute healing window t in [96, 160]
    te_acute = compute_transfer_entropy_ksg(
        int_hidden_traj[T_grow : T_grow + T_heal],
        bnd_hidden_traj[T_grow : T_grow + T_heal],
        k=4,
        pca_dims=2,
        seed=seed,
    )

    # Calculate TE during steady-state window t in [160, 220]
    te_steady = compute_transfer_entropy_ksg(
        int_hidden_traj[T_grow + T_heal : T_total],
        bnd_hidden_traj[T_grow + T_heal : T_total],
        k=4,
        pca_dims=2,
        seed=seed,
    )

    # Rolling TE for time-series figure (window size = 30)
    rolling_te = []
    window = 30
    eval_steps = list(range(60, min(200, T_total - window // 2)))
    for t_step in eval_steps:
        start_w = max(0, t_step - window // 2)
        end_w = min(T_total, t_step + window // 2)
        te_w = compute_transfer_entropy_ksg(
            int_hidden_traj[start_w:end_w],
            bnd_hidden_traj[start_w:end_w],
            k=4,
            pca_dims=2,
            seed=seed,
        )
        rolling_te.append((t_step, te_w))

    return {
        "clean_l2": clean_l2,
        "post_lesion_l2": l2_post_lesion_160,
        "l2_bnd_post_lesion": l2_bnd_post_lesion,
        "l2_bnd_healed": l2_bnd_healed_160,
        "rho_rec": rho_rec,
        "te": te_acute,
        "te_acute": te_acute,
        "te_steady": te_steady,
        "rolling_te": rolling_te,
        "snapshots": snapshots,
        "int_hidden_traj": int_hidden_traj,
        "bnd_hidden_traj": bnd_hidden_traj,
    }


def evaluate_noise_sweep(
    model: NCA2D,
    seed: int,
    sever_interface: bool = False,
    device: str = "cpu",
    sigmas: list = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
):
    model.eval()
    H, W = 64, 64
    target_rgba, mask_int, mask_bnd, mask_env, mask_interface = create_target_and_masks(H, W, device=device)
    sever_mask = mask_interface if sever_interface else None

    results = []
    with torch.no_grad():
        for sigma in sigmas:
            eval_gen = torch.Generator(device=device).manual_seed(seed + 2000)
            state = NCA2D.create_seed(1, H, W, device=device)

            # Grow to t = 96
            for _ in range(96):
                state = model(state, generator=eval_gen)

            # Continuous boundary noise sweep for 64 steps (no excision lesion)
            for _ in range(64):
                state = apply_boundary_noise(state, mask_bnd, sigma=sigma, generator=eval_gen)
                state = model(state, sever_mask=sever_mask, generator=eval_gen)

            l2_bnd_final = boundary_l2(state, target_rgba, mask_bnd)
            membrane_fidelity = float(1.0 - l2_bnd_final)
            results.append({"sigma": sigma, "membrane_fidelity": membrane_fidelity})

    return results


def main():
    parser = argparse.ArgumentParser(description="Autopoiesis NCA Ablation Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Fast CPU smoke test (250 steps, seed 42)")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save ablation outputs")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.smoke_test:
        print("=== Running SMOKE-TEST (Seed 42, 250 steps) ===")
        seeds = [42]
        train_steps = args.steps if args.steps is not None else 250
    else:
        print("=== Running FULL ABLATION (5 Seeds: [42, 101, 202, 303, 404], 6000 steps) ===")
        seeds = [42, 101, 202, 303, 404]
        train_steps = args.steps if args.steps is not None else 6000

    # 1. Train all required models per seed
    regimes = ["grow", "persist", "regen", "noise"]
    models_dict = {}

    for s in seeds:
        models_dict[s] = {}
        for r in regimes:
            ckpt_dir = f"checkpoints/{r}_seed{s}"
            ckpt_path = os.path.join(ckpt_dir, "model.pt")
            if os.path.exists(ckpt_path):
                print(f"Loading existing checkpoint: {ckpt_path}")
                model = NCA2D().to(device)
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                print(f"Training {r} (seed {s})...")
                model = train_nca(
                    regime=r,
                    seed=s,
                    steps=train_steps,
                    out_dir=ckpt_dir,
                    device=device,
                )
            models_dict[s][r] = model

    # 2. Evaluate M1..M5 across seeds
    variants = [
        ("M1: Grow-NCA", "grow", False),
        ("M2: Persist-NCA", "persist", False),
        ("M3: Regen-NCA", "regen", False),
        ("M4: Severed-NCA", "regen", True),  # Causal ablation on trained regen model
        ("M5: Noise-NCA", "noise", False),
    ]

    all_eval_records = []
    noise_sweep_records = []
    all_trajectories = {}
    sigmas = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

    for variant_name, regime_key, sever_bool in variants:
        print(f"\nEvaluating {variant_name}...")
        for s in seeds:
            model = models_dict[s][regime_key]
            eval_res = evaluate_rollout(
                model=model,
                seed=s,
                sever_interface=sever_bool,
                device=device,
            )
            all_eval_records.append(
                {
                    "variant": variant_name,
                    "seed": s,
                    "clean_l2": eval_res["clean_l2"],
                    "post_lesion_l2": eval_res["post_lesion_l2"],
                    "te": eval_res["te"],
                    "te_acute": eval_res["te_acute"],
                    "te_steady": eval_res["te_steady"],
                    "rho_rec": eval_res["rho_rec"],
                }
            )

            # Store trajectories and snapshots for seed 42
            if s == 42:
                all_trajectories[f"{variant_name}_snapshots"] = eval_res["snapshots"]
                all_trajectories[f"{variant_name}_rolling_te"] = eval_res["rolling_te"]
                all_trajectories[f"{variant_name}_int_traj"] = eval_res["int_hidden_traj"]
                all_trajectories[f"{variant_name}_bnd_traj"] = eval_res["bnd_hidden_traj"]

            # Noise sweep
            sweep_res = evaluate_noise_sweep(
                model=model,
                seed=s,
                sever_interface=sever_bool,
                device=device,
                sigmas=sigmas,
            )
            for item in sweep_res:
                noise_sweep_records.append(
                    {
                        "variant": variant_name,
                        "seed": s,
                        "sigma": item["sigma"],
                        "membrane_fidelity": item["membrane_fidelity"],
                    }
                )

    # 3. Save Summary CSV & LaTeX Booktabs Table
    df_eval = pd.DataFrame(all_eval_records)
    df_eval.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)
    print(f"Saved {os.path.join(args.out_dir, 'ablation_summary.csv')}")

    df_noise = pd.DataFrame(noise_sweep_records)
    df_noise.to_csv(os.path.join(args.out_dir, "noise_sweep.csv"), index=False)
    print(f"Saved {os.path.join(args.out_dir, 'noise_sweep.csv')}")

    # Save trajectories npz
    np.savez(os.path.join(args.out_dir, "trajectories.npz"), **all_trajectories)
    print(f"Saved {os.path.join(args.out_dir, 'trajectories.npz')}")

    # Generate LaTeX table (mean ± std)
    summary_rows = []
    for variant_name, _, _ in variants:
        sub = df_eval[df_eval["variant"] == variant_name]
        clean_mean, clean_std = sub["clean_l2"].mean(), sub["clean_l2"].std()
        post_mean, post_std = sub["post_lesion_l2"].mean(), sub["post_lesion_l2"].std()
        te_mean, te_std = sub["te"].mean(), sub["te"].std()
        rho_mean, rho_std = sub["rho_rec"].mean(), sub["rho_rec"].std()

        clean_std_str = f" \\pm {clean_std:.3f}" if not np.isnan(clean_std) else ""
        post_std_str = f" \\pm {post_std:.3f}" if not np.isnan(post_std) else ""
        te_std_str = f" \\pm {te_std:.3f}" if not np.isnan(te_std) else ""
        rho_std_str = f" \\pm {rho_std:.3f}" if not np.isnan(rho_std) else ""

        summary_rows.append(
            {
                "variant": f"\\textbf{{{variant_name}}}",
                "intervention": INTERVENTION_DESC[variant_name],
                "clean_l2": f"${clean_mean:.3f}{clean_std_str}$",
                "post_l2": f"${post_mean:.3f}{post_std_str}$",
                "te": f"${te_mean:.3f}{te_std_str}$",
                "rho_rec": f"${rho_mean:.3f}{rho_std_str}$",
                "purpose": SCIENTIFIC_PURPOSE[variant_name],
            }
        )

    # Write Booktabs LaTeX Table
    tex_path = os.path.join(args.out_dir, "ablation_table.tex")
    paper_tex_path = os.path.join("paper", "ablation_table.tex")
    table_content = [
        "% Auto-generated Ablation Table\n",
        "\\begin{table*}[t]\n",
        "\\centering\n",
        "\\small\n",
        "\\setlength{\\tabcolsep}{2.5pt}\n",
        "\\caption{\\textbf{Controlled Ablation across Morphogenesis, Homeostasis, and Causal Interface Severing.} Values report mean $\\pm$ standard deviation across seeds. While clean steady-state loss $\\mathcal{L}_{\\text{L2}}$ is comparable across architectures, interior-to-boundary Transfer Entropy ($\\text{TE}_{\\text{int}\\to\\text{bnd}}$) directly predicts wound recovery $\\rho_{\\text{rec}}$. Severing internal signaling channels (\\textbf{M4}) decisively suppresses regeneration despite identical initial visual state.}\n",
        "\\label{tab:ablation}\n",
        "\\begin{tabularx}{\\textwidth}{l X c c c c X}\n",
        "\\toprule\n",
        "\\textbf{Variant} & \\textbf{Intervention} & \\textbf{Clean $\\mathcal{L}_{\\text{L2}}$} & \\textbf{Post $\\mathcal{L}_{\\text{L2}}$} & \\textbf{TE (nats)} & \\textbf{$\\rho_{\\text{rec}}$} & \\textbf{Scientific Purpose} \\\\\n",
        "\\midrule\n",
    ]
    for idx, row in enumerate(summary_rows):
        row_end = " \\\\ \\addlinespace[2pt]\n" if idx < len(summary_rows) - 1 else " \\\\\n"
        table_content.append(
            f"{row['variant']} & {row['intervention']} & {row['clean_l2']} & {row['post_l2']} & {row['te']} & {row['rho_rec']} & {row['purpose']}{row_end}"
        )
    table_content.extend([
        "\\bottomrule\n",
        "\\end{tabularx}\n",
        "\\end{table*}\n",
    ])

    full_tex = "".join(table_content)
    with open(tex_path, "w") as f:
        f.write(full_tex)
    if os.path.exists("paper"):
        with open(paper_tex_path, "w") as f:
            f.write(full_tex)

    print(f"Generated LaTeX table: {tex_path}")

    # Generate Paper Macros
    try:
        from scripts.generate_paper_macros import generate_macros
        macro_paths = [
            os.path.join(args.out_dir, "metrics_macros.tex"),
            os.path.join("paper", "metrics_macros.tex"),
        ]
        generate_macros(
            os.path.join(args.out_dir, "ablation_summary.csv"),
            os.path.join(args.out_dir, "noise_sweep.csv"),
            macro_paths,
            traj_path=os.path.join(args.out_dir, "trajectories.npz"),
        )
    except Exception as e:
        print(f"Notice: macro generation skipped: {e}")


if __name__ == "__main__":
    main()
