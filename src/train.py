import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import NCA2D
from src.target import create_target_and_masks
from src.perturbations import apply_excision_lesion, apply_boundary_noise


class SamplePool:
    """
    Experience replay sample pool for NCA training.
    Capacity N = 1024 states.
    """

    def __init__(self, pool_size: int = 1024, H: int = 64, W: int = 64, device: str = "cpu"):
        self.pool_size = pool_size
        self.H = H
        self.W = W
        self.device = device
        # Initialize pool with fresh single seeds
        self.pool = torch.stack(
            [NCA2D.create_seed(1, H, W, device=device)[0] for _ in range(pool_size)], dim=0
        )

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.pool_size, (batch_size,))
        return indices, self.pool[indices].clone()

    def update(self, indices: torch.Tensor, states: torch.Tensor, replace_highest_idx: int = 0):
        """
        Writes states back into the pool, replacing the highest-loss sample with a fresh seed.
        """
        # Replace highest loss state with a fresh seed
        states[replace_highest_idx] = NCA2D.create_seed(1, self.H, self.W, device=self.device)[0]
        self.pool[indices] = states.detach()


def train_nca(
    regime: str = "regen",
    seed: int = 42,
    steps: int = 6000,
    out_dir: str = "checkpoints/regen_seed42",
    batch_size: int = 8,
    pool_size: int = 1024,
    device: str = None,
):
    """
    Trains an NCA2D model under one of four regimes:
      - 'grow': No sample pool; rolls forward 64..96 steps from clean seed every step.
      - 'persist': Sample pool N=1024, replaces highest-loss sample with clean seed, no lesions.
      - 'regen': Sample pool N=1024, applies random circular excision E_r (r in [4, 8]) to 50% batch items.
      - 'noise': Sample pool N=1024, applies continuous boundary Gaussian noise (sigma=0.05).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training regime '{regime}' with seed {seed} on device '{device}' for {steps} steps.")

    os.makedirs(out_dir, exist_ok=True)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator(device=device).manual_seed(seed)

    H, W = 64, 64
    target_rgba, mask_int, mask_bnd, mask_env, mask_interface = create_target_and_masks(H, W, device=device)

    model = NCA2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000], gamma=0.25)

    if regime != "grow":
        pool = SamplePool(pool_size=pool_size, H=H, W=W, device=device)

    log_records = []

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        rollout_steps = random.randint(64, 96)

        if regime == "grow":
            # Start fresh from single-pixel seed
            s = NCA2D.create_seed(batch_size, H, W, device=device)
            for _ in range(rollout_steps):
                s = model(s, generator=generator)
        else:
            # Sample from pool
            indices, s = pool.sample(batch_size)

            if regime == "regen":
                # Apply excision lesion to 50% of the batch items mid-rollout or at start
                split_step = rollout_steps // 2
                for _ in range(split_step):
                    s = model(s, generator=generator)

                # Lesion half of the batch
                half_b = batch_size // 2
                r = random.choice([4.0, 5.0, 6.0, 7.0, 8.0])
                s_lesion = apply_excision_lesion(s[:half_b], mask_bnd, radius=r, generator=generator)
                s = torch.cat([s_lesion, s[half_b:]], dim=0)

                for _ in range(rollout_steps - split_step):
                    s = model(s, generator=generator)

            elif regime == "noise":
                for _ in range(rollout_steps):
                    s = apply_boundary_noise(s, mask_bnd, sigma=0.05, generator=generator)
                    s = model(s, generator=generator)
            else:  # 'persist'
                for _ in range(rollout_steps):
                    s = model(s, generator=generator)

        # Morphological L2 loss against target RGBA
        diff_sq = (s[:, :4] - target_rgba) ** 2
        per_item_loss = diff_sq.mean(dim=[1, 2, 3])
        loss = per_item_loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        # Update pool
        if regime != "grow":
            worst_idx = int(torch.argmax(per_item_loss).item())
            pool.update(indices, s, replace_highest_idx=worst_idx)

        current_lr = scheduler.get_last_lr()[0]
        if step % 50 == 0 or step == steps:
            print(f"Step {step:04d}/{steps:04d} | Loss: {loss.item():.6f} | LR: {current_lr:.6f}")
            log_records.append({"step": step, "loss": loss.item(), "lr": current_lr})

    # Save checkpoint & training logs
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "regime": regime,
            "seed": seed,
            "steps": steps,
        },
        model_path,
    )
    df_log = pd.DataFrame(log_records)
    df_log.to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
    print(f"Model saved to {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Deterministic Sample-Pool NCA Trainer")
    parser.add_argument(
        "--regime",
        type=str,
        default="regen",
        choices=["grow", "persist", "regen", "noise"],
        help="Training regime",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=6000, help="Number of training steps")
    parser.add_argument("--out-dir", type=str, default=None, help="Output checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--pool-size", type=int, default=1024, help="Sample pool size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = f"checkpoints/{args.regime}_seed{args.seed}"

    train_nca(
        regime=args.regime,
        seed=args.seed,
        steps=args.steps,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
