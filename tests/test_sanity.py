import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.model import NCA2D
from src.target import create_target_and_masks
from src.perturbations import apply_excision_lesion, apply_boundary_noise
from src.metrics import (
    morphological_l2,
    boundary_l2,
    recovery_ratio,
    compute_transfer_entropy_ksg,
)


def test_zero_initialized_nca_identity():
    """
    Test 1: Zero-initialized NCA2D (fc2.weight == 0) acts as an identity operator
    on alive cells on step 1.
    """
    model = NCA2D()
    seed = NCA2D.create_seed(batch_size=1, H=64, W=64)
    s_next = model(seed)

    assert torch.allclose(s_next, seed, atol=1e-6), (
        "Zero-initialized NCA must leave state unchanged on step 1."
    )


def test_exact_determinism():
    """
    Test 2: Exact determinism given identical torch.Generator seed.
    """
    model = NCA2D()
    # Randomize MLP weights slightly to test dynamic updates
    torch.manual_seed(999)
    for p in model.parameters():
        if p.requires_grad:
            p.data.normal_(0.0, 0.05)

    seed_state = NCA2D.create_seed(batch_size=2, H=64, W=64)

    # Run 1
    gen1 = torch.Generator().manual_seed(42)
    s1 = seed_state.clone()
    for _ in range(5):
        s1 = model(s1, generator=gen1)

    # Run 2
    gen2 = torch.Generator().manual_seed(42)
    s2 = seed_state.clone()
    for _ in range(5):
        s2 = model(s2, generator=gen2)

    assert torch.equal(s1, s2), "NCA dynamics must be strictly deterministic given identical generator seed."


def test_ksg_transfer_entropy_coupled_vs_independent():
    """
    Test 3: compute_transfer_entropy_ksg yields TE > 0.25 nats on a coupled linear autoregressive
    system Y_{t+1} = 0.7 X_t + 0.2 Y_t + eps and TE < 0.05 nats on independent Gaussian processes.
    """
    np.random.seed(42)
    T = 300

    # 1. Coupled AR(1) system: Y_{t+1} = 0.7 X_t + 0.2 Y_t + eps
    X_coupled = np.zeros((T, 12))
    Y_coupled = np.zeros((T, 12))
    for t in range(T - 1):
        X_coupled[t + 1] = 0.6 * X_coupled[t] + np.random.normal(0, 0.5, size=12)
        Y_coupled[t + 1] = 0.7 * X_coupled[t] + 0.2 * Y_coupled[t] + np.random.normal(0, 0.3, size=12)

    te_coupled = compute_transfer_entropy_ksg(X_coupled, Y_coupled, k=4, pca_dims=2, seed=42)

    # 2. Independent Gaussian processes
    X_indep = np.random.normal(0, 1.0, size=(T, 12))
    Y_indep = np.random.normal(0, 1.0, size=(T, 12))

    te_indep = compute_transfer_entropy_ksg(X_indep, Y_indep, k=4, pca_dims=2, seed=42)

    assert te_coupled > 0.25, f"Expected TE_coupled > 0.25 nats, got {te_coupled:.4f}"
    assert te_indep < 0.05, f"Expected TE_indep < 0.05 nats, got {te_indep:.4f}"


def test_causal_severing_mask():
    """
    Test 4: sever_mask strictly zeroes hidden channels 4:16 on Omega_interface
    while preserving visible RGBA channels.
    """
    target_rgba, mask_int, mask_bnd, mask_env, mask_interface = create_target_and_masks(64, 64)
    model = NCA2D()

    # Create active state with non-zero RGBA and hidden channels across the whole grid
    state = torch.full((1, 16, 64, 64), 0.75, dtype=torch.float32)
    state[:, 3:4] = 1.0  # alpha = 1 so all cells are alive

    s_next = model(state, sever_mask=mask_interface)

    # Check that on mask_interface, hidden channels 4:16 are exactly 0.0
    interface_2d = mask_interface[0, 0]
    hidden_on_interface = s_next[0, 4:16, interface_2d]
    assert torch.all(hidden_on_interface == 0.0), "Hidden channels on Omega_interface must be exactly zero."

    # Check that visible RGBA channels are preserved (non-zero)
    rgba_on_interface = s_next[0, 0:4, interface_2d]
    assert torch.all(rgba_on_interface > 0.0), "Visible RGBA channels must remain preserved on Omega_interface."


def test_perturbations():
    """
    Sanity check for lesion and boundary noise operators.
    """
    target_rgba, mask_int, mask_bnd, mask_env, _ = create_target_and_masks(64, 64)
    state = torch.ones((1, 16, 64, 64), dtype=torch.float32)

    # Lesion test
    gen = torch.Generator().manual_seed(42)
    lesioned = apply_excision_lesion(state, mask_bnd, radius=6.0, generator=gen)
    assert (lesioned == 0.0).any(), "Lesion must zero out a region."

    # Noise test
    noisy = apply_boundary_noise(state, mask_bnd, sigma=0.05, generator=gen)
    assert not torch.equal(noisy, state), "Boundary noise must alter state."
    # Outside membrane, noise must be zero
    assert torch.equal(noisy[:, :, mask_env[0, 0]], state[:, :, mask_env[0, 0]]), (
        "Noise must only affect Omega_bnd."
    )
