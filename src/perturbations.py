import torch


def apply_excision_lesion(
    state: torch.Tensor,
    mask_bnd: torch.Tensor,
    radius: float = 6.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Applies an excision lesion E_r by sampling a random coordinate x_0 inside mask_bnd
    and zeroing out all 16 channels within disk B_radius(x_0).

    Args:
        state: State tensor [B, 16, H, W]
        mask_bnd: Boolean membrane mask [1, 1, H, W] or [H, W]
        radius: Lesion disk radius
        generator: Optional torch.Generator for reproducibility

    Returns:
        Perturbed state tensor [B, 16, H, W]
    """
    state_out = state.clone()
    B, C, H, W = state.shape

    if mask_bnd.dim() == 4:
        bnd_2d = mask_bnd[0, 0]
    elif mask_bnd.dim() == 3:
        bnd_2d = mask_bnd[0]
    else:
        bnd_2d = mask_bnd

    # Extract all candidate boundary coordinates (y, x)
    coords = torch.nonzero(bnd_2d, as_tuple=False)  # [N_bnd, 2]
    num_coords = coords.shape[0]
    if num_coords == 0:
        return state_out

    # Precompute coordinate grids for distance calculation
    y_grid = torch.arange(H, dtype=torch.float32, device=state.device)
    x_grid = torch.arange(W, dtype=torch.float32, device=state.device)
    grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing="ij")

    for b in range(B):
        if generator is not None:
            idx = torch.randint(0, num_coords, (1,), generator=generator, device=state.device).item()
        else:
            idx = torch.randint(0, num_coords, (1,), device=state.device).item()

        y0, x0 = coords[idx, 0].float(), coords[idx, 1].float()
        dist_sq = (grid_y - y0) ** 2 + (grid_x - x0) ** 2
        lesion_mask = dist_sq <= (radius**2)

        # Zero out all 16 channels inside lesion disk
        state_out[b, :, lesion_mask] = 0.0

    return state_out


def apply_boundary_noise(
    state: torch.Tensor,
    mask_bnd: torch.Tensor,
    sigma: float = 0.05,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Applies continuous boundary Gaussian noise G_sigma:
    s_t(x) <- s_t(x) + eps_t(x), eps_t(x) ~ N(0, sigma^2 I_C) for all x in Omega_bnd.

    Args:
        state: State tensor [B, 16, H, W]
        mask_bnd: Boolean membrane mask [1, 1, H, W] or [H, W]
        sigma: Noise standard deviation
        generator: Optional torch.Generator

    Returns:
        Perturbed state tensor [B, 16, H, W]
    """
    if sigma <= 0.0:
        return state

    if mask_bnd.dim() == 2:
        mask_bnd = mask_bnd.unsqueeze(0).unsqueeze(0)
    elif mask_bnd.dim() == 3:
        mask_bnd = mask_bnd.unsqueeze(0)

    noise = torch.randn(
        state.shape, generator=generator, device=state.device, dtype=state.dtype
    ) * sigma
    state_out = state + noise * mask_bnd.float()
    return state_out
