import torch


def create_target_and_masks(H: int = 64, W: int = 64, device: str = "cpu"):
    """
    Creates the two-compartment synthetic organism target and Markov blanket geometry masks.

    Lattice: H x W grid with center at (H/2, W/2) = (32, 32).
    - Omega_int (r <= 10.0): Interior core (RGBA: 0.15, 0.65, 0.70, 1.0)
    - Omega_bnd (10.0 < r <= 16.0): Bounding membrane (RGBA: 0.85, 0.45, 0.20, 1.0)
    - Omega_env (r > 16.0): External environment (RGBA: 0.0, 0.0, 0.0, 0.0)
    - Omega_interface (9.5 <= r <= 11.0): 1-to-2 pixel boundary interface ring

    Returns:
        target_rgba: torch.Tensor of shape [1, 4, H, W] (float32)
        mask_int: torch.Tensor of shape [1, 1, H, W] (bool)
        mask_bnd: torch.Tensor of shape [1, 1, H, W] (bool)
        mask_env: torch.Tensor of shape [1, 1, H, W] (bool)
        mask_interface: torch.Tensor of shape [1, 1, H, W] (bool)
    """
    cy, cx = H // 2, W // 2
    y = torch.arange(H, dtype=torch.float32, device=device)
    x = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt((grid_y - cy) ** 2 + (grid_x - cx) ** 2)

    mask_int = (r <= 10.0).unsqueeze(0).unsqueeze(0)
    mask_bnd = ((r > 10.0) & (r <= 16.0)).unsqueeze(0).unsqueeze(0)
    mask_env = (r > 16.0).unsqueeze(0).unsqueeze(0)
    mask_interface = ((r >= 9.5) & (r <= 11.0)).unsqueeze(0).unsqueeze(0)

    target_rgba = torch.zeros((1, 4, H, W), dtype=torch.float32, device=device)

    # Core RGBA
    target_rgba[0, 0, mask_int[0, 0]] = 0.15
    target_rgba[0, 1, mask_int[0, 0]] = 0.65
    target_rgba[0, 2, mask_int[0, 0]] = 0.70
    target_rgba[0, 3, mask_int[0, 0]] = 1.0

    # Membrane RGBA
    target_rgba[0, 0, mask_bnd[0, 0]] = 0.85
    target_rgba[0, 1, mask_bnd[0, 0]] = 0.45
    target_rgba[0, 2, mask_bnd[0, 0]] = 0.20
    target_rgba[0, 3, mask_bnd[0, 0]] = 1.0

    # Environment is already 0.0 across all 4 channels

    return target_rgba, mask_int, mask_bnd, mask_env, mask_interface
