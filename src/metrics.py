import numpy as np
import scipy.spatial
from scipy.special import digamma
import torch
import torch.nn.functional as F


def morphological_l2(state: torch.Tensor, target_rgba: torch.Tensor) -> float:
    """
    Computes steady-state morphological error (MSE) over visible channels 0:4 (RGBA)
    across all lattice sites.

    L2(t) = 1 / (4 * |Lambda|) * sum_{x in Lambda} || s_t^{(0:3)}(x) - y*(x) ||_2^2
    """
    state_rgba = state[:, :4]
    target_rgba_slice = target_rgba[:, :4].to(state.device)
    diff_sq = (state_rgba - target_rgba_slice) ** 2
    return float(diff_sq.mean().item())


def boundary_l2(
    state: torch.Tensor, target_rgba: torch.Tensor, mask_bnd: torch.Tensor
) -> float:
    """
    Computes morphological error (MSE) over visible channels 0:4 (RGBA)
    restricted to the bounding membrane Omega_bnd.
    """
    state_rgba = state[:, :4]
    target_rgba_slice = target_rgba[:, :4].to(state.device)
    diff_sq = (state_rgba - target_rgba_slice) ** 2

    if mask_bnd.dim() == 2:
        mask = mask_bnd.unsqueeze(0).unsqueeze(0)
    elif mask_bnd.dim() == 3:
        mask = mask_bnd.unsqueeze(0)
    else:
        mask = mask_bnd

    mask_f = mask.to(state.device).float()
    num_bnd_pixels = mask_f.sum().clamp(min=1.0)
    bnd_diff = (diff_sq * mask_f).sum()
    bnd_mse = bnd_diff / (4.0 * num_bnd_pixels * state.shape[0])
    return float(bnd_mse.item())


def recovery_ratio(l2_bnd_post_lesion: float, l2_bnd_healed: float) -> float:
    """
    Computes boundary recovery ratio rho_rec:
      rho_rec = 1 - (L2_bnd(T_grow + Delta_T_heal) / max(L2_bnd(T_grow^+), 1e-8))
    """
    denom = max(float(l2_bnd_post_lesion), 1e-8)
    return float(1.0 - (float(l2_bnd_healed) / denom))


def compute_transfer_entropy_ksg(
    s_int_traj: np.ndarray,
    s_bnd_traj: np.ndarray,
    k: int = 4,
    pca_dims: int = 2,
    seed: int = 42,
) -> float:
    """
    Estimates Transfer Entropy TE_{int -> bnd} = I(Y_{t+1}; X_t | Y_t)
    from interior hidden channel trajectory X to boundary hidden channel trajectory Y
    using the Kraskov-Stögbauer-Grassberger (KSG) estimator with Chebyshev (L_inf) metric.

    Args:
        s_int_traj: [T, 12] or [T, D] hidden channel trajectory in Omega_int
        s_bnd_traj: [T, 12] or [T, D] hidden channel trajectory in Omega_bnd
        k: Number of nearest neighbors (default: 4)
        pca_dims: Number of principal components for projection (default: 2)
        seed: Random seed for deterministic micro-jitter

    Returns:
        TE in nats (floored at 0.0)
    """
    if isinstance(s_int_traj, torch.Tensor):
        s_int_traj = s_int_traj.detach().cpu().numpy()
    if isinstance(s_bnd_traj, torch.Tensor):
        s_bnd_traj = s_bnd_traj.detach().cpu().numpy()

    s_int_traj = np.asarray(s_int_traj, dtype=np.float64)
    s_bnd_traj = np.asarray(s_bnd_traj, dtype=np.float64)

    T = len(s_int_traj)
    if T < k + 3:
        return 0.0

    # 1. Project each trajectory to top pca_dims principal components via SVD
    def _project_pca(series: np.ndarray, dims: int) -> np.ndarray:
        centered = series - np.mean(series, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        n_comp = min(dims, vt.shape[0])
        return centered @ vt[:n_comp].T

    x_pca = _project_pca(s_int_traj, pca_dims)
    y_pca = _project_pca(s_bnd_traj, pca_dims)

    # 2. Add tiny deterministic Gaussian jitter (1e-10) to break ties and prevent duplicate distances
    rng = np.random.RandomState(seed)
    x_pca += rng.normal(0.0, 1e-10, size=x_pca.shape)
    y_pca += rng.normal(0.0, 1e-10, size=y_pca.shape)

    # 3. Form time series pairs:
    # Source X_t: x_pca[0 : T-1]
    # Destination condition Y_t: y_pca[0 : T-1]
    # Destination target Y_{t+1}: y_pca[1 : T]
    N = T - 1
    u = y_pca[1:T]  # Y_{t+1}
    v = x_pca[0:N]  # X_t
    w = y_pca[0:N]  # Y_t

    # Joint space P = (U, V, W)
    p_joint = np.hstack([u, v, w])
    uw_sub = np.hstack([u, w])
    vw_sub = np.hstack([v, w])
    w_sub = w

    tree_p = scipy.spatial.cKDTree(p_joint)
    tree_uw = scipy.spatial.cKDTree(uw_sub)
    tree_vw = scipy.spatial.cKDTree(vw_sub)
    tree_w = scipy.spatial.cKDTree(w_sub)

    # Find Chebyshev distance to the k-th nearest neighbor in joint space
    # k+1 query returns distances to 0th (self), 1st, ..., kth neighbor
    dists, _ = tree_p.query(p_joint, k=k + 1, p=np.inf)
    eps = dists[:, -1]

    # In subspace trees, count points strictly within distance eps_i
    # query_ball_point with radius eps_i - 1e-15 includes self (distance 0)
    # hence len(...) equals (n_sub + 1)
    n_uw = np.array(
        [len(tree_uw.query_ball_point(uw_sub[i], r=eps[i] - 1e-15, p=np.inf)) for i in range(N)]
    )
    n_vw = np.array(
        [len(tree_vw.query_ball_point(vw_sub[i], r=eps[i] - 1e-15, p=np.inf)) for i in range(N)]
    )
    n_w = np.array(
        [len(tree_w.query_ball_point(w_sub[i], r=eps[i] - 1e-15, p=np.inf)) for i in range(N)]
    )

    # KSG CMI formula:
    # I(U; V | W) = psi(k) + < psi(n_w) - psi(n_uw) - psi(n_vw) >
    te = digamma(k) + np.mean(digamma(n_w) - digamma(n_uw) - digamma(n_vw))
    return float(max(0.0, te))
