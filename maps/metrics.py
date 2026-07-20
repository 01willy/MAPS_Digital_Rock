"""
Morphological / petrophysical metrics for digital rock evaluation.

All functions operate on torch tensors (B, 1, H, W) in [0, 1] range.
Training-time losses use differentiable (soft) versions.
Evaluation-time metrics use hard-thresholded (binary) versions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ════════════════════════════════════════════════════════════
# Otsu Threshold (non-differentiable — used as detached constant)
# ════════════════════════════════════════════════════════════

def otsu_threshold_batch(img: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """
    Compute Otsu threshold per sample in a batch.

    Args:
        img: (B, 1, H, W) in [0, 1]
        n_bins: number of histogram bins

    Returns:
        (B,) tensor of thresholds, detached (no gradient)
    """
    B = img.shape[0]
    thresholds = torch.zeros(B, device=img.device)

    # histc requires float32 (not supported in half precision / AMP)
    img_f32 = img.float()

    for b in range(B):
        flat = img_f32[b].reshape(-1)
        # Build histogram
        hist = torch.histc(flat, bins=n_bins, min=0.0, max=1.0)
        hist = hist / hist.sum()

        bin_centers = torch.linspace(0.0, 1.0, n_bins, device=img.device)

        # Cumulative sums
        w0 = torch.cumsum(hist, dim=0)
        w1 = 1.0 - w0
        mu0_num = torch.cumsum(hist * bin_centers, dim=0)

        # Avoid division by zero
        w0_safe = w0.clamp(min=1e-10)
        w1_safe = w1.clamp(min=1e-10)

        mu0 = mu0_num / w0_safe
        mu_total = (hist * bin_centers).sum()
        mu1 = (mu_total - mu0_num) / w1_safe

        # Inter-class variance
        sigma_between = w0 * w1 * (mu0 - mu1) ** 2

        # Best threshold
        best_idx = sigma_between.argmax()
        thresholds[b] = bin_centers[best_idx]

    return thresholds.detach()


# ════════════════════════════════════════════════════════════
# Soft Binarization (differentiable)
# ════════════════════════════════════════════════════════════

def soft_binarize(img: torch.Tensor, threshold: torch.Tensor,
                  temperature: float = 50.0) -> torch.Tensor:
    """
    Differentiable binarization using sigmoid.

    Args:
        img: (B, 1, H, W) in [0, 1]
        threshold: (B,) per-sample thresholds (detached)
        temperature: sharpness of sigmoid (higher = closer to hard threshold)

    Returns:
        (B, 1, H, W) soft-binarized image, values near 0 or 1
    """
    # Reshape threshold for broadcasting: (B,) -> (B, 1, 1, 1)
    tau = threshold.view(-1, 1, 1, 1)
    return torch.sigmoid(temperature * (img - tau))


# ════════════════════════════════════════════════════════════
# Porosity
# ════════════════════════════════════════════════════════════

def porosity_soft(img: torch.Tensor, temperature: float = 50.0) -> torch.Tensor:
    """
    Differentiable porosity via soft-Otsu binarization.
    Porosity = fraction of pore phase (dark voxels, below threshold).

    Args:
        img: (B, 1, H, W) in [0, 1], where 0=pore, 1=solid

    Returns:
        (B,) porosity per sample
    """
    tau = otsu_threshold_batch(img)
    soft_bin = soft_binarize(img, tau, temperature)  # 1=solid, 0=pore
    return 1.0 - soft_bin.mean(dim=[1, 2, 3])


def porosity_hard(img: torch.Tensor) -> torch.Tensor:
    """
    Hard (non-differentiable) porosity via Otsu threshold.

    Args:
        img: (B, 1, H, W) in [0, 1]

    Returns:
        (B,) porosity per sample
    """
    tau = otsu_threshold_batch(img)
    tau_4d = tau.view(-1, 1, 1, 1)
    binary = (img > tau_4d).float()  # 1=solid, 0=pore
    return 1.0 - binary.mean(dim=[1, 2, 3])


# ════════════════════════════════════════════════════════════
# Surface Area (2D perimeter proxy)
# ════════════════════════════════════════════════════════════

def surface_area_soft(img: torch.Tensor, temperature: float = 50.0) -> torch.Tensor:
    """
    Differentiable surface area proxy via boundary pixel counting
    on soft-binarized image.

    SA ∝ number of boundary pixels (pore-solid interfaces).

    Args:
        img: (B, 1, H, W) in [0, 1]

    Returns:
        (B,) normalized surface area per sample
    """
    tau = otsu_threshold_batch(img)
    soft_bin = soft_binarize(img, tau, temperature)

    # Count boundaries: |neighbor difference| on binarized image
    dy = (soft_bin[:, :, 1:, :] - soft_bin[:, :, :-1, :]).abs()
    dx = (soft_bin[:, :, :, 1:] - soft_bin[:, :, :, :-1]).abs()

    H, W = img.shape[2], img.shape[3]
    # Normalize by image area to make scale-invariant
    sa = dy.sum(dim=[1, 2, 3]) / (H * W) + dx.sum(dim=[1, 2, 3]) / (H * W)
    return sa


def surface_area_hard(img: torch.Tensor) -> torch.Tensor:
    """Hard (non-differentiable) surface area via Otsu + boundary counting."""
    tau = otsu_threshold_batch(img)
    tau_4d = tau.view(-1, 1, 1, 1)
    binary = (img > tau_4d).float()

    dy = (binary[:, :, 1:, :] - binary[:, :, :-1, :]).abs()
    dx = (binary[:, :, :, 1:] - binary[:, :, :, :-1]).abs()

    H, W = img.shape[2], img.shape[3]
    sa = dy.sum(dim=[1, 2, 3]) / (H * W) + dx.sum(dim=[1, 2, 3]) / (H * W)
    return sa


# ════════════════════════════════════════════════════════════
# Two-Point Correlation Function S2(r)
# ════════════════════════════════════════════════════════════

def two_point_correlation(img: torch.Tensor, max_lag: int = 32,
                          use_otsu: bool = True) -> torch.Tensor:
    """
    Compute radially averaged two-point correlation function S2(r).

    S2(r) = P(both points at distance r are in the same phase)
    For binary image I: S2(r) = <I(x) * I(x+r)>

    Computed via FFT for efficiency.

    Args:
        img: (B, 1, H, W) in [0, 1]
        max_lag: maximum lag distance in pixels
        use_otsu: if True, binarize with Otsu first

    Returns:
        (B, max_lag) S2 values for r = 0, 1, ..., max_lag-1
    """
    B, _, H, W = img.shape

    if use_otsu:
        tau = otsu_threshold_batch(img)
        tau_4d = tau.view(-1, 1, 1, 1)
        phase = (img > tau_4d).float()
    else:
        phase = img

    # Squeeze channel dim for FFT: (B, H, W)
    phase_2d = phase.squeeze(1)

    # Autocorrelation via FFT
    f = torch.fft.rfft2(phase_2d)
    power = f.real ** 2 + f.imag ** 2
    autocorr = torch.fft.irfft2(power, s=(H, W))

    # Normalize so S2(r) = autocorr(r) / (H*W).
    # For a binary image, S2(0) = <I(x)^2> = <I(x)> = volume fraction of phase 1.
    autocorr = autocorr / (H * W)

    # Radial average
    # Create distance map from origin (0, 0)
    cy, cx = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=torch.float32),
        torch.arange(W, device=img.device, dtype=torch.float32),
        indexing='ij'
    )
    # Wrap-around distances
    cy = torch.minimum(cy, H - cy)
    cx = torch.minimum(cx, W - cx)
    dist = torch.sqrt(cy ** 2 + cx ** 2)

    # Bin into integer radii
    S2 = torch.zeros(B, max_lag, device=img.device)
    for r in range(max_lag):
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if mask.sum() > 0:
            S2[:, r] = autocorr[:, mask].mean(dim=1)

    return S2


def two_point_correlation_error(pred: torch.Tensor, target: torch.Tensor,
                                max_lag: int = 32) -> torch.Tensor:
    """
    MSE between S2 curves of predicted and target images.

    Args:
        pred, target: (B, 1, H, W) in [0, 1]
        max_lag: maximum lag

    Returns:
        scalar MSE loss
    """
    S2_pred = two_point_correlation(pred, max_lag, use_otsu=True)
    S2_target = two_point_correlation(target, max_lag, use_otsu=True)
    return F.mse_loss(S2_pred, S2_target)


# ════════════════════════════════════════════════════════════
# Two-Point Correlation — Differentiable (soft) version for training
# ════════════════════════════════════════════════════════════

def two_point_correlation_soft(img: torch.Tensor, max_lag: int = 32,
                               temperature: float = 50.0) -> torch.Tensor:
    """
    Differentiable S2(r) using soft-Otsu binarization.
    Gradients flow through the soft binarization.
    """
    B, _, H, W = img.shape
    tau = otsu_threshold_batch(img)  # detached
    phase = soft_binarize(img, tau, temperature)  # differentiable
    phase_2d = phase.squeeze(1)

    # Autocorrelation via FFT (differentiable in PyTorch)
    f = torch.fft.rfft2(phase_2d)
    power = f.real ** 2 + f.imag ** 2
    autocorr = torch.fft.irfft2(power, s=(H, W))
    autocorr = autocorr / (H * W)

    # Radial average
    cy, cx = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=torch.float32),
        torch.arange(W, device=img.device, dtype=torch.float32),
        indexing='ij'
    )
    cy = torch.minimum(cy, H - cy)
    cx = torch.minimum(cx, W - cx)
    dist = torch.sqrt(cy ** 2 + cx ** 2)

    S2 = torch.zeros(B, max_lag, device=img.device)
    for r in range(max_lag):
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if mask.sum() > 0:
            S2[:, r] = autocorr[:, mask].mean(dim=1)

    return S2


def s2_loss(pred: torch.Tensor, target: torch.Tensor,
            max_lag: int = 32, temperature: float = 50.0) -> torch.Tensor:
    """
    Differentiable S2 loss for training.
    Computes MSE between S2 curves using soft binarization.
    """
    S2_pred = two_point_correlation_soft(pred, max_lag, temperature)
    S2_target = two_point_correlation_soft(target, max_lag, temperature)
    return F.mse_loss(S2_pred, S2_target)


# ════════════════════════════════════════════════════════════
# Euler Characteristic (2D)
# ════════════════════════════════════════════════════════════

def euler_number_2d(img: torch.Tensor, use_otsu: bool = True) -> torch.Tensor:
    """
    Compute 2D Euler number using quad-tree (2x2 pattern) counting.

    Euler number = #connected_components - #holes (for 8-connectivity)

    Uses the formula from Ohser & Mücklich (2000):
    χ = n1 - n2 + n3 - n4  (based on 2x2 pixel pattern counts)

    where n_k = number of 2x2 neighborhoods with exactly k foreground pixels,
    weighted by the connectivity rule.

    For 4-connectivity: χ = (Q1 - Q3 + 2*QD) / 4
    For 8-connectivity: χ = (Q1 - Q3 - 2*QD) / 4

    Q1 = patterns with 1 foreground pixel
    Q3 = patterns with 3 foreground pixels
    QD = diagonal patterns (checker-board 2x2)

    Args:
        img: (B, 1, H, W) in [0, 1]

    Returns:
        (B,) Euler number per sample
    """
    if use_otsu:
        tau = otsu_threshold_batch(img)
        tau_4d = tau.view(-1, 1, 1, 1)
        binary = (img > tau_4d).float()
    else:
        binary = (img > 0.5).float()

    b = binary.squeeze(1)  # (B, H, W)

    # Extract 2x2 neighborhoods
    tl = b[:, :-1, :-1]  # top-left
    tr = b[:, :-1, 1:]   # top-right
    bl = b[:, 1:, :-1]   # bottom-left
    br = b[:, 1:, 1:]    # bottom-right

    s = tl + tr + bl + br  # sum of 2x2 block

    # Q1: exactly 1 foreground pixel
    Q1 = (s == 1).float().sum(dim=[1, 2])

    # Q3: exactly 3 foreground pixels
    Q3 = (s == 3).float().sum(dim=[1, 2])

    # QD: diagonal patterns (exactly 2 pixels, diagonal)
    diag1 = (tl == 1) & (br == 1) & (tr == 0) & (bl == 0)
    diag2 = (tr == 1) & (bl == 1) & (tl == 0) & (br == 0)
    QD = (diag1 | diag2).float().sum(dim=[1, 2])

    # 8-connectivity Euler number
    euler = (Q1 - Q3 - 2 * QD) / 4.0

    return euler


# ════════════════════════════════════════════════════════════
# Lineal Path Function L(r) — differentiable
# ════════════════════════════════════════════════════════════

def lineal_path_function(img: torch.Tensor, max_lag: int = 32,
                         temperature: float = 50.0,
                         use_soft: bool = True) -> torch.Tensor:
    """
    Compute lineal path function L(r) for the pore phase.

    L(r) = probability that a line segment of length r lies entirely
    within the pore phase. More sensitive to connectivity than S2.

    Differentiable version: uses product of soft-binarized values along lines.
    L(r) = mean_x( prod_{i=0}^{r} (1 - sigma(T*(I(x+i) - tau))) )

    Computed in horizontal and vertical directions, then averaged.

    Args:
        img: (B, 1, H, W) in [0, 1]
        max_lag: maximum line length in pixels
        temperature: sigmoid sharpness for soft binarization
        use_soft: if True, differentiable; if False, hard threshold

    Returns:
        (B, max_lag) L(r) values for r = 1, 2, ..., max_lag
    """
    B, _, H, W = img.shape

    if use_soft:
        tau = otsu_threshold_batch(img)
        pore = 1.0 - soft_binarize(img, tau, temperature)  # 1=pore, 0=solid
    else:
        tau = otsu_threshold_batch(img)
        tau_4d = tau.view(-1, 1, 1, 1)
        pore = (img <= tau_4d).float()

    pore_2d = pore.squeeze(1)  # (B, H, W)

    L = torch.zeros(B, max_lag, device=img.device)

    for r in range(1, max_lag + 1):
        # Horizontal direction: product of pore values along rows
        if r <= W:
            # Sliding window product using cumulative product + division
            # For efficiency, compute running product
            h_products = pore_2d[:, :, :W - r + 1].clone()
            for i in range(1, r):
                h_products = h_products * pore_2d[:, :, i:W - r + 1 + i]
            L[:, r - 1] += h_products.mean(dim=[1, 2])

        # Vertical direction
        if r <= H:
            v_products = pore_2d[:, :H - r + 1, :].clone()
            for i in range(1, r):
                v_products = v_products * pore_2d[:, i:H - r + 1 + i, :]
            L[:, r - 1] += v_products.mean(dim=[1, 2])

        # Average of two directions
        L[:, r - 1] /= 2.0

    return L


def lineal_path_loss(pred: torch.Tensor, target: torch.Tensor,
                     max_lag: int = 16, temperature: float = 50.0
                     ) -> torch.Tensor:
    """
    Differentiable lineal path function loss.
    MSE between L(r) curves of predicted and target.
    """
    L_pred = lineal_path_function(pred, max_lag, temperature, use_soft=True)
    L_target = lineal_path_function(target, max_lag, temperature, use_soft=True)
    return F.mse_loss(L_pred, L_target)


# ════════════════════════════════════════════════════════════
# Connected Porosity Fraction
# ════════════════════════════════════════════════════════════

def connected_porosity_fraction(img: torch.Tensor) -> torch.Tensor:
    """
    Fraction of pore voxels that belong to the largest connected component.

    High value → well-connected pore network (good for flow).
    Low value → isolated pores (poor connectivity).

    NOT differentiable (uses scipy label). Evaluation only.

    Args:
        img: (B, 1, H, W) in [0, 1]

    Returns:
        (B,) connected porosity fraction per sample
    """
    from scipy import ndimage

    tau = otsu_threshold_batch(img)
    results = torch.zeros(img.shape[0], device=img.device)

    for b in range(img.shape[0]):
        binary = (img[b, 0].cpu().numpy() <= tau[b].cpu().item()).astype(np.int32)
        total_pore = binary.sum()
        if total_pore == 0:
            results[b] = 0.0
            continue

        labeled, n_features = ndimage.label(binary)
        if n_features == 0:
            results[b] = 0.0
            continue

        # Size of largest component
        component_sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))
        largest = max(component_sizes)
        results[b] = float(largest / total_pore)

    return results


def compute_all_morphological_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_lag: int = 32
) -> dict:
    """
    Compute all morphological metrics for evaluation.

    Args:
        pred, target: (B, 1, H, W) in [0, 1]

    Returns:
        dict with all metrics (scalars, averaged over batch)
    """
    B = pred.shape[0]

    # Porosity
    phi_pred = porosity_hard(pred)
    phi_target = porosity_hard(target)
    dphi = (phi_pred - phi_target).abs().mean().item()

    # Surface area
    sa_pred = surface_area_hard(pred)
    sa_target = surface_area_hard(target)
    dsa = (sa_pred - sa_target).abs().mean().item()

    # Two-point correlation
    S2_pred = two_point_correlation(pred, max_lag, use_otsu=True)
    S2_target = two_point_correlation(target, max_lag, use_otsu=True)
    s2_mse = F.mse_loss(S2_pred, S2_target).item()

    # Euler characteristic
    euler_pred = euler_number_2d(pred)
    euler_target = euler_number_2d(target)
    d_euler = (euler_pred - euler_target).abs().mean().item()

    # Lineal path function
    L_pred = lineal_path_function(pred, max_lag=min(max_lag, 16),
                                  use_soft=False)
    L_target = lineal_path_function(target, max_lag=min(max_lag, 16),
                                    use_soft=False)
    lpath_mse = F.mse_loss(L_pred, L_target).item()

    # Connected porosity fraction
    cp_pred = connected_porosity_fraction(pred)
    cp_target = connected_porosity_fraction(target)
    d_connected = (cp_pred - cp_target).abs().mean().item()

    return {
        'dphi': dphi,
        'phi_pred_mean': phi_pred.mean().item(),
        'phi_target_mean': phi_target.mean().item(),
        'dsa': dsa,
        'sa_pred_mean': sa_pred.mean().item(),
        'sa_target_mean': sa_target.mean().item(),
        's2_mse': s2_mse,
        'lpath_mse': lpath_mse,
        'd_euler': d_euler,
        'euler_pred_mean': euler_pred.mean().item(),
        'euler_target_mean': euler_target.mean().item(),
        'd_connected_porosity': d_connected,
        'connected_porosity_pred': cp_pred.mean().item(),
        'connected_porosity_target': cp_target.mean().item(),
    }
