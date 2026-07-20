"""
Loss functions for sparse micro-CT slice interpolation.

Conventions:
  1. SSIM: the same pytorch_msssim implementation is used for training
     and evaluation
  2. Morphology losses: soft-Otsu binarization (differentiable)
  3. All losses use the same [0, 1] data range
  4. GAN losses: BCE with label smoothing, or hinge
  5. Soft = differentiable (training), hard = thresholded (evaluation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim as pytorch_msssim_fn
from typing import Tuple, Dict, Optional

try:
    from .metrics import (
        porosity_soft, surface_area_soft, s2_loss, lineal_path_loss,
    )
except ImportError:
    from metrics import (
        porosity_soft, surface_area_soft, s2_loss, lineal_path_loss,
    )


# ════════════════════════════════════════════════════════════
# Reconstruction Losses
# ════════════════════════════════════════════════════════════

def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              data_range: float = 1.0) -> torch.Tensor:
    """
    SSIM loss using pytorch_msssim (11x11 Gaussian window).
    The same implementation is used for training and evaluation.

    Args:
        pred: (B, 1, H, W) in [0, 1]
        target: (B, 1, H, W) in [0, 1]
        data_range: max value range (1.0 for normalized images)

    Returns:
        scalar loss = 1 - SSIM
    """
    ssim_val = pytorch_msssim_fn(pred, target, data_range=data_range,
                                  size_average=True)
    return 1.0 - ssim_val


def ssim_value(pred: torch.Tensor, target: torch.Tensor,
               data_range: float = 1.0) -> float:
    """Compute SSIM value (for evaluation). Returns float."""
    with torch.no_grad():
        val = pytorch_msssim_fn(pred, target, data_range=data_range,
                                size_average=True)
    return float(val)


def grad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Gradient consistency loss (Sobel filter).
    Penalizes differences in edge structure.

    Args:
        pred, target: (B, 1, H, W)

    Returns:
        scalar loss
    """
    def _sobel(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                          dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                          dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)

    gx_p, gy_p = _sobel(pred)
    gx_t, gy_t = _sobel(target)

    return (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()


# ════════════════════════════════════════════════════════════
# Morphology-Preserving Losses
# ════════════════════════════════════════════════════════════

def porosity_loss(pred: torch.Tensor, target: torch.Tensor,
                  temperature: float = 10.0) -> Tuple[torch.Tensor, float]:
    """
    Differentiable porosity matching loss.
    Uses soft-Otsu binarization.

    Returns:
        (loss, dphi_value)
    """
    phi_p = porosity_soft(pred, temperature)
    phi_t = porosity_soft(target, temperature)
    dphi = (phi_p - phi_t).abs()
    return dphi.mean(), float(dphi.mean().detach())


def surface_area_loss(pred: torch.Tensor, target: torch.Tensor,
                      temperature: float = 10.0) -> Tuple[torch.Tensor, float]:
    """
    Differentiable surface area matching loss.
    Uses soft-Otsu boundary counting.

    Returns:
        (loss, dsa_value)
    """
    sa_p = surface_area_soft(pred, temperature)
    sa_t = surface_area_soft(target, temperature)
    dsa = (sa_p - sa_t).abs()
    return dsa.mean(), float(dsa.mean().detach())


# ════════════════════════════════════════════════════════════
# GAN Losses
# ════════════════════════════════════════════════════════════

def gan_loss_D(D: nn.Module, x_cond: torch.Tensor,
               y_real: torch.Tensor, y_fake: torch.Tensor,
               mode: str = "hinge") -> torch.Tensor:
    """
    Discriminator loss (BCE with label smoothing or Hinge).

    Args:
        D: discriminator network
        x_cond: condition input (B, C, H, W)
        y_real: real target (B, 1, H, W)
        y_fake: generated output (B, 1, H, W), detached
        mode: "hinge" or "bce"
    """
    real_out = D(x_cond, y_real)
    fake_out = D(x_cond, y_fake)

    if mode == "hinge":
        return F.relu(1 - real_out).mean() + F.relu(1 + fake_out).mean()
    else:  # bce
        real_loss = F.binary_cross_entropy_with_logits(
            real_out, torch.full_like(real_out, 0.9))
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_out, torch.full_like(fake_out, 0.1))
        return real_loss + fake_loss


def gan_loss_G(D: nn.Module, x_cond: torch.Tensor,
               y_fake: torch.Tensor, mode: str = "hinge") -> torch.Tensor:
    """Generator loss."""
    fake_out = D(x_cond, y_fake)
    if mode == "hinge":
        return (-fake_out).mean()
    else:
        return F.binary_cross_entropy_with_logits(
            fake_out, torch.full_like(fake_out, 1.0))


# ════════════════════════════════════════════════════════════
# Lambda Scheduling
# ════════════════════════════════════════════════════════════

def lambda_schedule(epoch: int, total_epochs: int, base: float,
                    warmup_epochs: int = 10) -> float:
    """
    GAN loss weight scheduling: warmup → linear decay.

    Args:
        epoch: current epoch (1-indexed)
        total_epochs: max epochs
        base: base lambda value
        warmup_epochs: warmup duration
    """
    if epoch < warmup_epochs:
        return base * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return base * (1.0 - 0.8 * progress)


# ════════════════════════════════════════════════════════════
# Combined Loss
# ════════════════════════════════════════════════════════════

class CombinedLoss:
    """
    Unified loss function combining:
    - Reconstruction: L1 + SSIM + Gradient
    - Morphology: Porosity + Surface Area + S2
    - GAN: Hinge or BCE

    All weights configurable via cfg dict.
    """

    def __init__(self, cfg: dict):
        # fallback defaults = paper configuration (Table S9)
        self.w_l1 = cfg.get('w_l1', 1.0)
        self.w_ssim = cfg.get('w_ssim', 0.349)
        self.w_grad = cfg.get('w_grad', 0.0)
        self.w_phi = cfg.get('w_phi', 0.174)
        self.w_sa = cfg.get('w_sa', 0.174)
        self.w_s2 = cfg.get('w_s2', 0.0177)  # two-point correlation loss
        self.w_lpath = cfg.get('w_lpath', 0.174)  # lineal path function loss
        self.s2_max_lag = cfg.get('s2_max_lag', 32)
        self.lpath_max_lag = cfg.get('lpath_max_lag', 16)
        self.soft_temperature = cfg.get('soft_temperature', 10.0)
        self.gan_mode = cfg.get('gan_mode', 'hinge')

    def compute_G_loss(
        self,
        G: nn.Module,
        D: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_gan: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute full generator loss.

        Returns:
            (total_loss, metrics_dict)
        """
        y_fake = G(x)

        # ── Reconstruction ──
        loss_l1 = F.l1_loss(y_fake, y)
        loss_ssim = ssim_loss(y_fake, y)
        loss_grad = grad_loss(y_fake, y)

        loss_recon = (self.w_l1 * loss_l1
                      + self.w_ssim * loss_ssim
                      + self.w_grad * loss_grad)

        # ── Morphology ──
        loss_phi, dphi_val = porosity_loss(y_fake, y, self.soft_temperature)
        loss_sa, dsa_val = surface_area_loss(y_fake, y, self.soft_temperature)

        loss_morph = self.w_phi * loss_phi + self.w_sa * loss_sa

        # S2 loss (optional, can be expensive)
        if self.w_s2 > 0:
            loss_s2_val = s2_loss(y_fake, y, self.s2_max_lag, self.soft_temperature)
            loss_morph = loss_morph + self.w_s2 * loss_s2_val
            s2_val = float(loss_s2_val.detach())
        else:
            s2_val = 0.0

        # Lineal path function loss (optional, sensitive to connectivity)
        if self.w_lpath > 0:
            loss_lpath_val = lineal_path_loss(y_fake, y, self.lpath_max_lag,
                                              self.soft_temperature)
            loss_morph = loss_morph + self.w_lpath * loss_lpath_val
            lpath_val = float(loss_lpath_val.detach())
        else:
            lpath_val = 0.0

        # ── GAN ──
        loss_gan = gan_loss_G(D, x, y_fake, self.gan_mode)

        # ── Total ──
        total = loss_recon + loss_morph + lambda_gan * loss_gan

        metrics = {
            'G_total': float(total.detach()),
            'G_recon': float(loss_recon.detach()),
            'G_l1': float(loss_l1.detach()),
            'G_ssim': float(loss_ssim.detach()),
            'G_grad': float(loss_grad.detach()),
            'G_phi': float(loss_phi.detach()),
            'G_sa': float(loss_sa.detach()),
            'G_s2': s2_val,
            'G_lpath': lpath_val,
            'G_gan': float(loss_gan.detach()),
            'dphi': dphi_val,
            'dsa': dsa_val,
        }

        return total, metrics

    def compute_D_loss(
        self,
        D: nn.Module,
        x: torch.Tensor,
        y_real: torch.Tensor,
        y_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute discriminator loss."""
        d_loss = gan_loss_D(D, x, y_real, y_fake, self.gan_mode)
        return d_loss, {'D_loss': float(d_loss.detach())}
