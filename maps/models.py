"""
Model architectures for sparse micro-CT slice interpolation.

The networks are deliberately standard (U-Net generator + PatchGAN
discriminator); the method's contributions lie in the problem formulation,
the morphology-preserving losses, the inference-time tri-axis aggregation,
and the deployment protocol rather than in the architecture.

UNetG: Canonical U-Net generator
  - Encoder: Conv-BN-LeakyReLU + MaxPool
  - Decoder: Conv-BN-LeakyReLU + Bilinear Upsample
  - Skip connections at each level
  - Output: Sigmoid [0, 1]

PatchD: PatchGAN discriminator with spectral normalization
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from typing import Optional


class UNetG(nn.Module):
    """
    U-Net Generator for 2D slice interpolation.

    Args:
        in_ch: input channels (2=pair, 6=multi-offset 2.5D)
        base: base channel count (80 for full model)
    """

    def __init__(self, in_ch: int = 6, base: int = 80):
        super().__init__()

        def cb(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, True),
            )

        b = base
        self.e1 = cb(in_ch, b)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), cb(b, b * 2))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), cb(b * 2, b * 4))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), cb(b * 4, b * 8))
        self.b = nn.Sequential(
            nn.MaxPool2d(2),
            cb(b * 8, b * 16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.d4 = nn.Sequential(
            cb(b * 16 + b * 8, b * 8),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.d3 = nn.Sequential(
            cb(b * 8 + b * 4, b * 4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.d2 = nn.Sequential(
            cb(b * 4 + b * 2, b * 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.d1 = nn.Sequential(
            cb(b * 2 + b, b),
            nn.Conv2d(b, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.b(e4)
        d4 = self.d4(torch.cat([b, e4], 1))
        d3 = self.d3(torch.cat([d4, e3], 1))
        d2 = self.d2(torch.cat([d3, e2], 1))
        return self.d1(torch.cat([d2, e1], 1))


class PatchD(nn.Module):
    """PatchGAN Discriminator with Spectral Normalization."""

    def __init__(self, in_ch: int = 7, base: int = 64):
        super().__init__()
        b = base
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_ch, b, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(b, b * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(b * 2, b * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(b * 4, 1, 4, 1, 1)),
        )

    def forward(self, x_cond: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_cond, y], dim=1))


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(), alpha=1 - self.decay)
            else:
                self.shadow[name].copy_(param)

    @torch.no_grad()
    def store(self):
        self.backup = {k: v.clone() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def apply(self):
        self.model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self):
        if self.backup:
            self.model.load_state_dict(self.backup, strict=True)

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_models(cfg: dict, device: torch.device):
    """Create G, D, EMA."""
    in_ch = cfg.get('in_ch', 6)
    base = cfg.get('base_ch', 80)

    G = UNetG(in_ch, base).to(device)
    D = PatchD(in_ch + 1, base_d := cfg.get('base_ch_d', 64)).to(device)
    ema = EMA(G, cfg.get('ema_decay', 0.999))

    n_params = count_parameters(G)
    print(f"[MODEL] UNetG: in_ch={in_ch}, base={base}, params={n_params:,}")
    print(f"[MODEL] PatchD: in_ch={in_ch + 1}, base={base_d}")

    return G, D, ema
