"""
3D U-Net baseline (b5 / b5-large) for slice interpolation.

Paper usage:
  - b5       = UNet3D(in_ch=2, out_ch=1, base=24), ~3.15 M parameters
  - b5-large = UNet3D(in_ch=2, out_ch=1, base=64), ~22.4 M parameters
               (capacity-matched to the 24.5 M MAPS generator)
  Appears in Table 1 (BB main comparison), Table 3 (LBM permeability),
  and Table 4 (compute cost).

Input:  (B, 2, D, H, W)  -- channel 0 = sparse volume (known slices only),
                            channel 1 = mask marking the known slices.
Output: (B, 1, D, H, W)  -- sigmoid-bounded reconstruction.

Training protocol ("fair single-target", b5): each sample is a 32^3 cube in
which ONLY the slices at z = t + o for o in OFFSETS_IN6 = [-15,-9,-3,3,9,15]
(relative to a single odd target t) are non-zero. This matches the input
information the 2D MAPS generator receives (closest known slice at t+-3),
avoiding the "leaky" variant in which the 3D model could read z = t+-1
directly.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import OFFSETS_IN6  # noqa: E402


# ════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════

class ConvBlock3D(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class UNet3D(nn.Module):
    """4-level 3D U-Net with sigmoid output (symmetric to the 2D UNetG)."""

    def __init__(self, in_ch=2, out_ch=1, base=24):
        super().__init__()
        b = base
        self.e1 = ConvBlock3D(in_ch, b)
        self.e2 = ConvBlock3D(b, b * 2)
        self.e3 = ConvBlock3D(b * 2, b * 4)
        self.e4 = ConvBlock3D(b * 4, b * 8)
        self.pool = nn.MaxPool3d(2)

        self.u3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.d3 = ConvBlock3D(b * 8, b * 4)
        self.u2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.d2 = ConvBlock3D(b * 4, b * 2)
        self.u1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.d1 = ConvBlock3D(b * 2, b)
        self.out = nn.Conv3d(b, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        d3 = self.d3(torch.cat([self.u3(e4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


# ════════════════════════════════════════════════════════════
# Dataset (fair single-target protocol)
# ════════════════════════════════════════════════════════════

class Sparse3DDatasetFair(Dataset):
    """
    Random 3D sub-cubes from the training slab -- SINGLE-TARGET PER CUBE.

    Each sample: pick a single odd target z=t inside the cube. Sparse input
    contains GT ONLY at z = t + o for o in OFFSETS_IN6. All other z zeroed.
    Mask channel marks the 6 known positions.

    The TARGET tensor is the full cube but the LOSS mask is the single target
    slice z=t (only z=t is what the model is asked to predict).

    Notes:
      - patch_size must be >= 2*15 + 1 = 31. Default 32 tightly wraps a single
        target's required context (z = t-15 .. t+15).
      - Worker-aware seeding via `worker_init_fn`.
    """

    def __init__(self, vol, slab_range, patch_size=32, n_samples=800,
                 offsets=OFFSETS_IN6):
        self.vol = vol
        self.lo, self.hi = slab_range
        self.ps = patch_size
        self.n = n_samples
        self.offsets = offsets
        self.k_max = max(abs(o) for o in offsets)
        D, H, W = vol.shape
        self.H, self.W = H, W
        self.valid_z_lo = self.lo
        self.valid_z_hi = self.hi - patch_size
        assert self.valid_z_hi > self.valid_z_lo, 'slab too small for patch_size'
        assert patch_size >= 2 * self.k_max + 1, \
            f'patch_size={patch_size} too small for k_max={self.k_max}'

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Pick a cube origin z0 such that an odd target fits inside.
        z0_min = self.valid_z_lo
        z0_max = self.valid_z_hi - 1
        z0 = random.randint(z0_min, z0_max)
        if (z0 + self.k_max) % 2 == 0:
            if z0 < z0_max:
                z0 += 1
            elif z0 > z0_min:
                z0 -= 1
            else:
                return self.__getitem__((idx + 1) % self.n)

        # Valid local target indices: zi in [k_max, ps-k_max) with odd global z
        valid_targets = [zi for zi in range(self.k_max, self.ps - self.k_max)
                         if (z0 + zi) % 2 == 1]
        if not valid_targets:
            return self.__getitem__((idx + 1) % self.n)
        t_local = random.choice(valid_targets)

        y0 = random.randint(0, self.H - self.ps)
        x0 = random.randint(0, self.W - self.ps)
        cube = self.vol[z0:z0 + self.ps, y0:y0 + self.ps,
                        x0:x0 + self.ps].astype(np.float32)
        cube = np.clip(cube, 0.0, 1.0)

        # Single-target sparse input
        sparse = np.zeros_like(cube)
        mask = np.zeros_like(cube)
        for o in self.offsets:
            kz = t_local + o
            if 0 <= kz < self.ps:
                sparse[kz] = cube[kz]
                mask[kz] = 1.0

        # Loss mask: only the single target slice
        loss_mask = np.zeros_like(cube)
        loss_mask[t_local] = 1.0

        x = np.stack([sparse, mask], axis=0)             # (2, D, H, W)
        y = cube[np.newaxis]                             # (1, D, H, W)
        lm = loss_mask[np.newaxis]                       # (1, D, H, W)
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(lm).float(),
                z0)


def worker_init_fn(worker_id):
    """Per-worker seeding for reproducibility."""
    base = torch.initial_seed() % 2 ** 31
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
