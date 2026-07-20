"""
Data Pipeline for Sparse Micro-CT Slice Interpolation.

Supports:
  - in_ch = 2 (adjacent pair), 4 (adjacent + far pair), 6 (multi-offset 2.5D)
  - Multi-axis: z, x, y
  - Global normalization: div255 (unified)
  - Leakage check: ensures no input slice overlaps with another split's targets
"""

import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════

# Default 2.5D multi-offset pattern (6 channels)
OFFSETS_IN6 = [-15, -9, -3, 3, 9, 15]


# ════════════════════════════════════════════════════════════
# Volume Loading
# ════════════════════════════════════════════════════════════

def load_volume(path: str, shape: Tuple[int, int, int],
                dtype=None) -> np.ndarray:
    """
    Load 3D volume from binary or TIFF file.

    Returns:
        float32 array in [0, 1] range (div255 normalized)
    """
    p = Path(path)

    if p.suffix in ('.tif', '.tiff'):
        import tifffile
        vol = tifffile.imread(str(p))
        print(f"[LOAD] {p.name}: TIFF, dtype={vol.dtype}, shape={vol.shape}")
    else:
        nvox = int(np.prod(shape))
        size = p.stat().st_size

        if dtype is None:
            if size == nvox:
                dtype = np.uint8
            elif size == nvox * 4:
                dtype = np.float32
            else:
                dtype = np.float32

        use_memmap = nvox > 256 ** 3
        if use_memmap:
            vol = np.memmap(p, dtype=dtype, mode='r', shape=shape)
        else:
            vol = np.fromfile(p, dtype=dtype, count=nvox).reshape(shape)

        print(f"[LOAD] {p.name}: BIN, dtype={dtype.__name__}, "
              f"shape={shape}, memmap={use_memmap}")

    # ── Normalize to [0, 1] ──
    vol = vol.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())

    if 1.5 < vmax < 200 and float(vmax).is_integer():
        # Small integer max: likely a multi-label segmentation (e.g. {0,1,2}) or a
        # mis-scaled volume, NOT a 0/255 binary. div255 would corrupt it. MAPS
        # expects a binary 0=pore/1=solid volume; convert with scripts/prepare_data.py.
        print(f"  [WARN] volume max={vmax:.0f} looks like a multi-label or "
              f"mis-scaled segmentation, not a 0/255 binary. Expected a binary "
              f"0=pore/1=solid volume; run scripts/prepare_data.py first. "
              f"Falling back to div-by-max normalization.")
        vol = vol / vmax
    elif vmax > 1.5:  # 0/255 binary or grayscale uint8
        vol = vol / 255.0
        print(f"  div255 norm: [{vmin:.1f}, {vmax:.1f}] -> "
              f"[{vol.min():.4f}, {vol.max():.4f}]")
    elif vmax - vmin > 1e-6:  # already roughly [0, 1] but may need clipping
        vol = np.clip(vol, 0.0, 1.0)
        print(f"  clip to [0,1]: [{vmin:.4f}, {vmax:.4f}]")
    else:
        print(f"  [WARN] constant volume? range=[{vmin}, {vmax}]")

    return vol


# ════════════════════════════════════════════════════════════
# Split Management
# ════════════════════════════════════════════════════════════

def compute_splits(Z: int, train_frac: float = 0.7,
                   val_frac: float = 0.15) -> Dict[str, Tuple[int, int]]:
    """
    Compute z-range splits for train/val/test.

    For Z=1000 (defaults): train [0, 700), val [700, 850), test [850, 1000).

    Returns:
        {'train': (lo, hi), 'val': (lo, hi), 'test': (lo, hi)}
    """
    z_train_end = int(Z * train_frac)
    z_val_end = int(Z * (train_frac + val_frac))
    return {
        'train': (0, z_train_end),
        'val': (z_train_end, z_val_end),
        'test': (z_val_end, Z),
    }


# ════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════

class SliceInterpDataset(Dataset):
    """
    Unified dataset for slice interpolation.

    Supports in_ch = 2, 4, or 6:
      - in_ch=2: [z-k, z+k]
      - in_ch=4: [z-2k, z-k, z+k, z+2k]
      - in_ch=6: [z+o for o in OFFSETS_IN6]  (multi-offset 2.5D)

    Multi-axis: volume is transposed so the interpolation axis is always dim 0.
    """

    def __init__(
        self,
        vol: np.ndarray,
        slab_range: Tuple[int, int],
        axis: str = 'z',
        in_ch: int = 6,
        k: int = 1,
        offsets: Optional[List[int]] = None,
        patch_size: int = 256,
        train: bool = True,
        odd_only: bool = True,
        augment: bool = True,
    ):
        """
        Args:
            vol: normalized volume (Z, Y, X) in [0, 1]
            slab_range: (lo, hi) z-range for this split
            axis: interpolation axis ('z', 'x', 'y')
            in_ch: input channels (2, 4, or 6)
            k: base offset for in_ch=2,4 mode
            offsets: explicit offsets for in_ch=6 (default: OFFSETS_IN6)
            patch_size: spatial crop size
            train: training mode (random crop + augmentation)
            odd_only: only odd target indices (even slices are the acquired
                      reference planes in the k=1 sparse scenario)
            augment: apply flip/rotation augmentation
        """
        assert in_ch >= 2, f"in_ch must be >= 2, got {in_ch}"
        assert axis in ('z', 'x', 'y'), f"axis must be z, x, or y, got {axis}"

        self.in_ch = in_ch
        self.k = k
        self.patch_size = patch_size
        self.train = train
        self.augment = augment and train
        self.axis = axis
        self.odd_only = odd_only
        self.slab_lo = slab_range[0]

        # Determine offsets
        if offsets is not None:
            # Explicit offsets provided — use them regardless of in_ch
            self.offsets = offsets
            assert len(offsets) == in_ch, \
                f"len(offsets)={len(offsets)} must match in_ch={in_ch}"
            self.k_max = max(abs(o) for o in self.offsets)
        elif in_ch == 6:
            self.offsets = OFFSETS_IN6
            self.k_max = max(abs(o) for o in self.offsets)
        elif in_ch == 4:
            self.offsets = [-2 * k, -k, k, 2 * k]
            self.k_max = 2 * k
        else:  # in_ch == 2
            self.offsets = [-k, k]
            self.k_max = k

        # Transpose volume so interpolation axis is always dim 0.
        # np.ascontiguousarray ensures contiguous memory layout after
        # transposition, avoiding random I/O on memmap for x/y axes.
        slab_vol = vol[slab_range[0]:slab_range[1], :, :]
        if axis == 'z':
            self.vol_view = slab_vol
        elif axis == 'x':
            self.vol_view = np.ascontiguousarray(
                np.transpose(slab_vol, (2, 0, 1)))
        elif axis == 'y':
            self.vol_view = np.ascontiguousarray(
                np.transpose(slab_vol, (1, 0, 2)))

        self.depth, self.H, self.W = self.vol_view.shape

        # Compute valid target indices (relative to this view)
        self.targets = []
        for z in range(self.k_max, self.depth - self.k_max):
            if odd_only and z % 2 == 0:
                continue
            self.targets.append(z)

        if len(self.targets) == 0:
            raise RuntimeError(
                f"No valid targets: slab={slab_range}, axis={axis}, "
                f"k_max={self.k_max}, depth={self.depth}")

        print(f"[DATA] axis={axis} in_ch={in_ch} train={train} "
              f"targets={len(self.targets)} "
              f"[{self.targets[0]}..{self.targets[-1]}] "
              f"vol_view={self.vol_view.shape}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.targets[idx]

        # Build input channels from offsets
        channels = []
        for o in self.offsets:
            s = self.vol_view[z + o]
            channels.append(s)

        target = self.vol_view[z]

        # Stack: (in_ch, H, W) and (1, H, W)
        inp = np.stack(channels, axis=0).astype(np.float32)
        tgt = target[np.newaxis, ...].astype(np.float32)

        # Crop
        inp, tgt = self._crop(inp, tgt)

        # To tensor
        inp = torch.from_numpy(inp.copy())
        tgt = torch.from_numpy(tgt.copy())

        # Augmentation
        if self.augment:
            inp, tgt = self._augment(inp, tgt)

        return inp, tgt

    def _crop(self, x: np.ndarray, y: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        s = self.patch_size
        if s >= self.H and s >= self.W:
            # Pad to patch_size if needed
            ph = max(0, s - self.H)
            pw = max(0, s - self.W)
            if ph > 0 or pw > 0:
                x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode='reflect')
                y = np.pad(y, ((0, 0), (0, ph), (0, pw)), mode='reflect')
            return x, y

        # Clamp crop origin to [0, dim - crop_size]
        crop_h = min(s, self.H)
        crop_w = min(s, self.W)

        if self.train:
            y0 = random.randint(0, max(0, self.H - crop_h))
            x0 = random.randint(0, max(0, self.W - crop_w))
        else:
            y0 = max(0, (self.H - crop_h) // 2)
            x0 = max(0, (self.W - crop_w) // 2)

        x = x[:, y0:y0 + crop_h, x0:x0 + crop_w]
        y = y[:, y0:y0 + crop_h, x0:x0 + crop_w]

        # Pad if spatial dims < patch_size
        ph = max(0, s - crop_h)
        pw = max(0, s - crop_w)
        if ph > 0 or pw > 0:
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode='reflect')
            y = np.pad(y, ((0, 0), (0, ph), (0, pw)), mode='reflect')

        return x, y

    def _augment(self, x: torch.Tensor, y: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Horizontal flip
        if random.random() < 0.5:
            x = x.flip(-1)
            y = y.flip(-1)
        # Vertical flip
        if random.random() < 0.5:
            x = x.flip(-2)
            y = y.flip(-2)
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k, [-2, -1])
            y = torch.rot90(y, k, [-2, -1])
        return x.contiguous(), y.contiguous()


# ════════════════════════════════════════════════════════════
# Balanced Multi-axis Dataset (Stage 2)
# ════════════════════════════════════════════════════════════

class BalancedMultiAxisDataset(Dataset):
    """
    Balanced multi-axis dataset: ensures each axis contributes equally.

    Instead of ConcatDataset (which over-represents axes with more targets),
    this dataset samples uniformly from each axis. Per epoch, each axis
    contributes min_len samples, so total = n_axes * min_len.

    DDP compatible (works with DistributedSampler).
    """

    def __init__(
        self,
        vol: np.ndarray,
        slab_range: Tuple[int, int],
        axes: List[str] = ['z', 'x', 'y'],
        in_ch: int = 6,
        **kwargs
    ):
        self.datasets = []
        for axis in axes:
            ds = SliceInterpDataset(vol, slab_range, axis=axis, in_ch=in_ch,
                                    **kwargs)
            self.datasets.append(ds)

        self.n_axes = len(axes)
        self.min_len = min(len(ds) for ds in self.datasets)
        self._total = self.min_len * self.n_axes

        axis_counts = {ax: len(ds) for ax, ds in zip(axes, self.datasets)}
        print(f"[DATA] BalancedMultiAxis: {axis_counts} -> "
              f"{self.min_len}/axis x {self.n_axes} = {self._total} samples")

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        axis_idx = idx % self.n_axes
        # Random sample from this axis (not sequential, for diversity)
        sample_idx = random.randint(0, len(self.datasets[axis_idx]) - 1)
        return self.datasets[axis_idx][sample_idx]


# ════════════════════════════════════════════════════════════
# DataLoader Factory
# ════════════════════════════════════════════════════════════

def create_dataloaders(
    vol: np.ndarray,
    splits: Dict[str, Tuple[int, int]],
    cfg: dict,
    axis: str = 'z',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        vol: normalized volume [0, 1]
        splits: {'train': (lo, hi), 'val': (lo, hi), 'test': (lo, hi)}
        cfg: config dict with in_ch, patch_size, batch_size, etc.
        axis: interpolation axis
    """
    in_ch = cfg.get('in_ch', 6)
    k = cfg.get('k', 1)
    offsets = cfg.get('offsets', None)
    patch_size = cfg.get('patch_size', 256)
    batch_size = cfg.get('batch_size', 4)
    num_workers = cfg.get('num_workers', 4)

    train_ds = SliceInterpDataset(
        vol, splits['train'], axis=axis, in_ch=in_ch, k=k,
        offsets=offsets, patch_size=patch_size, train=True)

    val_ds = SliceInterpDataset(
        vol, splits['val'], axis=axis, in_ch=in_ch, k=k,
        offsets=offsets, patch_size=patch_size, train=False)

    test_ds = SliceInterpDataset(
        vol, splits['test'], axis=axis, in_ch=in_ch, k=k,
        offsets=offsets, patch_size=patch_size, train=False)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ════════════════════════════════════════════════════════════
# Leakage Check
# ════════════════════════════════════════════════════════════

def check_leakage(vol_shape: Tuple[int, int, int],
                  splits: Dict[str, Tuple[int, int]],
                  in_ch: int = 6,
                  offsets: Optional[List[int]] = None) -> dict:
    """
    Verify that no input context slice falls in another split's target range.

    Returns:
        dict with leakage report
    """
    if offsets is not None:
        # Explicit offsets always take priority (any in_ch)
        offs = offsets
    elif in_ch == 6:
        offs = OFFSETS_IN6
    else:
        offs = list(range(-in_ch // 2, in_ch // 2 + 1))

    k_max = max(abs(o) for o in offs)

    violations = []
    for split_name, (lo, hi) in splits.items():
        targets = list(range(lo + k_max, hi - k_max))
        for z in targets:
            for o in offs:
                input_z = z + o
                # Check if input_z falls in a different split
                for other_name, (other_lo, other_hi) in splits.items():
                    if other_name == split_name:
                        continue
                    if other_lo <= input_z < other_hi:
                        violations.append({
                            'split': split_name,
                            'target_z': z,
                            'input_z': input_z,
                            'offset': o,
                            'leaked_into': other_name,
                        })

    return {
        'n_violations': len(violations),
        'clean': len(violations) == 0,
        'violations': violations[:10],  # first 10 only
    }
