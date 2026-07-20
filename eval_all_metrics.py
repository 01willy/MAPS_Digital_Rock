#!/usr/bin/env python3
"""
Unified Evaluation Script for Sparse Micro-CT Slice Interpolation.

Single entry point for slice-level metrics; used for all checkpoint and
baseline comparisons so metric definitions are consistent.

Metrics computed:
  1. SSIM (pytorch_msssim, 11x11 Gaussian, data_range=1.0)
  2. PSNR (dB)
  3. |dphi|  — porosity error (Otsu-thresholded)
  4. |dSA|   — surface area error (boundary counting on Otsu binary)
  5. S2 MSE  — two-point correlation function error
  6. |dEuler| — Euler characteristic error
  7. Lineal-path MSE, connected-porosity error

Usage:
  python eval_all_metrics.py --checkpoint path/to/ckpt.pt \\
                             --data path/to/volume.bin \\
                             --shape 1000 1000 1000 \\
                             --axis z --in_ch 6 \\
                             --output results.json
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as pytorch_msssim_fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maps.models import UNetG
from maps.data import load_volume, compute_splits, SliceInterpDataset
from maps.metrics import compute_all_morphological_metrics


def evaluate_checkpoint(
    ckpt_path: str,
    vol: np.ndarray,
    split_range: tuple,
    axis: str = 'z',
    in_ch: int = 6,
    offsets: Optional[List[int]] = None,
    device: str = 'cuda:0',
    batch_size: int = 1,
    s2_max_lag: int = 32,
    use_ema: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a checkpoint on a given split.

    Args:
        ckpt_path: path to checkpoint (.pt file)
        vol: normalized volume [0, 1]
        split_range: (lo, hi) z-range for evaluation
        axis: interpolation axis
        in_ch: input channels
        offsets: offset list (for in_ch=6)
        device: cuda device
        batch_size: eval batch size
        s2_max_lag: max lag for S2 computation
        use_ema: use EMA weights

    Returns:
        dict with all metrics
    """
    dev = torch.device(device)

    # Load checkpoint first, then build the model with its recorded width
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    base_ch = 80
    if isinstance(ckpt, dict) and 'config' in ckpt and isinstance(ckpt['config'], dict):
        base_ch = ckpt['config'].get('base_ch', 80)
    G = UNetG(in_ch=in_ch, base=base_ch).to(dev)

    if use_ema and isinstance(ckpt, dict) and 'ema_state_dict' in ckpt:
        shadow = ckpt['ema_state_dict']['shadow']
        G.load_state_dict(shadow, strict=True)
        print(f"[EVAL] Loaded EMA weights from {ckpt_path}")
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        G.load_state_dict(ckpt['model_state_dict'], strict=True)
        print(f"[EVAL] Loaded model weights from {ckpt_path}")
    else:
        # Direct state dict
        G.load_state_dict(ckpt, strict=True)
        print(f"[EVAL] Loaded direct state dict from {ckpt_path}")

    G.eval()

    # Create dataset
    ds = SliceInterpDataset(
        vol, split_range, axis=axis, in_ch=in_ch,
        offsets=offsets, patch_size=9999,  # full slice (no crop)
        train=False, odd_only=True, augment=False)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

    # Accumulate metrics
    ssim_vals = []
    psnr_vals = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)

            pred = G(x)

            # SSIM (unified: pytorch_msssim)
            ssim_val = float(pytorch_msssim_fn(pred, y, data_range=1.0,
                                               size_average=True))
            ssim_vals.append(ssim_val)

            # PSNR
            mse = F.mse_loss(pred, y).item()
            psnr = -10 * math.log10(mse + 1e-10)
            psnr_vals.append(psnr)

            # Store for morphological metrics
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Morphological metrics (on CPU, full resolution)
    morph = compute_all_morphological_metrics(
        all_preds, all_targets, max_lag=s2_max_lag)

    results = {
        'ssim_mean': float(np.mean(ssim_vals)),
        'ssim_std': float(np.std(ssim_vals)),
        'psnr_mean': float(np.mean(psnr_vals)),
        'psnr_std': float(np.std(psnr_vals)),
        'n_samples': len(ssim_vals),
        **morph,
        # Per-sample SSIM for bootstrap CI
        'ssim_per_sample': ssim_vals,
    }

    return results


def evaluate_baseline_linear(
    vol: np.ndarray,
    split_range: tuple,
    axis: str = 'z',
    k: int = 1,
    s2_max_lag: int = 32,
) -> Dict[str, float]:
    """
    Evaluate the linear interpolation baseline (average of z-k and z+k).
    """
    if axis == 'z':
        view = vol
    elif axis == 'x':
        view = np.transpose(vol, (2, 0, 1))
    elif axis == 'y':
        view = np.transpose(vol, (1, 0, 2))

    lo, hi = split_range

    ssim_vals = []
    psnr_vals = []
    all_preds = []
    all_targets = []

    for z in range(lo + k, hi - k):
        if z % 2 == 0:  # odd_only targets
            continue

        left = view[z - k].astype(np.float32)
        right = view[z + k].astype(np.float32)
        gt = view[z].astype(np.float32)

        pred = 0.5 * left + 0.5 * right

        # To tensor (B=1, C=1, H, W)
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

        ssim_val = float(pytorch_msssim_fn(pred_t, gt_t, data_range=1.0))
        ssim_vals.append(ssim_val)

        mse = F.mse_loss(pred_t, gt_t).item()
        psnr = -10 * math.log10(mse + 1e-10)
        psnr_vals.append(psnr)

        all_preds.append(pred_t)
        all_targets.append(gt_t)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    morph = compute_all_morphological_metrics(
        all_preds, all_targets, max_lag=s2_max_lag)

    return {
        'ssim_mean': float(np.mean(ssim_vals)),
        'ssim_std': float(np.std(ssim_vals)),
        'psnr_mean': float(np.mean(psnr_vals)),
        'psnr_std': float(np.std(psnr_vals)),
        'n_samples': len(ssim_vals),
        **morph,
        'ssim_per_sample': ssim_vals,
    }


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=[1000, 1000, 1000])
    parser.add_argument('--axis', default='z')
    parser.add_argument('--in_ch', type=int, default=6)
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='eval_results.json')
    parser.add_argument('--s2_max_lag', type=int, default=32)
    parser.add_argument('--no_ema', action='store_true')
    args = parser.parse_args()

    vol = load_volume(args.data, tuple(args.shape))
    splits = compute_splits(vol.shape[0])
    split_range = splits[args.split]

    print(f"\n[EVAL] Split={args.split}, range={split_range}")
    print(f"[EVAL] Checkpoint: {args.checkpoint}")
    print(f"[EVAL] Axis={args.axis}, in_ch={args.in_ch}\n")

    results = evaluate_checkpoint(
        args.checkpoint, vol, split_range,
        axis=args.axis, in_ch=args.in_ch,
        device=args.device,
        s2_max_lag=args.s2_max_lag,
        use_ema=not args.no_ema)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in results.items():
        if k == 'ssim_per_sample':
            continue
        print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print("=" * 60)

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n[SAVED] {args.output}")


if __name__ == '__main__':
    main()
