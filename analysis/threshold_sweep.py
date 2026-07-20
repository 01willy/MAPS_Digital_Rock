#!/usr/bin/env python3
"""
Binarization-threshold robustness sweep (sweep data archived; only
Fig. S6 appears in the compiled supplement).

The MAPS output is grayscale in [0, 1] and is binarized at tau = 0.5 for
all morphology and permeability metrics. This driver reconstructs the
center benchmark cube of a domain with tri_mean aggregation and sweeps the
binarization threshold, recording the reconstructed porosity phi(tau) and
the morphology errors (dphi / dSA / dchi-per-Mpx / S2 MSE) at every tau,
plus a prediction-derived Otsu threshold row. The paper's claim: the
reconstructed porosity changes by less than 3% over tau in [0.45, 0.55]
across BB, Bentheimer, and Ketton.

Usage (once per domain):
  CUDA_VISIBLE_DEVICES=0 python analysis/threshold_sweep.py \\
      --volume_path data/BB_1000c_f32.bin --domain BB \\
      --checkpoint runs/stage2/best.pt \\
      --out_csv outputs/analysis/threshold_sweep_BB.csv
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, OFFSETS_IN6  # noqa: E402
from maps.metrics import compute_all_morphological_metrics  # noqa: E402
from maps.triaxis import reconstruct_axis  # noqa: E402
from analysis.benchmark_cubes import get_cubes, CUBE_ZHW  # noqa: E402
from analysis.inference_tiled import load_unetg  # noqa: E402


def otsu_threshold(arr):
    """Otsu threshold of a flat [0,1] float array."""
    hist, edges = np.histogram(arr, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / hist.sum()
    bins = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * bins)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return float(bins[int(np.nanargmax(sigma_b2))])


def metrics_at_threshold(pred_cube, gt_cube, threshold, n_z=24,
                         device='cpu'):
    """Morphology of the prediction binarized at `threshold` vs the binary
    GT, on the standard z-slice batch."""
    D = gt_cube.shape[0]
    z_idxs = np.linspace(16, D - 16, n_z).astype(int)
    p = torch.stack([pred_cube[zi] for zi in z_idxs],
                    dim=0).unsqueeze(1).float().to(device)
    g = torch.stack([gt_cube[zi] for zi in z_idxs],
                    dim=0).unsqueeze(1).float().to(device)
    p_bin = (p >= threshold).float()
    morph = compute_all_morphological_metrics(p_bin, g, max_lag=16)
    area_mpx = (p.shape[-2] * p.shape[-1]) / 1e6
    return {
        'phi_pred': float(morph['phi_pred_mean']),
        'phi_gt': float(morph['phi_target_mean']),
        'dphi': float(morph['dphi']),
        'dsa': float(morph['dsa']),
        'd_euler_per_mpx': float(morph['d_euler']) / area_mpx,
        's2_mse': float(morph['s2_mse']),
    }


def main():
    ap = argparse.ArgumentParser(
        description='Binarization threshold sweep phi(tau) (Fig. S6)')
    ap.add_argument('--volume_path', required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--domain', default='volume')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.40, 0.45, 0.50, 0.55, 0.60],
                    help='swept tau values (paper band: [0.45, 0.55])')
    ap.add_argument('--offsets', type=int, nargs='+', default=None)
    ap.add_argument('--cubes_n', type=int, default=1,
                    help='1 = center cube only (figure convention); '
                         '3 adds the two random benchmark cubes')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_csv',
                    default='outputs/analysis/threshold_sweep.csv')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    offsets = args.offsets or OFFSETS_IN6
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape, n=max(args.cubes_n, 1))[:args.cubes_n]
    G = load_unetg(args.checkpoint, device)
    Dc, Hc, Wc = CUBE_ZHW

    rows = []
    for (z0, y0, x0, lab) in cubes:
        cube = np.clip(np.ascontiguousarray(
            vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32),
            0.0, 1.0)
        t0 = time.time()
        V_z = reconstruct_axis(G, cube, 'z', offsets, device)
        V_x = reconstruct_axis(G, cube, 'x', offsets, device)
        V_y = reconstruct_axis(G, cube, 'y', offsets, device)
        V_mean = (V_z + V_x + V_y) / 3.0
        t_recon = time.time() - t0
        log(f'[{args.domain}/{lab}] tri_mean recon {t_recon:.1f}s')

        otsu_t = otsu_threshold(V_mean.cpu().numpy().ravel())
        log(f'  Otsu threshold (from the prediction): {otsu_t:.4f}')
        gt_t = torch.from_numpy(cube).float()

        sweep = [(f'{t:.2f}', float(t)) for t in args.thresholds]
        sweep.append(('otsu', otsu_t))
        for label, thr in sweep:
            m = metrics_at_threshold(V_mean, gt_t, thr, device=device)
            rows.append({'domain': args.domain, 'cube': lab,
                         'cube_z0': z0, 'threshold_label': label,
                         'threshold': thr, **m,
                         'otsu_threshold': otsu_t})
            log(f'  tau {label:5s} ({thr:.4f}): phi={m["phi_pred"]:.4f} '
                f'(GT {m["phi_gt"]:.4f}) dphi={m["dphi"]:.5f} '
                f'dsa={m["dsa"]:.5f}')

    keys = list(rows[0].keys())
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f'[saved] {out_csv} ({len(rows)} rows)')

    # Band summary: max relative phi change over [0.45, 0.55]
    for lab in {r['cube'] for r in rows}:
        band = [r['phi_pred'] for r in rows
                if r['cube'] == lab and 0.45 <= r['threshold'] <= 0.55]
        if len(band) >= 2 and min(band) > 0:
            rel = (max(band) - min(band)) / min(band) * 100
            log(f'  {args.domain}/{lab}: phi change over tau in '
                f'[0.45, 0.55] = {rel:.2f}%')


if __name__ == '__main__':
    main()
