#!/usr/bin/env python3
"""
k-sweep stress evaluation -- porosity error vs. through-plane slice gap k
(Fig. 6 of the paper; the k=1 deployment-parity endpoint is Table S3,
reproduced by analysis/morphology3d_eval.py).

Protocol (all-replacement stress test, eval-only, no retraining):
  * paper k in {1, 3, 5, 7} by default (k = nearest-known-slice distance in
    voxels; k=1 -> every other slice known). Add 2 for the k=2 crossover
    point of Fig. 6.
  * DL offsets = [-5k, -3k, -k, +k, +3k, +5k] -- the training geometry
    OFFSETS_IN6 = [-15,-9,-3,3,9,15] corresponds to paper k=3, so every k is
    zero-shot for a single BB-trained checkpoint.
  * Classical B1 linear uses offsets +-k.
  * All metrics are computed on a FIXED core crop with margin = 35
    (= max k_max over the default sweep, 5*7) on all three axes:
        core = cube[35:93, 35:221, 35:221]  (58 x 186 x 186)
    identical for every method / k / aggregation -> clean across-k
    comparison (a moving margin would mix untouched GT slices into the
    metric batch at large k and inflate scores).
  * Aggregations: z_only and tri_mean only (both GT-free).

Cross-domain (Bentheimer / Estaillades / Ketton panels of Fig. 6): run this
script once per domain volume with the same BB-trained checkpoints
(zero-shot); all 1000^3 volumes yield the same seeded cube origins.

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/ksweep_eval.py \\
      --volume_path data/BB_1000c_f32.bin \\
      --ckpt_maps runs/stage2/best.pt --ckpt_b4 runs/b4/best.pt \\
      --k_values 1 2 3 5 7 --out_dir outputs/analysis/ksweep
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as pytorch_msssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume  # noqa: E402
from maps.models import UNetG  # noqa: E402
from maps.checkpoint import extract_model_state, load_state_checked  # noqa: E402
from maps.metrics import compute_all_morphological_metrics  # noqa: E402
from maps.triaxis import reconstruct_axis  # noqa: E402
from analysis.benchmark_cubes import get_cubes, CUBE_ZHW  # noqa: E402

K_PAPER_LIST = [1, 3, 5, 7]
MARGIN = 35  # = max k_max (paper k=7 -> 5*7=35); fixed eval core for ALL cells


def linear_reconstruct_axis(cube, axis, k):
    """B1 classical linear (all-replacement): every slice t in the interior
    band along `axis` replaced by 0.5*(GT[t-k] + GT[t+k]). GT elsewhere
    (excluded by the core crop)."""
    D, H, W = cube.shape
    out = torch.from_numpy(cube.copy()).float()
    g = torch.from_numpy(cube).float()
    if axis == 'z':
        for t in range(k, D - k):
            out[t] = 0.5 * (g[t - k] + g[t + k])
    elif axis == 'x':
        for t in range(k, W - k):
            out[:, :, t] = 0.5 * (g[:, :, t - k] + g[:, :, t + k])
    else:  # y
        for t in range(k, H - k):
            out[:, t, :] = 0.5 * (g[:, t - k] + g[:, t + k])
    return out


def metrics_core(pred_cube, gt_cube, device, margin=MARGIN, n_cross=8):
    """Metrics on the fixed core crop [margin:D-margin, margin:H-margin,
    margin:W-margin]. SSIM/morphology over ALL core z-slices."""
    D, H, W = gt_cube.shape
    p = pred_cube[margin:D - margin, margin:H - margin, margin:W - margin]
    g = gt_cube[margin:D - margin, margin:H - margin, margin:W - margin]
    Dc, Hc, Wc = g.shape
    p_stack = p.unsqueeze(1).float().to(device)   # (Dc, 1, Hc, Wc)
    g_stack = g.unsqueeze(1).float().to(device)
    ssim = float(pytorch_msssim_fn(p_stack, g_stack, data_range=1.0).item())
    mse = F.mse_loss(p_stack, g_stack).item()
    psnr = -10 * np.log10(mse + 1e-10)
    morph = compute_all_morphological_metrics(p_stack, g_stack, max_lag=16)
    area_mpx = (Hc * Wc) / 1e6
    xz_ssims, yz_ssims = [], []
    y_idxs = np.linspace(Hc // 4, 3 * Hc // 4, n_cross).astype(int)
    x_idxs = np.linspace(Wc // 4, 3 * Wc // 4, n_cross).astype(int)
    for yi in y_idxs:
        pz = p[:, yi, :].unsqueeze(0).unsqueeze(0).float().to(device)
        gz = g[:, yi, :].unsqueeze(0).unsqueeze(0).float().to(device)
        xz_ssims.append(float(pytorch_msssim_fn(pz, gz, data_range=1.0).item()))
    for xi in x_idxs:
        pz = p[:, :, xi].unsqueeze(0).unsqueeze(0).float().to(device)
        gz = g[:, :, xi].unsqueeze(0).unsqueeze(0).float().to(device)
        yz_ssims.append(float(pytorch_msssim_fn(pz, gz, data_range=1.0).item()))
    return {
        'ssim': ssim, 'psnr': psnr,
        'dphi': float(morph['dphi']), 'dsa': float(morph['dsa']),
        'd_euler': float(morph['d_euler']),
        'd_euler_per_mpx': float(morph['d_euler']) / area_mpx,
        's2_mse': float(morph['s2_mse']),
        'xz_ssim': float(np.mean(xz_ssims)), 'yz_ssim': float(np.mean(yz_ssims)),
        'n_core_slices': Dc,
    }


def main():
    ap = argparse.ArgumentParser(description='k-sweep stress evaluation (Fig. 6)')
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int, default=[1000, 1000, 1000])
    ap.add_argument('--ckpt_maps', type=str, default=None,
                    help='MAPS Stage-2 checkpoint (skipped if omitted)')
    ap.add_argument('--ckpt_b4', type=str, default=None,
                    help='b4 (2D U-Net L1) checkpoint (skipped if omitted)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--k_values', type=int, nargs='+', default=None,
                    help='paper-k list (default [1,3,5,7]); MARGIN stays 35 '
                         'for cross-k comparability')
    ap.add_argument('--out_dir', type=str, default='outputs/analysis/ksweep')
    args = ap.parse_args()

    k_list = args.k_values or K_PAPER_LIST
    assert max(5 * k for k in k_list) <= MARGIN, \
        f'k_max = 5*max(k) must be <= MARGIN={MARGIN}'

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')
    log_f = open(out_dir / 'run.log', 'a')

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + '\n')
        log_f.flush()

    dl_methods = {}
    if args.ckpt_maps:
        dl_methods['maps'] = Path(args.ckpt_maps)
    if args.ckpt_b4:
        dl_methods['b4_unet2d_l1'] = Path(args.ckpt_b4)

    log(f'k-sweep eval: k_paper={k_list}, margin={MARGIN}, '
        f'methods=[b1_linear, {", ".join(dl_methods)}]')
    log(f'Loading {args.volume_path}')
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape)
    log(f'Cubes (z0,y0,x0,label): {cubes}')

    rows = []

    def add_row(method, k, cube_label, agg, offsets, m, t_recon):
        rows.append({'method': method, 'aggregation': agg, 'k_paper': k,
                     'cube_id': cube_label,
                     'offsets': ' '.join(map(str, offsets)),
                     'recon_seconds': round(t_recon, 1), **m})
        log(f'    {method:18s} k={k} {cube_label:6s} {agg:8s} '
            f"SSIM={m['ssim']:.4f} dphi={m['dphi']:.5f} dsa={m['dsa']:.5f} "
            f"deul/Mpx={m['d_euler_per_mpx']:.1f}")

    # ── Classical B1 linear ──
    for (z0, y0, x0, label) in cubes:
        cube = np.ascontiguousarray(
            vol[z0:z0 + CUBE_ZHW[0], y0:y0 + CUBE_ZHW[1], x0:x0 + CUBE_ZHW[2]]
        ).astype(np.float32)
        cube = np.clip(cube, 0.0, 1.0)
        gt_t = torch.from_numpy(cube).float()
        for k in k_list:
            t0 = time.time()
            V_z = linear_reconstruct_axis(cube, 'z', k)
            V_x = linear_reconstruct_axis(cube, 'x', k)
            V_y = linear_reconstruct_axis(cube, 'y', k)
            V_mean = (V_z + V_x + V_y) / 3.0
            t_recon = time.time() - t0
            add_row('b1_linear', k, label, 'z_only', [-k, k],
                    metrics_core(V_z, gt_t, device), t_recon)
            add_row('b1_linear', k, label, 'tri_mean', [-k, k],
                    metrics_core(V_mean, gt_t, device), t_recon)

    # ── DL methods (zero-shot, scaled offset pattern) ──
    for method, ckpt in dl_methods.items():
        if not ckpt.exists():
            log(f'[SKIP] {method}: ckpt missing at {ckpt}')
            continue
        log(f'Loading {method} <- {ckpt}')
        G = UNetG(in_ch=6, base=80).to(device)
        ck = torch.load(ckpt, map_location=device)
        load_state_checked(G, extract_model_state(ck), label=str(ckpt))
        G.eval()
        for (z0, y0, x0, label) in cubes:
            cube = np.ascontiguousarray(
                vol[z0:z0 + CUBE_ZHW[0], y0:y0 + CUBE_ZHW[1],
                    x0:x0 + CUBE_ZHW[2]]).astype(np.float32)
            cube = np.clip(cube, 0.0, 1.0)
            gt_t = torch.from_numpy(cube).float()
            for k in k_list:
                offsets = [-5 * k, -3 * k, -k, k, 3 * k, 5 * k]
                t0 = time.time()
                V_z = reconstruct_axis(G, cube, 'z', offsets, device)
                V_x = reconstruct_axis(G, cube, 'x', offsets, device)
                V_y = reconstruct_axis(G, cube, 'y', offsets, device)
                V_mean = (V_z + V_x + V_y) / 3.0
                t_recon = time.time() - t0
                add_row(method, k, label, 'z_only', offsets,
                        metrics_core(V_z, gt_t, device), t_recon)
                add_row(method, k, label, 'tri_mean', offsets,
                        metrics_core(V_mean, gt_t, device), t_recon)
        del G
        torch.cuda.empty_cache()

    # ── Save wide CSV ──
    wide_path = out_dir / 'results_wide.csv'
    keys = list(rows[0].keys())
    with open(wide_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f'[SAVED] {wide_path} ({len(rows)} rows)')

    # ── Save long-format CSV ──
    metric_cols = ['ssim', 'psnr', 'dphi', 'dsa', 'd_euler', 'd_euler_per_mpx',
                   's2_mse', 'xz_ssim', 'yz_ssim']
    long_path = out_dir / 'results_long.csv'
    with open(long_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'aggregation', 'k_paper', 'cube_id',
                    'metric', 'value'])
        for r in rows:
            for mc in metric_cols:
                w.writerow([r['method'], r['aggregation'], r['k_paper'],
                            r['cube_id'], mc, r[mc]])
    log(f'[SAVED] {long_path}')
    log_f.close()


if __name__ == '__main__':
    main()
