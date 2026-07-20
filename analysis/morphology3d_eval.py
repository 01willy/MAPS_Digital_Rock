#!/usr/bin/env python3
"""
3D pore-morphology evaluation driver (Tables S3, S5-S7 of the Supplement).

Per (method, cube) it reconstructs the volume, then computes the 3D
morphology suite of `maps.metrics3d.compute_3d_morphology_metrics`:
S2(3D) MSE, correlation length, lineal-path(3D) MSE, connected porosity,
coordination number Z (porespy.snow2), PSD Wasserstein-1, tau_diffusive
(taufactor, 3 axes) and tau_hydraulic + permeability (D3Q19 LBM, 3 axes),
plus the slice-level suite (SSIM, dphi, ...) for cross-referencing.

Methods:
  maps          -- UNetG checkpoint (--checkpoint), tri-axis reconstruction
  classical_b1 / classical_b2 / classical_b3
                -- built from the GT cube directly (no checkpoint)
  linear_k1_tri -- tri-axis parity linear reference

Protocols:
  --protocol all_replacement (default): every interior slice along each
      axis re-predicted (the recon-cache convention behind Tables S5-S7).
  --protocol parity: only odd slices replaced, even slices kept = GT (the
      deployment scenario; with `--offsets -5 -3 -1 1 3 5` this reproduces
      the k=1 parity comparison of Table S3). Metrics are computed on the
      final deployed volume (acquired + synthesized slices), not on the
      synthesized odd slices alone.

Binarization: per-image Otsu (each of pred and GT thresholded
independently), the convention used throughout the paper.

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/morphology3d_eval.py \\
      --volume_path data/BB_1000c_f32.bin --domain BB \\
      --methods maps classical_b1 classical_b2 classical_b3 \\
      --checkpoint runs/stage2/best.pt \\
      --out_csv outputs/analysis/morph3d_BB.csv
"""

from __future__ import annotations

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
from maps.metrics3d import compute_3d_morphology_metrics  # noqa: E402
from maps.triaxis import reconstruct_axis, metrics_from_cube  # noqa: E402
from analysis.benchmark_cubes import get_cubes, CUBE_ZHW  # noqa: E402
from analysis.inference_tiled import (load_unetg,  # noqa: E402
                                      reconstruct_axis_parity)
from baselines.classical import (classical_linear_k1, classical_linear_k3,  # noqa: E402
                                 classical_cubic_k3, linear_recon_axis)


def build_prediction(method, gt_cube, G, protocol, offsets, aggregation,
                     device):
    """Return predicted float volume in [0,1] for one method/cube."""
    if method == 'classical_b1':
        return classical_linear_k1(gt_cube)
    if method == 'classical_b2':
        return classical_linear_k3(gt_cube)
    if method == 'classical_b3':
        return classical_cubic_k3(gt_cube)
    if method == 'linear_k1_tri':
        V_z = linear_recon_axis(gt_cube, 'z', k=1)
        V_x = linear_recon_axis(gt_cube, 'x', k=1)
        V_y = linear_recon_axis(gt_cube, 'y', k=1)
        return np.clip((V_z + V_x + V_y) / 3.0, 0.0, 1.0).astype(np.float32)
    if method == 'maps':
        if G is None:
            raise ValueError('method maps requires --checkpoint')
        recon = (reconstruct_axis_parity if protocol == 'parity'
                 else reconstruct_axis)
        V_z = recon(G, gt_cube, 'z', offsets, device)
        if aggregation == 'z_only':
            pred = V_z
        else:
            V_x = recon(G, gt_cube, 'x', offsets, device)
            V_y = recon(G, gt_cube, 'y', offsets, device)
            if aggregation == 'mean':
                pred = (V_z + V_x + V_y) / 3.0
            elif aggregation == 'median':
                pred = torch.stack([V_z, V_x, V_y], dim=0).median(dim=0).values
            else:
                raise ValueError(f'Unknown aggregation: {aggregation}')
        return np.clip(pred.cpu().numpy(), 0.0, 1.0).astype(np.float32)
    raise ValueError(f'Unknown method: {method}')


def main():
    ap = argparse.ArgumentParser(description='3D morphology evaluation '
                                             '(Tables S3, S5-S7)')
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--domain', type=str, default='volume',
                    help='Label written to the CSV (e.g. BB, Bentheimer)')
    ap.add_argument('--methods', nargs='+',
                    default=['maps', 'classical_b1', 'classical_b2',
                             'classical_b3'],
                    choices=['maps', 'classical_b1', 'classical_b2',
                             'classical_b3', 'linear_k1_tri'])
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='UNetG checkpoint for the maps method')
    ap.add_argument('--seed_label', type=int, default=2025,
                    help='Seed label recorded in the CSV (which multi-seed '
                         'checkpoint --checkpoint corresponds to)')
    ap.add_argument('--protocol', default='all_replacement',
                    choices=['all_replacement', 'parity'])
    ap.add_argument('--offsets', type=int, nargs='+', default=None,
                    help='Input offsets (default OFFSETS_IN6 = '
                         '[-15,-9,-3,3,9,15]; use -5 -3 -1 1 3 5 for the '
                         'k=1 parity comparison of Table S3)')
    ap.add_argument('--aggregation', default='mean',
                    choices=['mean', 'median', 'z_only'])
    ap.add_argument('--cubes_n', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--skip_tau_taufactor', action='store_true')
    ap.add_argument('--skip_tau_lbm', action='store_true')
    ap.add_argument('--skip_coord', action='store_true')
    ap.add_argument('--skip_psd', action='store_true')
    ap.add_argument('--tau_lbm_steps', type=int, default=3000)
    ap.add_argument('--max_r_s2', type=int, default=48)
    ap.add_argument('--max_r_lineal', type=int, default=24)
    ap.add_argument('--out_csv', type=str,
                    default='outputs/analysis/morphology3d.csv')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    offsets = args.offsets or OFFSETS_IN6
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

    log(f'morphology3d eval: domain={args.domain} methods={args.methods} '
        f'protocol={args.protocol} agg={args.aggregation} offsets={offsets}')

    G = None
    if 'maps' in args.methods:
        if not args.checkpoint:
            raise SystemExit('--checkpoint required for method maps')
        G = load_unetg(args.checkpoint, device)
        log(f'loaded checkpoint {args.checkpoint}')

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape, n=args.cubes_n)
    log(f'cubes: {[c[3] for c in cubes]}')

    keys = ['domain', 'method', 'seed', 'protocol', 'aggregation', 'cube',
            'cube_z0', 'cube_y0', 'cube_x0',
            # slice-level suite
            'ssim', 'psnr', 'dphi', 'dsa', 'd_euler_per_mpx',
            'xz_ssim', 'yz_ssim',
            # 3D morphology suite
            'porosity_pred', 'porosity_gt', 'dphi_3d',
            's2_3d_mse', 'lc_pred', 'lc_gt', 'dlc',
            'lpath_3d_mse',
            'connected_porosity_pred', 'connected_porosity_gt',
            'd_connected_porosity',
            'euler_3d_pred', 'euler_3d_gt', 'd_euler_3d',
            'ssa_3d_pred', 'ssa_3d_gt', 'd_ssa_3d',
            'Z_mean_pred', 'Z_mean_gt', 'dZ', 'n_pores_pred', 'n_pores_gt',
            'psd_wasserstein',
            'tau_tf_z_pred', 'tau_tf_z_gt', 'tau_tf_y_pred', 'tau_tf_y_gt',
            'tau_tf_x_pred', 'tau_tf_x_gt',
            'tau_lbm_z_pred', 'tau_lbm_z_gt', 'tau_lbm_y_pred', 'tau_lbm_y_gt',
            'tau_lbm_x_pred', 'tau_lbm_x_gt',
            'k_lbm_z_pred_mD', 'k_lbm_z_gt_mD',
            'k_lbm_y_pred_mD', 'k_lbm_y_gt_mD',
            'k_lbm_x_pred_mD', 'k_lbm_x_gt_mD',
            'wall_seconds', 'otsu_tau']
    new_file = not out_csv.exists()
    csv_f = open(out_csv, 'a', newline='')
    csv_w = csv.DictWriter(csv_f, fieldnames=keys, extrasaction='ignore')
    if new_file:
        csv_w.writeheader()
        csv_f.flush()

    n_rows = 0
    for cube_origin in cubes:
        z0, y0, x0, cube_label = cube_origin
        Dc, Hc, Wc = CUBE_ZHW
        gt_cube = np.ascontiguousarray(
            vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32)
        gt_cube = np.clip(gt_cube, 0.0, 1.0)
        gt_t = torch.from_numpy(gt_cube).float()

        for method in args.methods:
            t0 = time.time()
            try:
                pred = build_prediction(method, gt_cube, G, args.protocol,
                                        offsets, args.aggregation, device)
                m_slice = metrics_from_cube(torch.from_numpy(pred).float(),
                                            gt_t, device=device)
                m = compute_3d_morphology_metrics(
                    pred, gt_cube,
                    max_r_s2=args.max_r_s2,
                    max_r_lineal=args.max_r_lineal,
                    do_coord=not args.skip_coord,
                    do_psd=not args.skip_psd,
                    do_tau_taufactor=not args.skip_tau_taufactor,
                    do_tau_lbm=not args.skip_tau_lbm,
                    tau_lbm_steps=args.tau_lbm_steps,
                    device=str(device),
                )
            except Exception as e:
                log(f'  [ERR] {args.domain}/{method}/{cube_label}: {e}')
                continue
            wall = time.time() - t0
            row = {'domain': args.domain, 'method': method,
                   'seed': args.seed_label, 'protocol': args.protocol,
                   'aggregation': args.aggregation, 'cube': cube_label,
                   'cube_z0': z0, 'cube_y0': y0, 'cube_x0': x0,
                   'ssim': m_slice['ssim_z'], 'psnr': m_slice['psnr_z'],
                   'dphi': m_slice['dphi'], 'dsa': m_slice['dsa'],
                   'd_euler_per_mpx': m_slice['d_euler_per_mpx'],
                   'xz_ssim': m_slice['xz_ssim_mean'],
                   'yz_ssim': m_slice['yz_ssim_mean'],
                   'wall_seconds': round(wall, 1)}
            for k in keys:
                if k in row:
                    continue
                row[k] = m.get(k, '')
            csv_w.writerow(row)
            csv_f.flush()
            n_rows += 1
            log(f'  [ok] {args.domain}/{method}/{cube_label} t={wall:.1f}s '
                f'dphi3d={m["dphi_3d"]:.4f} s2mse={m["s2_3d_mse"]:.2e}')

    csv_f.close()
    log(f'[saved] {out_csv} ({n_rows} rows appended)')


if __name__ == '__main__':
    main()
