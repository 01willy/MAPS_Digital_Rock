#!/usr/bin/env python3
"""
Aggregation ablation (Table 2 of the paper): evaluate ALL ground-truth-free
tri-axis aggregation variants plus the GT-using oracle reference row on the
benchmark cubes.

Rows produced per (checkpoint, cube):
  z_only / x_only / y_only            -- single-axis references
  tri_mean / tri_median / tri_consensus / tri_weuler_self /
  tri_voxel_consensus                 -- GT-free aggregations (maps.triaxis)
  tri_weuler_oracle                   -- GT-using evaluation reference,
                                         imported from maps.oracle_eval and
                                         labeled explicitly (evaluation only,
                                         never deployable)

Optionally applies the deployment-parity protocol (--parity): acquired
even-z slices pasted back after aggregation, so only odd-z content is
model-synthesized (the Table 2 convention). Metrics are computed on the
final deployed volume (acquired + synthesized slices), not on the
synthesized odd slices alone.

Multi-seed: pass one checkpoint per seed, e.g.
  --checkpoints 2025=runs/s2025/best.pt 2026=runs/s2026/best.pt ...

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/aggregation_ablation.py \\
      --volume_path data/BB_1000c_f32.bin --domain BB \\
      --checkpoints 2025=runs/s2025/best.pt \\
      --parity --out_csv outputs/analysis/agg_ablation_BB.csv
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
from maps.triaxis import (reconstruct_axis, parity_paste,  # noqa: E402
                          compute_all_gtfree_aggregations, metrics_from_cube)
from maps.oracle_eval import weighted_euler_aggregation_oracle  # noqa: E402
from analysis.benchmark_cubes import get_cubes, CUBE_ZHW  # noqa: E402
from analysis.inference_tiled import load_unetg  # noqa: E402


AGG_KEYS = ['z_only', 'x_only', 'y_only', 'tri_mean', 'tri_median',
            'tri_consensus', 'tri_weuler_self', 'tri_voxel_consensus',
            'tri_weuler_oracle']


def parse_checkpoints(specs):
    """Parse 'seed=path' entries into an ordered dict."""
    out = {}
    for spec in specs:
        if '=' in spec:
            seed, path = spec.split('=', 1)
        else:
            seed, path = str(len(out)), spec
        out[seed] = Path(path)
    return out


def eval_single_config(seed, G, vol, cube_origin, device, use_parity, log):
    z0, y0, x0, cube_label = cube_origin
    Dc, Hc, Wc = CUBE_ZHW
    cube = np.ascontiguousarray(
        vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32)
    cube = np.clip(cube, 0.0, 1.0)
    gt_t = torch.from_numpy(cube).float()

    t0 = time.time()
    V_z = reconstruct_axis(G, cube, 'z', OFFSETS_IN6, device)
    V_x = reconstruct_axis(G, cube, 'x', OFFSETS_IN6, device)
    V_y = reconstruct_axis(G, cube, 'y', OFFSETS_IN6, device)
    t_recon = time.time() - t0
    log(f'    recon s{seed}/{cube_label} t={t_recon:.1f}s')

    def finalize(V):
        return parity_paste(V, gt_t, z0) if use_parity else V

    metrics = {}
    # Single-axis references
    metrics['z_only'] = metrics_from_cube(finalize(V_z), gt_t, device=device)
    metrics['x_only'] = metrics_from_cube(finalize(V_x), gt_t, device=device)
    metrics['y_only'] = metrics_from_cube(finalize(V_y), gt_t, device=device)

    # All GT-free aggregations
    aggs = compute_all_gtfree_aggregations(V_z, V_x, V_y, device)
    for name, (V_agg, info) in aggs.items():
        m = metrics_from_cube(finalize(V_agg), gt_t, device=device)
        m['_weights'] = info
        metrics[name] = m

    # Oracle reference (GT-using; for ablation comparison only, not deployable)
    V_weu, w_info = weighted_euler_aggregation_oracle(V_z, V_x, V_y, gt_t,
                                                      device)
    m = metrics_from_cube(finalize(V_weu), gt_t, device=device)
    m['_weights'] = w_info
    metrics['tri_weuler_oracle'] = m
    return metrics


def emit_rows(rows, domain, seed, protocol, cube_origin, m):
    cube_label = cube_origin[3]
    for agg in AGG_KEYS:
        if agg not in m:
            continue
        mm = m[agg]
        rows.append({
            'domain': domain, 'seed': seed, 'protocol': protocol,
            'cube': cube_label,
            'cube_z0': cube_origin[0], 'cube_y0': cube_origin[1],
            'cube_x0': cube_origin[2], 'agg': agg,
            'ssim': mm['ssim_z'], 'psnr': mm['psnr_z'],
            'dphi': mm['dphi'], 'dsa': mm['dsa'],
            'd_euler_per_mpx': mm['d_euler_per_mpx'],
            'xz_ssim': mm['xz_ssim_mean'],
            'yz_ssim': mm['yz_ssim_mean'],
        })


def main():
    ap = argparse.ArgumentParser(description='Aggregation ablation (Table 2)')
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--domain', type=str, default='BB',
                    help='Label written to the CSV')
    ap.add_argument('--checkpoints', nargs='+', required=True,
                    help='seed=path pairs (one per Stage-2 seed)')
    ap.add_argument('--cubes_n', type=int, default=3)
    ap.add_argument('--parity', action='store_true', default=True,
                    help='Deployment-parity protocol (default ON; Table 2)')
    ap.add_argument('--no_parity', dest='parity', action='store_false',
                    help='All-replacement protocol')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_csv', type=str,
                    default='outputs/analysis/aggregation_ablation.csv')
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')

    def log(msg):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

    ckpts = parse_checkpoints(args.checkpoints)
    protocol = 'parity' if args.parity else 'allrep'
    log(f'aggregation ablation: domain={args.domain} seeds={list(ckpts)} '
        f'protocol={protocol}')

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape, n=args.cubes_n)
    log(f'cubes: {[c[3] for c in cubes]}')

    rows = []
    for seed, ckpt in ckpts.items():
        if not ckpt.exists():
            log(f'  [skip] seed {seed}: ckpt missing at {ckpt}')
            continue
        G = load_unetg(ckpt, device)
        for cube_origin in cubes:
            try:
                m = eval_single_config(seed, G, vol, cube_origin, device,
                                       args.parity, log)
                emit_rows(rows, args.domain, seed, protocol, cube_origin, m)
                log(f'    [ok] s{seed}/{cube_origin[3]}')
            except Exception as e:
                log(f'    [ERR] s{seed}/{cube_origin[3]}: {e}')
        del G
        torch.cuda.empty_cache()

    if rows:
        keys = ['domain', 'seed', 'protocol', 'cube',
                'cube_z0', 'cube_y0', 'cube_x0', 'agg',
                'ssim', 'psnr', 'dphi', 'dsa', 'd_euler_per_mpx',
                'xz_ssim', 'yz_ssim']
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        log(f'[saved] {out_csv} ({len(rows)} rows)')

        # Per-aggregation summary (mean +/- std over seeds x cubes)
        import statistics
        log('summary (mean dphi / ssim per aggregation):')
        for agg in AGG_KEYS:
            d = [float(r['dphi']) for r in rows if r['agg'] == agg]
            s = [float(r['ssim']) for r in rows if r['agg'] == agg]
            if not d:
                continue
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            log(f'  {agg:20s} dphi {statistics.mean(d):.5f}+/-{sd:.5f} '
                f'ssim {statistics.mean(s):.4f}  n={len(d)}')
    log('aggregation ablation done.')


if __name__ == '__main__':
    main()
