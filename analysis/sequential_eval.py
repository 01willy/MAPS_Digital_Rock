#!/usr/bin/env python3
"""
Idealized vs strictly-sequential tri-axis evaluation at k=1 (Table S4).

At k=1 the idealized parity protocol lets the x/y passes condition on
cross-sectional planes that contain GT voxels at unacquired z positions
(aggregation-fair, but not deployable). The sequential protocol removes
this: the parity z pass fills the missing odd z slices first (boundary odd
slices filled GT-free by linear interpolation of the acquired neighbours),
then the x/y passes run on the z-filled volume.

Per (method, protocol, aggregation, cube) this driver reconstructs the
three seeded BB benchmark cubes (128x256x256) and reports the shallow
metric suite (SSIM / PSNR / dphi / dSA / dchi-per-Mpx):

  methods    : maps (UNetG checkpoint, k=1-scaled offsets
               [-5,-3,-1,1,3,5]) and linear_k1 (per-axis parity linear)
  protocols  : idealized (each pass conditions on GT planes) and
               sequential (z-fill first)
  aggregations: z_only (protocol-independent by construction),
               tri_mean, tri_weuler_self

The summary printed at the end gives the idealized -> sequential
degradation per method at tri_mean, the numbers behind the paper's
statement that MAPS degrades far less than tri-axis linear under the
deployable protocol.

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/sequential_eval.py \\
      --volume_path data/BB_1000c_f32.bin --checkpoint runs/stage2/best.pt \\
      --out_csv outputs/analysis/sequential_bb.csv
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume  # noqa: E402
from maps.triaxis import (reconstruct_axis_parity, sequential_triaxis,  # noqa: E402
                          gtfree_boundary_fill_z, aggregate_tri_weuler_self,
                          metrics_from_cube)
from analysis.benchmark_cubes import get_cubes, CUBE_ZHW  # noqa: E402
from analysis.inference_tiled import load_unetg  # noqa: E402
from baselines.classical import linear_recon_axis  # noqa: E402

OFFSETS_K1 = [-5, -3, -1, 1, 3, 5]  # the k=1-scaled offset pattern


def make_aggregations(V_z, V_x, V_y, device):
    out = {'z_only': V_z.clamp(0, 1),
           'tri_mean': ((V_z + V_x + V_y) / 3.0).clamp(0, 1)}
    V_w, _info = aggregate_tri_weuler_self(V_z, V_x, V_y, device)
    out['tri_weuler_self'] = V_w.clamp(0, 1)
    return out


def idealized_maps(G, gt_np, offsets, device):
    """Idealized parity recon: each axis pass conditions on GT planes."""
    V_z = reconstruct_axis_parity(G, gt_np, 'z', offsets, device)
    V_x = reconstruct_axis_parity(G, gt_np, 'x', offsets, device)
    V_y = reconstruct_axis_parity(G, gt_np, 'y', offsets, device)
    return V_z, V_x, V_y


def idealized_linear(gt_np):
    L_z = torch.from_numpy(linear_recon_axis(gt_np, 'z', 1).astype(np.float32))
    L_x = torch.from_numpy(linear_recon_axis(gt_np, 'x', 1).astype(np.float32))
    L_y = torch.from_numpy(linear_recon_axis(gt_np, 'y', 1).astype(np.float32))
    return L_z, L_x, L_y


def sequential_linear(gt_np):
    """Sequential linear counterpart: z-fill (odd z = mean of +-1 even GT,
    GT-free boundary fill), then x/y linear passes on the z-filled volume."""
    L_z = torch.from_numpy(linear_recon_axis(gt_np, 'z', 1).astype(np.float32))
    gtfree_boundary_fill_z(L_z, gt_np, k_max=1)
    zfill = L_z.numpy().astype(np.float32)
    L_x = torch.from_numpy(linear_recon_axis(zfill, 'x', 1).astype(np.float32))
    L_y = torch.from_numpy(linear_recon_axis(zfill, 'y', 1).astype(np.float32))
    return L_z, L_x, L_y


def main():
    ap = argparse.ArgumentParser(
        description='Idealized vs sequential tri-axis at k=1 (Table S4)')
    ap.add_argument('--volume_path', required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--checkpoint', required=True,
                    help='MAPS UNetG checkpoint (Table S4 uses seed 2025)')
    ap.add_argument('--seed_label', default='2025')
    ap.add_argument('--offsets', type=int, nargs='+', default=None,
                    help='MAPS offsets (default k=1 pattern -5 -3 -1 1 3 5)')
    ap.add_argument('--methods', nargs='+', default=['maps', 'linear_k1'],
                    choices=['maps', 'linear_k1'])
    ap.add_argument('--cubes_n', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_csv',
                    default='outputs/analysis/sequential_bb.csv')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    offsets = args.offsets or OFFSETS_K1
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

    log(f'sequential eval: methods={args.methods} offsets={offsets}')
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape, n=args.cubes_n)
    log(f'cubes: {[c[3] for c in cubes]}')
    Dc, Hc, Wc = CUBE_ZHW

    G = load_unetg(args.checkpoint, device) if 'maps' in args.methods else None

    rows = []
    for (z0, y0, x0, lab) in cubes:
        gt_np = np.clip(np.ascontiguousarray(
            vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32),
            0.0, 1.0)
        gt_t = torch.from_numpy(gt_np).float()

        arms = []  # (method, protocol, (V_z, V_x, V_y))
        if 'maps' in args.methods:
            t0 = time.time()
            arms.append(('maps', 'idealized',
                         idealized_maps(G, gt_np, offsets, device)))
            arms.append(('maps', 'sequential',
                         sequential_triaxis(G, gt_np, offsets, device)))
            log(f'[{lab}] maps recon both protocols '
                f'{time.time() - t0:.1f}s')
        if 'linear_k1' in args.methods:
            arms.append(('linear_k1', 'idealized', idealized_linear(gt_np)))
            arms.append(('linear_k1', 'sequential', sequential_linear(gt_np)))

        for method, protocol, (V_z, V_x, V_y) in arms:
            aggs = make_aggregations(V_z, V_x, V_y, device)
            for agg, V in aggs.items():
                m = metrics_from_cube(V, gt_t, device=device)
                rows.append(dict(
                    method=method,
                    seed=args.seed_label if method == 'maps' else '',
                    protocol=protocol, agg=agg, cube=lab,
                    cube_z0=z0, cube_y0=y0, cube_x0=x0,
                    ssim=m['ssim_z'], psnr=m['psnr_z'], dphi=m['dphi'],
                    dsa=m['dsa'], dchi=m['d_euler_per_mpx'],
                    xz_ssim=m['xz_ssim_mean'], yz_ssim=m['yz_ssim_mean']))
                log(f'  {method:10s} {protocol:10s} {agg:16s} '
                    f'ssim={m["ssim_z"]:.4f} dphi={m["dphi"]:.5f}')

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f'[saved] {out_csv} ({len(rows)} rows)')

    # Summary: mean over cubes per (method, protocol, agg) + degradation
    grp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r['method'], r['protocol'], r['agg'])
        for k in ('ssim', 'psnr', 'dphi', 'dsa', 'dchi'):
            grp[key][k].append(r[k])
    log('SUMMARY (mean over cubes) -- Table S4 columns:')
    for key, d in sorted(grp.items()):
        log(f'  {"/".join(key):42s} ssim={np.mean(d["ssim"]):.4f} '
            f'psnr={np.mean(d["psnr"]):.2f} dphi={np.mean(d["dphi"]):.3e} '
            f'dsa={np.mean(d["dsa"]):.3e} dchi={np.mean(d["dchi"]):.2f}')
    for method in args.methods:
        gi = grp.get((method, 'idealized', 'tri_mean'))
        gs = grp.get((method, 'sequential', 'tri_mean'))
        if gi and gs:
            parts = []
            for k in ('ssim', 'dphi', 'dsa', 'dchi'):
                vi, vs = np.mean(gi[k]), np.mean(gs[k])
                if vi != 0:
                    parts.append(f'{k} {vi:.4g}->{vs:.4g} '
                                 f'({100 * (vs - vi) / abs(vi):+.1f}%)')
            log(f'  {method} tri_mean idealized->sequential: '
                + '; '.join(parts))


if __name__ == '__main__':
    main()
