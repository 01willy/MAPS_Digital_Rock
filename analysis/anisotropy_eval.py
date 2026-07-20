#!/usr/bin/env python3
"""
Multi-cube anisotropy campaign (Table S16 and the Ketton permeability-tensor
ratio result: z_only 71.88 -> tri_weuler_self 1.58, a ~45x reduction on the
center cube).

Per (domain volume, 256^3 cube, aggregation): reconstruct the cube along
all three axes with a MAPS checkpoint, aggregate (z_only / tri_mean /
tri_weuler_self), binarize at --threshold, and run the D3Q19 LBM solver
along each of the three flow axes (5000 steps, tau=1.0, body force 1e-5).
The permeability tensor (k_ax0, k_ax1, k_ax2) is written one row per
(agg, flow_axis). GT tensors come from --include_gt / --gt_only runs.

Cube layouts:
  multicube   -- Table S16 layout ('center' + 'rand0' + 'rand1';
                 `benchmark_cubes.multicube_anisotropy_origins`).
  campaign256 -- the 8-domain-campaign origins
                 (`benchmark_cubes.find_cube_origins_256`; the sequential
                 anisotropy sentinel uses cube0/cube1 of this layout).

Protocols:
  idealized  -- all-replacement tri-axis reconstruction (Table S16
                convention: x/y passes condition on GT planes).
  sequential -- strictly-sequential GT-free reconstruction (z-fill first,
                then x/y passes on the filled volume; the sentinel behind
                the "sequential recovers only ~10% of the idealized
                anisotropy gain" statement).

Stage aggregate: reads the per-cube CSVs, forms the per-cube permeability
tensor per (domain, cube, method, agg, protocol), normalizes each axis by
the smallest component, and reports the L1 error of that ratio against the
GT ratio (ratio L1), plus the per-(domain, agg) mean +- std of Table S16.

Usage (one cube per invocation; parallelize across GPUs):
  CUDA_VISIBLE_DEVICES=0 python analysis/anisotropy_eval.py \\
      --volume_path data/Ketton_1000c_f32.bin --domain Ketton \\
      --voxel_um 3.0 --cube center --checkpoint runs/stage2/best.pt \\
      --include_gt --out_dir outputs/analysis/anisotropy
  python analysis/anisotropy_eval.py --stage aggregate \\
      --out_dir outputs/analysis/anisotropy
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
from maps.data import load_volume, OFFSETS_IN6  # noqa: E402
from maps.triaxis import (reconstruct_axis, sequential_triaxis,  # noqa: E402
                          aggregate_tri_mean, aggregate_tri_median,
                          aggregate_tri_consensus, aggregate_tri_weuler_self,
                          aggregate_tri_voxel_consensus)
from analysis.benchmark_cubes import (multicube_anisotropy_origins,  # noqa: E402
                                      find_cube_origins_256)
from analysis.inference_tiled import load_unetg  # noqa: E402
from lbm.d3q19 import D3Q19LBM  # noqa: E402

CUBE_SIZE = 256
AGG_CHOICES = ['z_only', 'tri_mean', 'tri_median', 'tri_consensus',
               'tri_weuler_self', 'tri_voxel_consensus']

CSV_KEYS = ['domain', 'method', 'protocol', 'agg', 'cube',
            'cube_z0', 'cube_y0', 'cube_x0', 'flow_axis', 'porosity',
            'k_mD', 'n_steps', 'voxel_um', 'wall_seconds']


def make_aggregation(V_z, V_x, V_y, agg, device):
    if agg == 'z_only':
        return V_z
    if agg == 'tri_mean':
        return aggregate_tri_mean(V_z, V_x, V_y)[0]
    if agg == 'tri_median':
        return aggregate_tri_median(V_z, V_x, V_y)[0]
    if agg == 'tri_consensus':
        return aggregate_tri_consensus(V_z, V_x, V_y)[0]
    if agg == 'tri_weuler_self':
        return aggregate_tri_weuler_self(V_z, V_x, V_y, device)[0]
    if agg == 'tri_voxel_consensus':
        return aggregate_tri_voxel_consensus(V_z, V_x, V_y)[0]
    raise ValueError(agg)


def lbm_3axis(solid01, voxel_um, device, n_steps, log):
    """D3Q19 permeability along each cube axis. Returns {axis: result}."""
    out = {}
    for axis in (0, 1, 2):
        t0 = time.time()
        sim = D3Q19LBM(solid01 > 0.5, device=str(device), tau=1.0,
                       body_force=1e-5, flow_axis=axis)
        for _ in range(n_steps):
            sim.step()
        r = sim.permeability(voxel_size_um=voxel_um)
        r['wall_seconds'] = round(time.time() - t0, 1)
        out[axis] = r
        log(f'    axis{axis}: k={r["k_mD"]:.2f} mD '
            f'({r["wall_seconds"]:.0f}s)')
        del sim
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return out


def emit_rows(rows, domain, method, protocol, agg, origin, porosity,
              ax_results, n_steps, voxel_um):
    z0, y0, x0, label = origin
    for axis, r in ax_results.items():
        rows.append({'domain': domain, 'method': method,
                     'protocol': protocol, 'agg': agg, 'cube': label,
                     'cube_z0': z0, 'cube_y0': y0, 'cube_x0': x0,
                     'flow_axis': axis, 'porosity': porosity,
                     'k_mD': r['k_mD'], 'n_steps': n_steps,
                     'voxel_um': voxel_um,
                     'wall_seconds': r['wall_seconds']})


def stage_run(args):
    device = torch.device(f'cuda:{args.gpu}')
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / 'per_cube_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    if args.cube_layout == 'multicube':
        cubes = multicube_anisotropy_origins(vol.shape, n=3)
    else:
        origins = find_cube_origins_256(vol.shape, n_cubes=3)
        cubes = [(z, y, x, f'cube{i}') for i, (z, y, x) in enumerate(origins)]
    cube_map = {c[3]: c for c in cubes}
    if args.cube not in cube_map:
        raise SystemExit(f'--cube {args.cube} not in layout '
                         f'{sorted(cube_map)}')
    origin = cube_map[args.cube]
    z0, y0, x0, label = origin
    log(f'anisotropy run: {args.domain}/{label} origin=({z0},{y0},{x0}) '
        f'protocol={args.protocol} aggs={args.aggs} voxel={args.voxel_um}um')

    gt_cube = np.clip(np.ascontiguousarray(
        vol[z0:z0 + CUBE_SIZE, y0:y0 + CUBE_SIZE, x0:x0 + CUBE_SIZE]
    ).astype(np.float32), 0.0, 1.0)

    rows = []
    if args.include_gt or args.gt_only:
        log('GT LBM permeability tensor...')
        gt_bin = (gt_cube > args.threshold).astype(np.float32)
        gt_axes = lbm_3axis(gt_bin, args.voxel_um, device, args.n_steps, log)
        emit_rows(rows, args.domain, 'GT', 'GT', 'GT', origin,
                  float(1.0 - gt_bin.mean()), gt_axes, args.n_steps,
                  args.voxel_um)

    if not args.gt_only:
        if not args.checkpoint:
            raise SystemExit('--checkpoint required unless --gt_only')
        G = load_unetg(args.checkpoint, device)
        offsets = args.offsets or OFFSETS_IN6
        t0 = time.time()
        if args.protocol == 'sequential':
            V_z, V_x, V_y = sequential_triaxis(G, gt_cube, offsets, device)
        else:
            V_z = reconstruct_axis(G, gt_cube, 'z', offsets, device)
            V_x = reconstruct_axis(G, gt_cube, 'x', offsets, device)
            V_y = reconstruct_axis(G, gt_cube, 'y', offsets, device)
        log(f'tri-axis reconstruction done in {time.time() - t0:.1f}s')
        del G
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        for agg in args.aggs:
            log(f'--- aggregation {agg} ---')
            V = make_aggregation(V_z, V_x, V_y, agg, device)
            bin_cube = (V.cpu().numpy() > args.threshold).astype(np.float32)
            porosity = float(1.0 - bin_cube.mean())
            log(f'  porosity={porosity:.4f} (GT '
                f'{1.0 - (gt_cube > args.threshold).mean():.4f})')
            if args.save_cubes:
                cdir = out_dir / 'cubes'
                cdir.mkdir(exist_ok=True)
                bin_cube.tofile(cdir / f'{args.domain}_{args.method_label}_'
                                       f'{args.protocol}_{agg}_{label}.bin')
            ax_results = lbm_3axis(bin_cube, args.voxel_um, device,
                                   args.n_steps, log)
            emit_rows(rows, args.domain, args.method_label, args.protocol,
                      agg, origin, porosity, ax_results, args.n_steps,
                      args.voxel_um)

    suffix = '_gt_only' if args.gt_only else f'_{args.protocol}'
    out_csv = raw_dir / f'{args.domain}_{args.method_label}_{label}{suffix}.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_KEYS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f'[saved] {out_csv} ({len(rows)} rows)')


def stage_aggregate(args):
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / 'per_cube_raw'
    csvs = sorted(raw_dir.glob('*.csv'))
    if not csvs:
        raise SystemExit(f'no per-cube CSVs under {raw_dir}')
    rows = []
    for p in csvs:
        with open(p, newline='') as f:
            rows.extend(csv.DictReader(f))
    print(f'loaded {len(rows)} rows from {len(csvs)} per-cube CSVs')

    # tensors[(domain, cube)][(method, protocol, agg)] = {axis: k}
    tensors = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r['domain'], r['cube'])
        mkey = (r['method'], r['protocol'], r['agg'])
        tensors[key][mkey][int(r['flow_axis'])] = float(r['k_mD'])

    def ratio(kmap):
        mn = min(kmap.values())
        mn = mn if mn > 0 else 1.0
        return {a: kmap[a] / mn for a in (0, 1, 2)}

    per_cube = []
    for (domain, cube), methods in sorted(tensors.items()):
        gt = methods.get(('GT', 'GT', 'GT'))
        if not gt or len(gt) < 3:
            print(f'[skip] {domain}/{cube}: GT tensor incomplete '
                  '(run with --gt_only first)')
            continue
        gt_ratio = ratio(gt)
        for (method, protocol, agg), kmap in methods.items():
            if method == 'GT' or len(kmap) < 3:
                continue
            rt = ratio(kmap)
            l1 = sum(abs(rt[a] - gt_ratio[a]) for a in (0, 1, 2))
            per_cube.append({
                'domain': domain, 'cube': cube, 'method': method,
                'protocol': protocol, 'agg': agg,
                'k_ax0_mD': kmap[0], 'k_ax1_mD': kmap[1],
                'k_ax2_mD': kmap[2],
                'ratio_ax0': rt[0], 'ratio_ax1': rt[1], 'ratio_ax2': rt[2],
                'gt_ratio_ax0': gt_ratio[0], 'gt_ratio_ax1': gt_ratio[1],
                'gt_ratio_ax2': gt_ratio[2], 'ratio_L1': l1})

    if not per_cube:
        raise SystemExit('no complete (method, GT) tensor pairs found')
    ratio_csv = out_dir / 'anisotropy_ratio_per_cube.csv'
    with open(ratio_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_cube[0].keys()))
        w.writeheader()
        for r in per_cube:
            w.writerow(r)
    print(f'[saved] {ratio_csv} ({len(per_cube)} rows)')

    # Table S16: mean +- std over cubes per (domain, protocol, agg)
    grp = defaultdict(list)
    for r in per_cube:
        grp[(r['domain'], r['method'], r['protocol'], r['agg'])].append(
            r['ratio_L1'])
    summary_csv = out_dir / 'anisotropy_ratio_summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['domain', 'method', 'protocol', 'agg', 'n',
                    'ratio_L1_mean', 'ratio_L1_std'])
        for (domain, method, protocol, agg), vals in sorted(grp.items()):
            w.writerow([domain, method, protocol, agg, len(vals),
                        f'{np.mean(vals):.4f}',
                        f'{np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.4f}'])
            print(f'  {domain:12s} {method:10s} {protocol:10s} {agg:18s} '
                  f'n={len(vals)}  ratio_L1={np.mean(vals):.3f}'
                  f'+-{np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.3f}')
    print(f'[saved] {summary_csv}')


def main():
    ap = argparse.ArgumentParser(
        description='Multi-cube anisotropy campaign (Table S16)')
    ap.add_argument('--stage', default='run', choices=['run', 'aggregate'])
    ap.add_argument('--volume_path')
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--domain', default='volume')
    ap.add_argument('--voxel_um', type=float, default=2.25)
    ap.add_argument('--cube', default='center',
                    help='center/rand0/rand1 (multicube layout) or '
                         'cube0/cube1/cube2 (campaign256 layout)')
    ap.add_argument('--cube_layout', default='multicube',
                    choices=['multicube', 'campaign256'])
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--method_label', default='maps')
    ap.add_argument('--protocol', default='idealized',
                    choices=['idealized', 'sequential'])
    ap.add_argument('--aggs', nargs='+',
                    default=['z_only', 'tri_mean', 'tri_weuler_self'],
                    choices=AGG_CHOICES)
    ap.add_argument('--offsets', type=int, nargs='+', default=None)
    ap.add_argument('--n_steps', type=int, default=5000)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--include_gt', action='store_true',
                    help='also compute the GT tensor for this cube')
    ap.add_argument('--gt_only', action='store_true',
                    help='only compute the GT tensor (no reconstruction)')
    ap.add_argument('--save_cubes', action='store_true',
                    help='dump binarized aggregation cubes (.bin)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_dir', default='outputs/analysis/anisotropy')
    args = ap.parse_args()

    if args.stage == 'aggregate':
        stage_aggregate(args)
    else:
        if not args.volume_path:
            raise SystemExit('--volume_path required for --stage run')
        stage_run(args)


if __name__ == '__main__':
    main()
