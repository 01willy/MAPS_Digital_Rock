#!/usr/bin/env python3
"""
8-domain k=1 LBM permeability campaign (Table S15 of the paper): generate
256^3 reconstructed cubes under the deployment-parity protocol, run the
D3Q19 LBM on all three axes, and aggregate trace errors vs. GT.

Three stages (run per domain; --stage all runs them in sequence):
  generate  -- cut GT 256^3 cube(s) from the volume; build the method cells
               (b1_linear_z, linear_k1_tri, maps_z, maps_tri) and write
               float32 {0,1} .bin files (binarize at 0.5 AFTER aggregation).
  lbm       -- run D3Q19 (5000 steps, tau=1.0, body_force=1e-5) on every
               .bin along all 3 axes; one *_lbm.json per cube.
  aggregate -- collect every *_lbm.json under --out into a long CSV and a
               per-(domain, method) trace-error summary. IMPORTANT: the
               trace is rebuilt as the mean of the per-axis k values (never
               trusting a stored single-axis value), the convention fixed
               in the paper's campaign.

Protocol notes:
  * paper k=1 parity: EVEN slices along the reconstruction axis are known
    (GT anchors, kept); ODD slices are targets. Boundary odd slices without
    full offset support keep GT.
  * b1_linear_z: odd z -> 0.5*(GT[z-1] + GT[z+1]).
  * linear_k1_tri: same linear op along z, x, y -> mean of the 3 volumes.
  * maps_z / maps_tri: UNetG (OFFSETS_IN6), odd-slice replacement along one
    or three axes.
  * Cube origins: seeded (analysis/benchmark_cubes.find_cube_origins_256);
    for 1000^3 volumes cube0 = (744, 372, 372).

Usage (one domain; repeat for the 8 domains with their voxel sizes):
  CUDA_VISIBLE_DEVICES=0 python analysis/lbm_8domain_eval.py --stage all \\
      --volume_path data/BB_1000c_f32.bin --domain BB --voxel_um 2.25 \\
      --checkpoint runs/stage2/best.pt --out outputs/analysis/lbm_8domain
  # after all domains:
  python analysis/lbm_8domain_eval.py --stage aggregate \\
      --out outputs/analysis/lbm_8domain
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, OFFSETS_IN6  # noqa: E402
from analysis.benchmark_cubes import find_cube_origins_256  # noqa: E402
from analysis.inference_tiled import load_unetg, ours_recon_axis  # noqa: E402
from analysis.lbm_multiseed_eval import run_lbm_cube  # noqa: E402
from baselines.classical import linear_recon_axis  # noqa: E402

CUBE_SIZE = 256

ALL_METHODS = ['GT', 'b1_linear_z', 'linear_k1_tri', 'maps_z', 'maps_tri']


# ──────────────────────────────────────────────────────────────────────
# Stage: generate
# ──────────────────────────────────────────────────────────────────────

def stage_generate(args, device):
    out_root = Path(args.out) / 'cubes'
    dom_dir = out_root / args.domain
    dom_dir.mkdir(parents=True, exist_ok=True)

    needs_maps = any(m in args.methods for m in ('maps_z', 'maps_tri'))
    G = None
    if needs_maps:
        if not args.checkpoint:
            raise SystemExit('--checkpoint required for maps_z / maps_tri')
        print(f'[load] UNetG <- {args.checkpoint}')
        G = load_unetg(args.checkpoint, device)

    meta_path = dom_dir / 'metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {'domain': args.domain, 'volume_path': args.volume_path,
                'cube_size': CUBE_SIZE, 'voxel_size_um': args.voxel_um,
                'protocol': 'paper k=1 parity (even anchors GT, odd '
                            'targets); binarize >0.5 after aggregation',
                'offsets_maps': OFFSETS_IN6,
                'checkpoint': str(args.checkpoint),
                'cells': {}}

    print(f'\n== {args.domain}: load {args.volume_path}')
    t0 = time.time()
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    print(f'   loaded in {time.time() - t0:.0f}s')
    origins = find_cube_origins_256(vol.shape, CUBE_SIZE,
                                    args.cubes_per_domain)

    for ci, (z0, y0, x0) in enumerate(origins):
        gt = np.ascontiguousarray(
            vol[z0:z0 + CUBE_SIZE, y0:y0 + CUBE_SIZE, x0:x0 + CUBE_SIZE]
        ).astype(np.float32)
        gt = np.clip(gt, 0.0, 1.0)
        gt_phi = float(1.0 - gt.mean())
        print(f'- cube{ci} origin=({z0},{y0},{x0}) GT phi={gt_phi:.4f}')

        # Per-cube caches so maps_z / maps_tri share the z-pass
        lin_axes = {}
        maps_axes = {}

        def lin_axis(ax):
            if ax not in lin_axes:
                lin_axes[ax] = linear_recon_axis(gt, ax, k=1)
            return lin_axes[ax]

        def maps_axis(ax):
            if ax not in maps_axes:
                t1 = time.time()
                maps_axes[ax] = ours_recon_axis(G, gt, ax, OFFSETS_IN6,
                                                device)
                print(f'    maps axis={ax} recon {time.time() - t1:.1f}s')
            return maps_axes[ax]

        cells = []
        if 'GT' in args.methods:
            cells.append((f'{args.domain}_GT_cube{ci}', lambda: gt))
        if 'b1_linear_z' in args.methods:
            cells.append((f'{args.domain}_b1_linear_z_cube{ci}',
                          lambda: (lin_axis('z') > 0.5).astype(np.float32)))
        if 'linear_k1_tri' in args.methods:
            cells.append((f'{args.domain}_linear_k1_tri_cube{ci}',
                          lambda: (((lin_axis('z') + lin_axis('x')
                                     + lin_axis('y')) / 3.0)
                                   > 0.5).astype(np.float32)))
        if 'maps_z' in args.methods:
            cells.append((f'{args.domain}_maps_z_cube{ci}',
                          lambda: (maps_axis('z') > 0.5).astype(np.float32)))
        if 'maps_tri' in args.methods:
            cells.append((f'{args.domain}_maps_tri_cube{ci}',
                          lambda: (((maps_axis('z') + maps_axis('x')
                                     + maps_axis('y')) / 3.0)
                                   > 0.5).astype(np.float32)))

        for stem, build in cells:
            out_bin = dom_dir / f'{stem}.bin'
            if out_bin.exists() and not args.overwrite:
                print(f'    [skip  ] {stem} (exists)')
                continue
            t1 = time.time()
            rec = build()
            rec.astype(np.float32).tofile(out_bin)
            phi = float(1.0 - rec.mean())
            meta['cells'][stem] = {
                'status': 'generated', 'origin': [z0, y0, x0],
                'phi': phi, 'gt_phi': gt_phi,
                'phi_diff_from_gt': abs(phi - gt_phi),
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'wall_seconds': round(time.time() - t1, 1)}
            print(f'    [gen   ] {stem} phi={phi:.4f} '
                  f'(GT {gt_phi:.4f}, d={phi - gt_phi:+.4f}) '
                  f'{time.time() - t1:.1f}s')

        del lin_axes, maps_axes

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'   [meta] {meta_path}')
    del vol


# ──────────────────────────────────────────────────────────────────────
# Stage: lbm
# ──────────────────────────────────────────────────────────────────────

def stage_lbm(args, device):
    cube_dir = Path(args.out) / 'cubes' / args.domain
    lbm_dir = Path(args.out) / 'lbm' / args.domain
    lbm_dir.mkdir(parents=True, exist_ok=True)
    bins = sorted(cube_dir.glob('*.bin'))
    if not bins:
        raise SystemExit(f'no cubes in {cube_dir} -- run --stage generate '
                         'first')
    print(f'[lbm] {len(bins)} cubes | voxel {args.voxel_um} um | '
          f'n_steps={args.n_steps}')
    for b in bins:
        out_json = lbm_dir / f'{b.stem}_lbm.json'
        if out_json.exists() and not args.overwrite:
            print(f'  [skip] {b.stem}')
            continue
        t0 = time.time()
        summ = run_lbm_cube(b, args.n_steps, args.voxel_um, device,
                            tau=args.tau, body_force=args.body_force)
        summ['domain'] = args.domain
        with open(out_json, 'w') as f:
            json.dump(summ, f, indent=2, default=str)
        print(f'  [done] {b.stem} k_trace={summ["k_trace_mD"]:.2f}mD '
              f'({time.time() - t0:.0f}s)')


# ──────────────────────────────────────────────────────────────────────
# Stage: aggregate
# ──────────────────────────────────────────────────────────────────────

STEM_RE = re.compile(
    r'^(?P<domain>[A-Za-z0-9]+)_(?P<method>.+?)(?:_s(?P<seed>\d{4}))?'
    r'_cube(?P<cube>\d+)$')


def parse_stem(stem):
    m = STEM_RE.match(stem)
    if not m:
        return None
    return (m.group('domain'), m.group('method'), m.group('seed') or '',
            int(m.group('cube')))


def stage_aggregate(args):
    import pandas as pd
    root = Path(args.out) / 'lbm'
    rows = []
    for p in sorted(root.glob('**/*_lbm.json')):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f'[WARN] unreadable {p}: {e}', file=sys.stderr)
            continue
        parsed = parse_stem(d.get('cube_stem', p.stem.replace('_lbm', '')))
        if parsed is None:
            print(f'[WARN] unparsable stem {p}', file=sys.stderr)
            continue
        domain, method, seed, cube = parsed
        k_axis = d.get('k_per_axis_mD') or {}
        base = dict(domain=domain, method=method, seed=seed, cube=cube,
                    porosity=d.get('porosity_total'),
                    k_trace_mD=d.get('k_trace_mD'),
                    k_aniso_pct=d.get('k_anisotropy_pct'))
        if k_axis:
            for ax, k in k_axis.items():
                rows.append({**base, 'axis': str(ax), 'k_mD': k})
        else:
            rows.append({**base, 'axis': '', 'k_mD': float('nan')})
    if not rows:
        print('No LBM JSONs found -- run the lbm stage first.')
        return 1
    df = pd.DataFrame(rows).drop_duplicates(
        subset=['domain', 'method', 'seed', 'cube', 'axis'], keep='first')
    long_p = Path(args.out) / 'lbm_all_long.csv'
    df.to_csv(long_p, index=False)
    print(f'[WROTE] {long_p} ({len(df)} rows)')

    # Rebuild the trace from the per-axis k_mD rows as the mean over the
    # (up to 3) available axes, UNIFORMLY -- never trust a stored per-file
    # k_trace field, which can silently be a single-axis value.
    ax = df[df['axis'].astype(str).str.contains('axis')].copy()
    per_cube = (ax.groupby(['domain', 'method', 'seed', 'cube'])
                  .agg(trace_mD=('k_mD', 'mean'), n_axes=('k_mD', 'size'),
                       porosity=('porosity', 'first')).reset_index())
    incomplete = per_cube[per_cube.n_axes < 3]
    if len(incomplete):
        print(f'[WARN] {len(incomplete)} cube cells have <3 axes '
              f'(non-percolating?):')
        print(incomplete[['domain', 'method', 'cube',
                          'n_axes']].to_string(index=False))
    tr = per_cube
    gt = tr[tr.method == 'GT'].set_index(['domain', 'cube'])['trace_mD']
    rec = []
    for (dom, meth, seed), g in tr[tr.method != 'GT'].groupby(
            ['domain', 'method', 'seed']):
        errs = []
        for _, r in g.iterrows():
            kgt = gt.get((dom, r.cube))
            if kgt and kgt > 0 and pd.notna(r.trace_mD):
                errs.append(abs(r.trace_mD - kgt) / kgt * 100)
        if errs:
            rec.append(dict(domain=dom, method=meth, seed=seed,
                            n_cubes=len(errs),
                            k_err_pct_mean=sum(errs) / len(errs),
                            k_err_pct_max=max(errs)))
    summ = pd.DataFrame(rec).sort_values(['domain', 'method'])
    summ_p = Path(args.out) / 'lbm_summary.csv'
    summ.to_csv(summ_p, index=False)
    print(f'[WROTE] {summ_p}')
    print(summ.to_string(index=False))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['generate', 'lbm', 'aggregate', 'all'],
                    required=True)
    ap.add_argument('--volume_path', type=str, default=None)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--domain', type=str, default='BB')
    ap.add_argument('--voxel_um', type=float, default=2.25)
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='UNetG checkpoint for maps_z / maps_tri')
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS,
                    choices=ALL_METHODS)
    ap.add_argument('--cubes_per_domain', type=int, default=1,
                    help='1 = center cube only (Table S15 uses 1-3 per domain)')
    ap.add_argument('--n_steps', type=int, default=5000)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--body_force', type=float, default=1e-5)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', type=str, default='outputs/analysis/lbm_8domain')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    if args.stage in ('generate', 'lbm', 'all') and not args.volume_path:
        raise SystemExit('--volume_path required for generate/lbm stages')

    device = torch.device(f'cuda:{args.gpu}'
                          if torch.cuda.is_available() else 'cpu')

    if args.stage in ('generate', 'all'):
        stage_generate(args, device)
    if args.stage in ('lbm', 'all'):
        stage_lbm(args, device)
    if args.stage in ('aggregate', 'all'):
        stage_aggregate(args)


if __name__ == '__main__':
    main()
