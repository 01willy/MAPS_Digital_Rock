#!/usr/bin/env python3
"""
TEST-ONLY multi-seed LBM permeability comparison at k=1
(Table 3 of the paper; the per-axis breakdown is Table S10).

Protocol
--------
* TEST-ONLY cubes: split = train [0,700) / val [700,850) / test [850,1000).
  A 256^3 cube cannot fit the 150-slice test band, so CUBE_SIZE=128 anchored
  at z0=850 -> cube spans z=[850,978), fully inside the test split. Multiple
  cubes per domain vary (y0,x0) at fixed z0 (seeded; see
  analysis/benchmark_cubes.testonly_origins).
* k=1 parity: EVEN z slices are kept as GT anchors, ODD z slices are the
  targets each method reconstructs. Boundary odd slices without full offset
  support keep GT. All reconstructions binarized at 0.5.
* LBM: D3Q19 BGK, n_steps=5000, tau=1.0, body_force=1e-5, all 3 axes
  (identical to the 8-domain campaign of Table S15, so numbers are directly
  comparable).
* Primary metric: k_zz = interpolation-axis permeability error vs. the GT
  cube's LBM. Also recorded: k_trace (mean of k_x/k_y/k_z).

Methods: GT (reference), MAPS (2D UNetG, tri-offsets, odd-z replacement),
b4 (2D U-Net L1, same operator), b5 (3D U-Net fair single-target).
Checkpoints are passed per seed, aligned with --seeds:

  --seeds 2025 2026 2027 \\
  --maps_ckpts sA.pt sB.pt sC.pt --b4_ckpts ... --b5_ckpts ...

Cross-domain rows of Table 3 (Bentheimer/CastleGate/Ketton): run once per
domain volume with the SAME BB-trained checkpoints (zero-shot) and the
domain's voxel size.

Aggregate the per-cube JSONs with analysis/lbm_multiseed_aggregate.py.

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/lbm_multiseed_eval.py \\
      --volume_path data/BB_1000c_f32.bin --domain BB --voxel_um 2.25 \\
      --n_cubes 4 --seeds 2025 2026 2027 \\
      --maps_ckpts m25.pt m26.pt m27.pt \\
      --b4_ckpts b25.pt b26.pt b27.pt \\
      --b5_ckpts c25.pt c26.pt c27.pt \\
      --out outputs/analysis/lbm_testonly
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, compute_splits, OFFSETS_IN6  # noqa: E402
from lbm.d3q19 import D3Q19LBM  # noqa: E402
from analysis.benchmark_cubes import testonly_origins  # noqa: E402
from analysis.inference_tiled import (lock_determinism, load_unetg,  # noqa: E402
                                      load_unet3d, recon_unetg_z,
                                      recon_unet3d_z)

CUBE_SIZE = 128


def run_lbm_cube(cube_bin, n_steps, voxel_um, device, tau=1.0,
                 body_force=1e-5, axes=(0, 1, 2)):
    """Run D3Q19 LBM on a cube .bin over `axes`. Returns summary dict with
    k_per_axis_mD (axis_0 = z = k_zz, the primary metric), k_trace_mD,
    porosity, anisotropy."""
    arr = np.fromfile(cube_bin, dtype=np.float32)
    side = int(round(arr.size ** (1 / 3)))
    assert side ** 3 == arr.size, f'{cube_bin}: not cubic ({arr.size})'
    cube = arr.reshape(side, side, side)
    solid = cube > 0.5
    per_axis = {}
    for ax in axes:
        sim = D3Q19LBM(solid, device=device, tau=tau, body_force=body_force,
                       flow_axis=ax)
        t0 = time.time()
        for _ in range(n_steps):
            sim.step()
        r = sim.permeability(voxel_size_um=voxel_um)
        r['wall_seconds'] = round(time.time() - t0, 1)
        r['flow_axis'] = ax
        per_axis[f'axis_{ax}'] = r
        del sim
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    ks = [per_axis[f'axis_{a}']['k_mD'] for a in axes]
    return {
        'cube_path': str(cube_bin),
        'cube_stem': Path(cube_bin).stem,
        'cube_side': side,
        'n_steps': n_steps,
        'porosity_total': per_axis[f'axis_{axes[0]}']['porosity_total'],
        'k_per_axis_mD': {f'axis_{a}': per_axis[f'axis_{a}']['k_mD']
                          for a in axes},
        'k_zz_mD': per_axis['axis_0']['k_mD'] if 0 in axes else None,
        'k_trace_mD': float(np.mean(ks)),
        'k_anisotropy_pct': float(np.std(ks) / max(np.mean(ks), 1e-30)
                                  * 100.0),
        'per_axis_full': per_axis,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--domain', type=str, default='BB',
                    help='Label used in file stems and metadata')
    ap.add_argument('--voxel_um', type=float, default=2.25,
                    help='Voxel size in micrometers (k scales with voxel^2)')
    ap.add_argument('--n_cubes', type=int, default=4)
    ap.add_argument('--seeds', nargs='+', type=int, default=[2025, 2026, 2027])
    ap.add_argument('--maps_ckpts', nargs='+', default=[],
                    help='MAPS checkpoints, one per seed (in --seeds order)')
    ap.add_argument('--b4_ckpts', nargs='+', default=[],
                    help='b4 checkpoints, one per seed')
    ap.add_argument('--b5_ckpts', nargs='+', default=[],
                    help='b5 (3D U-Net fair) checkpoints, one per seed')
    ap.add_argument('--b5_base', type=int, default=24)
    ap.add_argument('--n_steps', type=int, default=5000)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--body_force', type=float, default=1e-5)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', type=str, default='outputs/analysis/lbm_testonly')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    for name, cks in (('maps', args.maps_ckpts), ('b4', args.b4_ckpts),
                      ('b5', args.b5_ckpts)):
        if cks and len(cks) != len(args.seeds):
            raise SystemExit(f'--{name}_ckpts must have one entry per seed '
                             f'({len(args.seeds)} seeds, {len(cks)} ckpts)')

    lock_determinism(2025)
    device = torch.device(f'cuda:{args.gpu}')
    out_root = Path(args.out)
    cube_dir = out_root / 'cubes' / args.domain
    lbm_dir = out_root / 'lbm' / args.domain
    cube_dir.mkdir(parents=True, exist_ok=True)
    lbm_dir.mkdir(parents=True, exist_ok=True)

    splits = compute_splits(args.volume_shape[0])
    print(f'[split] test-only z-range = {splits["test"]} '
          f'(val={splits["val"]})')
    origins = testonly_origins(tuple(args.volume_shape), CUBE_SIZE,
                               args.n_cubes, splits)

    # Load models per seed (BB-trained; cross-domain use = zero-shot).
    maps_G = {s: load_unetg(c, device)
              for s, c in zip(args.seeds, args.maps_ckpts)}
    b4_G = {s: load_unetg(c, device)
            for s, c in zip(args.seeds, args.b4_ckpts)}
    b5_G = {s: load_unet3d(c, device, base=args.b5_base)
            for s, c in zip(args.seeds, args.b5_ckpts)}
    print(f'[ckpt] loaded maps={len(maps_G)} b4={len(b4_G)} b5={len(b5_G)} '
          f'onto {device}')

    print(f'\n== {args.domain}: load {args.volume_path}  '
          f'test_z={splits["test"]}  voxel={args.voxel_um}um')
    t0 = time.time()
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    print(f'   loaded in {time.time() - t0:.0f}s ; origins={origins}')

    meta = {'domain': args.domain, 'volume_path': args.volume_path,
            'cube_size': CUBE_SIZE, 'voxel_size_um': args.voxel_um,
            'test_z_range': list(splits['test']),
            'val_z_range': list(splits['val']),
            'origins': origins, 'seeds': args.seeds,
            'protocol': 'k=1 parity (even-z GT anchors, odd-z targets); '
                        'boundary keeps GT; binarize >0.5; TEST-ONLY cubes',
            'lbm': f'D3Q19 BGK n_steps={args.n_steps} tau={args.tau} '
                   f'body_force={args.body_force} axes=z,y,x',
            'primary_metric': 'k_zz (axis_0 = interpolation axis)',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'cells': []}

    t_all = time.time()
    for ci, (z0, y0, x0) in enumerate(origins):
        gt = np.ascontiguousarray(
            vol[z0:z0 + CUBE_SIZE, y0:y0 + CUBE_SIZE,
                x0:x0 + CUBE_SIZE]).astype(np.float32)
        gt = np.clip(gt, 0.0, 1.0)
        gt_phi = float(1.0 - gt.mean())
        print(f'- cube{ci} origin=({z0},{y0},{x0}) GT phi={gt_phi:.4f}')

        # (stem, kind, seed) -- GT once; maps/b4/b5 per seed.
        cells = [(f'{args.domain}_GT_cube{ci}', None, None)]
        for s in args.seeds:
            if s in maps_G:
                cells.append((f'{args.domain}_maps_s{s}_cube{ci}', 'maps', s))
            if s in b4_G:
                cells.append((f'{args.domain}_b4_unet2d_l1_s{s}_cube{ci}',
                              'b4', s))
            if s in b5_G:
                cells.append((f'{args.domain}_b5_unet3d_fair_s{s}_cube{ci}',
                              'b5', s))

        for stem, kind, seed in cells:
            out_bin = cube_dir / f'{stem}.bin'
            out_json = lbm_dir / f'{stem}_lbm.json'
            if out_json.exists() and not args.overwrite:
                print(f'    [skip ] {stem} (lbm json exists)')
                continue
            t1 = time.time()
            if kind is None:                       # GT
                rec = gt.astype(np.float32)
            elif kind == 'maps':
                rec = (recon_unetg_z(maps_G[seed], gt, OFFSETS_IN6, device)
                       > 0.5).astype(np.float32)
            elif kind == 'b4':
                rec = (recon_unetg_z(b4_G[seed], gt, OFFSETS_IN6, device)
                       > 0.5).astype(np.float32)
            elif kind == 'b5':
                rec = (recon_unet3d_z(b5_G[seed], gt, OFFSETS_IN6, device)
                       > 0.5).astype(np.float32)
            else:
                raise ValueError(kind)
            rec.tofile(out_bin)
            phi = float(1.0 - rec.mean())
            recon_s = time.time() - t1

            # LBM
            t2 = time.time()
            summ = run_lbm_cube(out_bin, args.n_steps, args.voxel_um, device,
                                tau=args.tau, body_force=args.body_force)
            summ.update({'domain': args.domain, 'method': kind or 'GT',
                         'seed': seed, 'cube': ci, 'origin': [z0, y0, x0],
                         'phi': phi, 'gt_phi': gt_phi,
                         'recon_seconds': round(recon_s, 1)})
            with open(out_json, 'w') as f:
                json.dump(summ, f, indent=2, default=str)
            meta['cells'].append({'stem': stem, 'method': kind or 'GT',
                                  'seed': seed, 'cube': ci, 'phi': phi,
                                  'k_zz_mD': summ['k_zz_mD'],
                                  'k_trace_mD': summ['k_trace_mD']})
            print(f'    [done ] {stem} phi={phi:.4f} '
                  f'k_zz={summ["k_zz_mD"]:.2f}mD '
                  f'k_trace={summ["k_trace_mD"]:.2f}mD '
                  f'(recon {recon_s:.1f}s + lbm {time.time() - t2:.1f}s)')

    with open(cube_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f'   [meta] {cube_dir / "metadata.json"}')
    print(f'\n[DONE] total {time.time() - t_all:.0f}s -> {out_root}')


if __name__ == '__main__':
    main()
