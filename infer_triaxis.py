#!/usr/bin/env python3
"""
Tri-axis inference: reconstruct a volume with a trained MAPS checkpoint
along z, x, and y, then fuse the three reconstructions with a GT-free
aggregation rule (default: tri_mean, the paper's recommended deployable
aggregation).

Protocols:
  --parity (default ON): deployment-parity protocol. Acquired even-index
      z-slices (global parity) are pasted back after aggregation, so the
      output contains model prediction only where data was missing. With
      --metrics, metrics are computed on this final deployed volume after
      the parity paste (acquired even-z slices restored; odd-z slices
      contain model predictions) — not on odd slices only. This is the
      protocol behind the paper's reported numbers.
  --no_parity: all-replacement protocol (every interior slice replaced by
      the model output; used for some ablation comparisons).
  --sequential: strictly-sequential GT-free protocol (Table S4). The parity
      z pass fills the odd (missing) z slices first (boundary odd slices
      filled GT-free by linear interpolation of the acquired neighbours);
      the x/y passes then run on the z-filled volume, so no pass ever reads
      ground truth at an unacquired position. No parity paste is applied
      afterwards (the aggregation itself is the deployable output).

By default the script cuts the paper's benchmark cube: a 128x256x256
center crop of the test z-slab [850, 1000) of a 1000^3 volume, and reports
metrics against the ground-truth cube. Use --cube_origin / --cube_size to
choose a different region.

Usage:
  CUDA_VISIBLE_DEVICES=0 python infer_triaxis.py \\
      --checkpoint outputs/stage2/<run>/checkpoints/best.pt \\
      --volume_path data/BB_1000c_f32.bin \\
      --agg tri_mean --out outputs/recon_BB_trimean.npy --metrics
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maps.models import UNetG
from maps.data import load_volume, compute_splits, OFFSETS_IN6
from maps.triaxis import (reconstruct_axis, parity_paste, sequential_triaxis,
                          compute_all_gtfree_aggregations,
                          aggregate_tri_mean, metrics_from_cube)

AGG_CHOICES = ['tri_mean', 'tri_median', 'tri_consensus',
               'tri_weuler_self', 'tri_weuler', 'tri_voxel_consensus', 'z_only']
# 'tri_weuler' is the paper's name for the GT-free Euler-consensus rule
# implemented as 'tri_weuler_self'; accepted here as an alias (mapped below).
AGG_ALIAS = {'tri_weuler': 'tri_weuler_self'}


def load_generator(ckpt_path, device, in_ch=6, use_ema=True):
    ck = torch.load(ckpt_path, map_location=device)
    base_ch = 80
    if isinstance(ck, dict) and 'config' in ck and isinstance(ck['config'], dict):
        base_ch = ck['config'].get('base_ch', 80)
    G = UNetG(in_ch=in_ch, base=base_ch).to(device)
    if use_ema and isinstance(ck, dict) and 'ema_state_dict' in ck:
        G.load_state_dict(ck['ema_state_dict']['shadow'], strict=True)
        print(f'[LOAD] EMA weights from {ckpt_path} (base={base_ch})')
    elif isinstance(ck, dict) and 'model_state_dict' in ck:
        G.load_state_dict(ck['model_state_dict'], strict=True)
        print(f'[LOAD] model weights from {ckpt_path} (base={base_ch})')
    else:
        G.load_state_dict(ck, strict=True)
        print(f'[LOAD] direct state dict from {ckpt_path} (base={base_ch})')
    G.eval()
    return G


def main():
    ap = argparse.ArgumentParser(description='MAPS tri-axis inference')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--volume_path', required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--cube_origin', nargs=3, type=int, default=None,
                    metavar=('Z0', 'Y0', 'X0'),
                    help='Cube origin (default: center of the test z-slab)')
    ap.add_argument('--cube_size', nargs=3, type=int, default=[128, 256, 256],
                    metavar=('D', 'H', 'W'))
    ap.add_argument('--agg', default='tri_mean', choices=AGG_CHOICES)
    ap.add_argument('--offsets', type=int, nargs='+', default=None,
                    help='Input offsets (default: -15 -9 -3 3 9 15)')
    parity = ap.add_mutually_exclusive_group()
    parity.add_argument('--parity', dest='parity', action='store_true',
                        help='Deployment-parity: keep acquired even-z slices '
                             '(default). Metrics are then computed on the '
                             'final deployed volume after the parity paste '
                             '(even-z acquired, odd-z model predictions), '
                             'not on odd slices only')
    parity.add_argument('--no_parity', dest='parity', action='store_false',
                        help='All-replacement protocol')
    parity.add_argument('--sequential', action='store_true', default=False,
                        help='Strictly-sequential GT-free protocol: z-fill '
                             'first, then x/y passes on the filled volume '
                             '(Table S4)')
    ap.set_defaults(parity=True)
    ap.add_argument('--use_ema', action='store_true', default=True)
    ap.add_argument('--no_ema', dest='use_ema', action='store_false')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default=None,
                    help='Output volume path (.npy, float32). Optional.')
    ap.add_argument('--metrics', action='store_true',
                    help='Report SSIM/morphology metrics vs the GT cube')
    ap.add_argument('--metrics_json', default=None,
                    help='Save metrics to this JSON path')
    args = ap.parse_args()
    args.agg = AGG_ALIAS.get(args.agg, args.agg)  # map 'tri_weuler' -> 'tri_weuler_self'

    device = torch.device(f'cuda:{args.gpu}')
    offsets = args.offsets if args.offsets is not None else OFFSETS_IN6

    # ── Cut the evaluation cube ──
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    D, H, W = args.cube_size
    if args.cube_origin is not None:
        z0, y0, x0 = args.cube_origin
    else:
        splits = compute_splits(vol.shape[0])
        z_lo, z_hi = splits['test']
        D = min(D, z_hi - z_lo)
        z0 = z_lo + (z_hi - z_lo - D) // 2
        y0 = (vol.shape[1] - H) // 2
        x0 = (vol.shape[2] - W) // 2
    cube = np.ascontiguousarray(
        vol[z0:z0 + D, y0:y0 + H, x0:x0 + W]).astype(np.float32)
    cube = np.clip(cube, 0.0, 1.0)
    print(f'[CUBE] origin=({z0},{y0},{x0}) size=({D},{H},{W})')

    # ── Model ──
    G = load_generator(args.checkpoint, device, in_ch=len(offsets),
                       use_ema=args.use_ema)

    # ── Tri-axis reconstruction ──
    t0 = time.time()
    if args.sequential:
        V_z, V_x, V_y = sequential_triaxis(G, cube, offsets, device,
                                           args.batch_size)
        if args.agg == 'z_only':
            V_agg, info = V_z, {'agg': 'z_only (z-fill)'}
        elif args.agg == 'tri_mean':
            V_agg, info = aggregate_tri_mean(V_z, V_x, V_y)
        else:
            aggs = compute_all_gtfree_aggregations(V_z, V_x, V_y, device)
            V_agg, info = aggs[args.agg]
        V_agg = V_agg.clamp(0.0, 1.0)
    else:
        V_z = reconstruct_axis(G, cube, 'z', offsets, device, args.batch_size)
        if args.agg == 'z_only':
            V_agg, info = V_z, {'agg': 'z_only'}
        else:
            V_x = reconstruct_axis(G, cube, 'x', offsets, device,
                                   args.batch_size)
            V_y = reconstruct_axis(G, cube, 'y', offsets, device,
                                   args.batch_size)
            if args.agg == 'tri_mean':
                V_agg, info = aggregate_tri_mean(V_z, V_x, V_y)
            else:
                aggs = compute_all_gtfree_aggregations(V_z, V_x, V_y, device)
                V_agg, info = aggs[args.agg]
    t_recon = time.time() - t0
    protocol_name = ('sequential' if args.sequential
                     else ('parity' if args.parity else 'allrep'))
    print(f'[RECON] agg={args.agg} protocol={protocol_name} '
          f'took {t_recon:.1f}s  weights={info}')

    # ── Deployment-parity: paste back acquired even-z slices ──
    gt_t = torch.from_numpy(cube).float()
    if args.sequential:
        print('[SEQUENTIAL] z-filled first, x/y passes on the filled volume '
              '(no GT read at unacquired positions; no parity paste)')
    elif args.parity:
        V_agg = parity_paste(V_agg, gt_t, z0)
        print('[PARITY] acquired even-z slices restored '
              '(model output only at odd z)')

    # ── Save ──
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, V_agg.numpy().astype(np.float32))
        print(f'[SAVED] {out_path}')

    # ── Metrics vs GT ──
    # Computed on the final output volume V_agg — i.e. after the parity
    # paste when --parity is enabled (acquired even-z slices restored,
    # odd-z slices are model predictions), not on odd slices only.
    if args.metrics or args.metrics_json:
        m = metrics_from_cube(V_agg, gt_t, device=device)
        m['_agg'] = args.agg
        m['_protocol'] = protocol_name
        m['_recon_seconds'] = t_recon
        m['_cube'] = {'z0': z0, 'y0': y0, 'x0': x0, 'D': D, 'H': H, 'W': W}
        print('\n=== METRICS (vs GT cube) ===')
        for k, v in m.items():
            if not k.startswith('_'):
                print(f'  {k:18s}: {v:.6f}' if isinstance(v, float)
                      else f'  {k:18s}: {v}')
        if args.metrics_json:
            with open(args.metrics_json, 'w') as f:
                json.dump(m, f, indent=2, default=str)
            print(f'[SAVED] {args.metrics_json}')


if __name__ == '__main__':
    main()
