#!/usr/bin/env python3
"""
Stage-1 x Stage-2 seed-matrix evaluation under deployment parity
(Table S2 of the Supplement; source of the reported robustness number
dphi = 0.00134 +/- 0.00033 over the 3x3 = 9 independent initializations).

Each cell of the matrix is one MAPS pipeline trained with an independent
(Stage-1 seed, Stage-2 seed) pair. This driver evaluates every provided
checkpoint with tri_mean aggregation on the 3 BB benchmark cubes under both
the all-replacement and the deployment-parity protocol (n = 9 checkpoints x
3 cubes = 27 evaluations per protocol). Under the parity protocol the
acquired even-z slices are pasted back after aggregation, and metrics are
computed on the final deployed volume (acquired + synthesized slices), not
on the synthesized odd slices alone.

Checkpoints are passed either as label=path pairs:
  --checkpoints s1_2025_s2_2025=runs/a/best.pt s1_2025_s2_2026=runs/b/best.pt ...
or via a two-column TSV manifest (label <tab> checkpoint path):
  --manifest matrix_manifest.tsv

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/parity_matrix_eval.py \\
      --volume_path data/BB_1000c_f32.bin \\
      --manifest matrix_manifest.tsv \\
      --out_csv outputs/analysis/parity_matrix.csv
"""
import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, OFFSETS_IN6  # noqa: E402
from maps.models import UNetG  # noqa: E402
from maps.checkpoint import extract_model_state, load_state_checked  # noqa: E402
from maps.triaxis import (reconstruct_axis, parity_paste,  # noqa: E402
                          aggregate_tri_mean, metrics_from_cube)
from analysis.benchmark_cubes import get_cubes, CUBE_ZHW  # noqa: E402


def load_cells(args):
    """Return list of (label, ckpt_path)."""
    cells = []
    if args.manifest:
        for line in Path(args.manifest).read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            cells.append((parts[0], parts[1]))
    for spec in (args.checkpoints or []):
        label, path = spec.split('=', 1)
        cells.append((label, path))
    if not cells:
        raise SystemExit('provide --manifest or --checkpoints')
    return cells


def main():
    ap = argparse.ArgumentParser(
        description='Stage1 x Stage2 parity matrix (Table S2)')
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--manifest', type=str, default=None,
                    help='TSV: label <tab> checkpoint path')
    ap.add_argument('--checkpoints', nargs='+', default=None,
                    help='label=path pairs')
    ap.add_argument('--cubes_n', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_csv', type=str,
                    default='outputs/analysis/parity_matrix.csv')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    cells = load_cells(args)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape, n=args.cubes_n)
    Dc, Hc, Wc = CUBE_ZHW
    print(f'cells: {[c[0] for c in cells]}')
    print(f'cubes: {cubes}')

    rows = []
    for label, ck_path in cells:
        if not Path(ck_path).exists():
            print(f'SKIP {label}: missing {ck_path}')
            continue
        G = UNetG(in_ch=6, base=80).to(device)
        c = torch.load(ck_path, map_location=device)
        load_state_checked(G, extract_model_state(c), label=str(ck_path))
        G.eval()
        for (z0, y0, x0, lab) in cubes:
            cube = np.clip(np.ascontiguousarray(
                vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32),
                0, 1)
            gt = torch.from_numpy(cube).float()
            V_z = reconstruct_axis(G, cube, 'z', OFFSETS_IN6, device)
            V_x = reconstruct_axis(G, cube, 'x', OFFSETS_IN6, device)
            V_y = reconstruct_axis(G, cube, 'y', OFFSETS_IN6, device)
            V_m = aggregate_tri_mean(V_z, V_x, V_y)[0]
            for proto, V_p in [('allrep', V_m),
                               ('parity', parity_paste(V_m, gt, z0))]:
                m = metrics_from_cube(V_p, gt, device=device)
                rows.append(dict(label=label, cube=lab, protocol=proto,
                                 ssim=m['ssim_z'], dphi=m['dphi'],
                                 dsa=m['dsa'],
                                 d_euler_per_mpx=m['d_euler_per_mpx']))
            print(f'[{datetime.now():%H:%M:%S}] {label}/{lab}')
        del G
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit('no rows produced (all checkpoints missing?)')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    for proto in ['allrep', 'parity']:
        d = [float(r['dphi']) for r in rows if r['protocol'] == proto]
        s = [float(r['ssim']) for r in rows if r['protocol'] == proto]
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        print(f'MATRIX {proto}: dphi {statistics.mean(d):.5f}+/-{sd:.5f}  '
              f'ssim {statistics.mean(s):.4f}  n={len(d)}')
    print('->', out_csv)


if __name__ == '__main__':
    main()
