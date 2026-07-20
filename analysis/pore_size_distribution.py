#!/usr/bin/env python3
"""
Pore-size distribution f(r) via the Euclidean distance transform (Fig. S3).

Definition (matching the paper's caption): binarize at --threshold
(solid > tau), take the pore mask, compute the Euclidean distance from each
pore voxel to the nearest solid voxel
(`scipy.ndimage.distance_transform_edt`), histogram those distances in
log-spaced bins from 0.5 voxel to the maximum distance, and normalize the
counts by the bin widths to obtain a per-unit-radius frequency (PDF) f(r).
Curves are plotted log-log; in the paper the MAPS reconstructions overlay
the GT curve across nearly four decades.

Inputs: one or more labeled volumes -- raw float32 .bin cubes (--cube_size)
or .npy arrays. The GT cube can also be cut directly from a full volume
(--volume_path with --cube_origin/--cube_size), which reproduces the
center-cube convention of the figure.

Usage:
  python analysis/pore_size_distribution.py \\
      --inputs GT=outputs/cubes/BB_gt_center.npy \\
               tri_mean=outputs/cubes/BB_maps_tri_mean_center.npy \\
      --out_csv outputs/analysis/psd_BB.csv --fig outputs/figures/psd_BB.png
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume  # noqa: E402


def compute_psd(pore_mask: np.ndarray, n_bins: int = 30):
    """Log-binned pore-size PDF from the EDT of the pore space.

    Returns (bin centers in voxel units, PDF values per unit radius).
    """
    if pore_mask.sum() == 0:
        return np.array([1.0]), np.array([0.0])
    dist = ndi.distance_transform_edt(pore_mask)
    pore_d = dist[pore_mask]
    r_max = float(pore_d.max())
    if r_max <= 0:
        return np.array([1.0]), np.array([0.0])
    edges = np.geomspace(max(0.5, 1e-3), r_max + 1e-6, n_bins + 1)
    counts, edges = np.histogram(pore_d, bins=edges, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    pdf = counts / (counts.sum() * widths + 1e-12)
    return centers, pdf


def load_labeled_volume(path: Path, cube_size):
    if path.suffix == '.npy':
        return np.load(path).astype(np.float32)
    arr = np.fromfile(path, dtype=np.float32)
    n = arr.size
    if cube_size is not None:
        return arr.reshape(cube_size, cube_size, cube_size)
    side = round(n ** (1.0 / 3.0))
    if side ** 3 != n:
        raise SystemExit(f'{path}: not a cube ({n} voxels); pass --cube_size')
    return arr.reshape(side, side, side)


def main():
    ap = argparse.ArgumentParser(
        description='EDT pore-size distribution f(r) (Fig. S3)')
    ap.add_argument('--inputs', nargs='+', default=[],
                    help='label=path pairs (.npy or float32 .bin cubes)')
    ap.add_argument('--cube_size', type=int, default=None,
                    help='side length for raw .bin cubes')
    ap.add_argument('--volume_path', default=None,
                    help='optionally cut a GT cube from a full volume')
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--cube_origin', nargs=3, type=int, default=None,
                    metavar=('Z0', 'Y0', 'X0'))
    ap.add_argument('--gt_cube_size', type=int, default=256)
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='solid = value > threshold; pore = complement')
    ap.add_argument('--n_bins', type=int, default=30)
    ap.add_argument('--out_csv', default='outputs/analysis/psd.csv')
    ap.add_argument('--fig', default=None,
                    help='optional log-log overlay figure (.png/.pdf)')
    args = ap.parse_args()

    volumes = []  # (label, ndarray)
    if args.volume_path:
        vol = load_volume(args.volume_path, tuple(args.volume_shape))
        if args.cube_origin is None:
            c = args.gt_cube_size
            z0 = vol.shape[0] - c
            y0 = (vol.shape[1] - c) // 2
            x0 = (vol.shape[2] - c) // 2
        else:
            z0, y0, x0 = args.cube_origin
        c = args.gt_cube_size
        gt = np.ascontiguousarray(
            vol[z0:z0 + c, y0:y0 + c, x0:x0 + c]).astype(np.float32)
        volumes.append(('GT', gt))
        print(f'[GT] cube origin=({z0},{y0},{x0}) size={c}')
    for spec in args.inputs:
        if '=' in spec:
            label, path = spec.split('=', 1)
        else:
            label, path = Path(spec).stem, spec
        volumes.append((label, load_labeled_volume(Path(path),
                                                   args.cube_size)))
    if not volumes:
        raise SystemExit('no inputs (pass --inputs and/or --volume_path)')

    rows = []
    curves = []
    for label, arr in volumes:
        pore = ~(arr > args.threshold)
        r, pdf = compute_psd(pore, n_bins=args.n_bins)
        curves.append((label, r, pdf))
        print(f'  {label:24s} pore fraction={pore.mean():.4f}  '
              f'r_max={r[-1]:.2f} vox')
        for ri, pi in zip(r, pdf):
            rows.append({'label': label, 'r_voxel': ri, 'pdf': pi})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['label', 'r_voxel', 'pdf'])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'[saved] {out_csv} ({len(rows)} rows)')

    if args.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        for label, r, pdf in curves:
            m = (pdf > 0) & (r > 0)
            lw = 1.6 if label == 'GT' else 1.0
            color = 'k' if label == 'GT' else None
            ax.plot(r[m], pdf[m], lw=lw, color=color, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('pore radius (voxel)')
        ax.set_ylabel('frequency  $f(r)$')
        ax.legend(frameon=False, fontsize=7)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig_path = Path(args.fig)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300)
        print(f'[saved] {fig_path}')


if __name__ == '__main__':
    main()
