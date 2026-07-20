#!/usr/bin/env python3
"""
Classical interpolation baselines B1 / B2 / B3 (no learning).

Paper naming (Section 4.6; Tables S5-S7):
  B1 = Linear k=1 : each odd-z slice = average of the immediate even
       neighbors (z-1, z+1).
  B2 = Linear k=3 : inverse-distance weighted average of the 3 closest even
       slices on each side.
  B3 = Cubic k=3  : 1D cubic spline along z anchored on all even slices.

All three operate under the deployment-parity convention of the k=1 sparse
scenario: even-index slices are the acquired data and are kept verbatim;
only odd-index slices are synthesized.

`linear_recon_axis` is the per-axis parity linear operator used to build the
tri-axis linear reference ("linear_k1_tri" of Table S15 and the k-sweep
figure): apply along z, x, y and average the three volumes.

Running this file evaluates B1/B2/B3 (z-axis) and tri-axis linear on the
paper's benchmark cubes with the slice-metric suite.

Usage:
  python baselines/classical.py --volume_path data/BB_1000c_f32.bin \\
      --out_dir outputs/baselines/classical
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, compute_splits  # noqa: E402
from maps.triaxis import metrics_from_cube  # noqa: E402


# ──────────────── B1 / B2 / B3 constructors (z-axis) ────────────────

def classical_linear_k1(gt_cube: np.ndarray) -> np.ndarray:
    """B1: predict odd-z slices as average of immediate even neighbors.
    Even slices are kept as-is (known input)."""
    Z = gt_cube.shape[0]
    pred = gt_cube.copy()
    for z in range(1, Z - 1, 2):
        pred[z] = 0.5 * (gt_cube[z - 1] + gt_cube[z + 1])
    return pred


def classical_linear_k3(gt_cube: np.ndarray) -> np.ndarray:
    """B2: 1D linear interpolation along z using k=3 even neighbors
    (inverse-distance weighted average of the 3 closest even slices on
    each side)."""
    Z = gt_cube.shape[0]
    pred = gt_cube.copy()
    for z in range(1, Z - 1, 2):
        even_below = [z - d for d in (1, 3, 5) if (z - d) >= 0 and (z - d) % 2 == 0]
        even_above = [z + d for d in (1, 3, 5) if (z + d) < Z and (z + d) % 2 == 0]
        neighbors = even_below + even_above
        weights = [1.0 / max(abs(z - n), 1) for n in neighbors]
        wsum = sum(weights)
        pred[z] = sum(w * gt_cube[n] for w, n in zip(weights, neighbors)) / wsum
    return pred


def classical_cubic_k3(gt_cube: np.ndarray) -> np.ndarray:
    """B3: 1D cubic-spline interpolation along z, using all even slices as
    anchors."""
    from scipy.interpolate import CubicSpline
    Z = gt_cube.shape[0]
    even_z = np.arange(0, Z, 2)
    flat = gt_cube[even_z].reshape(len(even_z), -1)
    cs = CubicSpline(even_z, flat, axis=0, extrapolate=False)
    pred = gt_cube.copy().reshape(Z, -1)
    odd_z = np.arange(1, Z - 1, 2)
    pred[odd_z] = cs(odd_z)
    pred = pred.reshape(gt_cube.shape)
    return np.clip(pred, 0.0, 1.0)


# ──────────────── per-axis parity linear (tri-axis linear) ────────────────

def linear_recon_axis(cube: np.ndarray, axis: str, k: int = 1) -> np.ndarray:
    """Classical linear along one axis, parity protocol: odd targets t in
    [k, n-k) replaced by 0.5*(GT[t-k] + GT[t+k]); everything else (even
    slices and boundary odd slices) keeps GT. Vectorized numpy; returns
    float32 (continuous; binarize after aggregation where required)."""
    D, H, W = cube.shape
    out = cube.copy()
    n = {'z': D, 'y': H, 'x': W}[axis]
    ts = np.array([t for t in range(k, n - k) if t % 2 == 1], dtype=np.int64)
    if ts.size == 0:
        return out
    if axis == 'z':
        out[ts] = 0.5 * (cube[ts - k] + cube[ts + k])
    elif axis == 'y':
        out[:, ts, :] = 0.5 * (cube[:, ts - k, :] + cube[:, ts + k, :])
    else:
        out[:, :, ts] = 0.5 * (cube[:, :, ts - k] + cube[:, :, ts + k])
    return out


def tri_axis_linear_k1(cube: np.ndarray) -> np.ndarray:
    """Tri-axis linear reference: per-axis parity linear along z, x, y,
    then voxel-wise mean of the three volumes (continuous)."""
    V_z = linear_recon_axis(cube, 'z', k=1)
    V_x = linear_recon_axis(cube, 'x', k=1)
    V_y = linear_recon_axis(cube, 'y', k=1)
    return (V_z + V_x + V_y) / 3.0


# ──────────────── benchmark-cube evaluation ────────────────

CUBE_ZHW = (128, 256, 256)
SEED_FOR_CUBES = 2025


def get_cubes(vol_shape, n=3):
    """Benchmark cube origins (identical to the MAPS evaluation)."""
    Z, Y, X = vol_shape
    Dc, Hc, Wc = CUBE_ZHW
    splits = compute_splits(Z)
    z_lo, z_hi = splits['test']
    Dc = min(Dc, z_hi - z_lo)
    rng = np.random.default_rng(SEED_FOR_CUBES)
    z0_c = z_lo + (z_hi - z_lo - Dc) // 2
    y0_c = (Y - Hc) // 2
    x0_c = (X - Wc) // 2
    cubes = [(z0_c, y0_c, x0_c, 'center')]
    while len(cubes) < n:
        z0 = int(rng.integers(z_lo, z_hi - Dc))
        y0 = int(rng.integers(0, Y - Hc))
        x0 = int(rng.integers(0, X - Wc))
        ok = True
        for (zp, yp, xp, _) in cubes:
            if (abs(zp - z0) < Dc // 2 and abs(yp - y0) < Hc // 2
                    and abs(xp - x0) < Wc // 2):
                ok = False
                break
        if ok:
            cubes.append((z0, y0, x0, f'rand{len(cubes) - 1}'))
    return cubes


METHODS = {
    'classical_b1': classical_linear_k1,
    'classical_b2': classical_linear_k3,
    'classical_b3': classical_cubic_k3,
    'linear_k1_tri': tri_axis_linear_k1,
}


def main():
    ap = argparse.ArgumentParser(description='Classical baselines B1/B2/B3 eval')
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int, default=[1000, 1000, 1000])
    ap.add_argument('--methods', nargs='+', default=list(METHODS),
                    choices=list(METHODS))
    ap.add_argument('--cubes_n', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0,
                    help='GPU for the SSIM/morphology metric suite')
    ap.add_argument('--out_dir', type=str, default='outputs/baselines/classical')
    args = ap.parse_args()

    device = (torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available()
              else torch.device('cpu'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape, n=args.cubes_n)
    print(f'cubes: {cubes}')

    rows = []
    for (z0, y0, x0, label) in cubes:
        Dc, Hc, Wc = CUBE_ZHW
        cube = np.ascontiguousarray(
            vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32)
        cube = np.clip(cube, 0.0, 1.0)
        gt_t = torch.from_numpy(cube).float()
        for method in args.methods:
            t0 = time.time()
            pred = METHODS[method](cube)
            t_recon = time.time() - t0
            m = metrics_from_cube(torch.from_numpy(pred).float(), gt_t,
                                  device=device)
            rows.append({
                'method': method, 'cube': label, 'cube_z0': z0,
                'ssim': m['ssim_z'], 'psnr': m['psnr_z'],
                'dphi': m['dphi'], 'dsa': m['dsa'],
                'd_euler_per_mpx': m['d_euler_per_mpx'],
                'xz_ssim': m['xz_ssim_mean'], 'yz_ssim': m['yz_ssim_mean'],
                'recon_seconds': round(t_recon, 2),
            })
            print(f'  [{label}] {method:14s} SSIM={m["ssim_z"]:.4f} '
                  f'dphi={m["dphi"]:.5f} ({t_recon:.1f}s)')

    csv_path = out_dir / 'classical_eval.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[SAVED] {csv_path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
