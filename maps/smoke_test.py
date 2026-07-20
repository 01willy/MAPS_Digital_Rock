#!/usr/bin/env python3
"""
CPU smoke test for the MAPS release code.

Runs the core inference pipeline on a small synthetic binary volume with a
randomly initialized generator — no checkpoints, no data files, no
downloads. Exercised stages:

  1. UNetG forward pass (paper config: in_ch=6, base=80)
  2. per-axis volume reconstruction (`reconstruct_axis`)
  3. deployment-parity paste (`parity_paste`)
  4. GT-free tri-mean aggregation (`aggregate_tri_mean`)
  5. GT-free Euler-weighted aggregation (`aggregate_tri_weuler_self`,
     the rule behind `infer_triaxis.py --agg tri_weuler_self`)
  6. cube-level metric suite (`metrics_from_cube`)

Each stage asserts basic shape/range/finiteness properties and prints a
PASS line. Exit code 0 means all stages passed. Metric values themselves
are meaningless here (random weights, random volume); only finiteness and
protocol behavior are checked.

Usage:
    python -m maps.smoke_test --device cpu
"""

import argparse
import sys
import time

try:
    import numpy as np
    import torch
except ImportError as e:  # pragma: no cover
    sys.exit(f'missing dependency ({e.name}); install with: pip install -r requirements.txt')

try:
    from .models import UNetG, count_parameters
    from .data import OFFSETS_IN6
    from .triaxis import (reconstruct_axis, parity_paste,
                          aggregate_tri_mean, aggregate_tri_weuler_self,
                          metrics_from_cube)
except ImportError:
    from models import UNetG, count_parameters
    from data import OFFSETS_IN6
    from triaxis import (reconstruct_axis, parity_paste,
                         aggregate_tri_mean, aggregate_tri_weuler_self,
                         metrics_from_cube)

VOL_SHAPE = (64, 96, 96)  # (D, H, W); H, W divisible by 16 (4 pool levels)
SEED = 0


def _assert_finite(t: torch.Tensor, name: str):
    assert torch.isfinite(t).all(), f'{name} contains non-finite values'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cpu',
                    help="'cpu' (default) or a CUDA device string")
    ap.add_argument('--batch_size', type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)
    t_start = time.time()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # ── Stage 1: UNetG forward pass (paper config) ──
    G = UNetG(in_ch=6, base=80).to(device).eval()
    n_params = count_parameters(G)
    x = torch.rand(2, 6, 96, 96, generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        y = G(x.to(device)).cpu()
    assert y.shape == (2, 1, 96, 96), f'unexpected output shape {tuple(y.shape)}'
    _assert_finite(y, 'UNetG output')
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0, \
        'UNetG output outside [0, 1] despite sigmoid head'
    print(f'PASS  [1/6] UNetG forward: in_ch=6 base=80 params={n_params:,} '
          f'out={tuple(y.shape)} range=[{y.min():.3f},{y.max():.3f}]')

    # ── Synthetic binary volume (float32 in {0, 1}, fixed seed) ──
    vol = (rng.random(VOL_SHAPE) < 0.30).astype(np.float32)
    offsets = OFFSETS_IN6  # [-15, -9, -3, 3, 9, 15]
    k_max = max(abs(o) for o in offsets)

    # ── Stage 2: per-axis reconstruction ──
    V_z = reconstruct_axis(G, vol, 'z', offsets, device,
                           batch_size=args.batch_size)
    assert tuple(V_z.shape) == VOL_SHAPE
    _assert_finite(V_z, 'V_z')
    gt_t = torch.from_numpy(vol).float()
    # Slices outside the offset margin keep the input volume.
    assert torch.equal(V_z[:k_max], gt_t[:k_max]), \
        'boundary slices should be unmodified'
    # Interior slices are replaced by model predictions.
    assert not torch.equal(V_z[k_max:VOL_SHAPE[0] - k_max],
                           gt_t[k_max:VOL_SHAPE[0] - k_max]), \
        'interior slices were not replaced'
    print(f'PASS  [2/6] reconstruct_axis(z): shape={tuple(V_z.shape)}, '
          f'interior [{k_max},{VOL_SHAPE[0] - k_max}) replaced, '
          f'boundary preserved')

    # ── Stage 3: deployment-parity paste ──
    V_par = parity_paste(V_z, gt_t, z0=0)
    for zi in (0, 16, 62):   # even global z: acquired slices restored
        assert torch.equal(V_par[zi], gt_t[zi]), \
            f'even slice {zi} not restored to acquired data'
    for zi in (17, 31, 47):  # odd interior z: model output retained
        assert torch.equal(V_par[zi], V_z[zi]), \
            f'odd slice {zi} does not hold the model prediction'
    print('PASS  [3/6] parity_paste: even-z acquired slices restored, '
          'odd-z model output retained')

    # ── Stage 4: tri-mean aggregation (default of infer_triaxis.py) ──
    V_x = reconstruct_axis(G, vol, 'x', offsets, device,
                           batch_size=args.batch_size)
    V_y = reconstruct_axis(G, vol, 'y', offsets, device,
                           batch_size=args.batch_size)
    V_mean, info_mean = aggregate_tri_mean(V_z, V_x, V_y)
    _assert_finite(V_mean, 'tri_mean volume')
    assert torch.allclose(V_mean, (V_z + V_x + V_y) / 3.0)
    assert abs(sum(info_mean[k] for k in ('wz', 'wx', 'wy')) - 1.0) < 1e-9
    print(f'PASS  [4/6] tri_mean aggregation: weights='
          f"({info_mean['wz']:.3f}, {info_mean['wx']:.3f}, "
          f"{info_mean['wy']:.3f})")

    # ── Stage 5: GT-free Euler-weighted aggregation ──
    V_we, info_we = aggregate_tri_weuler_self(V_z, V_x, V_y, device)
    _assert_finite(V_we, 'tri_weuler_self volume')
    assert tuple(V_we.shape) == VOL_SHAPE
    w = [info_we['wz'], info_we['wx'], info_we['wy']]
    assert all(wi > 0 for wi in w) and abs(sum(w) - 1.0) < 1e-6, \
        f'tri_weuler_self weights not a positive partition of unity: {w}'
    assert all(np.isfinite(info_we[k])
               for k in ('e_z', 'e_x', 'e_y', 'e_median'))
    print(f'PASS  [5/6] tri_weuler_self aggregation: weights='
          f'({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}) '
          f"euler=({info_we['e_z']:.1f}, {info_we['e_x']:.1f}, "
          f"{info_we['e_y']:.1f})")

    # ── Stage 6: cube-level metrics ──
    V_final = parity_paste(V_mean.clamp(0.0, 1.0), gt_t, z0=0)
    m = metrics_from_cube(V_final, gt_t, device=device)
    bad = [k for k, v in m.items()
           if isinstance(v, float) and not np.isfinite(v)]
    assert not bad, f'non-finite metrics: {bad}'
    assert m['n_z_samples'] > 0 and m['n_cross_samples'] > 0
    print(f"PASS  [6/6] metrics_from_cube: all values finite "
          f"(ssim_z={m['ssim_z']:.3f}, dphi={m['dphi']:.4f}, "
          f"d_euler={m['d_euler']:.1f})")

    print(f'SMOKE TEST OK ({time.time() - t_start:.1f}s, '
          f'device={device.type})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
