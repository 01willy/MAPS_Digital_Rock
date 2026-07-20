#!/usr/bin/env python3
"""
Contiguous-gap (vertical-stitching) restoration probe (Supplementary Note E,
"a contiguous-gap probe ... marks fully-skipped sub-scans as out of scope").

Models missing-sub-scan acquisition: planes along z arrive in periodic
blocks of width B (kept as GT anchors) separated by contiguous gaps of G
missing planes -- NOT the uniform every-(k+1) sparsity of the k-sweep.
Retained fraction = B / (B + G); block 8 with gap 8 gives 50% retained,
block 8 with gap 16 gives ~34%.

Gaps are filled deployably (no GT read inside a gap):
  ours_seq -- sequential nearest-6 fill: unknown planes are filled in
      ascending distance to the nearest acquired plane; each target is
      predicted by the MAPS UNetG from its 6 nearest already-known-or-
      filled planes (3 below + 3 above, spatial order). This is
      out-of-distribution vs the trained offsets and corresponds to the
      deployable "use the nearest anchors you have" strategy.
  linear   -- 1-D linear interpolation across each gap between its
      bounding acquired planes.

Acquired planes keep GT (parity); volumes are binarized at 0.5. Metrics:
morphology per (gap, method, cube), plus an optional LBM permeability
TRACE error vs the GT cube (--lbm; D3Q19, 3 axes, 5000 steps) at the
--lbm_gaps widths -- the numbers behind "learned fill 1.6% vs linear 10.2%
LBM-trace error at 50% retained; both collapse at 34%".

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/slab_stitching_probe.py \\
      --volume_path data/BB_1000c_f32.bin --domain BB --voxel_um 2.25 \\
      --checkpoint runs/stage2/best.pt --lbm --lbm_gaps 8 16 \\
      --out_dir outputs/analysis/slab_stitching
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ssim as pytorch_msssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume  # noqa: E402
from maps.metrics import compute_all_morphological_metrics  # noqa: E402
from analysis.benchmark_cubes import find_cube_origins_256  # noqa: E402
from analysis.inference_tiled import load_unetg  # noqa: E402
from lbm.d3q19 import D3Q19LBM  # noqa: E402

CUBE_SIZE = 256
N_NEIGH = 3  # nearest known/filled planes per side (6 total)


def acquired_mask(D: int, B: int, G: int) -> np.ndarray:
    """True where a plane is acquired (inside a block); period B + G."""
    return (np.arange(D) % (B + G)) < B


def linear_fill_z(cube: np.ndarray, known: np.ndarray) -> np.ndarray:
    """1-D linear interpolation of unknown z planes between bounding known
    planes; boundary unknowns copy the nearest known plane."""
    D = cube.shape[0]
    out = cube.copy()
    known_idx = np.where(known)[0]
    for t in range(D):
        if known[t]:
            continue
        lo = known_idx[known_idx < t]
        hi = known_idx[known_idx > t]
        if lo.size and hi.size:
            a, b = lo[-1], hi[0]
            w = (t - a) / (b - a)
            out[t] = (1.0 - w) * cube[a] + w * cube[b]
        elif lo.size:
            out[t] = cube[lo[-1]]
        else:
            out[t] = cube[hi[0]]
    return out


@torch.no_grad()
def ours_fill_z_sequential(G_model, cube: np.ndarray, known: np.ndarray,
                           device) -> np.ndarray:
    """Deployable sequential nearest-6 fill; never reads an unfilled plane.
    Fill order = ascending distance to the nearest acquired plane."""
    D = cube.shape[0]
    vol = cube.copy()
    filled = known.copy()
    known_idx = np.where(known)[0]
    assert known_idx.size >= N_NEIGH, 'need >= 3 acquired planes'
    dist = np.array([np.min(np.abs(known_idx - t)) for t in range(D)])
    order = [t for t in np.argsort(dist, kind='stable') if not known[t]]
    for t in order:
        f_idx = np.where(filled)[0]
        below = f_idx[f_idx < t]
        above = f_idx[f_idx > t]
        b = list(below[-N_NEIGH:]) if below.size else []
        a = list(above[:N_NEIGH]) if above.size else []
        while len(b) < N_NEIGH:
            b.insert(0, (b[0] if b else (a[0] if a else t)))
        while len(a) < N_NEIGH:
            a.append(a[-1] if a else (b[-1] if b else t))
        inp = np.stack([vol[j] for j in b + a], axis=0).astype(np.float32)
        x = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1, 6, H, W)
        vol[t] = G_model(x).float().cpu().numpy()[0, 0]
        filled[t] = True
    return vol


def morph_metrics(rec_bin: np.ndarray, gt_bin: np.ndarray, device):
    p = torch.from_numpy(rec_bin)[:, None].float().to(device)
    g = torch.from_numpy(gt_bin)[:, None].float().to(device)
    ssim = float(pytorch_msssim_fn(p, g, data_range=1.0).item())
    morph = compute_all_morphological_metrics(p, g, max_lag=16)
    area_mpx = (gt_bin.shape[1] * gt_bin.shape[2]) / 1e6
    return {'ssim': ssim, 'dphi': float(morph['dphi']),
            'dsa': float(morph['dsa']),
            'd_euler_per_mpx': float(morph['d_euler']) / area_mpx,
            's2_mse': float(morph['s2_mse']),
            'phi': float(1.0 - rec_bin.mean()),
            'gt_phi': float(1.0 - gt_bin.mean())}


def lbm_trace(solid01: np.ndarray, voxel_um: float, device, n_steps: int):
    """Permeability trace k0 + k1 + k2 (mD) over the three flow axes."""
    trace = 0.0
    for axis in (0, 1, 2):
        sim = D3Q19LBM(solid01 > 0.5, device=str(device), tau=1.0,
                       body_force=1e-5, flow_axis=axis)
        for _ in range(n_steps):
            sim.step()
        trace += sim.permeability(voxel_size_um=voxel_um)['k_mD']
        del sim
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return trace


def main():
    ap = argparse.ArgumentParser(
        description='Contiguous-gap restoration probe (Supplementary Note E)')
    ap.add_argument('--volume_path', required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--domain', default='BB')
    ap.add_argument('--voxel_um', type=float, default=2.25)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cubes_per_domain', type=int, default=2)
    ap.add_argument('--block', type=int, default=8,
                    help='acquired sub-scan block width (planes)')
    ap.add_argument('--gaps', type=int, nargs='+', default=[4, 8, 16, 32],
                    help='contiguous gap widths swept')
    ap.add_argument('--lbm', action='store_true',
                    help='also run the LBM trace-error probe')
    ap.add_argument('--lbm_gaps', type=int, nargs='+', default=[8, 16],
                    help='gap widths for the LBM probe (block 8: gap 8 = '
                         '50%% retained, gap 16 = ~34%%)')
    ap.add_argument('--lbm_cubes', type=int, default=1,
                    help='number of cubes (from cube0) in the LBM probe')
    ap.add_argument('--n_steps', type=int, default=5000)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_dir',
                    default='outputs/analysis/slab_stitching')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

    log(f'slab-stitching probe: {args.domain} block={args.block} '
        f'gaps={args.gaps} lbm={args.lbm}')
    G = load_unetg(args.checkpoint, device)
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    origins = find_cube_origins_256(vol.shape, CUBE_SIZE,
                                    args.cubes_per_domain)

    rows = []
    lbm_rows = []
    for ci, (z0, y0, x0) in enumerate(origins):
        gt = np.clip(np.ascontiguousarray(
            vol[z0:z0 + CUBE_SIZE, y0:y0 + CUBE_SIZE, x0:x0 + CUBE_SIZE]
        ).astype(np.float32), 0.0, 1.0)
        gt_bin = (gt > 0.5).astype(np.float32)
        log(f'cube{ci} origin=({z0},{y0},{x0}) '
            f'GT phi={1.0 - gt_bin.mean():.4f}')

        gt_trace = None
        if args.lbm and ci < args.lbm_cubes:
            log('  GT LBM trace...')
            gt_trace = lbm_trace(gt_bin, args.voxel_um, device, args.n_steps)
            log(f'  GT trace={gt_trace:.2f} mD')

        for Gw in args.gaps:
            known = acquired_mask(CUBE_SIZE, args.block, Gw)
            acq_frac = float(known.mean())
            t0 = time.time()
            ours = ours_fill_z_sequential(G, gt, known, device)
            ours[known] = gt[known]  # parity: acquired planes keep GT
            ours_bin = (ours > 0.5).astype(np.float32)
            t_ours = time.time() - t0
            lin = linear_fill_z(gt, known)
            lin[known] = gt[known]
            lin_bin = (lin > 0.5).astype(np.float32)

            for method, rec_bin in (('ours_seq', ours_bin),
                                    ('linear', lin_bin)):
                m = morph_metrics(rec_bin, gt_bin, device)
                rows.append({'domain': args.domain, 'cube': ci,
                             'block': args.block, 'gap': Gw,
                             'acq_frac': round(acq_frac, 3),
                             'method': method,
                             **{k: round(v, 6) for k, v in m.items()}})
            log(f'  B={args.block} G={Gw} acq={acq_frac:.2f} | '
                f'ours dphi={rows[-2]["dphi"]:.5f} | '
                f'linear dphi={rows[-1]["dphi"]:.5f} ({t_ours:.1f}s)')

            if (args.lbm and ci < args.lbm_cubes and Gw in args.lbm_gaps
                    and gt_trace):
                for method, rec_bin in (('ours_seq', ours_bin),
                                        ('linear', lin_bin)):
                    tr = lbm_trace(rec_bin, args.voxel_um, device,
                                   args.n_steps)
                    err_pct = abs(tr - gt_trace) / gt_trace * 100.0
                    lbm_rows.append({'domain': args.domain, 'cube': ci,
                                     'block': args.block, 'gap': Gw,
                                     'acq_frac': round(acq_frac, 3),
                                     'method': method,
                                     'trace_mD': round(tr, 3),
                                     'gt_trace_mD': round(gt_trace, 3),
                                     'trace_err_pct': round(err_pct, 2)})
                    log(f'    LBM {method}: trace={tr:.2f} mD '
                        f'(err {err_pct:.1f}%)')

    csv_p = out_dir / 'slab_morphology.csv'
    with open(csv_p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f'[saved] {csv_p} ({len(rows)} rows)')
    if lbm_rows:
        csv_l = out_dir / 'slab_lbm_trace.csv'
        with open(csv_l, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(lbm_rows[0].keys()))
            w.writeheader()
            w.writerows(lbm_rows)
        log(f'[saved] {csv_l} ({len(lbm_rows)} rows)')


if __name__ == '__main__':
    main()
