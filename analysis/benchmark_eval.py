#!/usr/bin/env python3
"""
Cross-model benchmark evaluation driver (Table 1 learned-baseline rows and
the Section 5.6 convergence-parity comparison).

Reconstructs the three seeded BB benchmark cubes (128x256x256, drawn from
the test z-slab; `analysis/benchmark_cubes.get_cubes`) with a trained
checkpoint of any of the paper's learned methods and reports
SSIM / dphi / dSA / dchi-per-Mpx under the deployment-parity and/or
all-replacement protocols.

Models:
  unetg    -- the 2D UNetG backbone. With a MAPS checkpoint this is the
              primary method; with an L1-only checkpoint (trained via
              `train_stage1.py` with all extra losses zeroed) this is the
              architecture-matched b4 row -- set `--method_label b4`.
              Aggregations: z_only / tri_mean / tri_weuler_self.
  unet3d   -- b5 (base 24) / b5-large (`--base 64`, `--method_label
              b5_large`). Fair single-target 3D protocol, parity by
              construction: only odd global-z slices are predicted (32-deep
              patch, xy tile 256 + pad 16); z_only.
  swinunet -- SwinUNet 2D hybrid baseline (shape-agnostic, so full tri-axis
              aggregations are evaluated like unetg).
  i3net    -- I3Net medical-CT baseline; z_only ONLY (its cross-view block
              requires square inputs, which structurally blocks tri-axis on
              the non-square x/y slabs -- the paper's Section 5.7 argument).

Protocols:
  parity -- acquired even global-z slices pasted back after aggregation;
            metrics are computed on the final deployed volume (acquired +
            synthesized slices), not on the synthesized odd slices alone
            (reported convention; the 3D path is parity by construction).
  allrep -- every interior slice model-predicted (used by some published
            per-baseline figures; not available for unet3d).

Convergence-parity use (Section 5.6, "3D capacity does not close the gap"):
train b5 / b5-large to the full MAPS wall-clock budget (~4.5 h) with
`baselines/train_unet3d.py --max_seconds 16200`, then evaluate the
checkpoints here; compare dphi against the MAPS rows from
`analysis/aggregation_ablation.py`.

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/benchmark_eval.py \\
      --model swinunet --volume_path data/BB_1000c_f32.bin \\
      --checkpoints 2025=runs/swin_s2025/best.pt 2026=runs/swin_s2026/best.pt \\
      --out_csv outputs/analysis/benchmark_swinunet.csv
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
from maps.triaxis import (reconstruct_axis, parity_paste,  # noqa: E402
                          compute_all_gtfree_aggregations, metrics_from_cube)
from analysis.benchmark_cubes import get_cubes, CUBE_ZHW  # noqa: E402
from analysis.inference_tiled import (lock_determinism, load_unetg,  # noqa: E402
                                      load_unet3d, predict_3d_target_slice)
from maps.checkpoint import extract_model_state, load_state_checked  # noqa: E402

MODEL_CHOICES = ['unetg', 'unet3d', 'swinunet', 'i3net']
DEFAULT_LABEL = {'unetg': 'maps', 'unet3d': 'b5',
                 'swinunet': 'swinunet', 'i3net': 'i3net'}
AGGS_2D = ['z_only', 'tri_mean', 'tri_weuler_self']


def parse_checkpoints(specs):
    out = {}
    for spec in specs:
        if '=' in spec:
            seed, path = spec.split('=', 1)
        else:
            seed, path = str(len(out)), spec
        out[seed] = Path(path)
    return out


def load_model(args, device):
    if args.model == 'unetg':
        return None  # loaded per-seed via load_unetg
    if args.model == 'swinunet':
        from baselines.swinunet import SwinUNet
        return lambda ckpt: _load_generic(
            SwinUNet(in_ch=6, base=args.swin_base, num_heads=args.num_heads,
                     window_size=args.window_size), ckpt, device)
    if args.model == 'i3net':
        from baselines.i3net import I3NetRock
        return lambda ckpt: _load_generic(
            I3NetRock(in_ch=6, out_ch=1, n_feats=args.n_feats,
                      num_blocks=args.num_blocks,
                      window_size=args.window_size), ckpt, device)
    return None


def _load_generic(model, ckpt_path, device):
    model = model.to(device)
    ck = torch.load(str(ckpt_path), map_location=device)
    load_state_checked(model, extract_model_state(ck), label=str(ckpt_path))
    model.eval()
    return model


@torch.no_grad()
def recon_unet3d_parity_cube(G3, vol, cube_origin, offsets, device):
    """b5 fair single-target reconstruction of one benchmark cube: only odd
    GLOBAL-z slices predicted (32-deep patch, xy tile 256 + pad 16); even
    and boundary slices keep GT (parity by construction)."""
    z0, y0, x0, _ = cube_origin
    Dc, Hc, Wc = CUBE_ZHW
    view = vol[:, y0:y0 + Hc, x0:x0 + Wc]  # memmap view, full z context
    gt_cube = np.clip(np.ascontiguousarray(
        vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32), 0., 1.)
    pred = gt_cube.copy()
    k_max = max(abs(o) for o in offsets)
    for zi in range(Dc):
        z_abs = z0 + zi
        if z_abs % 2 == 0:
            continue                      # acquired slice keeps GT
        if zi < k_max or zi >= Dc - k_max:
            continue                      # boundary keeps GT
        pred[zi] = predict_3d_target_slice(
            G3, view, z_abs, device, input_mode='fair_offsets',
            offsets=offsets, patch_size=32, xy_tile=256, xy_pad=16)
    return pred, gt_cube


def eval_2d(model, vol, cubes, offsets, aggs, protocols, device, log):
    """2D backbone path (unetg / swinunet / i3net). Returns rows of
    (cube_label, agg, protocol, metrics)."""
    Dc, Hc, Wc = CUBE_ZHW
    out = []
    for (z0, y0, x0, lab) in cubes:
        cube = np.clip(np.ascontiguousarray(
            vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32),
            0., 1.)
        gt_t = torch.from_numpy(cube).float()
        t0 = time.time()
        V_z = reconstruct_axis(model, cube, 'z', offsets, device)
        recon = {'z_only': V_z}
        if any(a.startswith('tri') for a in aggs):
            V_x = reconstruct_axis(model, cube, 'x', offsets, device)
            V_y = reconstruct_axis(model, cube, 'y', offsets, device)
            gf = compute_all_gtfree_aggregations(V_z, V_x, V_y, device)
            for name, (V, _info) in gf.items():
                recon[name] = V
        t_recon = time.time() - t0
        for agg in aggs:
            V = recon[agg]
            for proto in protocols:
                Vp = parity_paste(V, gt_t, z0) if proto == 'parity' else V
                m = metrics_from_cube(Vp, gt_t, device=device)
                m['_recon_seconds'] = t_recon
                out.append((lab, agg, proto, m))
        log(f'  [ok] cube={lab} recon={t_recon:.1f}s')
    return out


def eval_3d(G3, vol, cubes, offsets, device, log):
    out = []
    for cube_origin in cubes:
        t0 = time.time()
        pred, gt_cube = recon_unet3d_parity_cube(G3, vol, cube_origin,
                                                 offsets, device)
        t_recon = time.time() - t0
        m = metrics_from_cube(torch.from_numpy(pred).float(),
                              torch.from_numpy(gt_cube).float(),
                              device=device)
        m['_recon_seconds'] = t_recon
        out.append((cube_origin[3], 'z_only', 'parity', m))
        log(f'  [ok] cube={cube_origin[3]} recon={t_recon:.1f}s')
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Cross-model benchmark eval (Table 1 / Section 5.6)')
    ap.add_argument('--model', required=True, choices=MODEL_CHOICES)
    ap.add_argument('--method_label', default=None,
                    help='method name written to the CSV (default: '
                         'maps/b5/swinunet/i3net; use e.g. b4, b5_large)')
    ap.add_argument('--checkpoints', nargs='+', required=True,
                    help='seed=path pairs (one per training seed)')
    ap.add_argument('--volume_path', required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--domain', default='BB')
    ap.add_argument('--protocols', nargs='+', default=['parity', 'allrep'],
                    choices=['parity', 'allrep'])
    ap.add_argument('--aggs', nargs='+', default=None, choices=AGGS_2D,
                    help='2D models only (default: all three; i3net is '
                         'forced to z_only)')
    ap.add_argument('--offsets', type=int, nargs='+', default=None,
                    help='input offsets (default -15 -9 -3 3 9 15)')
    ap.add_argument('--cubes_n', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2025,
                    help='determinism lock for the evaluation itself')
    ap.add_argument('--gpu', type=int, default=0)
    # unet3d
    ap.add_argument('--base', type=int, default=24,
                    help='unet3d base channels (24 = b5, 64 = b5-large)')
    # swinunet
    ap.add_argument('--swin_base', type=int, default=96)
    ap.add_argument('--num_heads', type=int, default=4)
    # swinunet uses window_size=8, i3net uses 16
    ap.add_argument('--window_size', type=int, default=None)
    # i3net
    ap.add_argument('--n_feats', type=int, default=64)
    ap.add_argument('--num_blocks', type=int, default=16)
    ap.add_argument('--out_csv', default='outputs/analysis/benchmark_eval.csv')
    args = ap.parse_args()

    if args.window_size is None:
        args.window_size = 16 if args.model == 'i3net' else 8
    method = args.method_label or DEFAULT_LABEL[args.model]
    offsets = args.offsets or OFFSETS_IN6

    aggs = args.aggs or list(AGGS_2D)
    protocols = list(args.protocols)
    if args.model == 'i3net' and aggs != ['z_only']:
        aggs = ['z_only']  # square-input constraint blocks tri-axis
    if args.model == 'unet3d':
        aggs = ['z_only']
        protocols = ['parity']  # parity by construction

    lock_determinism(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

    log(f'benchmark eval: model={args.model} method={method} aggs={aggs} '
        f'protocols={protocols} offsets={offsets}')

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape, n=args.cubes_n)
    log(f'cubes: {cubes}')

    builder = load_model(args, device)
    ckpts = parse_checkpoints(args.checkpoints)

    rows = []
    for seed, ckpt in ckpts.items():
        if not ckpt.exists():
            log(f'[skip] seed {seed}: missing {ckpt}')
            continue
        log(f'[{method} s{seed}] {ckpt}')
        if args.model == 'unetg':
            model = load_unetg(ckpt, device)
            results = eval_2d(model, vol, cubes, offsets, aggs, protocols,
                              device, log)
        elif args.model == 'unet3d':
            model = load_unet3d(ckpt, device, base=args.base)
            results = eval_3d(model, vol, cubes, offsets, device, log)
        else:
            model = builder(ckpt)
            results = eval_2d(model, vol, cubes, offsets, aggs, protocols,
                              device, log)
        for lab, agg, proto, m in results:
            rows.append(dict(
                method=method, model=args.model, domain=args.domain,
                seed=seed, cube=lab, agg=agg, protocol=proto,
                ssim=m['ssim_z'], psnr=m['psnr_z'], dphi=m['dphi'],
                dsa=m['dsa'], dchi=m['d_euler_per_mpx'],
                xz_ssim=m['xz_ssim_mean'], yz_ssim=m['yz_ssim_mean'],
                recon_seconds=round(m['_recon_seconds'], 1)))
        del model
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit('no rows produced (all checkpoints missing?)')
    keys = list(rows[0].keys())
    new_file = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f'[saved] {out_csv} ({len(rows)} rows appended)')

    # Aggregate summary (mean +- std over seeds x cubes)
    grp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r['agg'], r['protocol'])
        for k in ('dphi', 'ssim', 'dsa', 'dchi'):
            grp[key][k].append(r[k])
    log('AGGREGATE (mean +- std over seeds x cubes):')
    for (agg, proto), d in sorted(grp.items()):
        n = len(d['dphi'])
        log(f'  {method:10s} {agg:16s} {proto:7s} n={n:2d}  '
            f'dphi={np.mean(d["dphi"]):.5f}+-{np.std(d["dphi"]):.5f}  '
            f'ssim={np.mean(d["ssim"]):.4f}+-{np.std(d["ssim"]):.4f}  '
            f'dsa={np.mean(d["dsa"]):.5f}  dchi={np.mean(d["dchi"]):.3f}')


if __name__ == '__main__':
    main()
