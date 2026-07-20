#!/usr/bin/env python3
"""
Input channel/offset ablation evaluation -- common-target protocol
(Table S8 and Fig. S2).

Every trained configuration (EMA weights from a `train_stage1.py` run
directory) predicts the IDENTICAL target set, removing the task-difficulty
confound of per-config k_max-dependent target lists:

  48 odd z slices in [877, 971] of the BB test slab [850, 1000)
  (z - 27 >= 850 and z + 27 <= 999, the intersection of all configs with
  k_max <= 27), centre 512x512 crop (y, x in [244, 756)).

Metrics per config:
  2D  -- per-slice SSIM mean/std, PSNR, and the slice-batch morphology
         suite (dphi / dSA / dchi-per-Mpx / S2 MSE).
  3D  -- stacked 48x512x512 volume, per-volume Otsu binarization of pred
         and GT independently: dphi_3D, dSA_3D (6-connectivity interface
         faces per voxel), dchi_3D per Mvox (skimage Euler number,
         connectivity=3, solid phase).

Training recipe of the ablation runs (Table S8): BB, 50 epochs, GAN off
(--lambda_gan_base 0), --w_ssim 0.3 --w_grad 0 --w_phi 0.1 --w_sa 0.1
--w_s2 0 --w_lpath 0.1 --soft_temperature 10, batch 4, patch 256, one
`train_stage1.py` run per (offsets, seed) configuration -- e.g. the
6-channel default A6 uses `--in_ch 6 --offsets -15 -9 -3 3 9 15`, the
10-channel A10 `--in_ch 10 --offsets -27 -21 -15 -9 -3 3 9 15 21 27`.

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/channel_ablation_eval.py \\
      --volume_path data/BB_1000c_f32.bin \\
      --runs A6_default=runs/chab_A6 A10=runs/chab_A10 ... \\
      --out_dir outputs/analysis/channel_ablation
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, OFFSETS_IN6  # noqa: E402
from maps.models import UNetG  # noqa: E402
from maps.checkpoint import extract_model_state, load_state_checked  # noqa: E402
from maps.metrics import compute_all_morphological_metrics  # noqa: E402
from maps.metrics3d import (binarize_per_image_otsu,  # noqa: E402
                            specific_surface_3d, euler_3d)

# Common target set (all configs with k_max <= 27 on a 1000^3 volume)
Z_TARGETS = list(range(877, 972, 2))  # 48 odd z slices
CROP = (244, 756)                     # centre 512x512

WIDE_FIELDS = ['config', 'in_ch', 'offsets', 'k_max', 'seed',
               'ssim_2d', 'ssim_2d_std', 'psnr_2d',
               'dphi_2d', 'dsa_2d', 'd_euler_2d_per_mpx', 's2_mse_2d',
               'dphi_3d', 'dsa_3d', 'd_euler_3d_per_mvox',
               'porosity_pred_3d', 'porosity_gt_3d',
               'otsu_tau_pred', 'otsu_tau_gt', 'wall_seconds', 'run_dir']


@torch.no_grad()
def predict_targets(G, vol, offsets, device, batch_size=8):
    """Predict every common target with the config's own offsets."""
    y0, y1 = CROP
    preds = []
    for i in range(0, len(Z_TARGETS), batch_size):
        zs = Z_TARGETS[i:i + batch_size]
        ins = []
        for z in zs:
            ch = [np.asarray(vol[z + o, y0:y1, y0:y1], dtype=np.float32)
                  for o in offsets]
            ins.append(np.stack(ch, axis=0))
        x = torch.from_numpy(np.stack(ins, axis=0)).to(device)
        preds.append(G(x).float().cpu()[:, 0])
    return torch.cat(preds, dim=0)  # (48, 512, 512)


def eval_run(run_dir: Path, vol, device):
    cfg = json.loads((run_dir / 'config.json').read_text())
    offsets = cfg.get('offsets') or OFFSETS_IN6
    in_ch = cfg.get('in_ch', len(offsets))
    k_max = max(abs(o) for o in offsets)

    ck = torch.load(run_dir / 'checkpoints' / 'best.pt', map_location='cpu')
    G = UNetG(in_ch=in_ch, base=cfg.get('base_ch', 80)).to(device)
    load_state_checked(G, extract_model_state(ck),
                       label=str(run_dir / 'checkpoints' / 'best.pt'))
    G.eval()

    t0 = time.time()
    pred = predict_targets(G, vol, offsets, device)
    y0, y1 = CROP
    gt = torch.from_numpy(np.stack(
        [np.asarray(vol[z, y0:y1, y0:y1], dtype=np.float32)
         for z in Z_TARGETS], axis=0))

    # 2D per-slice suite
    p4 = pred.unsqueeze(1).to(device)
    g4 = gt.unsqueeze(1).to(device)
    ssim_per_slice = [float(ssim_fn(p4[j:j + 1], g4[j:j + 1],
                                    data_range=1.0).item())
                      for j in range(p4.shape[0])]
    mse = F.mse_loss(p4, g4).item()
    morph = compute_all_morphological_metrics(p4, g4, max_lag=16)
    area_mpx = (p4.shape[-2] * p4.shape[-1]) / 1e6

    # 3D stacked-volume suite (per-volume Otsu, protocol of Table S8)
    pred_solid, gt_solid, taus = binarize_per_image_otsu(pred.numpy(),
                                                         gt.numpy())
    phi_p = 1.0 - float(pred_solid.mean())
    phi_g = 1.0 - float(gt_solid.mean())
    mvox = pred_solid.size / 1e6

    del G
    torch.cuda.empty_cache()

    return {
        'in_ch': in_ch,
        'offsets': ' '.join(str(o) for o in offsets),
        'k_max': k_max,
        'seed': cfg.get('seed', ''),
        'ssim_2d': float(np.mean(ssim_per_slice)),
        'ssim_2d_std': float(np.std(ssim_per_slice)),
        'psnr_2d': float(-10 * np.log10(mse + 1e-10)),
        'dphi_2d': float(morph['dphi']),
        'dsa_2d': float(morph['dsa']),
        'd_euler_2d_per_mpx': float(morph['d_euler']) / area_mpx,
        's2_mse_2d': float(morph['s2_mse']),
        'dphi_3d': abs(phi_p - phi_g),
        'dsa_3d': abs(specific_surface_3d(pred_solid)
                      - specific_surface_3d(gt_solid)),
        'd_euler_3d_per_mvox': abs(euler_3d(pred_solid)
                                   - euler_3d(gt_solid)) / mvox,
        'porosity_pred_3d': phi_p,
        'porosity_gt_3d': phi_g,
        'otsu_tau_pred': float(taus[0]),
        'otsu_tau_gt': float(taus[1]),
        'wall_seconds': round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(
        description='Channel/offset ablation common-target eval (Table S8)')
    ap.add_argument('--volume_path', required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--runs', nargs='+', required=True,
                    help='name=run_dir pairs; each run_dir is a '
                         'train_stage1.py output dir (config.json + '
                         'checkpoints/best.pt)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_dir',
                    default='outputs/analysis/channel_ablation')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    print(f'[eval] common targets: {len(Z_TARGETS)} odd z in '
          f'[{Z_TARGETS[0]}, {Z_TARGETS[-1]}], crop {CROP}')

    rows = []
    for spec in args.runs:
        if '=' in spec:
            name, rd = spec.split('=', 1)
        else:
            name, rd = Path(spec).name, spec
        rd = Path(rd)
        if not (rd / 'checkpoints' / 'best.pt').exists():
            print(f'[skip] {name}: no checkpoints/best.pt under {rd}')
            continue
        try:
            m = eval_run(rd, vol, device)
        except Exception as e:
            print(f'[fail] {name}: {type(e).__name__}: {e}')
            continue
        m['config'] = name
        m['run_dir'] = str(rd)
        rows.append(m)
        print(f'[done] {name:14s} ssim={m["ssim_2d"]:.4f} '
              f'dphi3d={m["dphi_3d"]:.4f} dsa3d={m["dsa_3d"]:.5f} '
              f'dchi3d/Mvox={m["d_euler_3d_per_mvox"]:.2f} '
              f'({m["wall_seconds"]:.0f}s)', flush=True)

    if not rows:
        raise SystemExit('no results')

    wide_path = out_dir / 'channel_ablation_wide.csv'
    with open(wide_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=WIDE_FIELDS, extrasaction='ignore')
        w.writeheader()
        for m in rows:
            w.writerow(m)
    print(f'[saved] {wide_path}')

    metric_cols = ['ssim_2d', 'psnr_2d', 'dphi_2d', 'dsa_2d',
                   'd_euler_2d_per_mpx', 's2_mse_2d',
                   'dphi_3d', 'dsa_3d', 'd_euler_3d_per_mvox']
    long_path = out_dir / 'channel_ablation_long.csv'
    with open(long_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'in_ch', 'offsets', 'k_max', 'seed',
                    'metric', 'value'])
        for m in rows:
            for mc in metric_cols:
                w.writerow([m['config'], m['in_ch'], m['offsets'],
                            m['k_max'], m['seed'], mc, m[mc]])
    print(f'[saved] {long_path}')


if __name__ == '__main__':
    main()
