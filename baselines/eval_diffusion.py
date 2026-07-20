#!/usr/bin/env python3
"""
Diffusion-baseline evaluation (V1 / V2 pixel-space DDIM and V3 latent
diffusion): DDIM-sample every interior slice of a benchmark cube and score
against GT.

Paper roles:
  - the Section 5.7 V3 latent-diffusion results (runtime in Table 4)
    (z_only, 50 DDIM steps, 3 BB cubes, all-replacement protocol)
  - the V1/V2/V3 diffusion comparison of Section 5.7 (--variant v1 / v2)
  - DDIM step-sweep Supplement figure: run this script once per value of
    --ddim_steps (e.g. 10 20 50 100 200 500 1000) and collect the CSVs.

Variants:
  v1 -- pixel-space conditional DDIM (baselines/diffusion_v1_pixel.py)
  v2 -- pixel-space DDIM with channel-concat time conditioning
        (baselines/diffusion_v2_pixel.py)
  v3 -- latent diffusion (VAE + latent DDIM; needs --vae_ckpt). Default.

Same benchmark cubes as the MAPS evaluation (3 cubes of 128x256x256 drawn
from the test slab with seed 2025).

Usage:
  CUDA_VISIBLE_DEVICES=0 python baselines/eval_diffusion.py \\
      --variant v3 --vae_ckpt <vae_best.pt> --ddim_ckpt <ddim_final.pt> \\
      --volume_path data/BB_1000c_f32.bin --ddim_steps 50 \\
      --out_dir outputs/baselines/diffusion_eval
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, compute_splits, OFFSETS_IN6  # noqa: E402
from maps.triaxis import metrics_from_cube  # noqa: E402
from maps.checkpoint import extract_model_state, load_state_checked  # noqa: E402

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


@torch.no_grad()
def reconstruct_axis_pixel(model, diffusion, vol_cube, axis, offsets,
                           device, ddim_steps=50):
    """V1/V2 pixel-space path: for each interior target along `axis`,
    stack the 6 conditioning slices and DDIM-sample the target directly in
    pixel space."""
    D, H, W = vol_cube.shape
    k_max = max(abs(o) for o in offsets)
    out = torch.from_numpy(vol_cube.copy()).float()
    if axis == 'z':
        target_range = range(k_max, D - k_max)
    elif axis == 'x':
        target_range = range(k_max, W - k_max)
    elif axis == 'y':
        target_range = range(k_max, H - k_max)
    else:
        raise ValueError(axis)
    for t_idx in target_range:
        if axis == 'z':
            cond = np.stack([vol_cube[t_idx + o] for o in offsets], axis=0)
        elif axis == 'x':
            cond = np.stack([vol_cube[:, :, t_idx + o] for o in offsets],
                            axis=0)
        else:
            cond = np.stack([vol_cube[:, t_idx + o, :] for o in offsets],
                            axis=0)
        cond_t = torch.from_numpy(cond.astype(np.float32))[None].to(device)
        x0 = diffusion.ddim_sample(model, cond_t, n_steps=ddim_steps).cpu()
        if axis == 'z':
            out[t_idx] = x0[0, 0]
        elif axis == 'x':
            out[:, :, t_idx] = x0[0, 0]
        else:
            out[:, t_idx, :] = x0[0, 0]
    return out


@torch.no_grad()
def reconstruct_axis_latent(vae, model, diffusion, vol_cube, axis, offsets,
                            device, ddim_steps=50, n_time_ch=16):
    """For each interior target along `axis`, encode the 6 cond slices via
    the VAE, run DDIM in latent space, decode the denoised target latent."""
    D, H, W = vol_cube.shape
    k_max = max(abs(o) for o in offsets)
    out = torch.from_numpy(vol_cube.copy()).float()
    if axis == 'z':
        target_range = range(k_max, D - k_max)
    elif axis == 'x':
        target_range = range(k_max, W - k_max)
    elif axis == 'y':
        target_range = range(k_max, H - k_max)
    else:
        raise ValueError(axis)
    for t_idx in target_range:
        if axis == 'z':
            cond_slices = np.stack([vol_cube[t_idx + o] for o in offsets], axis=0)
        elif axis == 'x':
            cond_slices = np.stack([vol_cube[:, :, t_idx + o] for o in offsets], axis=0)
        else:
            cond_slices = np.stack([vol_cube[:, t_idx + o, :] for o in offsets], axis=0)
        cond_t = torch.from_numpy(cond_slices.astype(np.float32))[None].to(device)
        # Encode each cond slice via VAE
        cond_lats = []
        for c in range(cond_t.shape[1]):
            mu, _ = vae.encode(cond_t[:, c:c + 1])
            cond_lats.append(mu)
        cond_lat = torch.cat(cond_lats, dim=1)                    # (1, 6*4, 32, 32)
        # DDIM sample in latent space
        target_lat = diffusion.ddim_sample(model, cond_lat, n_steps=ddim_steps,
                                           latent_ch=4)          # (1, 4, 32, 32)
        # Decode latent
        decoded = vae.decode(target_lat).clamp(0.0, 1.0)          # (1, 1, 256, 256)
        if axis == 'z':
            out[t_idx] = decoded[0, 0]
        elif axis == 'x':
            out[:, :, t_idx] = decoded[0, 0]
        else:
            out[:, t_idx, :] = decoded[0, 0]
    return out


def main():
    ap = argparse.ArgumentParser(description='Diffusion baseline DDIM eval '
                                             '(V1/V2 pixel, V3 latent)')
    ap.add_argument('--variant', default='v3', choices=['v1', 'v2', 'v3'])
    ap.add_argument('--ddim_ckpt', required=True, type=str,
                    help='denoiser checkpoint (pixel UNet for v1/v2, '
                         'latent UNet for v3)')
    ap.add_argument('--vae_ckpt', type=str, default=None,
                    help='VAE checkpoint (required for --variant v3)')
    ap.add_argument('--ddim_steps', type=int, default=50)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int, default=[1000, 1000, 1000])
    ap.add_argument('--out_dir', type=str,
                    default='outputs/baselines/diffusion_eval')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')

    ck = torch.load(args.ddim_ckpt, map_location=device)
    args_train = ck.get('args', {})
    T = args_train.get('T', 1000)
    n_time_ch = args_train.get('n_time_ch', 16)
    vae = None
    if args.variant == 'v1':
        from baselines.diffusion_v1_pixel import (DiffusionUNet,
                                                  GaussianDiffusion)
        print(f'[V1 pixel DDIM eval] ddim={args.ddim_ckpt}')
        model = DiffusionUNet(in_ch=7, base=80).to(device)
        # diffusion checkpoints store raw weights only (no EMA)
        load_state_checked(model, extract_model_state(ck, prefer_ema=False),
                           label=str(args.ddim_ckpt))
        model.eval()
        diffusion = GaussianDiffusion(T=T, device=device)
    elif args.variant == 'v2':
        from baselines.diffusion_v2_pixel import (DiffusionUNetV2,
                                                  GaussianDiffusionV2)
        print(f'[V2 pixel DDIM eval] ddim={args.ddim_ckpt}')
        model = DiffusionUNetV2(n_cond=6, n_time=n_time_ch,
                                base=80).to(device)
        load_state_checked(model, extract_model_state(ck, prefer_ema=False),
                           label=str(args.ddim_ckpt))
        model.eval()
        diffusion = GaussianDiffusionV2(T=T, device=device,
                                        n_time_ch=n_time_ch)
    else:
        from baselines.latent_diffusion import (VAE2D, LatentUNet,
                                                LatentDiffusion)
        if not args.vae_ckpt:
            raise SystemExit('--vae_ckpt is required for --variant v3')
        print(f'[V3 latent DDIM eval] vae={args.vae_ckpt}  '
              f'ddim={args.ddim_ckpt}')
        vae = VAE2D(base=64, latent_ch=4).to(device)
        vae.load_state_dict(torch.load(
            args.vae_ckpt, map_location=device)['model_state_dict'])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        model = LatentUNet(latent_ch=4, n_cond=6, n_time=n_time_ch,
                           base=128).to(device)
        load_state_checked(model, extract_model_state(ck, prefer_ema=False),
                           label=str(args.ddim_ckpt))
        model.eval()
        diffusion = LatentDiffusion(T=T, device=device, n_time_ch=n_time_ch)

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    cubes = get_cubes(vol.shape)
    print(f'cubes: {cubes}')

    rows = []
    for origin in cubes:
        z0, y0, x0, label = origin
        Dc, Hc, Wc = CUBE_ZHW
        cube = np.ascontiguousarray(
            vol[z0:z0 + Dc, y0:y0 + Hc, x0:x0 + Wc]).astype(np.float32)
        cube = np.clip(cube, 0.0, 1.0)
        t0 = time.time()
        # z-only for diffusion (tri-axis is prohibitively expensive)
        if args.variant == 'v3':
            V_z = reconstruct_axis_latent(vae, model, diffusion, cube, 'z',
                                          OFFSETS_IN6, device,
                                          args.ddim_steps, n_time_ch)
        else:
            V_z = reconstruct_axis_pixel(model, diffusion, cube, 'z',
                                         OFFSETS_IN6, device,
                                         args.ddim_steps)
        t_recon = time.time() - t0
        gt_t = torch.from_numpy(cube).float()
        m = metrics_from_cube(V_z, gt_t, device=device)
        rows.append({
            'method': f'diffusion_{args.variant}', 'cube': label,
            'cube_z0': z0, 'ddim_steps': args.ddim_steps,
            'agg': 'z_only', 'ssim': m['ssim_z'], 'psnr': m['psnr_z'],
            'dphi': m['dphi'], 'dsa': m['dsa'],
            'd_euler_per_mpx': m['d_euler_per_mpx'],
            'recon_seconds': t_recon,
        })
        print(f'  cube={label}  SSIM={m["ssim_z"]:.4f} dphi={m["dphi"]:.5f} '
              f'dsa={m["dsa"]:.5f}  recon {t_recon:.1f}s')

    csv_path = out_dir / (f'diffusion_{args.variant}_eval_'
                          f'steps{args.ddim_steps}.csv')
    keys = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[SAVED] {csv_path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
