#!/usr/bin/env python3
"""
Stage 1: z-axis GAN pretraining for Sparse Micro-CT Slice Interpolation.

Trains UNetG (in_ch=6, 2.5D multi-offset inputs) on a single volume with:
  - Unified SSIM (pytorch_msssim)
  - Morphology-preserving regularization (soft-Otsu porosity, surface area)
  - Optional S2 / lineal-path losses, hinge GAN loss
  - EMA checkpointing

All defaults follow the paper's primary recipe (Table S9, the
physics-balanced pareto4 preset of the loss-weight HPO): lr_G 4.5e-4,
lr_D 1.9e-4, Adam beta1 0.5, L1/SSIM 1.0/0.349, phi=SA=lineal-path 0.174,
S2 0.0177, soft-Otsu temperature 10, lambda_adv 0.131 with warmup 27
epochs and decay 0.77.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_stage1.py \\
      --volume_path data/BB_1000c_f32.bin \\
      --max_epochs 200 --seed 2025 --run_name stage1_BB_6ch

Expected data: 1000^3 float32 binary volume in [0, 1] (0=pore, 1=solid);
see README for preparation. Splits are z-slabs train [0,700) /
val [700,850) / test [850,1000).
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maps.models import UNetG, PatchD, EMA, count_parameters
from maps.losses import CombinedLoss, ssim_value
from maps.data import (load_volume, compute_splits, create_dataloaders,
                       check_leakage, OFFSETS_IN6)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Stage1 Training')
    # Data
    parser.add_argument('--volume_path', required=True,
                        help='Path to float32/uint8 binary (or TIFF) volume')
    parser.add_argument('--volume_shape', nargs=3, type=int,
                        default=[1000, 1000, 1000])
    parser.add_argument('--seed', type=int, default=2025)
    # Architecture
    parser.add_argument('--in_ch', type=int, default=6)
    parser.add_argument('--base_ch', type=int, default=80)
    parser.add_argument('--k', type=int, default=1,
                        help='Base offset for in_ch=2,4 (e.g., k=3 -> offsets [-3,+3])')
    parser.add_argument('--offsets', type=int, nargs='+', default=None,
                        help='Explicit offsets for in_ch=6 '
                             '(default: -15 -9 -3 3 9 15)')
    # Training (defaults = the paper's pareto4 recipe, Table S9)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--max_seconds', type=int, default=0,
                        help='wall-clock training budget in seconds; 0 disables '
                             '(train for --max_epochs). Time-matched baselines '
                             'use this, e.g. b4 = 3600.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--lr_G', type=float, default=4.5e-4)
    parser.add_argument('--lr_D', type=float, default=1.9e-4)
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1 for both G and D')
    # Loss weights (paper defaults: L1 1.0 / SSIM 0.349 / phi=SA=lp 0.174 /
    # S2 0.0177; gradient loss excluded by ablation)
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=0.349)
    parser.add_argument('--w_grad', type=float, default=0.0)
    parser.add_argument('--w_phi', type=float, default=0.174)
    parser.add_argument('--w_sa', type=float, default=0.174)
    parser.add_argument('--w_s2', type=float, default=0.0177,
                        help='S2 loss weight (0=off)')
    parser.add_argument('--w_lpath', type=float, default=0.174,
                        help='Lineal path function loss weight (0=off)')
    parser.add_argument('--lambda_gan_base', type=float, default=0.131)
    parser.add_argument('--gan_warmup', type=int, default=27,
                        help='GAN warmup epochs (linear ramp of lambda_adv)')
    parser.add_argument('--lambda_decay', type=float, default=0.77,
                        help='lambda_adv decay factor after warmup '
                             '(floored at 10%% of the base value)')
    parser.add_argument('--gan_mode', default='hinge',
                        choices=['hinge', 'bce'])
    parser.add_argument('--soft_temperature', type=float, default=10.0,
                        help='Soft-Otsu temperature (paper default 10; '
                             'sharper values such as 50 vanish gradients '
                             'when morphology weights are large)')
    # Hardware
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    # Checkpointing
    parser.add_argument('--ckpt_interval', type=int, default=25,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--out_dir', default='outputs/stage1',
                        help='Root directory for run outputs')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}')

    # ── Run directory ──
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = args.run_name or f'stage1_{Path(args.volume_path).stem}_6ch'
    run_dir = (Path(args.out_dir) / f'{run_name}_{timestamp}').resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"Stage1 Training: {args.volume_path}")
    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")

    # ── Data ──
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])
    print(f"Splits: {splits}")

    # Leakage check
    leak = check_leakage(vol.shape, splits, in_ch=args.in_ch,
                         offsets=args.offsets)
    print(f"Leakage check: clean={leak['clean']}")
    assert leak['clean'], f"Data leakage detected! {leak['violations']}"

    # Dataloaders
    cfg_data = {
        'in_ch': args.in_ch,
        'k': args.k,
        'offsets': args.offsets,
        'patch_size': args.patch_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
    }
    train_loader, val_loader, test_loader = create_dataloaders(
        vol, splits, cfg_data, axis='z')

    print(f"Train: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")

    # ── Models ──
    G = UNetG(in_ch=args.in_ch, base=args.base_ch).to(device)
    D = PatchD(in_ch=args.in_ch + 1, base=64).to(device)
    ema = EMA(G, decay=0.999)

    print(f"G params: {count_parameters(G):,}")
    print(f"D params: {count_parameters(D):,}")

    # ── Optimizers ──
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_G,
                             betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_D,
                             betas=(args.beta1, 0.999))
    scaler = GradScaler(enabled=args.use_amp)

    # ── Loss ──
    loss_cfg = {
        'w_l1': args.w_l1, 'w_ssim': args.w_ssim, 'w_grad': args.w_grad,
        'w_phi': args.w_phi, 'w_sa': args.w_sa, 'w_s2': args.w_s2,
        'w_lpath': args.w_lpath,
        'gan_mode': args.gan_mode,
        'soft_temperature': args.soft_temperature,
    }
    criterion = CombinedLoss(loss_cfg)

    # ── Save config ──
    config = {**vars(args), 'splits': splits, 'timestamp': timestamp,
              'g_params': count_parameters(G), 'loss_cfg': loss_cfg}
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ── Training log ──
    log_path = run_dir / 'train_log.csv'
    with open(log_path, 'w') as f:
        f.write('epoch,train_G,train_D,train_dphi,train_dsa,'
                'val_ssim,val_psnr,ema_ssim,ema_psnr,'
                'lambda_gan,duration_s\n')

    # ── Training loop ──
    best_ema_ssim = -1.0
    total_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        G.train()
        D.train()
        epoch_start = time.time()

        # GAN weight schedule: linear warmup, then decay floored at 10% of
        # the base value (paper recipe: warmup 27 epochs, decay 0.77)
        if epoch < args.gan_warmup:
            lambda_gan = args.lambda_gan_base * epoch / args.gan_warmup
        else:
            progress = ((epoch - args.gan_warmup)
                        / max(1, args.max_epochs - args.gan_warmup))
            lambda_gan = args.lambda_gan_base * (1.0
                                                 - args.lambda_decay * progress)
            lambda_gan = max(lambda_gan, args.lambda_gan_base * 0.1)

        g_total_sum = d_loss_sum = dphi_sum = dsa_sum = 0.0
        n_steps = 0

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ── D update ──
            opt_D.zero_grad(set_to_none=True)
            with autocast(enabled=args.use_amp):
                y_fake_d = G(x).detach()
                d_loss, d_metrics = criterion.compute_D_loss(D, x, y, y_fake_d)
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)

            # ── G update ──
            opt_G.zero_grad(set_to_none=True)
            with autocast(enabled=args.use_amp):
                g_loss, g_metrics = criterion.compute_G_loss(
                    G, D, x, y, lambda_gan)
            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler.step(opt_G)
            scaler.update()

            # EMA
            ema.update()

            # Stats
            g_total_sum += g_metrics['G_total']
            d_loss_sum += d_metrics['D_loss']
            dphi_sum += g_metrics['dphi']
            dsa_sum += g_metrics['dsa']
            n_steps += 1

            if step % args.log_interval == 0:
                print(f"  [E{epoch:03d} S{step:04d}] "
                      f"G={g_metrics['G_total']:.3f} "
                      f"D={d_metrics['D_loss']:.3f} "
                      f"phi={g_metrics['dphi']:.4f} "
                      f"sa={g_metrics['dsa']:.4f} "
                      f"lam={lambda_gan:.4f}")

        # ── Validation ──
        G.eval()
        val_ssim_sum = val_psnr_sum = 0
        n_val = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for x_v, y_v in val_loader:
                x_v = x_v.to(device).float()
                y_v = y_v.to(device).float()
                pred_v = G(x_v)
                val_ssim_sum += ssim_value(pred_v.float(), y_v.float())
                mse = F.mse_loss(pred_v.float(), y_v.float()).item()
                val_psnr_sum += -10 * np.log10(mse + 1e-10)
                n_val += 1

        val_ssim = val_ssim_sum / max(n_val, 1)
        val_psnr = val_psnr_sum / max(n_val, 1)

        # ── EMA Validation ──
        ema.store()
        ema.apply()
        ema_ssim_sum = ema_psnr_sum = 0
        n_ema = 0
        G.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for x_v, y_v in val_loader:
                x_v = x_v.to(device).float()
                y_v = y_v.to(device).float()
                pred_v = G(x_v)
                ema_ssim_sum += ssim_value(pred_v.float(), y_v.float())
                mse = F.mse_loss(pred_v.float(), y_v.float()).item()
                ema_psnr_sum += -10 * np.log10(mse + 1e-10)
                n_ema += 1
        ema.restore()

        ema_ssim = ema_ssim_sum / max(n_ema, 1)
        ema_psnr = ema_psnr_sum / max(n_ema, 1)

        duration = time.time() - epoch_start

        # Log
        print(f"[E{epoch:03d}] Val SSIM={val_ssim:.4f} "
              f"EMA SSIM={ema_ssim:.4f} "
              f"PSNR={ema_psnr:.1f} "
              f"G={g_total_sum/n_steps:.3f} "
              f"D={d_loss_sum/n_steps:.3f} "
              f"phi={dphi_sum/n_steps:.4f} "
              f"sa={dsa_sum/n_steps:.4f} "
              f"time={duration:.0f}s")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},"
                    f"{g_total_sum/n_steps:.6f},{d_loss_sum/n_steps:.6f},"
                    f"{dphi_sum/n_steps:.6f},{dsa_sum/n_steps:.6f},"
                    f"{val_ssim:.6f},{val_psnr:.4f},"
                    f"{ema_ssim:.6f},{ema_psnr:.4f},"
                    f"{lambda_gan:.6f},{duration:.2f}\n")

        # ── Checkpointing ──
        if ema_ssim > best_ema_ssim:
            best_ema_ssim = ema_ssim
            ckpt = {
                'epoch': epoch,
                'model_state_dict': G.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'scaler': scaler.state_dict(),
                'best_ema_ssim': best_ema_ssim,
                'config': config,
            }
            torch.save(ckpt, ckpt_dir / 'best.pt')
            print(f"  * New best EMA SSIM: {best_ema_ssim:.4f}")

        if epoch % args.ckpt_interval == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': G.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'scaler': scaler.state_dict(),
                'best_ema_ssim': best_ema_ssim,
                'config': config,
            }
            torch.save(ckpt, ckpt_dir / f'epoch_{epoch:03d}.pt')

        # Memory cleanup
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        # Wall-clock budget (time-matched baselines, e.g. b4 = 3600 s)
        if args.max_seconds and (time.time() - total_start) >= args.max_seconds:
            print(f"  [WC] wall-clock budget {args.max_seconds}s reached "
                  f"at epoch {epoch}; stopping.")
            break

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best EMA SSIM: {best_ema_ssim:.4f}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}")

    # ── Save final summary ──
    summary = {
        'best_ema_ssim': best_ema_ssim,
        'total_time_hours': total_time / 3600,
        'total_epochs': epoch,
        'run_dir': str(run_dir),
    }
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
