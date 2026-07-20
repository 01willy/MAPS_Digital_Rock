#!/usr/bin/env python3
"""Training entry for the I3Net baseline (Table 1, z_only row).

Mirrors the structure of the SwinUNet training loop:
- Same SliceInterpDataset, OFFSETS_IN6 geometry, same z-slab splits
- Time-matched wall-clock training (90 min default)
- Pure L1 loss (faithful to the I3Net paper's `select_loss.py`)
- AdamW + weight decay (per I3Net config.py defaults: lr=3e-4, wd=1e-4)

Requires the official I3Net clone (see baselines/i3net.py docstring).

Usage:
  CUDA_VISIBLE_DEVICES=0 python baselines/train_i3net.py \\
      --volume_path data/BB_1000c_f32.bin --max_seconds 5400 --seed 2025 \\
      --out_dir outputs/baselines/i3net
"""
import argparse
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, compute_splits, SliceInterpDataset  # noqa: E402
from maps.losses import ssim_value  # noqa: E402
from baselines.i3net import I3NetRock  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='I3Net baseline training')
    ap.add_argument('--max_seconds', type=int, default=5400)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int, default=[1000, 1000, 1000])
    ap.add_argument('--run_tag', default='i3net')
    ap.add_argument('--out_dir', type=str, default='outputs/baselines/i3net')
    ap.add_argument('--ckpt_every_min', type=int, default=15)
    ap.add_argument('--n_feats', type=int, default=64)
    ap.add_argument('--num_blocks', type=int, default=16)
    ap.add_argument('--window_size', type=int, default=16)
    ap.add_argument('--no_amp', action='store_true',
                    help='Disable mixed precision (DCT module can NaN under FP16)')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_dir = Path(args.out_dir) / f'{args.run_tag}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    log_path = run_dir / 'train.log'

    def log(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'I3Net training. seed={args.seed} batch={args.batch_size} '
        f'patch={args.patch_size}')
    log(f'Loading {args.volume_path}')
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])
    train_ds = SliceInterpDataset(vol, splits['train'], axis='z', in_ch=6,
                                  patch_size=args.patch_size, train=True, augment=True)
    val_ds = SliceInterpDataset(vol, splits['val'], axis='z', in_ch=6,
                                patch_size=args.patch_size, train=False)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=2, pin_memory=True)
    log(f'train n={len(train_ds)} val n={len(val_ds)}')

    model = I3NetRock(in_ch=6, out_ch=1,
                      n_feats=args.n_feats, num_blocks=args.num_blocks,
                      window_size=args.window_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'I3NetRock params: {n_params / 1e6:.2f}M')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    use_amp = not args.no_amp
    log(f'AMP enabled: {use_amp}')

    with open(run_dir / 'train_log.csv', 'w') as f:
        f.write('step,epoch,loss,val_ssim,wall_seconds\n')

    t0 = time.time()
    last_ckpt = t0
    step = 0
    epoch = 0
    best_val_ssim = -1.0
    while time.time() - t0 < args.max_seconds:
        epoch += 1
        model.train()
        loss_acc = 0.0
        n_acc = 0
        for cond, target in loader:
            if time.time() - t0 >= args.max_seconds:
                break
            cond = cond.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(cond)
                loss = F.l1_loss(pred, target)  # pure L1 per I3Net paper
            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            # NaN guard
            if not torch.isfinite(loss):
                log(f'  ! NaN loss at step {step}, ep {epoch} -- skipping')
                opt.zero_grad()
                continue
            loss_acc += loss.item() * cond.shape[0]
            n_acc += cond.shape[0]
            step += 1
            if step % 100 == 0:
                log(f'step {step} ep {epoch} loss {loss.item():.4f} '
                    f'cum {(time.time() - t0) / 60:.1f}min')

        # Val
        model.eval()
        val_ssim = 0.0
        n_val = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv = xv.to(device).float()
                yv = yv.to(device).float()
                pv = model(xv)
                val_ssim += ssim_value(pv, yv)
                n_val += 1
        val_ssim /= max(n_val, 1)
        log(f'  ep {epoch} val_ssim={val_ssim:.4f}  best={best_val_ssim:.4f}')

        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_val_ssim': best_val_ssim, 'args': vars(args)},
                       run_dir / 'checkpoints' / 'best.pt')
            log('  * best ckpt saved')

        with open(run_dir / 'train_log.csv', 'a') as f:
            f.write(f'{step},{epoch},{loss_acc / max(n_acc, 1):.6f},'
                    f'{val_ssim:.6f},{time.time() - t0:.1f}\n')

        if time.time() - last_ckpt > args.ckpt_every_min * 60:
            last_ckpt = time.time()
            mins = int((time.time() - t0) / 60)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'wall_minutes': mins},
                       run_dir / 'checkpoints' / f'wc_{mins:03d}min.pt')

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_val_ssim': best_val_ssim},
               run_dir / 'checkpoints' / 'final.pt')
    log(f'I3Net training done. ep={epoch} best_val_ssim={best_val_ssim:.4f}')


if __name__ == '__main__':
    main()
