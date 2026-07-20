#!/usr/bin/env python3
"""
Train the 3D U-Net baseline (b5 / b5-large) under the fair single-target
protocol -- z-axis only, L1 loss on the single target slice per sample.

Paper rows:
  b5       (Tables 1, 3, 4):   --base 24  (~3.15 M params), 3 h wall-clock
  b5-large (Table 4, Fig. 10): --base 64  (~22.4 M params, capacity-matched)

Usage:
  CUDA_VISIBLE_DEVICES=0 python baselines/train_unet3d.py \\
      --volume_path data/BB_1000c_f32.bin --base 24 --seed 2025 \\
      --max_seconds 10800 --out_dir outputs/baselines/b5
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, compute_splits  # noqa: E402
from maps.losses import ssim_value  # noqa: E402
from baselines.unet3d import (UNet3D, Sparse3DDatasetFair,  # noqa: E402
                              worker_init_fn)


DEFAULT_WC_THRESHOLDS_MIN = [5, 10, 20, 30, 45, 60, 75, 90, 105, 120, 150]


class WallclockCheckpointer:
    """Saves a model snapshot once when each wall-clock threshold is crossed
    (used for the time-matched baseline comparison in the paper)."""

    def __init__(self, ckpt_dir, thresholds_min=None, prefix='wc_'):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = sorted(thresholds_min or DEFAULT_WC_THRESHOLDS_MIN)
        self.prefix = prefix
        self.fired = set()

    def maybe_save(self, model, elapsed_seconds, extras=None):
        elapsed_min = elapsed_seconds / 60.0
        for thr in self.thresholds:
            if thr in self.fired or elapsed_min < thr:
                continue
            payload = {'model_state_dict': model.state_dict(),
                       'elapsed_seconds': elapsed_seconds,
                       'threshold_min': thr}
            if extras:
                payload.update(extras)
            path = self.ckpt_dir / f'{self.prefix}{thr:03d}min.pt'
            torch.save(payload, path)
            self.fired.add(thr)
            print(f'  [WC] saved {path.name} at elapsed={elapsed_min:.2f}min')


def lock_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='b5 / b5-large 3D U-Net training')
    parser.add_argument('--volume_path', required=True,
                        help='float32/uint8 binary volume in [0,1] (see README)')
    parser.add_argument('--volume_shape', nargs=3, type=int,
                        default=[1000, 1000, 1000])
    parser.add_argument('--max_seconds', type=int, default=10800,
                        help='Wall-clock budget (default 3 h, paper protocol)')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=32,
                        help='32 = tight wrap of single target +-15 context')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base', type=int, default=24,
                        help='24 = b5 (~3.15M); 64 = b5-large (~22.4M)')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--run_tag', default='b5_fair')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--init_ckpt', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='outputs/baselines/unet3d')
    args = parser.parse_args()

    lock_determinism(args.seed)
    device = torch.device(f'cuda:{args.gpu}')

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_dir = Path(args.out_dir) / f'3d_unet_{args.run_tag}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)

    cfg = vars(args).copy()
    cfg['env'] = {
        'gpu_name': torch.cuda.get_device_name(args.gpu),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'timestamp_start': datetime.now().isoformat(),
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])
    print(f'[SPLITS] {splits}')

    train_ds = Sparse3DDatasetFair(vol, splits['train'], args.patch_size,
                                   args.n_samples)
    val_ds = Sparse3DDatasetFair(vol, splits['val'], args.patch_size, 50)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    G = UNet3D(in_ch=2, out_ch=1, base=args.base).to(device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f'[3D-UNet FAIR] base={args.base} params={n_params / 1e6:.2f}M '
          f'device={device}')
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device)
        G.load_state_dict(ck['model_state_dict'])
        print(f'[INIT] loaded from {args.init_ckpt}')

    opt = torch.optim.Adam(G.parameters(), lr=args.lr)
    scaler = GradScaler()

    log_path = run_dir / 'train_log.csv'
    with open(log_path, 'w') as f:
        f.write('epoch,train_l1,val_l1,val_ssim,duration_s,cum_seconds,'
                'samples_seen\n')

    wcc = WallclockCheckpointer(run_dir / 'checkpoints')
    start_time = time.time()
    best_val_ssim = -1.0
    epoch = 0
    peak_mem = 0
    samples_seen = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > args.max_seconds:
            print(f'[b5] time budget reached at epoch {epoch}')
            break
        epoch += 1
        G.train()
        ep_loss = 0.0
        n_steps = 0
        t0 = time.time()
        for x, y, lm, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lm = lm.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                pred = G(x)
                # Loss only on the SINGLE target slice per sample
                l1 = (F.l1_loss(pred, y, reduction='none') * lm).sum() \
                     / (lm.sum() + 1e-6)
            scaler.scale(l1).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(l1.detach())
            n_steps += 1
            samples_seen += x.shape[0]
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))
        duration = time.time() - t0
        wcc.maybe_save(G, time.time() - start_time,
                       extras={'epoch': epoch, 'samples_seen': samples_seen,
                               'config': {k: v for k, v in cfg.items()
                                          if k != 'env'}})

        # Validation: same single-target protocol
        G.eval()
        val_l1, val_ssim_vals = 0.0, []
        with torch.no_grad():
            for x, y, lm, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                lm = lm.to(device)
                pred = G(x.float()).float()
                l1 = (F.l1_loss(pred, y, reduction='none') * lm).sum() \
                     / (lm.sum() + 1e-6)
                val_l1 += float(l1)
                # SSIM at the single target slice
                for bi in range(pred.shape[0]):
                    for zi in range(pred.shape[2]):
                        if lm[bi, 0, zi, 0, 0] > 0.5:
                            p = pred[bi:bi + 1, :, zi]
                            g = y[bi:bi + 1, :, zi]
                            val_ssim_vals.append(ssim_value(p, g))
        val_l1 /= len(val_loader)
        val_ssim = float(np.mean(val_ssim_vals)) if val_ssim_vals else 0.0
        cum = time.time() - start_time
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{ep_loss / n_steps:.6f},{val_l1:.6f},'
                    f'{val_ssim:.4f},{duration:.1f},{cum:.1f},{samples_seen}\n')
        print(f'[E{epoch:03d}] tl1={ep_loss / n_steps:.4f} vl1={val_l1:.4f} '
              f'vssim={val_ssim:.4f} {duration:.1f}s cum={cum / 60:.1f}min '
              f'peak_mem={peak_mem / 1e9:.2f}GB')

        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save({'epoch': epoch, 'model_state_dict': G.state_dict(),
                        'best_val_ssim': best_val_ssim, 'config': cfg},
                       run_dir / 'checkpoints/best.pt')

    torch.save({'epoch': epoch, 'model_state_dict': G.state_dict(),
                'best_val_ssim': best_val_ssim, 'config': cfg},
               run_dir / 'checkpoints/final.pt')
    summary = {'best_val_ssim': best_val_ssim, 'total_epochs': epoch,
               'total_seconds': time.time() - start_time,
               'n_params': n_params, 'peak_memory_bytes': peak_mem,
               'run_dir': str(run_dir),
               'timestamp_end': datetime.now().isoformat()}
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[b5 DONE] best_val_ssim={best_val_ssim:.4f} after {epoch} epochs '
          f'in {(time.time() - start_time) / 60:.1f}min '
          f'peak_mem={peak_mem / 1e9:.2f}GB')


if __name__ == '__main__':
    main()
