#!/usr/bin/env python3
"""
Joint multi-k training (Section 5.4, per-k retraining vs. single joint
multi-k model comparison).

Trains the UNetG backbone with a RANDOM offset-scale k per sample: k is
drawn uniformly from --k_pool and the input slices sit at t + o*k for the
base offsets o in OFFSETS_IN6 = [-15, -9, -3, 3, 9, 15] (the target stays
the single odd slice t). Exposure to every scale during training yields
one joint checkpoint, which is then evaluated across k with
`analysis/ksweep_eval.py` and compared against per-k retrained
checkpoints (`train_stage1.py --offsets <scaled pattern>`) -- the
per-k-vs-joint comparison of Section 5.4.

Losses: --loss l1_only (b4-style) or --loss morph (L1 + SSIM + soft-Otsu
porosity/surface-area). Training is wall-clock budgeted (--max_seconds)
and cycles the z/x/y axes per epoch, matching the MAPS training protocol.

Usage:
  CUDA_VISIBLE_DEVICES=0 python baselines/train_multik.py \\
      --volume_path data/BB_1000c_f32.bin --max_seconds 3600 \\
      --loss morph --out_dir outputs/baselines/multik
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.models import UNetG  # noqa: E402
from maps.data import load_volume, compute_splits, OFFSETS_IN6  # noqa: E402
from maps.losses import (ssim_value, ssim_loss, porosity_loss,  # noqa: E402
                         surface_area_loss)


class MultiKSliceDataset(Dataset):
    """Per sample: draw k from k_pool, pick an odd target t along `axis`,
    stack the 6 slices at t + o*k for o in OFFSETS_IN6 (input), target =
    the slice at t. All k must satisfy the 15*k offset margin."""

    def __init__(self, vol, slab_range, axis='z', patch_size=256,
                 k_pool=(1, 2, 3, 5), n_samples=2000):
        self.vol = vol
        self.slab_lo, self.slab_hi = slab_range
        self.axis = axis
        self.ps = patch_size
        self.n = n_samples
        self.offsets = OFFSETS_IN6
        if axis == 'z':
            view_axes = (0, 1, 2)
        elif axis == 'x':
            view_axes = (2, 0, 1)
        else:
            view_axes = (1, 0, 2)
        self.view_shape = tuple(vol.shape[a] for a in view_axes)
        self.D_view, self.H_view, self.W_view = self.view_shape
        max_k_fit = (self.slab_hi - self.slab_lo - 2) // (2 * 15)
        self.k_pool = [k for k in k_pool if k <= max_k_fit]
        assert self.k_pool, f'all k in {k_pool} too large for {slab_range}'

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        k = random.choice(self.k_pool)
        k_max = 15 * k
        t_lo = max(self.slab_lo, k_max)
        t_hi = min(self.slab_hi, self.D_view) - k_max
        if t_hi <= t_lo:
            return self.__getitem__((idx + 1) % self.n)
        for _ in range(20):
            t = random.randint(t_lo, t_hi - 1)
            if t % 2 == 1:
                break
        else:
            t = (t_lo + t_hi) // 2 | 1
        y0 = random.randint(0, self.H_view - self.ps)
        x0 = random.randint(0, self.W_view - self.ps)
        if self.axis == 'z':
            inp = np.stack([self.vol[t + o * k, y0:y0 + self.ps,
                                     x0:x0 + self.ps]
                            for o in self.offsets], axis=0)
            tgt = self.vol[t, y0:y0 + self.ps, x0:x0 + self.ps]
        elif self.axis == 'x':
            inp = np.stack([self.vol[y0:y0 + self.ps, x0:x0 + self.ps,
                                     t + o * k]
                            for o in self.offsets], axis=0)
            tgt = self.vol[y0:y0 + self.ps, x0:x0 + self.ps, t]
        else:
            inp = np.stack([self.vol[y0:y0 + self.ps, t + o * k,
                                     x0:x0 + self.ps]
                            for o in self.offsets], axis=0)
            tgt = self.vol[y0:y0 + self.ps, t, x0:x0 + self.ps]
        inp = np.clip(inp.astype(np.float32), 0.0, 1.0)
        tgt = np.clip(tgt.astype(np.float32), 0.0, 1.0)
        return (torch.from_numpy(inp), torch.from_numpy(tgt).unsqueeze(0),
                k)


def main():
    ap = argparse.ArgumentParser(description='Joint multi-k training')
    ap.add_argument('--volume_path', required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--max_seconds', type=int, default=3600)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--loss', choices=['l1_only', 'morph'], default='morph',
                    help='l1_only = b4-style; morph = MAPS-style')
    ap.add_argument('--w_l1', type=float, default=1.0)
    ap.add_argument('--w_ssim', type=float, default=0.3)
    ap.add_argument('--w_phi', type=float, default=0.15)
    ap.add_argument('--w_sa', type=float, default=0.15)
    ap.add_argument('--soft_temperature', type=float, default=10.0)
    ap.add_argument('--k_pool', nargs='+', type=int, default=[1, 2, 3, 5])
    ap.add_argument('--n_samples', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--run_tag', default='multik')
    ap.add_argument('--out_dir', default='outputs/baselines/multik')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{args.gpu}')
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_dir = Path(args.out_dir) / f'{args.run_tag}_{timestamp}'
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg['env'] = {
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES',
                                               'unset'),
        'torch_version': torch.__version__,
        'timestamp_start': datetime.now().isoformat(),
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])

    # If the largest k needs a wider slab than validation offers, extend the
    # val slab into the train tail (val-SSIM logging only; test untouched).
    max_k = max(args.k_pool)
    req_slab = 2 * 15 * max_k + 10
    val_lo, val_hi = splits['val']
    if (val_hi - val_lo) < req_slab:
        new_lo = max(0, val_hi - req_slab)
        print(f'[k={max_k}] val slab {val_lo}-{val_hi} too small; extending '
              f'to {new_lo}-{val_hi}')
        splits = dict(splits)
        splits['val'] = (new_lo, val_hi)

    train_ds = {ax: MultiKSliceDataset(vol, splits['train'], axis=ax,
                                       patch_size=args.patch_size,
                                       k_pool=tuple(args.k_pool),
                                       n_samples=args.n_samples)
                for ax in ('z', 'x', 'y')}
    val_ds = MultiKSliceDataset(vol, splits['val'], axis='z',
                                patch_size=args.patch_size,
                                k_pool=tuple(args.k_pool), n_samples=100)
    train_loaders = {ax: DataLoader(train_ds[ax],
                                    batch_size=args.batch_size,
                                    shuffle=True, drop_last=True,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
                     for ax in ('z', 'x', 'y')}
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)

    G = UNetG(in_ch=6, base=80).to(device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f'[multik] params={n_params / 1e6:.2f}M loss={args.loss} '
          f'k_pool={args.k_pool}')
    opt = torch.optim.Adam(G.parameters(), lr=args.lr)
    scaler = GradScaler()

    log_path = run_dir / 'train_log.csv'
    with open(log_path, 'w') as f:
        f.write('epoch,axis,train_loss,val_ssim,duration_s,cum_seconds\n')

    start = time.time()
    best_val_ssim = -1.0
    epoch = 0
    axes_cycle = ('z', 'x', 'y')
    while True:
        if time.time() - start > args.max_seconds:
            print(f'[multik] budget reached at epoch {epoch}')
            break
        epoch += 1
        axis = axes_cycle[(epoch - 1) % 3]
        loader = train_loaders[axis]
        G.train()
        ep_loss, n = 0.0, 0
        t0 = time.time()
        for x, y, _k in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                pred = G(x)
                if args.loss == 'l1_only':
                    total = F.l1_loss(pred, y)
                else:
                    total = (args.w_l1 * F.l1_loss(pred, y)
                             + args.w_ssim * ssim_loss(pred, y)
                             + args.w_phi * porosity_loss(
                                 pred, y, args.soft_temperature)[0]
                             + args.w_sa * surface_area_loss(
                                 pred, y, args.soft_temperature)[0])
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(total.detach())
            n += 1
        duration = time.time() - t0

        G.eval()
        val_ssims = []
        with torch.no_grad():
            for xv, yv, _k in val_loader:
                xv, yv = xv.to(device).float(), yv.to(device).float()
                val_ssims.append(ssim_value(G(xv), yv))
        val_ssim = float(np.mean(val_ssims))
        cum = time.time() - start
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{axis},{ep_loss / n:.6f},{val_ssim:.4f},'
                    f'{duration:.1f},{cum:.1f}\n')
        print(f'[E{epoch:03d}] ax={axis} loss={ep_loss / n:.4f} '
              f'val_ssim={val_ssim:.4f} {duration:.1f}s '
              f'cum={cum / 60:.1f}min')
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save({'epoch': epoch, 'model_state_dict': G.state_dict(),
                        'best_val_ssim': best_val_ssim, 'config': cfg},
                       run_dir / 'checkpoints' / 'best.pt')

    torch.save({'epoch': epoch, 'model_state_dict': G.state_dict(),
                'best_val_ssim': best_val_ssim, 'config': cfg},
               run_dir / 'checkpoints' / 'final.pt')
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump({'best_val_ssim': best_val_ssim, 'total_epochs': epoch,
                   'total_seconds': time.time() - start,
                   'n_params': n_params, 'run_dir': str(run_dir)}, f,
                  indent=2)
    print(f'[multik done] best_val_ssim={best_val_ssim:.4f} '
          f'after {epoch} epochs')


if __name__ == '__main__':
    main()
