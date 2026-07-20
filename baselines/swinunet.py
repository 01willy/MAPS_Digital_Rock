#!/usr/bin/env python3
"""
SwinUNet 2D baseline: hybrid CNN + Swin-Transformer for 6-channel sparse-slice
interpolation. Appears in Table 1 (BB main comparison) and the SwinUNet
hyper-parameter sweep (archived; not in the compiled supplement).

Architecture:
  - Patch embed: 4x4 conv stride 4 (256^2 -> 64^2 x base ch)
  - 2 Swin blocks (W-MSA + SW-MSA) at 64^2 x base
  - Patch merge: 64^2 x base -> 32^2 x 2*base
  - 2 Swin blocks at 32^2 x 2*base
  - Patch expand x2 with skip connection, 2 more Swin blocks
  - Final two x2 transpose convs -> 256^2 x 1 ch, Sigmoid

Same 6-channel input / 1-channel output and the same time-matched wall-clock
training budget as the other 2D baselines (L1 loss).

Usage:
  CUDA_VISIBLE_DEVICES=0 python baselines/swinunet.py \\
      --volume_path data/BB_1000c_f32.bin --max_seconds 5400 --seed 2025 \\
      --out_dir outputs/baselines/swinunet
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, compute_splits, SliceInterpDataset  # noqa: E402
from maps.losses import ssim_value  # noqa: E402


# ─── Swin Transformer block (simplified) ──────────────────────────────

def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim))

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        if self.shift_size > 0:
            x_shifted = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x_shifted = x_norm
        win = window_partition(x_shifted, self.window_size)
        win = win.view(-1, self.window_size * self.window_size, C)
        attn_out = self.attn(win)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        out = window_reverse(attn_out, self.window_size, H, W)
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + out
        x = x + self.mlp(self.norm2(x))
        return x


class SwinUNet(nn.Module):
    """Hybrid CNN + Swin Transformer for 256^2 sparse slice interpolation.
    Input: (B, 6, 256, 256), Output: (B, 1, 256, 256) sigmoid."""

    def __init__(self, in_ch=6, base=96, num_heads=4, window_size=8):
        super().__init__()
        # Patch embed: conv stride 4 -> 64x64xbase
        self.embed = nn.Conv2d(in_ch, base, 4, stride=4)
        # Stage 1: 64x64xbase
        self.s1 = nn.ModuleList([
            SwinBlock(base, num_heads, window_size, shift=(i % 2 == 1))
            for i in range(2)])
        # Down: 64x64xbase -> 32x32x(base*2)
        self.down = nn.Conv2d(base, base * 2, 2, stride=2)
        # Stage 2: 32x32x(base*2)
        self.s2 = nn.ModuleList([
            SwinBlock(base * 2, num_heads * 2, window_size, shift=(i % 2 == 1))
            for i in range(2)])
        # Up: 32x32x(base*2) -> 64x64xbase
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        # Skip merge
        self.skip_conv = nn.Conv2d(base * 2, base, 1)
        # Stage 3: 64x64xbase
        self.s3 = nn.ModuleList([
            SwinBlock(base, num_heads, window_size, shift=(i % 2 == 1))
            for i in range(2)])
        # Final upsample 64 -> 256 (x4) via two x2 stages
        self.up2 = nn.ConvTranspose2d(base, base // 2, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(base // 2, base // 4, 2, stride=2)
        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(base // 4, base // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base // 4, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.embed(x)                 # (B, base, 64, 64)
        x = x.permute(0, 2, 3, 1)         # (B, 64, 64, base)
        for blk in self.s1:
            x = blk(x)
        skip1 = x
        x = x.permute(0, 3, 1, 2)         # (B, base, 64, 64)
        x = self.down(x)                  # (B, base*2, 32, 32)
        x = x.permute(0, 2, 3, 1)
        for blk in self.s2:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)
        x = self.up1(x)                   # (B, base, 64, 64)
        # Skip connection
        x = torch.cat([x, skip1.permute(0, 3, 1, 2)], dim=1)
        x = self.skip_conv(x)
        x = x.permute(0, 2, 3, 1)
        for blk in self.s3:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)
        x = self.up2(x)                   # (B, base/2, 128, 128)
        x = self.up3(x)                   # (B, base/4, 256, 256)
        return self.head(x)               # (B, 1, 256, 256) sigmoid


def main():
    ap = argparse.ArgumentParser(description='SwinUNet 2D baseline training')
    ap.add_argument('--max_seconds', type=int, default=5400)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int, default=[1000, 1000, 1000])
    ap.add_argument('--run_tag', default='swin')
    ap.add_argument('--out_dir', type=str, default='outputs/baselines/swinunet')
    ap.add_argument('--ckpt_every_min', type=int, default=15)
    ap.add_argument('--base', type=int, default=96, help='SwinUNet base channels')
    ap.add_argument('--num_heads', type=int, default=4, help='Attention heads')
    ap.add_argument('--window_size', type=int, default=8, help='Window attention size')
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

    log(f'SwinUNet baseline. seed={args.seed} batch={args.batch_size} '
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

    model = SwinUNet(in_ch=6, base=args.base, num_heads=args.num_heads,
                     window_size=args.window_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'SwinUNet params: {n_params / 1e6:.2f}M')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

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
            with torch.cuda.amp.autocast():
                pred = model(cond)
                loss = F.l1_loss(pred, target)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
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
    log(f'SwinUNet training done. ep={epoch} best_val_ssim={best_val_ssim:.4f}')


if __name__ == '__main__':
    main()
