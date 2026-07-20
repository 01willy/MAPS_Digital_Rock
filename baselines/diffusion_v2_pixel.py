#!/usr/bin/env python3
"""
Diffusion baseline V2 -- pixel-space conditional 2D DDIM with proper
channel-concat sinusoidal time conditioning (fixes over V1).

Paper role: second of the three diffusion variants (V1/V2/V3) compared in
the "Latent diffusion, Transformer, and medical-CT baselines" results
section.

Key differences vs V1 (baselines/diffusion_v1_pixel.py):
  1. Time conditioning (Ho et al. 2020; Nichol & Dhariwal 2021):
     16-channel sinusoidal embedding broadcast to spatial dims and
     channel-concatenated. UNet input = 6 (cond) + 1 (noisy) + 16 (time) = 23.
  2. DDIM update stability (Song et al. 2020 Eq. 12): +1e-8 clamp on
     sqrt(alpha_t).
  3. x0 clamp to [0,1] only after the first 5% of sampling steps.
Invariants preserved from V1: stable anchor (condition channels never
noised), cosine schedule, T=1000, DDIM 50-step deployment.

Usage:
  torchrun --nproc_per_node=3 baselines/diffusion_v2_pixel.py \\
      --volume_path data/BB_1000c_f32.bin --max_seconds 86400 \\
      --out_dir outputs/baselines/diffusion_v2
"""
import argparse
import json
import math
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import load_volume, compute_splits, SliceInterpDataset  # noqa: E402


# ─── Time embedding ──────────────────────────────────────────────────────

def sinusoidal_time_embedding(t, dim, max_period=10000.0):
    """Standard DDPM sinusoidal position embedding for the diffusion
    timestep. Returns (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32,
                                     device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb                                                  # (B, dim)


def build_time_channels(t, T, H, W, n_channels=16):
    """Sinusoidal time encoding broadcast to (B, n_channels, H, W)."""
    t_norm = t.float() / T                                      # (B,) in [0,1]
    emb = sinusoidal_time_embedding(t_norm * T, n_channels)     # (B, n_ch)
    emb = emb.view(emb.shape[0], n_channels, 1, 1)
    emb = emb.expand(-1, -1, H, W)
    return emb


# ─── UNet with channel-concat time conditioning ──────────────────────────

class DiffusionUNetV2(nn.Module):
    """UNetG-shaped backbone, in_ch = 6 (sparse cond) + 1 (noisy target)
    + n_time (sinusoidal) = 23. NO sigmoid output (epsilon-prediction)."""

    def __init__(self, n_cond=6, n_time=16, base=80):
        super().__init__()
        in_ch = n_cond + 1 + n_time

        def cb(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1),
                                 nn.GroupNorm(8, o),
                                 nn.SiLU())
        b = base
        self.e1 = cb(in_ch, b)
        self.e2 = nn.Sequential(nn.AvgPool2d(2), cb(b, b * 2))
        self.e3 = nn.Sequential(nn.AvgPool2d(2), cb(b * 2, b * 4))
        self.e4 = nn.Sequential(nn.AvgPool2d(2), cb(b * 4, b * 8))
        self.bottleneck = nn.Sequential(
            nn.AvgPool2d(2), cb(b * 8, b * 16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.d4 = nn.Sequential(
            cb(b * 16 + b * 8, b * 8),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.d3 = nn.Sequential(
            cb(b * 8 + b * 4, b * 4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.d2 = nn.Sequential(
            cb(b * 4 + b * 2, b * 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.d1 = nn.Sequential(cb(b * 2 + b, b), nn.Conv2d(b, 1, 1))

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        bn = self.bottleneck(e4)
        d4 = self.d4(torch.cat([bn, e4], 1))
        d3 = self.d3(torch.cat([d4, e3], 1))
        d2 = self.d2(torch.cat([d3, e2], 1))
        return self.d1(torch.cat([d2, e1], 1))


# ─── Diffusion utilities ─────────────────────────────────────────────────

class GaussianDiffusionV2:
    def __init__(self, T=1000, schedule='cosine', device='cuda', n_time_ch=16):
        self.T = T
        self.device = device
        self.n_time_ch = n_time_ch
        if schedule == 'linear':
            betas = torch.linspace(1e-4, 2e-2, T)
        else:
            s = 0.008
            x = torch.linspace(0, T, T + 1)
            ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
            ac = ac / ac[0]
            betas = torch.clip(1 - ac[1:] / ac[:-1], 1e-5, 0.999)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, cond_6ch, n_steps=50, eta=0.0):
        """Stable DDIM sampling. Conditioning channels held fixed
        (stable anchor). Time channel rebuilt at each step."""
        B, _, H, W = cond_6ch.shape
        x = torch.randn(B, 1, H, W, device=cond_6ch.device)
        steps = torch.linspace(self.T - 1, 0, n_steps + 1, dtype=torch.long,
                               device=cond_6ch.device)
        for i in range(n_steps):
            t_cur = steps[i]
            t_next = steps[i + 1]
            t_batch = torch.full((B,), t_cur.item(), device=x.device,
                                 dtype=torch.long)
            ac_cur = self.alphas_cumprod[t_cur]
            ac_next = (self.alphas_cumprod[t_next]
                       if t_next >= 0 else torch.tensor(1.0, device=x.device))
            tc = build_time_channels(t_batch, self.T, H, W, self.n_time_ch)
            inp = torch.cat([cond_6ch, x, tc], dim=1)
            eps_pred = model(inp)
            # DDIM update with epsilon clamp on sqrt(alpha_t)
            sqrt_ac = torch.sqrt(ac_cur + 1e-8)
            sqrt_one_minus_ac = torch.sqrt(1 - ac_cur)
            x0_pred = (x - sqrt_one_minus_ac * eps_pred) / sqrt_ac
            # Clamp x0 only after the first 5% of steps
            if i > int(0.05 * n_steps):
                x0_pred = x0_pred.clamp(0.0, 1.0)
            x = (torch.sqrt(ac_next) * x0_pred
                 + torch.sqrt(1 - ac_next) * eps_pred)
        return x.clamp(0.0, 1.0)


# ─── DDP setup ──────────────────────────────────────────────────────────

def setup_ddp():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local)
        return rank, world, local, True
    return 0, 1, 0, False


# ─── Training ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='V2 pixel-space diffusion baseline')
    ap.add_argument('--max_seconds', type=int, default=86400)  # 24h
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--ddim_steps', type=int, default=50)
    ap.add_argument('--n_time_ch', type=int, default=16)
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int, default=[1000, 1000, 1000])
    ap.add_argument('--run_tag', default='v2')
    ap.add_argument('--out_dir', type=str, default='outputs/baselines/diffusion_v2')
    ap.add_argument('--ckpt_every_min', type=int, default=30)
    args = ap.parse_args()

    rank, world, local, is_ddp = setup_ddp()
    device = torch.device(f'cuda:{local}' if is_ddp else f'cuda:{args.gpu}')

    s = args.seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_dir = Path(args.out_dir) / f'diff_{args.run_tag}_{timestamp}'
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'checkpoints').mkdir(exist_ok=True)
        (run_dir / 'config.json').write_text(
            json.dumps(vars(args), indent=2, default=str))

    log_path = run_dir / 'train.log' if rank == 0 else None

    def log(msg):
        if rank != 0:
            return
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'V2 diffusion start. world={world} rank={rank} device={device} '
        f'batch={args.batch_size} patch={args.patch_size} T={args.T} '
        f'n_time_ch={args.n_time_ch}')
    log(f'Loading {args.volume_path}')
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])

    train_ds = SliceInterpDataset(vol, splits['train'], axis='z', in_ch=6,
                                  patch_size=args.patch_size, train=True)
    sampler = (torch.utils.data.distributed.DistributedSampler(train_ds)
               if is_ddp else None)
    loader = DataLoader(train_ds, batch_size=args.batch_size,
                        sampler=sampler, shuffle=(sampler is None),
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True)
    log(f'train n={len(train_ds)}')

    model = DiffusionUNetV2(n_cond=6, n_time=args.n_time_ch, base=80).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'DiffusionUNetV2 params: {n_params / 1e6:.2f}M')

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=1e-4)
    diffusion = GaussianDiffusionV2(T=args.T, device=device,
                                    n_time_ch=args.n_time_ch)

    if rank == 0:
        with open(run_dir / 'train_log.csv', 'w') as f:
            f.write('step,epoch,loss,wall_seconds\n')

    t0 = time.time()
    last_ckpt = t0
    step = 0
    epoch = 0
    avg = None
    while time.time() - t0 < args.max_seconds:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        loss_acc = 0.0
        n_acc = 0
        for cond, target in loader:
            if time.time() - t0 >= args.max_seconds:
                break
            cond = cond.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            B = target.shape[0]
            t = torch.randint(0, diffusion.T, (B,), device=device)
            noisy_target, noise = diffusion.q_sample(target, t)
            tc = build_time_channels(t, diffusion.T, target.shape[-2],
                                     target.shape[-1], args.n_time_ch)
            inp = torch.cat([cond, noisy_target, tc], dim=1)
            pred = model(inp)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_acc += loss.item() * B
            n_acc += B
            step += 1
            if step % 200 == 0 and rank == 0:
                log(f'step {step} ep {epoch} loss {loss.item():.4f} '
                    f'cum {(time.time() - t0) / 60:.1f}min')

        if rank == 0 and n_acc > 0:
            avg = loss_acc / n_acc
            with open(run_dir / 'train_log.csv', 'a') as f:
                f.write(f'{step},{epoch},{avg:.6f},{time.time() - t0:.1f}\n')

        if rank == 0 and time.time() - last_ckpt > args.ckpt_every_min * 60:
            last_ckpt = time.time()
            mins = int((time.time() - t0) / 60)
            so = (model.module.state_dict() if is_ddp else model.state_dict())
            torch.save({'step': step, 'epoch': epoch,
                        'model_state_dict': so,
                        'opt': opt.state_dict(), 'args': vars(args),
                        'wall_minutes': mins, 'avg_loss': avg},
                       run_dir / 'checkpoints' / f'wc_{mins:03d}min.pt')
            log(f'  [ckpt] wc_{mins:03d}min.pt')

    if rank == 0:
        so = (model.module.state_dict() if is_ddp else model.state_dict())
        torch.save({'step': step, 'epoch': epoch,
                    'model_state_dict': so,
                    'opt': opt.state_dict(), 'args': vars(args)},
                   run_dir / 'checkpoints' / 'final.pt')
        log(f'V2 training done. step={step} ep={epoch} '
            f'wall={(time.time() - t0) / 3600:.2f}h')

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
