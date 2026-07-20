#!/usr/bin/env python3
"""
Diffusion baseline V1 -- pixel-space conditional 2D DDIM
(Sparse3Diff-inspired) for sparse-slice interpolation.

Paper role: first of the three diffusion variants (V1/V2/V3) compared in
the "Latent diffusion, Transformer, and medical-CT baselines" results
section.

Architecture:
  - Backbone: UNetG-shaped network (in_ch=7, base=80) -- 6 sparse
    conditioning channels concatenated with 1 noisy target channel, NO
    sigmoid (epsilon-prediction needs unbounded reals).
  - Time conditioning: coarse multiplicative modulation of the noisy channel
    (1 + 0.01 * cos-embedding). V2 replaces this with proper channel-concat
    sinusoidal embedding.
  - Loss: epsilon-prediction MSE; cosine noise schedule (Nichol & Dhariwal
    2021), T=1000; DDIM sampling (50 steps at deployment).
  - Sparse3Diff "stable anchor": conditioning channels are never noised.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 python baselines/diffusion_v1_pixel.py \\
      --volume_path data/BB_1000c_f32.bin --max_seconds 14400 \\
      --out_dir outputs/baselines/diffusion_v1
DDP:
  torchrun --nproc_per_node=3 baselines/diffusion_v1_pixel.py ...
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


# UNet without sigmoid output -- DDPM epsilon-prediction needs unbounded reals.
class DiffusionUNet(nn.Module):
    def __init__(self, in_ch=7, base=80):
        super().__init__()

        def cb(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1),
                                 nn.BatchNorm2d(o),
                                 nn.LeakyReLU(0.2, True))
        b = base
        self.e1 = cb(in_ch, b)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), cb(b, b * 2))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), cb(b * 2, b * 4))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), cb(b * 4, b * 8))
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), cb(b * 8, b * 16),
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
        # Final head: NO sigmoid -- outputs unbounded epsilon prediction
        self.d1 = nn.Sequential(cb(b * 2 + b, b),
                                nn.Conv2d(b, 1, 1))

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        bn = self.bottleneck(e4)
        d4 = self.d4(torch.cat([bn, e4], 1))
        d3 = self.d3(torch.cat([d4, e3], 1))
        d2 = self.d2(torch.cat([d3, e2], 1))
        return self.d1(torch.cat([d2, e1], 1))


# ─── Diffusion utilities ──────────────────────────────────────────────────

def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


class GaussianDiffusion:
    def __init__(self, T=1000, schedule='cosine', device='cuda'):
        self.T = T
        self.device = device
        if schedule == 'linear':
            betas = linear_beta_schedule(T)
        else:
            # Cosine schedule (Nichol & Dhariwal 2021), tighter at low t
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 1e-5, 0.999)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """Forward noising x_t = sqrt(alpha_t) x0 + sqrt(1-alpha_t) eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, cond_6ch, n_steps=50, shape=None, eta=0.0):
        """DDIM sampling. cond_6ch fixed; only noisy-target channel denoised."""
        B, _, H, W = cond_6ch.shape
        if shape is None:
            shape = (B, 1, H, W)
        x = torch.randn(shape, device=cond_6ch.device)
        steps = torch.linspace(self.T - 1, 0, n_steps + 1, dtype=torch.long,
                               device=cond_6ch.device)
        for i in range(n_steps):
            t_cur = steps[i]
            t_next = steps[i + 1]
            ac_cur = self.alphas_cumprod[t_cur]
            ac_next = (self.alphas_cumprod[t_next]
                       if t_next >= 0 else torch.tensor(1.0, device=x.device))
            t_batch = torch.full((B,), t_cur, device=x.device, dtype=torch.long)
            t_emb = build_time_channel(t_batch, self.T, H, W)
            inp = torch.cat([cond_6ch, x], dim=1)  # (B, 7, H, W)
            # Time embedding via input modulation of the noisy channel
            # (V1's coarse trick; V2 replaces this with channel-concat).
            inp_t = inp.clone()
            inp_t[:, -1:] = inp_t[:, -1:] * (1.0 + 0.01 * t_emb)
            eps_pred = model(inp_t)            # (B, 1, H, W) -- predicts noise
            x0_pred = (x - torch.sqrt(1 - ac_cur) * eps_pred) / torch.sqrt(ac_cur)
            x0_pred = x0_pred.clamp(0.0, 1.0)
            # Deterministic DDIM (eta=0): no noise injection in reverse
            x = torch.sqrt(ac_next) * x0_pred + torch.sqrt(1 - ac_next) * eps_pred
        return x.clamp(0.0, 1.0)


def build_time_channel(t, T, H, W):
    """Cosine time embedding broadcast to (B, 1, H, W)."""
    t_norm = t.float() / T              # (B,)
    emb = torch.cos(t_norm * math.pi).view(-1, 1, 1, 1)
    return emb.expand(-1, 1, H, W)


# ─── DDP setup ───────────────────────────────────────────────────────────

def setup_ddp():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local)
        return rank, world, local, True
    return 0, 1, 0, False


# ─── Training ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='V1 pixel-space diffusion baseline')
    ap.add_argument('--max_seconds', type=int, default=14400)  # 4h
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--T', type=int, default=1000, help='Diffusion timesteps')
    ap.add_argument('--ddim_steps', type=int, default=50)
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int, default=[1000, 1000, 1000])
    ap.add_argument('--run_tag', default='v1')
    ap.add_argument('--out_dir', type=str, default='outputs/baselines/diffusion_v1')
    ap.add_argument('--ckpt_every_min', type=int, default=15)
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
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    log_path = run_dir / 'train.log' if rank == 0 else None

    def log(msg):
        if rank != 0:
            return
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'V1 diffusion baseline start. world={world} rank={rank} '
        f'device={device} batch={args.batch_size} patch={args.patch_size} '
        f'T={args.T}')
    log(f'Loading {args.volume_path}')
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])
    log(f'splits: {splits}')

    train_ds = SliceInterpDataset(vol, splits['train'], axis='z', in_ch=6,
                                  patch_size=args.patch_size, train=True,
                                  augment=True)
    sampler = (torch.utils.data.distributed.DistributedSampler(train_ds)
               if is_ddp else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    log(f'train n={len(train_ds)}')

    model = DiffusionUNet(in_ch=7, base=80).to(device)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=1e-4)
    diffusion = GaussianDiffusion(T=args.T, device=device)

    log_csv = run_dir / 'train_log.csv'
    if rank == 0:
        with open(log_csv, 'w') as f:
            f.write('step,epoch,loss,wall_seconds\n')

    t0 = time.time()
    last_ckpt = t0
    step = 0
    epoch = 0
    avg_loss = None
    while time.time() - t0 < args.max_seconds:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        loss_acc = 0.0
        n_acc = 0
        for cond, target in train_loader:
            if time.time() - t0 >= args.max_seconds:
                break
            cond = cond.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            B = target.shape[0]
            t = torch.randint(0, diffusion.T, (B,), device=device)
            noisy_target, noise = diffusion.q_sample(target, t)
            t_emb = build_time_channel(t, diffusion.T,
                                       target.shape[-2], target.shape[-1])
            inp = torch.cat([cond, noisy_target], dim=1)
            inp_t = inp.clone()
            inp_t[:, -1:] = inp_t[:, -1:] * (1.0 + 0.01 * t_emb)
            pred = model(inp_t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_acc += loss.item() * B
            n_acc += B
            step += 1
            if step % 100 == 0 and rank == 0:
                log(f'step {step} epoch {epoch} loss {loss.item():.4f} '
                    f'cum {(time.time() - t0) / 60:.1f}min')

        if rank == 0 and n_acc > 0:
            avg_loss = loss_acc / n_acc
            with open(log_csv, 'a') as f:
                f.write(f'{step},{epoch},{avg_loss:.6f},'
                        f'{time.time() - t0:.1f}\n')

        # Periodic ckpt
        if rank == 0 and time.time() - last_ckpt > args.ckpt_every_min * 60:
            last_ckpt = time.time()
            mins = int((time.time() - t0) / 60)
            save_obj = (model.module.state_dict() if is_ddp
                        else model.state_dict())
            torch.save({'step': step, 'epoch': epoch,
                        'model_state_dict': save_obj,
                        'opt': opt.state_dict(), 'args': vars(args),
                        'wall_minutes': mins, 'avg_loss': avg_loss},
                       run_dir / 'checkpoints' / f'wc_{mins:03d}min.pt')
            log(f'  [ckpt] wc_{mins:03d}min.pt')

    if rank == 0:
        save_obj = (model.module.state_dict() if is_ddp
                    else model.state_dict())
        torch.save({'step': step, 'epoch': epoch,
                    'model_state_dict': save_obj,
                    'opt': opt.state_dict(), 'args': vars(args)},
                   run_dir / 'checkpoints' / 'final.pt')
        log(f'V1 training complete. step={step} epoch={epoch} '
            f'wall={(time.time() - t0) / 3600:.2f}h')

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
