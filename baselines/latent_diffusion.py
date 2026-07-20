#!/usr/bin/env python3
"""
Diffusion baseline V3 -- latent diffusion for digital rock (following the
approach of Naiff et al. 2025 / DiffSci), the best-performing of the three
diffusion variants in our comparison.

Paper rows: Section 5.7 (V3 latent diffusion, z_only 50-step; runtime in
Table 4), Table 4 (compute cost), qualitative Fig. panel, and the
DDIM-step sweep figure (sampled via baselines/eval_diffusion.py).

Two-stage training:
  STAGE 1 (--stage vae): train a 2D VAE on binary rock slices to compress
           256^2 -> 32^2 x 4ch latent (reconstruction MSE + KL).
  STAGE 2 (--stage ddim): train a conditional DDIM in the 32^2 x 4ch latent
           space: encode the 6 sparse conditioning slices and the target
           slice with the frozen VAE, learn eps-prediction on the noisy
           target latent conditioned on the flattened cond latents.
           At inference: DDIM-sample the target latent, VAE-decode to 256^2.

Paper training budgets: VAE ~6 h, DDIM ~18 h (single RTX 3090).

Usage:
  python baselines/latent_diffusion.py --stage vae \\
      --volume_path data/BB_1000c_f32.bin --max_seconds 21600 \\
      --out_dir outputs/baselines/latent_diffusion
  python baselines/latent_diffusion.py --stage ddim \\
      --vae_ckpt outputs/baselines/latent_diffusion/vae_*/checkpoints/best.pt \\
      --volume_path data/BB_1000c_f32.bin --max_seconds 64800 \\
      --out_dir outputs/baselines/latent_diffusion
"""
import argparse
import json
import math
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


# ─── 2D VAE for binary digital rock slices ──────────────────────────────

class VAE2D(nn.Module):
    """256^2 binary slice -> 32^2 x latent_ch latent (3 stride-2 stages)."""

    def __init__(self, base=64, latent_ch=4):
        super().__init__()
        # Encoder: 256->128->64->32, channels 1->64->128->256->latent
        self.enc = nn.Sequential(
            nn.Conv2d(1, base, 3, 2, 1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, 2, 1), nn.GroupNorm(16, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 3, 2, 1), nn.GroupNorm(16, base * 4), nn.SiLU(),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1), nn.GroupNorm(16, base * 4), nn.SiLU(),
        )
        self.mu = nn.Conv2d(base * 4, latent_ch, 1)
        self.logvar = nn.Conv2d(base * 4, latent_ch, 1)
        # Decoder: 32->64->128->256
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, base * 4, 3, 1, 1), nn.GroupNorm(16, base * 4), nn.SiLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.GroupNorm(16, base * 2), nn.SiLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.ConvTranspose2d(base, base, 4, 2, 1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(mu)
        recon = self.decode(z)
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return recon, mu, logvar, kld


# ─── Latent DDIM (32^2 x 4ch latent space, flattened cond latents) ───────

class LatentUNet(nn.Module):
    """U-Net on the 32x32 latent grid.
    Input channels: 6 cond latents x 4ch + 1 target x 4ch + 16 time = 44."""

    def __init__(self, latent_ch=4, n_cond=6, n_time=16, base=128):
        super().__init__()
        in_ch = (n_cond + 1) * latent_ch + n_time

        def cb(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1),
                                 nn.GroupNorm(8, o), nn.SiLU())
        self.e1 = cb(in_ch, base)
        self.e2 = nn.Sequential(nn.AvgPool2d(2), cb(base, base * 2))      # 16
        self.e3 = nn.Sequential(nn.AvgPool2d(2), cb(base * 2, base * 4))  # 8
        self.bn = nn.Sequential(nn.AvgPool2d(2), cb(base * 4, base * 8),  # 4
                                nn.Upsample(scale_factor=2, mode='bilinear',
                                            align_corners=False))
        self.d3 = nn.Sequential(cb(base * 8 + base * 4, base * 4),
                                nn.Upsample(scale_factor=2, mode='bilinear',
                                            align_corners=False))
        self.d2 = nn.Sequential(cb(base * 4 + base * 2, base * 2),
                                nn.Upsample(scale_factor=2, mode='bilinear',
                                            align_corners=False))
        self.d1 = nn.Sequential(cb(base * 2 + base, base),
                                nn.Conv2d(base, latent_ch, 1))

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2)
        b = self.bn(e3)
        d3 = self.d3(torch.cat([b, e3], 1))
        d2 = self.d2(torch.cat([d3, e2], 1))
        return self.d1(torch.cat([d2, e1], 1))


def sinusoidal_emb(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32,
                                     device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def build_time_channels(t, T, H, W, n_channels=16):
    emb = sinusoidal_emb(t.float() / T * T, n_channels)
    return emb.view(emb.shape[0], n_channels, 1, 1).expand(-1, -1, H, W)


class LatentDiffusion:
    def __init__(self, T=1000, device='cuda', n_time_ch=16):
        self.T = T
        self.device = device
        self.n_time_ch = n_time_ch
        # Cosine schedule
        s = 0.008
        x = torch.linspace(0, T, T + 1)
        ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = torch.clip(1 - ac[1:] / ac[:-1], 1e-5, 0.999)
        self.betas = betas.to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1mac = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_ac[t].view(-1, 1, 1, 1)
        b = self.sqrt_1mac[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, cond_latents, n_steps=50, latent_ch=4):
        """cond_latents: (B, 6*latent_ch, 32, 32).
        Returns target latent (B, latent_ch, 32, 32)."""
        B = cond_latents.shape[0]
        H = W = cond_latents.shape[-1]
        x = torch.randn(B, latent_ch, H, W, device=cond_latents.device)
        steps = torch.linspace(self.T - 1, 0, n_steps + 1, dtype=torch.long,
                               device=x.device)
        for i in range(n_steps):
            t_cur = steps[i]
            t_next = steps[i + 1]
            t_batch = torch.full((B,), t_cur.item(), device=x.device,
                                 dtype=torch.long)
            ac_cur = self.alphas_cumprod[t_cur]
            ac_next = (self.alphas_cumprod[t_next]
                       if t_next >= 0 else torch.tensor(1.0, device=x.device))
            tc = build_time_channels(t_batch, self.T, H, W, self.n_time_ch)
            inp = torch.cat([cond_latents, x, tc], dim=1)
            eps_pred = model(inp)
            sqrt_ac = torch.sqrt(ac_cur + 1e-8)
            sqrt_1mac = torch.sqrt(1 - ac_cur)
            x0_pred = (x - sqrt_1mac * eps_pred) / sqrt_ac
            # Latent space: x0 not clamped (latents are unbounded)
            x = torch.sqrt(ac_next) * x0_pred + torch.sqrt(1 - ac_next) * eps_pred
        return x


# ─── VAE training ──────────────────────────────

def train_vae(args):
    device = torch.device(f'cuda:{args.gpu}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_dir = Path(args.out_dir) / f'vae_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    log_path = run_dir / 'train.log'

    def log(m):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {m}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'VAE training. seed={args.seed} max_seconds={args.max_seconds}')
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])
    # VAE is unconditional reconstruction; slices from the train range only.
    z_lo, z_hi = splits['train']
    print(f'VAE train slices: z=[{z_lo}, {z_hi})')

    class SliceDataset(torch.utils.data.Dataset):
        def __init__(self, vol, z_lo, z_hi, patch_size=256):
            self.vol = vol
            self.z_idx = list(range(z_lo, z_hi))
            self.patch_size = patch_size
            self.H, self.W = vol.shape[1], vol.shape[2]

        def __len__(self):
            return len(self.z_idx)

        def __getitem__(self, idx):
            z = self.z_idx[idx]
            sl = self.vol[z].astype(np.float32)
            sl = np.clip(sl, 0.0, 1.0)
            ps = self.patch_size
            y0 = random.randint(0, self.H - ps)
            x0 = random.randint(0, self.W - ps)
            return torch.from_numpy(sl[y0:y0 + ps, x0:x0 + ps].copy())[None]

    ds = SliceDataset(vol, z_lo, z_hi, patch_size=args.patch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True)
    log(f'VAE train slices: {len(ds)}')

    vae = VAE2D(base=64, latent_ch=4).to(device)
    n_params = sum(p.numel() for p in vae.parameters())
    log(f'VAE params: {n_params / 1e6:.2f}M')
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    t0 = time.time()
    step = 0
    epoch = 0
    best_recon = float('inf')
    while time.time() - t0 < args.max_seconds:
        epoch += 1
        vae.train()
        for x in loader:
            if time.time() - t0 >= args.max_seconds:
                break
            x = x.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast():
                recon, mu, logvar, kld = vae(x)
                # MSE (autocast-safe); binary in/out so MSE is reasonable.
                rec = F.mse_loss(recon, x)
                loss = rec + 1e-4 * kld
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 200 == 0:
                log(f'step {step} ep {epoch} mse {rec.item():.4f} '
                    f'kld {kld.item():.4f} cum {(time.time() - t0) / 60:.1f}min')
                if rec.item() < best_recon:
                    best_recon = rec.item()
                    torch.save({'epoch': epoch,
                                'model_state_dict': vae.state_dict(),
                                'best_recon': best_recon},
                               run_dir / 'checkpoints' / 'best.pt')

    torch.save({'epoch': epoch, 'model_state_dict': vae.state_dict()},
               run_dir / 'checkpoints' / 'final.pt')
    log(f'VAE done. ep={epoch} step={step} best_recon={best_recon:.4f}')


# ─── Latent DDIM training ──────────────────────────────

def train_ddim(args):
    device = torch.device(f'cuda:{args.gpu}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_dir = Path(args.out_dir) / f'ddim_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    log_path = run_dir / 'train.log'

    def log(m):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {m}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'Latent DDIM training. vae_ckpt={args.vae_ckpt}')
    vae = VAE2D(base=64, latent_ch=4).to(device)
    vae_ck = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(vae_ck['model_state_dict'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])
    train_ds = SliceInterpDataset(vol, splits['train'], axis='z', in_ch=6,
                                  patch_size=args.patch_size, train=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True)
    log(f'train n={len(train_ds)}')

    model = LatentUNet(latent_ch=4, n_cond=6, n_time=args.n_time_ch,
                       base=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'LatentUNet params: {n_params / 1e6:.2f}M')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = LatentDiffusion(T=args.T, device=device,
                                n_time_ch=args.n_time_ch)
    scaler = torch.cuda.amp.GradScaler()

    t0 = time.time()
    step = 0
    epoch = 0
    last_ckpt = t0
    while time.time() - t0 < args.max_seconds:
        epoch += 1
        model.train()
        for cond, target in loader:
            if time.time() - t0 >= args.max_seconds:
                break
            cond = cond.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            B = target.shape[0]
            # Encode the 6 cond slices + target via the frozen VAE (mu only)
            with torch.no_grad():
                cond_lats = []
                for c in range(cond.shape[1]):
                    mu, _ = vae.encode(cond[:, c:c + 1])
                    cond_lats.append(mu)
                cond_lat = torch.cat(cond_lats, dim=1)   # (B, 6*4, 32, 32)
                tgt_mu, _ = vae.encode(target)
                tgt_lat = tgt_mu                          # (B, 4, 32, 32)
            t = torch.randint(0, diffusion.T, (B,), device=device)
            noisy_lat, noise = diffusion.q_sample(tgt_lat, t)
            tc = build_time_channels(t, diffusion.T, tgt_lat.shape[-2],
                                     tgt_lat.shape[-1], args.n_time_ch)
            inp = torch.cat([cond_lat, noisy_lat, tc], dim=1)
            with torch.cuda.amp.autocast():
                pred = model(inp)
                loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 200 == 0:
                log(f'step {step} ep {epoch} loss {loss.item():.4f} '
                    f'cum {(time.time() - t0) / 60:.1f}min')

        if time.time() - last_ckpt > args.ckpt_every_min * 60:
            last_ckpt = time.time()
            mins = int((time.time() - t0) / 60)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'wall_minutes': mins,
                        'vae_ckpt': args.vae_ckpt, 'args': vars(args)},
                       run_dir / 'checkpoints' / f'wc_{mins:03d}min.pt')
            log(f'  [ckpt] wc_{mins:03d}min.pt')

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'vae_ckpt': args.vae_ckpt, 'args': vars(args)},
               run_dir / 'checkpoints' / 'final.pt')
    log(f'Latent DDIM done. ep={epoch} step={step}')


def main():
    ap = argparse.ArgumentParser(description='V3 latent diffusion baseline')
    ap.add_argument('--stage', choices=['vae', 'ddim'], required=True)
    ap.add_argument('--vae_ckpt', type=str, help='VAE ckpt for the ddim stage')
    ap.add_argument('--max_seconds', type=int, default=21600)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--n_time_ch', type=int, default=16)
    ap.add_argument('--volume_path', type=str, required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int, default=[1000, 1000, 1000])
    ap.add_argument('--out_dir', type=str,
                    default='outputs/baselines/latent_diffusion')
    ap.add_argument('--ckpt_every_min', type=int, default=30)
    args = ap.parse_args()

    if args.stage == 'vae':
        train_vae(args)
    else:
        if not args.vae_ckpt:
            raise SystemExit('--vae_ckpt required for ddim stage')
        train_ddim(args)


if __name__ == '__main__':
    main()
