#!/usr/bin/env python3
"""
Stage 2: Multi-axis fine-tuning for Sparse Micro-CT Slice Interpolation.

Trains a single UNetG on z + x + y axis data simultaneously (balanced
sampling). Initializes G (and its EMA) from the Stage 1 best checkpoint;
the discriminator is RE-INITIALIZED from scratch.

Key features:
  - Optional DDP multi-GPU training (single-GPU fallback is the default)
  - BalancedMultiAxisDataset(z, x, y): each axis contributes equally
  - Cosine LR schedule (fine-tuning from Stage 1, LR scaled by --lr_scale)
  - Validation on all 3 axes separately
  - Morphological metrics per axis; checkpoints by best EMA z-SSIM and
    best validation z-axis dphi
  - Optional wall-clock budget (--max_seconds; the paper's Stage 2 runs
    used a 90-minute budget, i.e. --max_seconds 5400)

Usage (single GPU, as in the paper's deployment setting):
  CUDA_VISIBLE_DEVICES=0 python train_stage2.py \\
      --volume_path data/BB_1000c_f32.bin \\
      --stage1_ckpt outputs/stage1/<run>/checkpoints/best.pt \\
      --preset pareto4 --max_epochs 100 --max_seconds 5400 --gpu 0

Multi-GPU DDP:
  torchrun --nproc_per_node=4 train_stage2.py \\
      --volume_path data/BB_1000c_f32.bin \\
      --stage1_ckpt <path_to_stage1_best.pt> --preset pareto4
"""

import os, sys, json, time, random, math, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maps.models import UNetG, PatchD, EMA, count_parameters
from maps.losses import CombinedLoss, ssim_value
from maps.data import (load_volume, compute_splits, SliceInterpDataset,
                       BalancedMultiAxisDataset)
from maps.metrics import compute_all_morphological_metrics

# Pareto presets from the 60-trial multi-objective HPO (SSIM / dphi / dsa).
# The paper's final model uses 'pareto4' (physics-balanced preset).
PRESETS = {
    'pareto0': {
        'name': 'pareto0_best_ssim',
        'lr_G': 0.00042, 'lr_D': 0.00033, 'beta1': 0.5,
        'w_ssim': 0.374, 'morph_scale': 0.264, 'w_s2': 0.0614,
        'soft_temperature': 10,
        'lambda_gan': 0.0940, 'gan_warmup': 22, 'lambda_decay': 0.89,
    },
    'pareto5': {
        'name': 'pareto5_balanced',
        'lr_G': 0.00049, 'lr_D': 0.00017, 'beta1': 0.5,
        'w_ssim': 0.380, 'morph_scale': 0.219, 'w_s2': 0.0112,
        'soft_temperature': 20,
        'lambda_gan': 0.0921, 'gan_warmup': 33, 'lambda_decay': 0.82,
    },
    'pareto4': {
        'name': 'pareto4_best_physics',
        'lr_G': 0.00045, 'lr_D': 0.00019, 'beta1': 0.5,
        'w_ssim': 0.349, 'morph_scale': 0.174, 'w_s2': 0.0177,
        'soft_temperature': 10,
        'lambda_gan': 0.1311, 'gan_warmup': 27, 'lambda_decay': 0.77,
    },
}


def set_seed(seed, rank=0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_ddp():
    """Initialize DDP. Returns (rank, world_size, local_rank, is_ddp)."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_ddp(is_ddp):
    if is_ddp:
        dist.destroy_process_group()


def print_rank0(msg, rank=0):
    if rank == 0:
        print(msg)


def create_axis_val_loader(vol, splits, cfg, axis):
    """Create a single-axis validation loader."""
    ds = SliceInterpDataset(
        vol, splits['val'], axis=axis,
        in_ch=cfg['in_ch'], patch_size=cfg['patch_size'],
        train=False, offsets=cfg.get('offsets'))
    return DataLoader(ds, batch_size=cfg['batch_size'],
                      num_workers=cfg['num_workers'],
                      pin_memory=True, shuffle=False)


def validate_axis(G, loader, device):
    """Run validation on a single axis. Returns mean SSIM."""
    G.eval()
    ssim_sum = 0.0
    n = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for xv, yv in loader:
            xv = xv.to(device).float()
            yv = yv.to(device).float()
            pv = G(xv)
            ssim_sum += ssim_value(pv, yv)
            n += 1
    return ssim_sum / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description='Stage2: Multi-axis Training')
    parser.add_argument('--stage1_ckpt', required=True, type=str,
                        help='Path to Stage1 best.pt checkpoint')
    parser.add_argument('--preset', required=True, choices=list(PRESETS.keys()))
    parser.add_argument('--volume_path', required=True, type=str,
                        help='Path to float32/uint8 binary (or TIFF) volume')
    parser.add_argument('--volume_shape', nargs=3, type=int,
                        default=[1000, 1000, 1000])
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id for single-GPU mode (ignored in DDP)')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--offsets', type=int, nargs='+', default=None,
                        help='6-channel input offsets (default: '
                             '-15 -9 -3 3 9 15). For per-k retraining pass '
                             'scaled offsets.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Per-GPU batch size')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Full morphological eval every N epochs')
    parser.add_argument('--ckpt_interval', type=int, default=25)
    parser.add_argument('--lr_scale', type=float, default=0.5,
                        help='Scale Stage1 LR for fine-tuning (default 0.5x)')
    parser.add_argument('--lambda_gan_override', type=float, default=None,
                        help='Override preset lambda_gan (set to 0.0 for the '
                             'no-GAN ablation)')
    parser.add_argument('--run_tag', type=str, default='',
                        help='Optional tag appended to run dir name')
    parser.add_argument('--out_dir', type=str, default='outputs/stage2',
                        help='Root directory for run outputs')
    parser.add_argument('--max_seconds', type=int, default=None,
                        help='If set, training stops at this wall-clock budget '
                             '(in addition to max_epochs). Paper: 5400 (90 min)')
    args = parser.parse_args()

    # ── DDP Setup ──
    rank, world_size, local_rank, is_ddp = setup_ddp()

    preset = PRESETS[args.preset]
    set_seed(args.seed, rank)

    if is_ddp:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(f'cuda:{args.gpu}')

    timestamp = datetime.now().strftime('%m%d_%H%M')
    tag = f'_{args.run_tag}' if args.run_tag else ''
    run_dir = Path(args.out_dir) / \
        f'stage2_multiaxis_{preset["name"]}{tag}_{timestamp}'

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
    if is_ddp:
        dist.barrier()
    ckpt_dir = run_dir / 'checkpoints'

    print_rank0(f"{'='*70}", rank)
    print_rank0(f"Stage2 Multi-axis (DDP={is_ddp}, world_size={world_size})", rank)
    print_rank0(f"Preset: {preset['name']}", rank)
    print_rank0(f"Stage1 checkpoint: {args.stage1_ckpt}", rank)
    print_rank0(f"Run dir: {run_dir}", rank)
    print_rank0(f"Epochs: {args.max_epochs} | LR scale: {args.lr_scale} "
                f"| Batch/GPU: {args.batch_size}", rank)
    print_rank0(f"Effective batch size: {args.batch_size * world_size}", rank)
    print_rank0(f"{'='*70}", rank)

    # ── Data ──
    _vol_shape = tuple(args.volume_shape)
    # Each rank loads independently (memmap is cheap)
    vol = load_volume(args.volume_path, _vol_shape)
    splits = compute_splits(vol.shape[0])

    cfg_data = {
        'in_ch': 6, 'patch_size': args.patch_size,
        'batch_size': args.batch_size, 'num_workers': args.num_workers,
        'offsets': args.offsets,
    }

    # Multi-axis training dataset (z + x + y, balanced sampling)
    # Per-k retrain: pass --offsets to override the default OFFSETS_IN6.
    ds_offsets = args.offsets if args.offsets is not None else None
    train_ds = BalancedMultiAxisDataset(
        vol, splits['train'], axes=['z', 'x', 'y'],
        in_ch=6, patch_size=args.patch_size, train=True,
        offsets=ds_offsets)

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            drop_last=True, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=args.num_workers > 0)
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=args.num_workers > 0)

    # Validation/Test loaders (rank 0 only for evaluation)
    if rank == 0:
        val_z = create_axis_val_loader(vol, splits, cfg_data, 'z')
        val_x = create_axis_val_loader(vol, splits, cfg_data, 'x')
        val_y = create_axis_val_loader(vol, splits, cfg_data, 'y')

        test_ds = SliceInterpDataset(
            vol, splits['test'], axis='z', in_ch=6,
            patch_size=args.patch_size, train=False,
            offsets=args.offsets)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True, shuffle=False)

    print_rank0(f"[DATA] Train: {len(train_ds)} samples (z+x+y combined)", rank)

    # ── Models ──
    G = UNetG(in_ch=6, base=80).to(device)
    # Discriminator is re-initialized (NOT loaded from Stage 1)
    D = PatchD(in_ch=7, base=64).to(device)

    # Load Stage1 checkpoint (all ranks)
    print_rank0(f"\n[LOAD] Loading Stage1 checkpoint: {args.stage1_ckpt}", rank)
    ckpt = torch.load(args.stage1_ckpt, map_location=device)
    G.load_state_dict(ckpt['model_state_dict'])
    print_rank0(f"  G loaded from epoch {ckpt.get('epoch', '?')}, "
                f"best EMA SSIM = {ckpt.get('best_ema_ssim', '?')}", rank)

    # EMA (on rank 0 for eval; all ranks maintain for consistency)
    ema = EMA(G, decay=0.999)
    if 'ema_state_dict' in ckpt:
        ema.load_state_dict(ckpt['ema_state_dict'])
        print_rank0("  EMA weights loaded", rank)

    # Wrap with DDP
    if is_ddp:
        G_ddp = DDP(G, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        D_ddp = DDP(D, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    else:
        G_ddp = G
        D_ddp = D

    # Optimizers with scaled LR. Fine-tuning is conservative:
    # scale by sqrt(world_size) instead of the linear scaling rule.
    lr_scale_ddp = math.sqrt(world_size) if is_ddp else 1.0
    lr_G = preset['lr_G'] * args.lr_scale * lr_scale_ddp
    lr_D = preset['lr_D'] * args.lr_scale * lr_scale_ddp
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_G,
                             betas=(preset['beta1'], 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_D,
                             betas=(preset['beta1'], 0.999))

    # Cosine annealing
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=args.max_epochs, eta_min=lr_G * 0.01)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=args.max_epochs, eta_min=lr_D * 0.01)

    scaler = GradScaler(enabled=True)

    loss_cfg = {
        'w_l1': 1.0, 'w_ssim': preset['w_ssim'], 'w_grad': 0.0,
        'w_phi': preset['morph_scale'], 'w_sa': preset['morph_scale'],
        'w_s2': preset['w_s2'], 'w_lpath': preset['morph_scale'],
        'soft_temperature': preset['soft_temperature'],
        'gan_mode': 'hinge',
    }
    criterion = CombinedLoss(loss_cfg)

    print_rank0(f"\n[CONFIG] lr_G={lr_G:.6f}, lr_D={lr_D:.6f} "
                f"(scale={args.lr_scale} x ddp_scale={lr_scale_ddp:.2f})", rank)
    print_rank0(f"G params: {count_parameters(G):,}", rank)

    # Save config (rank 0)
    config = {
        **vars(args), 'preset': preset, 'loss_cfg': loss_cfg,
        'base_ch': 80, 'in_ch': 6, 'splits': splits,
        'stage': 'stage2_multiaxis',
        'lr_G_effective': lr_G, 'lr_D_effective': lr_D,
        'world_size': world_size,
        'effective_batch_size': args.batch_size * world_size,
    }
    if rank == 0:
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # ── Logs (rank 0) ──
    if rank == 0:
        log_path = run_dir / 'train_log.csv'
        with open(log_path, 'w') as f:
            f.write('epoch,train_G,train_D,train_dphi,train_dsa,'
                    'val_z_ssim,val_x_ssim,val_y_ssim,'
                    'ema_z_ssim,ema_x_ssim,ema_y_ssim,'
                    'lambda_gan,lr_G,duration_s\n')

        morph_log_path = run_dir / 'morph_log.csv'
        with open(morph_log_path, 'w') as f:
            # Per-axis morphology: z, x, y separately + mean
            f.write('epoch,ema_z_ssim,'
                    'dphi_z,dphi_x,dphi_y,dphi_mean,'
                    'dsa_z,dsa_x,dsa_y,dsa_mean,'
                    's2_mse_z,s2_mse_x,s2_mse_y,'
                    'lpath_mse_z,lpath_mse_x,lpath_mse_y,'
                    'd_euler_z,d_euler_x,d_euler_y\n')

    # ── Training ──
    best_ema_ssim = -1.0
    best_val_dphi = float('inf')   # parallel ckpt by min(val z-axis dphi)
    total_start = time.time()

    # GAN warmup: shorter for Stage2 (already pretrained)
    gan_warmup_s2 = max(5, preset['gan_warmup'] // 3)

    for epoch in range(1, args.max_epochs + 1):
        # Time-budget early stop (in addition to max_epochs)
        if args.max_seconds is not None:
            elapsed = time.time() - total_start
            if elapsed > args.max_seconds:
                if rank == 0:
                    print(f"[STAGE2] time budget {args.max_seconds}s "
                          f"reached at epoch {epoch}")
                break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        G_ddp.train(); D_ddp.train()

        # Lambda schedule
        lam_base = (args.lambda_gan_override
                    if args.lambda_gan_override is not None
                    else preset['lambda_gan'])
        if epoch < gan_warmup_s2:
            lam = lam_base * epoch / gan_warmup_s2
        else:
            progress = (epoch - gan_warmup_s2) / max(1, args.max_epochs - gan_warmup_s2)
            lam = lam_base * (1.0 - preset['lambda_decay'] * progress)
            lam = max(lam, lam_base * 0.1)

        g_sum = d_sum = dphi_sum = dsa_sum = 0.0
        n_steps = 0
        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # D step (use unwrapped G with no_grad to avoid DDP overhead)
            opt_D.zero_grad(set_to_none=True)
            with torch.no_grad(), autocast():
                fake_detach = G(x).detach()
            with autocast():
                d_loss, d_m = criterion.compute_D_loss(D_ddp, x, y, fake_detach)
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)

            # G step
            opt_G.zero_grad(set_to_none=True)
            with autocast():
                g_loss, g_m = criterion.compute_G_loss(G_ddp, D_ddp, x, y, lam)
            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler.step(opt_G)
            scaler.update()

            ema.update()
            g_sum += g_m['G_total']; d_sum += d_m['D_loss']
            dphi_sum += g_m['dphi']; dsa_sum += g_m['dsa']
            n_steps += 1

        duration = time.time() - t0
        sched_G.step(); sched_D.step()
        cur_lr = sched_G.get_last_lr()[0]

        # ── Validation (rank 0 only) ──
        if rank == 0:
            # EMA validation (all axes)
            ema.store(); ema.apply()
            ema_z = validate_axis(G, val_z, device)
            ema_x = validate_axis(G, val_x, device)
            ema_y = validate_axis(G, val_y, device)
            ema.restore()

            # Raw model validation
            G.eval()
            val_z_raw = validate_axis(G, val_z, device)
            val_x_raw = validate_axis(G, val_x, device)
            val_y_raw = validate_axis(G, val_y, device)

            with open(log_path, 'a') as f:
                f.write(f'{epoch},{g_sum/n_steps:.6f},{d_sum/n_steps:.6f},'
                        f'{dphi_sum/n_steps:.6f},{dsa_sum/n_steps:.6f},'
                        f'{val_z_raw:.6f},{val_x_raw:.6f},{val_y_raw:.6f},'
                        f'{ema_z:.6f},{ema_x:.6f},{ema_y:.6f},'
                        f'{lam:.6f},{cur_lr:.8f},{duration:.2f}\n')

            print(f'[E{epoch:03d}] z={ema_z:.4f} x={ema_x:.4f} y={ema_y:.4f} '
                  f'G={g_sum/n_steps:.3f} D={d_sum/n_steps:.3f} '
                  f'dphi={dphi_sum/n_steps:.4f} lam={lam:.4f} lr={cur_lr:.6f} '
                  f'{duration:.0f}s')

            # ── Morphological Eval (per-axis: z, x, y) ──
            if epoch % args.eval_interval == 0 or epoch == args.max_epochs:
                ema.store(); ema.apply(); G.eval()

                def eval_axis_morph(loader):
                    pp, tt = [], []
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                        for xv, yv in loader:
                            xv = xv.to(device).float(); yv = yv.to(device).float()
                            pp.append(G(xv).cpu()); tt.append(yv.cpu())
                    return compute_all_morphological_metrics(
                        torch.cat(pp), torch.cat(tt), max_lag=32)

                m_z = eval_axis_morph(val_z)
                m_x = eval_axis_morph(val_x)
                m_y = eval_axis_morph(val_y)
                ema.restore()

                with open(morph_log_path, 'a') as f:
                    f.write(f'{epoch},{ema_z:.6f},'
                            f'{m_z["dphi"]:.6f},{m_x["dphi"]:.6f},{m_y["dphi"]:.6f},'
                            f'{(m_z["dphi"]+m_x["dphi"]+m_y["dphi"])/3:.6f},'
                            f'{m_z["dsa"]:.6f},{m_x["dsa"]:.6f},{m_y["dsa"]:.6f},'
                            f'{(m_z["dsa"]+m_x["dsa"]+m_y["dsa"])/3:.6f},'
                            f'{m_z["s2_mse"]:.8f},{m_x["s2_mse"]:.8f},{m_y["s2_mse"]:.8f},'
                            f'{m_z["lpath_mse"]:.8f},{m_x["lpath_mse"]:.8f},{m_y["lpath_mse"]:.8f},'
                            f'{m_z["d_euler"]:.4f},{m_x["d_euler"]:.4f},{m_y["d_euler"]:.4f}\n')

                print(f'  [MORPH z] dphi={m_z["dphi"]:.4f} dsa={m_z["dsa"]:.4f}')
                print(f'  [MORPH x] dphi={m_x["dphi"]:.4f} dsa={m_x["dsa"]:.4f}')
                print(f'  [MORPH y] dphi={m_y["dphi"]:.4f} dsa={m_y["dsa"]:.4f}')

                # ── best-by-val-dphi (z-axis) checkpoint ──
                if m_z['dphi'] < best_val_dphi:
                    best_val_dphi = m_z['dphi']
                    ema.store(); ema.apply()
                    torch.save({
                        'epoch': epoch, 'model_state_dict': G.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'D_state_dict': D.state_dict(),
                        'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_val_dphi': best_val_dphi,
                        'val_dphi_z': m_z['dphi'], 'val_dphi_x': m_x['dphi'],
                        'val_dphi_y': m_y['dphi'],
                        'val_ema_z_ssim': ema_z,
                        'config': config, 'stage': 'stage2',
                        'criterion': 'best_dphi',
                    }, ckpt_dir / 'best_dphi.pt')
                    ema.restore()
                    print(f'  * New best val dphi (z): {best_val_dphi:.5f}')

            # ── Checkpoint ──
            if ema_z > best_ema_ssim:
                best_ema_ssim = ema_z
                ema.store(); ema.apply()
                torch.save({
                    'epoch': epoch, 'model_state_dict': G.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_ema_ssim': best_ema_ssim, 'config': config,
                    'stage': 'stage2',
                }, ckpt_dir / 'best.pt')
                ema.restore()
                print(f'  * New best EMA z-SSIM: {best_ema_ssim:.4f}')

            if epoch % args.ckpt_interval == 0:
                ema.store(); ema.apply()
                torch.save({
                    'epoch': epoch, 'model_state_dict': G.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'best_ema_ssim': best_ema_ssim, 'config': config,
                    'stage': 'stage2',
                }, ckpt_dir / f'epoch_{epoch:03d}.pt')
                ema.restore()

        # Sync all ranks after checkpoint
        if is_ddp:
            dist.barrier()

        if epoch % 10 == 0:
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # ── Final Test Eval (rank 0 only) ──
    if rank == 0:
        print(f'\n{"="*70}')
        print(f'Stage2 complete! Best EMA z-SSIM: {best_ema_ssim:.4f}')
        print(f'Total time: {total_time/3600:.1f} hours')
        print(f'Running final test evaluation...')

        ckpt_best = torch.load(ckpt_dir / 'best.pt', map_location=device)
        G.load_state_dict(ckpt_best['model_state_dict'])
        G.eval()

        # Build x and y test datasets
        test_x_ds = SliceInterpDataset(
            vol, splits['test'], axis='x', in_ch=6,
            patch_size=args.patch_size, train=False,
            offsets=args.offsets)
        test_y_ds = SliceInterpDataset(
            vol, splits['test'], axis='y', in_ch=6,
            patch_size=args.patch_size, train=False,
            offsets=args.offsets)
        test_x_loader = DataLoader(test_x_ds, batch_size=args.batch_size,
                                   num_workers=args.num_workers, pin_memory=True)
        test_y_loader = DataLoader(test_y_ds, batch_size=args.batch_size,
                                   num_workers=args.num_workers, pin_memory=True)

        def test_axis_full(loader):
            pp, tt, ss = [], [], []
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                for xv, yv in loader:
                    xv = xv.to(device).float(); yv = yv.to(device).float()
                    pv = G(xv)
                    pp.append(pv.cpu()); tt.append(yv.cpu())
                    ss.append(ssim_value(pv, yv))
            preds = torch.cat(pp); targets = torch.cat(tt)
            morph = compute_all_morphological_metrics(preds, targets, max_lag=32)
            return float(np.mean(ss)), morph

        test_z_ssim, test_z_morph = test_axis_full(test_loader)
        test_x_ssim, test_x_morph = test_axis_full(test_x_loader)
        test_y_ssim, test_y_morph = test_axis_full(test_y_loader)

        print(f'\n=== TEST RESULTS (Stage2 {preset["name"]}) ===')
        print(f'  SSIM     z={test_z_ssim:.4f} x={test_x_ssim:.4f} y={test_y_ssim:.4f}')
        for k in ['dphi', 'dsa', 's2_mse', 'lpath_mse', 'd_euler']:
            print(f'  {k:10s} z={test_z_morph[k]:.6f} '
                  f'x={test_x_morph[k]:.6f} y={test_y_morph[k]:.6f}')

        summary = {
            'preset': preset['name'],
            'stage': 'stage2_multiaxis',
            'best_ema_ssim': best_ema_ssim,
            'test_z_ssim': test_z_ssim,
            'test_x_ssim': test_x_ssim,
            'test_y_ssim': test_y_ssim,
            'test_morphology_z': test_z_morph,
            'test_morphology_x': test_x_morph,
            'test_morphology_y': test_y_morph,
            'total_time_hours': total_time / 3600,
            'total_epochs': args.max_epochs,
            'stage1_ckpt': args.stage1_ckpt,
            'world_size': world_size,
        }
        with open(run_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f'\n[SAVED] {run_dir / "summary.json"}')
        print(f'{"="*70}')

    cleanup_ddp(is_ddp)


if __name__ == '__main__':
    main()
