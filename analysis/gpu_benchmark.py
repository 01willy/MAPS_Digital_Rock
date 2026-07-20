#!/usr/bin/env python3
"""
Per-GPU / multi-GPU benchmark suite (Tables S11-S12 of the Supplement and
the DDP-scaling figure).

Experiments:
  exp1   Inference latency: MAPS generator, single 1024^2 slice
  exp2   Tri-axis aggregation cost (256^2 forwards, z-only vs 3x)
  exp3   Per-method inference latency (MAPS/b4 UNetG 256^2; b5 3D slab)
  exp4   Batch throughput at 256^2, batch 1..32 (memory-permitting)
  exp5   DDP Stage-1 train-step throughput (multi-GPU; the Table S11 /
         scaling-figure numbers: speedup = thr(N)/thr(1),
         efficiency = speedup/N)
  exp6   Precision matrix FP32/TF32/FP16/BF16 inference
  exp7   Memory ceiling: max patch x batch fitting the GPU

Table S11 (RTX 3090 DDP scaling): run exp5 with N = 1..8 GPUs, e.g.
  for N in 1 2 3 4 5 6 7 8; do
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N-1))) torchrun \\
        --nproc_per_node=$N analysis/gpu_benchmark.py --exp exp5 \\
        --gpu_name RTX3090_${N}x
  done
Table S12 (GPU-tier comparison): run exp1/exp3/exp5 on each GPU model.

Latency is weight-independent, so checkpoints are optional: pass
--checkpoint to load trained MAPS weights, otherwise random init is used.

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/gpu_benchmark.py --exp all \\
      --gpu_name RTX3090 --out_dir outputs/analysis/gpu_benchmark
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.models import UNetG  # noqa: E402
from maps.checkpoint import extract_model_state, load_state_checked  # noqa: E402
from baselines.unet3d import UNet3D  # noqa: E402

SEED = 2025
IN_CH = 6
BASE_CH = 80


def get_env_info():
    info = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'gpu_count': torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info['gpus'] = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info['gpus'].append({
                'idx': i, 'name': p.name,
                'mem_total_gb': p.total_memory / 1e9,
                'compute_capability': f'{p.major}.{p.minor}',
                'multi_processor_count': p.multi_processor_count,
            })
    return info


def time_op_gpu(fn, n_warmup=20, n_iters=100, n_reps=3, device='cuda'):
    """Time a GPU op using torch.cuda.Event. Returns (mean_ms, std_ms,
    all_reps_ms)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize(device)
    rep_means = []
    for _rep in range(n_reps):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        for i in range(n_iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize(device)
        ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        rep_means.append(float(np.mean(ms)))
    return float(np.mean(rep_means)), float(np.std(rep_means)), rep_means


def build_generator(args, device):
    G = UNetG(in_ch=IN_CH, base=BASE_CH).to(device)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=device)
        load_state_checked(G, extract_model_state(ck),
                           label=str(args.checkpoint))
    G.eval()
    return G


# ─── EXP 1: inference latency on a full slice ───
@torch.no_grad()
def exp1_inference_latency(args, device):
    G = build_generator(args, device)
    inp = torch.randn(1, IN_CH, 1024, 1024, device=device,
                      dtype=torch.float32)

    def _fwd():
        with torch.no_grad():
            return G(inp)

    mean_ms, std_ms, all_reps = time_op_gpu(_fwd, n_warmup=10, n_iters=30,
                                            n_reps=3, device=device)
    n_params = sum(p.numel() for p in G.parameters())
    return {
        'method': 'maps',
        'input_shape': list(inp.shape),
        'n_params': n_params,
        'latency_ms_mean': mean_ms,
        'latency_ms_std': std_ms,
        'latency_per_rep_ms': all_reps,
        'n_warmup': 10, 'n_iters': 30, 'n_reps': 3,
    }


# ─── EXP 2: tri-axis aggregation cost ───
@torch.no_grad()
def exp2_tri_axis_cost(args, device):
    """Time z-only vs tri_mean (z+x+y reconstruction) forward counts for a
    256^3 cube (sampled forwards, extrapolated)."""
    G = build_generator(args, device)
    cube_side = 256
    n_targets_full = (cube_side - 2 * 15) // 2  # ~113 odd targets per axis
    n_targets_sample = 10
    inp = torch.randn(1, IN_CH, cube_side, cube_side, device=device)

    def _z_sample():
        for _ in range(n_targets_sample):
            with torch.no_grad():
                G(inp)
    z_mean, z_std, _ = time_op_gpu(_z_sample, n_warmup=3, n_iters=5,
                                   n_reps=3, device=device)

    def _tri_sample():
        for _ in range(3 * n_targets_sample):
            with torch.no_grad():
                G(inp)
    tri_mean, tri_std, _ = time_op_gpu(_tri_sample, n_warmup=2, n_iters=3,
                                       n_reps=3, device=device)

    factor = n_targets_full / n_targets_sample
    return {
        'cube_side': cube_side,
        'n_targets_per_axis_full': n_targets_full,
        'n_targets_sample': n_targets_sample,
        'z_only_ms_sample': z_mean, 'z_only_ms_std_sample': z_std,
        'tri_mean_ms_sample': tri_mean, 'tri_mean_ms_std_sample': tri_std,
        'z_only_ms_extrapolated_full_cube': z_mean * factor,
        'tri_mean_ms_extrapolated_full_cube': tri_mean * factor,
        'overhead_ratio': tri_mean / max(z_mean, 1e-9),
    }


# ─── EXP 3: per-method inference ───
@torch.no_grad()
def exp3_per_method(args, device):
    out = {}
    inp_2d = torch.randn(1, IN_CH, 256, 256, device=device)

    # MAPS / b4 share the UNetG architecture (identical latency)
    m = build_generator(args, device)

    def _fwd():
        with torch.no_grad():
            m(inp_2d)
    mean_ms, std_ms, _ = time_op_gpu(_fwd, n_warmup=10, n_iters=20,
                                     n_reps=3, device=device)
    out['maps_b4_unetg'] = {
        'n_params': sum(p.numel() for p in m.parameters()),
        'latency_ms_mean': mean_ms, 'latency_ms_std': std_ms,
    }
    del m
    torch.cuda.empty_cache()

    # b5: 3D model, slab input (32, 256, 256)
    m3 = UNet3D(in_ch=2, out_ch=1, base=24).to(device)
    m3.eval()
    inp_3d = torch.randn(1, 2, 32, 256, 256, device=device)

    def _fwd_3d():
        with torch.no_grad():
            m3(inp_3d)
    mean_ms, std_ms, _ = time_op_gpu(_fwd_3d, n_warmup=5, n_iters=15,
                                     n_reps=3, device=device)
    out['b5_unet3d_fair'] = {
        'n_params': sum(p.numel() for p in m3.parameters()),
        'input_shape': [1, 2, 32, 256, 256],
        'latency_ms_mean': mean_ms, 'latency_ms_std': std_ms,
    }
    del m3
    torch.cuda.empty_cache()
    return out


# ─── EXP 4: batch throughput ───
@torch.no_grad()
def exp4_batch_throughput(args, device):
    G = build_generator(args, device)
    out = {'patch_size': 256, 'batches': []}
    for batch in [1, 2, 4, 8, 16, 32]:
        try:
            torch.cuda.empty_cache()
            inp = torch.randn(batch, IN_CH, 256, 256, device=device)

            def _fwd():
                with torch.no_grad():
                    G(inp)
            mean_ms, std_ms, _ = time_op_gpu(_fwd, n_warmup=5, n_iters=15,
                                             n_reps=3, device=device)
            throughput = batch / (mean_ms / 1000.0)
            mem_used = torch.cuda.max_memory_allocated(device) / 1e9
            out['batches'].append({
                'batch': batch, 'latency_ms': mean_ms, 'std_ms': std_ms,
                'throughput_imgs_per_sec': throughput,
                'peak_mem_gb': mem_used,
            })
            del inp
            torch.cuda.reset_peak_memory_stats(device)
        except torch.cuda.OutOfMemoryError as e:
            out['batches'].append({'batch': batch, 'oom': True,
                                   'msg': str(e)[:200]})
            torch.cuda.empty_cache()
            break
    return out


# ─── EXP 5: DDP Stage-1 train throughput (Table S11) ───
def exp5_ddp_train_throughput(args, device, world_size, rank, local_rank):
    """Stage-1 train-step throughput, scales with world_size."""
    G = UNetG(in_ch=IN_CH, base=BASE_CH).to(device)
    if world_size > 1:
        G = DDP(G, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.Adam(G.parameters(), lr=3e-4)
    scaler = GradScaler()
    inp = torch.randn(args.batch_size, IN_CH, 256, 256, device=device)
    tgt = torch.randn(args.batch_size, 1, 256, 256, device=device)

    # Warmup
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        with autocast():
            pred = G(inp)
            loss = F.l1_loss(pred, tgt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize(device)

    n_steps = 30
    t0 = time.time()
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        with autocast():
            pred = G(inp)
            loss = F.l1_loss(pred, tgt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    samples_per_sec = (n_steps * args.batch_size * world_size) / dt
    if rank == 0:
        return {
            'world_size': world_size,
            'batch_per_gpu': args.batch_size,
            'effective_batch': args.batch_size * world_size,
            'n_steps': n_steps,
            'wall_seconds': dt,
            'step_ms_mean': dt / n_steps * 1000,
            'samples_per_sec_total': samples_per_sec,
        }
    return None


# ─── EXP 6: precision matrix ───
def exp6_precision(args, device):
    G = build_generator(args, device)
    inp_fp32 = torch.randn(1, IN_CH, 1024, 1024, device=device,
                           dtype=torch.float32)
    out = {}

    # FP32 (no TF32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def _fp32():
        with torch.no_grad():
            G(inp_fp32)
    mean_ms, std_ms, _ = time_op_gpu(_fp32, n_warmup=20, n_iters=50,
                                     n_reps=3, device=device)
    out['fp32'] = {'latency_ms_mean': mean_ms, 'latency_ms_std': std_ms}

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def _tf32():
        with torch.no_grad():
            G(inp_fp32)
    mean_ms, std_ms, _ = time_op_gpu(_tf32, n_warmup=20, n_iters=50,
                                     n_reps=3, device=device)
    out['tf32'] = {'latency_ms_mean': mean_ms, 'latency_ms_std': std_ms}

    # FP16
    G_fp16 = G.half()
    inp_fp16 = inp_fp32.half()

    def _fp16():
        with torch.no_grad():
            G_fp16(inp_fp16)
    try:
        mean_ms, std_ms, _ = time_op_gpu(_fp16, n_warmup=20, n_iters=50,
                                         n_reps=3, device=device)
        out['fp16'] = {'latency_ms_mean': mean_ms, 'latency_ms_std': std_ms}
    except Exception as e:
        out['fp16'] = {'err': str(e)[:200]}
    G_fp16 = G_fp16.float()  # restore

    # BF16
    try:
        G_bf16 = G.to(torch.bfloat16)
        inp_bf16 = inp_fp32.to(torch.bfloat16)

        def _bf16():
            with torch.no_grad():
                G_bf16(inp_bf16)
        mean_ms, std_ms, _ = time_op_gpu(_bf16, n_warmup=20, n_iters=50,
                                         n_reps=3, device=device)
        out['bf16'] = {'latency_ms_mean': mean_ms, 'latency_ms_std': std_ms}
        G_bf16 = G_bf16.float()
    except Exception as e:
        out['bf16'] = {'err': str(e)[:200]}

    return out


# ─── EXP 7: memory ceiling ───
def exp7_memory_ceiling(args, device):
    G = UNetG(in_ch=IN_CH, base=BASE_CH).to(device)
    G.eval()
    out = {'grid': []}
    for patch in [256, 512, 1024, 2048]:
        for batch in [1, 2, 4, 8, 16]:
            try:
                torch.cuda.empty_cache()
                inp = torch.randn(batch, IN_CH, patch, patch, device=device)
                with torch.no_grad():
                    _ = G(inp)
                torch.cuda.synchronize(device)
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                out['grid'].append({'patch': patch, 'batch': batch,
                                    'fit': True, 'peak_mem_gb': mem_gb})
                del inp
                torch.cuda.reset_peak_memory_stats(device)
            except torch.cuda.OutOfMemoryError:
                out['grid'].append({'patch': patch, 'batch': batch,
                                    'fit': False})
                torch.cuda.empty_cache()
            except Exception as e:
                out['grid'].append({'patch': patch, 'batch': batch,
                                    'err': str(e)[:100]})
                torch.cuda.empty_cache()
    return out


# ─── DDP setup ───
def setup_ddp():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def main():
    ap = argparse.ArgumentParser(description='GPU benchmark suite '
                                             '(Tables S11-S12)')
    ap.add_argument('--exp', default='all',
                    choices=['exp1', 'exp2', 'exp3', 'exp4', 'exp5',
                             'exp6', 'exp7', 'all'])
    ap.add_argument('--gpu_name', required=True,
                    help='Label for the output dir, e.g. RTX3090, RTX3090_4x')
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='Optional trained MAPS checkpoint (latency is '
                         'weight-independent; random init if omitted)')
    ap.add_argument('--out_dir', type=Path,
                    default=Path('outputs/analysis/gpu_benchmark'))
    ap.add_argument('--batch_size', type=int, default=4,
                    help='Per-GPU batch for exp5')
    args = ap.parse_args()

    rank, world_size, local_rank, is_ddp = setup_ddp()
    if is_ddp:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0')
    out_dir = args.out_dir / args.gpu_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if rank == 0:
        env = get_env_info()
        env['gpu_name_arg'] = args.gpu_name
        env['world_size'] = world_size
        with open(out_dir / 'env.json', 'w') as f:
            json.dump(env, f, indent=2)
        print(f'[ENV] {env}')

    # Single-GPU experiments
    if not is_ddp or rank == 0:
        if args.exp in ('exp1', 'all'):
            print('[EXP1] inference latency')
            r = exp1_inference_latency(args, device)
            with open(out_dir / 'exp1_inference_latency.json', 'w') as f:
                json.dump(r, f, indent=2)
            print(f'  done: {r["latency_ms_mean"]:.2f} +/- '
                  f'{r["latency_ms_std"]:.2f} ms')

        if args.exp in ('exp2', 'all'):
            print('[EXP2] tri-axis cost')
            r = exp2_tri_axis_cost(args, device)
            with open(out_dir / 'exp2_tri_axis_cost.json', 'w') as f:
                json.dump(r, f, indent=2)
            print(f'  z_only_sample={r["z_only_ms_sample"]:.0f}ms '
                  f'tri_sample={r["tri_mean_ms_sample"]:.0f}ms '
                  f'ratio={r["overhead_ratio"]:.2f}')

        if args.exp in ('exp3', 'all'):
            print('[EXP3] per-method inference')
            r = exp3_per_method(args, device)
            with open(out_dir / 'exp3_per_method.json', 'w') as f:
                json.dump(r, f, indent=2)

        if args.exp in ('exp4', 'all'):
            print('[EXP4] batch throughput')
            r = exp4_batch_throughput(args, device)
            with open(out_dir / 'exp4_batch_throughput.json', 'w') as f:
                json.dump(r, f, indent=2)

        if args.exp in ('exp6', 'all'):
            print('[EXP6] precision matrix')
            r = exp6_precision(args, device)
            with open(out_dir / 'exp6_precision.json', 'w') as f:
                json.dump(r, f, indent=2)

        if args.exp in ('exp7', 'all'):
            print('[EXP7] memory ceiling')
            r = exp7_memory_ceiling(args, device)
            with open(out_dir / 'exp7_memory_ceiling.json', 'w') as f:
                json.dump(r, f, indent=2)

    if args.exp in ('exp5', 'all'):
        print(f'[EXP5] DDP train throughput (rank={rank}/{world_size})')
        r = exp5_ddp_train_throughput(args, device, world_size, rank,
                                      local_rank)
        if rank == 0 and r is not None:
            with open(out_dir / 'exp5_ddp_train.json', 'w') as f:
                json.dump(r, f, indent=2)
            print(f'  done: {r["samples_per_sec_total"]:.1f} samples/s '
                  f'on {world_size} GPUs')

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
