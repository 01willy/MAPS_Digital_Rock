#!/usr/bin/env python3
"""
End-to-end compute-cost benchmark (Table 4 of the paper and the
cost--quality Pareto figure).

Two independent measurements on a common basis (every method reconstructs
the SAME cube):

--mode latency : real, batched forward passes with random weights (latency
    is weight-independent). MAPS (2.5D tri-axis): the same UNetG run along
    z, x, y; each plane xy-tiled (tile + 2*pad) and batched. b5 / b5-large
    (3D U-Net): sliding 32-deep slab over z x xy-tiles -- both the EFFICIENT
    amortized deployment (one slab forward yields 32 z-planes) and the
    as-implemented non-amortized cost (one 32-cube per target slice) are
    reported. Optional --planes1000 also times a 1000x1000x64 slab (the
    deployment resolution row of Table 4).

--mode flops : per-forward FLOPs via `thop` at the deployed input shapes
    (MAPS, SwinUNet, b5, b5-large), converted to end-to-end TFLOPs and
    per-synthesized-slice GFLOPs for a 256^3 cube (non-amortized and
    amortized 3D deployments) plus the per-1000^2-slice normalization of
    the Table 4 FLOP column. (Requires `pip install thop`.)

--mode summary : params / peak GPU memory / single-forward latency for
    every method incl. SwinUNet and the V3 latent-diffusion sampler
    (multi-step inference estimated as vae_decode + n_steps x latent-UNet).

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --mode latency \\
      --cubes 256,512 --out_dir outputs/analysis/compute_cost
  CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --mode flops
  CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --mode summary
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.models import UNetG  # noqa: E402
from baselines.unet3d import UNet3D  # noqa: E402

OFFS = [-15, -9, -3, 3, 9, 15]
KMAX = 15


def sync(d):
    torch.cuda.synchronize(d)


def tiles(L, tile):
    out, s = [], 0
    while s < L:
        e = min(s + tile, L)
        out.append((s, e))
        s = e
    return out


# ════════════════════════════════════════════════════════════
# Mode: latency (end-to-end wall-clock, common basis)
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def run_batched(model, make_input, n_calls, device, batch, warmup=2):
    """Execute n_calls forwards of inputs from make_input(), batched, timed
    (ms total). Falls back to batch=1 on OOM (realistic for large 3D
    slabs)."""
    sample = make_input().to(device)
    while batch >= 1:
        try:
            for _ in range(warmup):
                _ = model(torch.cat([sample] * min(batch, 2), 0))
            sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            done = 0
            while done < n_calls:
                b = min(batch, n_calls - done)
                x = torch.cat([sample] * b, 0) if b > 1 else sample
                _ = model(x)
                done += b
            sync(device)
            ms = (time.time() - t0) * 1000.0
            mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            return ms, mem, batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch = batch // 2
    raise RuntimeError('OOM even at batch=1')


@torch.no_grad()
def maps_triaxis(G, D, H, W, device, tile=256, pad=16, batch=16):
    """Tri-axis MAPS reconstruction cost on a (D,H,W) cube, real batched
    forwards."""
    inp_hw = tile + 2 * pad
    total_calls = 0
    for (A, P, Q) in [(D, H, W), (W, D, H), (H, D, W)]:  # z, x, y passes
        n_planes = max(0, A - 2 * KMAX)
        n_tiles = len(tiles(P, tile)) * len(tiles(Q, tile))
        total_calls += n_planes * n_tiles
    mk = lambda: torch.rand(1, 6, inp_hw, inp_hw)  # noqa: E731
    ms, mem, bs = run_batched(G, mk, total_calls, device, batch)
    return {'forwards': total_calls, 'ms_total': ms, 'ms_per_slice': ms / D,
            'gpu_mem_MB': mem, 'tile': inp_hw, 'batch': bs}


@torch.no_grad()
def b5_reconstruct(G3, D, H, W, device, patch_z=32, tile=256, pad=16,
                   batch=1):
    """3D U-Net cube reconstruction. EFFICIENT (amortized) deployment: slide
    a 32-deep slab over z (disjoint) x xy-tiles; one slab forward yields 32
    z-planes. Also computes the NON-amortized (as-implemented) cost = one
    32-cube per target slice."""
    inp_hw = tile + 2 * pad
    n_z_slabs = int(np.ceil(D / patch_z))
    n_tiles = len(tiles(H, tile)) * len(tiles(W, tile))
    amort_calls = n_z_slabs * n_tiles       # amortized over 32 planes
    n_odd = D // 2                          # missing (odd) planes
    nonamort_calls = n_odd * n_tiles        # as-implemented
    mk = lambda: torch.rand(1, 2, patch_z, inp_hw, inp_hw)  # noqa: E731
    ms_amort, mem, _bs = run_batched(G3, mk, amort_calls, device, batch)
    # scale non-amortized from the same per-forward time (same shape)
    per_fwd = ms_amort / max(1, amort_calls)
    ms_nonamort = per_fwd * nonamort_calls
    return {'forwards_amort': amort_calls,
            'forwards_nonamort': nonamort_calls,
            'ms_total_amort': ms_amort, 'ms_per_slice_amort': ms_amort / D,
            'ms_total_nonamort': ms_nonamort,
            'ms_per_slice_nonamort': ms_nonamort / D,
            'gpu_mem_MB': mem, 'tile': inp_hw, 'per_forward_ms': per_fwd}


def mode_latency(args, device):
    torch.backends.cudnn.benchmark = True
    G = UNetG(in_ch=6, base=80).to(device).eval()
    b5 = UNet3D(in_ch=2, out_ch=1, base=24).to(device).eval()
    b5l = UNet3D(in_ch=2, out_ch=1, base=64).to(device).eval()

    out = {}
    sizes = [int(s) for s in args.cubes.split(',')]
    shapes = [(s, s, s) for s in sizes]
    if args.planes1000:
        shapes.append((64, 1000, 1000))
    for (D, H, W) in shapes:
        key = f'{D}x{H}x{W}'
        print(f'\n=== cube {key} ===')
        m = maps_triaxis(G, D, H, W, device)
        print(f'MAPS tri-axis: {m["forwards"]} fwd, {m["ms_total"]:.0f} ms '
              f'total, {m["ms_per_slice"]:.2f} ms/slice, '
              f'{m["gpu_mem_MB"]:.0f} MB')
        r5 = b5_reconstruct(b5, D, H, W, device)
        print(f'b5 amortized:  {r5["forwards_amort"]} fwd, '
              f'{r5["ms_total_amort"]:.0f} ms total, '
              f'{r5["ms_per_slice_amort"]:.2f} ms/slice | non-amort '
              f'{r5["ms_per_slice_nonamort"]:.2f} ms/slice')
        r5l = b5_reconstruct(b5l, D, H, W, device)
        print(f'b5-large amort:{r5l["forwards_amort"]} fwd, '
              f'{r5l["ms_total_amort"]:.0f} ms total, '
              f'{r5l["ms_per_slice_amort"]:.2f} ms/slice | non-amort '
              f'{r5l["ms_per_slice_nonamort"]:.2f} ms/slice')
        ratio_amort = r5['ms_per_slice_amort'] / m['ms_per_slice']
        ratio_nonamort = r5['ms_per_slice_nonamort'] / m['ms_per_slice']
        print(f'  ==> b5/MAPS per-slice ratio: amortized {ratio_amort:.2f}x, '
              f'as-implemented(non-amort) {ratio_nonamort:.2f}x')
        out[key] = {'maps_triaxis': m, 'b5': r5, 'b5_large': r5l,
                    'ratio_b5_over_maps_amort': ratio_amort,
                    'ratio_b5_over_maps_nonamort': ratio_nonamort}

    outpath = Path(args.out_dir) / 'e2e_latency.json'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f'\n[saved] {outpath}')


# ════════════════════════════════════════════════════════════
# Mode: flops (deployment-independent fair metric)
# ════════════════════════════════════════════════════════════

def gflops(model, shape, device):
    from thop import profile
    x = torch.zeros(*shape, device=device)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    return 2 * macs / 1e9


def mode_flops(args, device):
    G = UNetG(in_ch=6, base=80).to(device).eval()
    b5 = UNet3D(in_ch=2, out_ch=1, base=24).to(device).eval()
    b5l = UNet3D(in_ch=2, out_ch=1, base=64).to(device).eval()

    N = 256                     # cube side (fair aspect ratio)
    tile = 256
    pad = 16                    # in-plane tiling
    n_planes_axis = N - 2 * KMAX
    n_odd = len([t for t in range(KMAX, N - KMAX) if t % 2 == 1])
    xy_tiles = int(np.ceil(N / tile)) ** 2          # =1 at N=256
    z_slabs = int(np.ceil(N / 32))                  # =8

    # per-forward GFLOPs at the actual deployed input shapes
    f_maps = gflops(G, (1, 6, tile, tile), device)
    f_b5 = gflops(b5, (1, 2, 32, tile + 2 * pad, tile + 2 * pad), device)
    f_b5l = gflops(b5l, (1, 2, 32, tile + 2 * pad, tile + 2 * pad), device)
    print(f'per-forward GFLOPs: MAPS(6,{tile},{tile})={f_maps:.1f}  '
          f'b5(2,32,{tile + 2 * pad}^2)={f_b5:.1f}  b5-large={f_b5l:.1f}')
    f_swin = None
    try:
        from baselines.swinunet import SwinUNet
        swin = SwinUNet(in_ch=6, base=96, num_heads=4,
                        window_size=8).to(device).eval()
        f_swin = gflops(swin, (1, 6, tile, tile), device)
        print(f'                    SwinUNet(6,{tile},{tile})={f_swin:.1f}')
        del swin
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  SwinUNet FLOPs skipped: {e}')

    # forward counts to reconstruct the N^3 cube
    maps_tri_fwd = (n_odd + 2 * n_planes_axis) * xy_tiles  # z:odd; x,y:all
    maps_z_fwd = n_odd * xy_tiles
    b5_nonamort_fwd = n_odd * xy_tiles       # as-implemented
    b5_amort_fwd = z_slabs * xy_tiles        # one slab fwd -> 32 planes

    def report(name, fwd, per_fwd):
        tflop = fwd * per_fwd / 1000.0
        per_slice = fwd * per_fwd / n_odd
        print(f'  {name:30s}: {fwd:5d} fwd | {tflop:8.1f} TFLOP | '
              f'{per_slice:8.1f} GFLOP/synth-slice')
        return {'forwards': fwd, 'tflop': tflop,
                'gflop_per_slice': per_slice}

    print(f'\n=== end-to-end FLOPs to reconstruct {N}^3 cube '
          f'({n_odd} synth odd-z planes) ===')
    out = {}
    out['maps_triaxis'] = report('MAPS tri-axis', maps_tri_fwd, f_maps)
    out['maps_zonly'] = report('MAPS z-only', maps_z_fwd, f_maps)
    out['b5_nonamort'] = report('b5 (3.15M) AS-IMPLEMENTED',
                                b5_nonamort_fwd, f_b5)
    out['b5_amort'] = report('b5 (3.15M) amortized', b5_amort_fwd, f_b5)
    out['b5large_nonamort'] = report('b5-large (22.4M) AS-IMPL',
                                     b5_nonamort_fwd, f_b5l)
    out['b5large_amort'] = report('b5-large (22.4M) amortized',
                                  b5_amort_fwd, f_b5l)

    mt = out['maps_triaxis']['gflop_per_slice']
    print('\n=== MAPS tri-axis vs 3D (per-synth-slice GFLOP ratio; '
          '>1 = MAPS cheaper) ===')
    for k in ['b5_nonamort', 'b5_amort', 'b5large_nonamort', 'b5large_amort']:
        print(f'  vs {k:22s}: {out[k]["gflop_per_slice"] / mt:6.2f}x')

    # ── per-1000^2-slice normalization (Table 4 FLOP column) ──
    # 2D methods: a 1000^2 plane needs ceil(1000/tile)^2 tiled forwards;
    # tri-axis is charged the full cube-amortized forwards-per-synthesized-
    # slice factor (~5 at 256^3: one z forward + all-plane x/y passes),
    # which is resolution-independent. 3D baselines: as-deployed
    # (non-amortized sliding window) = one forward per tile per slice.
    tiles_1000 = int(np.ceil(1000 / tile)) ** 2     # = 16
    tri_factor = (n_odd + 2 * n_planes_axis) / n_odd
    per1000 = {}
    per1000['maps_triaxis'] = tri_factor * tiles_1000 * f_maps
    per1000['maps_zonly'] = tiles_1000 * f_maps
    if f_swin is not None:
        per1000['swinunet_triaxis'] = tri_factor * tiles_1000 * f_swin
        per1000['swinunet_zonly'] = tiles_1000 * f_swin
    per1000['b5_nonamort'] = tiles_1000 * f_b5
    per1000['b5large_nonamort'] = tiles_1000 * f_b5l
    print(f'\n=== per-1000^2 synthesized slice (tri-axis factor '
          f'{tri_factor:.1f} forwards/slice, {tiles_1000} tiles/plane) ===')
    for k, v in per1000.items():
        print(f'  {k:22s}: {v:10.1f} GFLOP/slice ({v / 1000:.1f} kGFLOP)')
    out['per_1000sq_slice_gflop'] = per1000

    outpath = Path(args.out_dir) / 'e2e_flops.json'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f'[saved] {outpath}')


# ════════════════════════════════════════════════════════════
# Mode: summary (params / memory / single-forward per method)
# ════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_mem_used_mb(device):
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


@torch.no_grad()
def measure_2d(model, in_ch, hw=(256, 256), device='cuda', warmup=3, n=50):
    model.eval()
    x = torch.zeros(1, in_ch, hw[0], hw[1], device=device)
    for _ in range(warmup):
        _ = model(x)
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for _ in range(n):
        _ = model(x)
    torch.cuda.synchronize(device)
    elapsed = (time.time() - t0) / n
    return elapsed * 1000, gpu_mem_used_mb(device)


@torch.no_grad()
def measure_3d(model, in_ch, dhw=(32, 32, 32), device='cuda', warmup=2, n=20):
    model.eval()
    x = torch.zeros(1, in_ch, dhw[0], dhw[1], dhw[2], device=device)
    for _ in range(warmup):
        _ = model(x)
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for _ in range(n):
        _ = model(x)
    torch.cuda.synchronize(device)
    elapsed = (time.time() - t0) / n
    return elapsed * 1000, gpu_mem_used_mb(device)


def mode_summary(args, device):
    results = {}

    print('[measure] MAPS / b4 (UNetG 6->1, base=80)')
    G = UNetG(in_ch=6, base=80).to(device)
    ms, mem = measure_2d(G, 6, device=device)
    results['maps_b4_UNetG'] = {
        'params': count_params(G), 'gpu_mem_MB': mem,
        'single_forward_ms': ms,
        'per_slice_z_only_ms': ms,
        'per_slice_tri_axis_ms': ms * 3.0,
        'input_shape': '1x6x256x256'}
    del G
    torch.cuda.empty_cache()
    print(f'  -> {results["maps_b4_UNetG"]}')

    for name, base in (('b5_fair', 24), ('b5_large', 64)):
        print(f'[measure] {name} (3D UNet, base={base})')
        m3 = UNet3D(in_ch=2, out_ch=1, base=base).to(device)
        ms, mem = measure_3d(m3, 2, device=device)
        results[name] = {
            'params': count_params(m3), 'gpu_mem_MB': mem,
            'single_forward_ms': ms,
            'input_shape': '1x2x32x32x32',
            'note': 'end-to-end per-slice cost: use --mode latency '
                    '(amortized vs non-amortized deployment)'}
        del m3
        torch.cuda.empty_cache()
        print(f'  -> {results[name]}')

    print('[measure] SwinUNet 2D baseline')
    try:
        from baselines.swinunet import SwinUNet
        model = SwinUNet(in_ch=6, base=96, num_heads=4, window_size=8).to(device)
        ms, mem = measure_2d(model, 6, device=device)
        results['swinunet'] = {
            'params': count_params(model), 'gpu_mem_MB': mem,
            'single_forward_ms': ms,
            'per_slice_z_only_ms': ms,
            'per_slice_tri_axis_ms': ms * 3.0,
            'input_shape': '1x6x256x256'}
        del model
        torch.cuda.empty_cache()
        print(f'  -> {results["swinunet"]}')
    except Exception as e:
        print(f'  -> SwinUNet measure failed: {e}')
        results['swinunet'] = {'error': str(e)}

    # Latent diffusion V3: multi-step inference; measure VAE decode +
    # single denoise step, estimate per-slice = vae + nsteps x latent-UNet.
    print('[measure] V3 latent diffusion (VAE + DDIM single-step proxy)')
    try:
        from baselines.latent_diffusion import VAE2D, LatentUNet
        vae = VAE2D(base=64, latent_ch=4).to(device)
        vae.eval()
        unet = LatentUNet(latent_ch=4, n_cond=6, n_time=16, base=128).to(device)
        unet.eval()
        n_params_total = count_params(vae) + count_params(unet)

        torch.cuda.reset_peak_memory_stats(device)
        z_lat = torch.zeros(1, 4, 64, 64, device=device)
        with torch.no_grad():
            for _ in range(3):
                _ = vae.decode(z_lat)
                torch.cuda.synchronize(device)
            t0 = time.time()
            for _ in range(20):
                _ = vae.decode(z_lat)
            torch.cuda.synchronize(device)
            vae_ms = (time.time() - t0) / 20 * 1000

        # LatentUNet single step: 6 cond * 4ch + 1 target * 4ch + 16 = 44 ch
        full_input = torch.zeros(1, 44, 64, 64, device=device)
        with torch.no_grad():
            for _ in range(3):
                _ = unet(full_input)
                torch.cuda.synchronize(device)
            t0 = time.time()
            for _ in range(20):
                _ = unet(full_input)
            torch.cuda.synchronize(device)
            unet_ms = (time.time() - t0) / 20 * 1000
        mem = gpu_mem_used_mb(device)
        del vae, unet
        torch.cuda.empty_cache()

        nsteps = 50
        per_slice_ms = vae_ms + nsteps * unet_ms
        results['diffusion_v3'] = {
            'params': n_params_total, 'gpu_mem_MB': mem,
            'single_forward_ms': unet_ms,
            'per_slice_z_only_ms': per_slice_ms,
            'per_slice_tri_axis_ms': per_slice_ms * 3.0,
            'input_shape': f'latent 1x4x64x64; cond 1x6x64x64; '
                           f'{nsteps} DDIM steps'}
        print(f'  -> {results["diffusion_v3"]}')
    except Exception as e:
        print(f'  -> diffusion measure failed: {e}')
        results['diffusion_v3'] = {'error': str(e)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'compute_cost_summary.csv'
    keys = ['method', 'params', 'gpu_mem_MB', 'single_forward_ms',
            'per_slice_z_only_ms', 'per_slice_tri_axis_ms',
            'input_shape', 'note', 'error']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for name, r in results.items():
            row = {'method': name}
            for k in keys[1:]:
                row[k] = r.get(k, '')
            w.writerow(row)
    print(f'[saved] {csv_path}')


def main():
    ap = argparse.ArgumentParser(description='Compute-cost benchmark (Table 4)')
    ap.add_argument('--mode', choices=['latency', 'flops', 'summary'],
                    default='latency')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cubes', type=str, default='256,512',
                    help='cube sides for --mode latency')
    ap.add_argument('--planes1000', action='store_true',
                    help='also time a 1000x1000x64 slab (deployment '
                         'resolution)')
    ap.add_argument('--out_dir', type=str,
                    default='outputs/analysis/compute_cost')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    if args.mode == 'latency':
        mode_latency(args, device)
    elif args.mode == 'flops':
        mode_flops(args, device)
    else:
        mode_summary(args, device)


if __name__ == '__main__':
    main()
