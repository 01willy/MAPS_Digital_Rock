"""
Tiled inference and parity-protocol reconstruction utilities shared by the
evaluation drivers (LBM Table 3 and Tables S10/S15, morphology Tables
S3/S5-S7).

Functions:
    lock_determinism         -- global seed lock
    load_unetg / load_unet3d -- checkpoint loading (EMA-preferred, fail-loud)
    predict_2d_disjoint      -- disjoint-tile 2D inference on one target slice
    predict_3d_target_slice  -- sliding-window 3D inference (fair or leaky mode)
    recon_unetg_z            -- 2D-model odd-z parity replacement (128^3 LBM run)
    recon_unet3d_z           -- 3D-model odd-z parity replacement (128^3 LBM run)
    ours_recon_axis          -- 2D-model odd-slice parity replacement along any
                                axis, batched (256^3 8-domain LBM run)
    reconstruct_axis_parity  -- alias of ours_recon_axis with prediction clamp
                                (deployment-parity morphology runs)

Parity convention throughout: EVEN slices along the reconstruction axis are
acquired (kept = GT); ODD slices are the targets. Odd targets whose full
offset support does not fit in the cube keep GT (boundary convention of the
paper's LBM protocol).
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import OFFSETS_IN6  # noqa: E402
from maps.models import UNetG  # noqa: E402
from maps.checkpoint import extract_model_state, load_state_checked  # noqa: E402
from baselines.unet3d import UNet3D  # noqa: E402


def lock_determinism(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_unetg(ckpt_path, device):
    G = UNetG(in_ch=6, base=80).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    load_state_checked(G, extract_model_state(ck), label=str(ckpt_path))
    G.eval()
    return G


def load_unet3d(ckpt_path, device, base=24):
    G3 = UNet3D(in_ch=2, out_ch=1, base=base).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    load_state_checked(G3, extract_model_state(ck), label=str(ckpt_path))
    G3.eval()
    return G3


# ─── 2D model: disjoint-tile eval ───────────────────────────────

@torch.no_grad()
def predict_2d_disjoint(G, view, t, offsets, device, tile=256, pad=128):
    """
    Disjoint-tile inference for one target slice. If H or W < inp_size
    (small volumes), reflection-pad the slice up to inp_size and run a
    single forward.
    """
    inp_size = tile + 2 * pad
    _, H, W = view.shape

    # Small-image fast path: pad once, single forward, crop result
    if H < inp_size or W < inp_size:
        slc = np.stack([view[t + o].astype(np.float32) for o in offsets], axis=0)
        py = max(0, inp_size - H)
        px = max(0, inp_size - W)
        py0, py1 = py // 2, py - py // 2
        px0, px1 = px // 2, px - px // 2
        slc = np.pad(slc, ((0, 0), (py0, py1), (px0, px1)), mode='reflect')
        inp_t = torch.from_numpy(slc).unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=False):
            out = G(inp_t).cpu().numpy()[0, 0]
        return out[py0:py0 + H, px0:px0 + W]

    def tr(L):
        rngs, s = [], 0
        while s < L:
            e = min(s + tile, L)
            rngs.append((s, e))
            s = e
        return rngs

    pred = np.zeros((H, W), dtype=np.float32)
    for ty0, ty1 in tr(H):
        iy0 = max(0, ty0 - pad)
        iy1 = iy0 + inp_size
        if iy1 > H:
            iy1 = H
            iy0 = H - inp_size
        for tx0, tx1 in tr(W):
            ix0 = max(0, tx0 - pad)
            ix1 = ix0 + inp_size
            if ix1 > W:
                ix1 = W
                ix0 = W - inp_size
            inp = np.stack([view[t + o, iy0:iy1, ix0:ix1].astype(np.float32)
                            for o in offsets], axis=0)[np.newaxis]
            inp_t = torch.from_numpy(inp).to(device)
            with torch.cuda.amp.autocast(enabled=False):
                out = G(inp_t).cpu().numpy()[0, 0]
            local_ty0 = ty0 - iy0
            local_tx0 = tx0 - ix0
            h_t = ty1 - ty0
            w_t = tx1 - tx0
            pred[ty0:ty1, tx0:tx1] = out[local_ty0:local_ty0 + h_t,
                                         local_tx0:local_tx0 + w_t]
    return pred


# ─── 3D model: sliding-window eval, two input modes ─────────────

@torch.no_grad()
def predict_3d_target_slice(G3, view, t, device, input_mode='fair_offsets',
                            offsets=OFFSETS_IN6, patch_size=32,
                            xy_tile=256, xy_pad=16):
    """
    Predict a SINGLE odd-target slice z=t using a 3D model.

    Strategy: build a small cube of depth `patch_size` covering the offset
    positions around t; slide xy disjointly to cover the full HxW.

    input_mode (matching the training counterparts in baselines/unet3d.py):
      'fair_offsets': sparse has GT ONLY at z=t+o for o in `offsets`
                      (b5 fair protocol, the paper's default).
      'leaky_k1':     sparse has all even-z slices in the cube = GT, odd = 0
                      (unfair upper-bound reference; model can read z=t+-1).

    Returns (H, W) float32 prediction at z=t.
    """
    _, H, W = view.shape
    k_max = max(abs(o) for o in offsets)
    # Place target at local zi = k_max so cube covers [t-k_max, ...]
    z_local_target = k_max
    z0_abs = t - k_max  # first absolute z of cube
    if z0_abs < 0 or z0_abs + patch_size > view.shape[0]:
        # fall back: shift cube to fit
        z0_abs = max(0, min(view.shape[0] - patch_size, z0_abs))
        z_local_target = t - z0_abs

    cube_z = view[z0_abs:z0_abs + patch_size].astype(np.float32)
    cube_z = np.clip(cube_z, 0.0, 1.0)

    def tr(L):
        rngs, s = [], 0
        while s < L:
            e = min(s + xy_tile, L)
            rngs.append((s, e))
            s = e
        return rngs

    pred_slice = np.zeros((H, W), dtype=np.float32)

    inp_xy = xy_tile + 2 * xy_pad
    for ty0, ty1 in tr(H):
        iy0 = max(0, ty0 - xy_pad)
        iy1 = iy0 + inp_xy
        if iy1 > H:
            iy1 = H
            iy0 = max(0, H - inp_xy)
        for tx0, tx1 in tr(W):
            ix0 = max(0, tx0 - xy_pad)
            ix1 = ix0 + inp_xy
            if ix1 > W:
                ix1 = W
                ix0 = max(0, W - inp_xy)
            sub_cube = cube_z[:, iy0:iy1, ix0:ix1]  # (D, h, w)
            sparse = np.zeros_like(sub_cube)
            mask = np.zeros_like(sub_cube)
            if input_mode == 'leaky_k1':
                for zi in range(patch_size):
                    if (z0_abs + zi) % 2 == 0:
                        sparse[zi] = sub_cube[zi]
                        mask[zi] = 1.0
            elif input_mode == 'fair_offsets':
                for o in offsets:
                    kz = z_local_target + o
                    if 0 <= kz < patch_size:
                        sparse[kz] = sub_cube[kz]
                        mask[kz] = 1.0
            else:
                raise ValueError(f'unknown input_mode {input_mode}')
            inp = np.stack([sparse, mask], axis=0)[np.newaxis]
            inp_t = torch.from_numpy(inp).to(device).float()
            with torch.cuda.amp.autocast(enabled=False):
                out = G3(inp_t).cpu().numpy()[0, 0, z_local_target]  # (h, w)
            local_ty0 = ty0 - iy0
            local_tx0 = tx0 - ix0
            h_t = ty1 - ty0
            w_t = tx1 - tx0
            pred_slice[ty0:ty1, tx0:tx1] = out[local_ty0:local_ty0 + h_t,
                                               local_tx0:local_tx0 + w_t]
    return pred_slice


# ─── Parity-protocol cube reconstruction ────────────────────────

@torch.no_grad()
def recon_unetg_z(G, cube, offsets, device):
    """2D-model odd-z replacement (k=1 parity; boundary keeps GT).
    Continuous output. Used by the Table 3 LBM run (128^3 cubes)."""
    D, H, W = cube.shape
    k_max = max(abs(o) for o in offsets)
    out = cube.copy()
    tile = min(128, H, W)
    pad = min(64, max(8, tile // 2))
    for t in range(D):
        if t % 2 != 1:
            continue
        if t - k_max < 0 or t + k_max >= D:
            continue  # boundary keeps GT
        out[t] = predict_2d_disjoint(G, cube, t, offsets, device,
                                     tile=tile, pad=pad)
    return out


@torch.no_grad()
def recon_unet3d_z(G3, cube, offsets, device, base=24):
    """3D-model fair single-target odd-z replacement (k=1 parity).
    Used by the Table 3 LBM run (128^3 cubes)."""
    D, H, W = cube.shape
    k_max = max(abs(o) for o in offsets)
    out = cube.copy()
    xy_tile = min(128, H, W)
    for t in range(D):
        if t % 2 != 1:
            continue
        if t - k_max < 0 or t + k_max >= D:
            continue
        out[t] = predict_3d_target_slice(
            G3, cube, t, device, input_mode='fair_offsets', offsets=offsets,
            patch_size=32, xy_tile=xy_tile, xy_pad=16)
    return out


@torch.no_grad()
def ours_recon_axis(G, cube, axis, offsets, device, batch_size=16):
    """2D-model odd-slice parity replacement along `axis`, batched.
    Numpy in/out (continuous sigmoid output on odd slices). Used by the
    8-domain LBM campaign (Table S15, 256^3 cubes)."""
    D, H, W = cube.shape
    k_max = max(abs(o) for o in offsets)
    out = cube.copy()
    n = {'z': D, 'y': H, 'x': W}[axis]
    targets = [t for t in range(k_max, n - k_max) if t % 2 == 1]
    for i in range(0, len(targets), batch_size):
        batch = targets[i:i + batch_size]
        ins = []
        for t in batch:
            if axis == 'z':
                inp = np.stack([cube[t + o, :, :] for o in offsets], axis=0)
            elif axis == 'x':
                inp = np.stack([cube[:, :, t + o] for o in offsets], axis=0)
            else:
                inp = np.stack([cube[:, t + o, :] for o in offsets], axis=0)
            ins.append(inp.astype(np.float32))
        ins = torch.from_numpy(np.stack(ins, axis=0)).to(device)
        pr = G(ins).float().cpu().numpy()
        for j, t in enumerate(batch):
            p = pr[j, 0]
            if axis == 'z':
                out[t] = p
            elif axis == 'x':
                out[:, :, t] = p
            else:
                out[:, t, :] = p
    return out


@torch.no_grad()
def reconstruct_axis_parity(G, vol_cube, axis, offsets, device,
                            batch_size=16):
    """Parity reconstruction returning a torch tensor with predictions
    clamped to [0, 1] (convention of the deployment-parity morphology
    evaluations, Table S3)."""
    out_np = ours_recon_axis(G, vol_cube, axis, offsets, device,
                             batch_size=batch_size)
    out = torch.from_numpy(np.clip(out_np, 0.0, 1.0)).float()
    return out
