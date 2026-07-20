"""
Tri-axis volume reconstruction and GT-free aggregation.

A single trained 2D generator is applied slice-by-slice along each of the
three Cartesian axes of a volume, producing three reconstructions
V_z, V_x, V_y. These are then fused with a ground-truth-free aggregation
rule. All aggregation variants in this module use only the three predicted
volumes — no ground truth is required, so every variant is deployable.

Aggregation variants:
    tri_mean       : V_z, V_x, V_y arithmetic mean (the paper's recommended
                     deployable aggregation).
    tri_median     : per-voxel median.
    tri_consensus  : per-axis scalar weight from inverse cross-axis
                     disagreement w_a = 1 / (mean(|V_a - mean(V_other)|) + eps).
    tri_weuler_self: GT-free weighted-Euler aggregation. Uses the median of
                     the three predicted Euler characteristics as the
                     consensus anchor:
                     w_a = 1 / (|Euler_a - median(Euler_z, Euler_x, Euler_y)| + eps).
    tri_voxel_consensus : per-voxel soft weights
                     w_a(x) = exp(-|V_a(x) - mu_other(x)| / tau),
                     renormalized over axes.

A ground-truth-using oracle variant (weights from ground-truth Euler
numbers) exists as a GT-using evaluation reference only — see
`maps/oracle_eval.py`. It is not part of the deployable pipeline.

Deployment protocol ("parity"): in the k=1 sparse scenario the even-index
slices along the acquisition axis are physically acquired; only odd-index
slices are synthesized. `parity_paste` restores the acquired slices after
aggregation so that the final volume contains model output only where data
was actually missing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as pytorch_msssim_fn

try:
    from .metrics import compute_all_morphological_metrics, euler_number_2d
except ImportError:
    from metrics import compute_all_morphological_metrics, euler_number_2d


__all__ = [
    'reconstruct_axis',
    'reconstruct_axis_parity',
    'parity_paste',
    'gtfree_boundary_fill_z',
    'sequential_triaxis',
    'aggregate_tri_mean',
    'aggregate_tri_median',
    'aggregate_tri_consensus',
    'aggregate_tri_weuler_self',
    'aggregate_tri_voxel_consensus',
    'compute_all_gtfree_aggregations',
    'metrics_from_cube',
]


# ════════════════════════════════════════════════════════════
# Axis-wise reconstruction
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def reconstruct_axis(G, vol_cube, axis, offsets, device, batch_size=16):
    """Reconstruct full cube values along axis by running 2D forward at each slice.

    Args:
        G: trained UNetG (eval mode)
        vol_cube: numpy float32 (D, H, W) in [0, 1]
        axis: 'z', 'x', or 'y'
        offsets: input slice offsets (e.g., OFFSETS_IN6 = [-15,-9,-3,3,9,15])
        device: torch device
        batch_size: slices per forward pass

    Returns:
        torch tensor (D, H, W): copy of vol_cube with every interior slice
        along `axis` replaced by the model prediction (all-replacement
        protocol). Apply `parity_paste` afterwards for the deployment-parity
        protocol, where acquired even-index slices are retained.
    """
    D, H, W = vol_cube.shape
    k_max = max(abs(o) for o in offsets)
    out = torch.from_numpy(vol_cube.copy()).float()
    if axis == 'z':
        target_range = range(k_max, D - k_max)
    elif axis == 'x':
        target_range = range(k_max, W - k_max)
    elif axis == 'y':
        target_range = range(k_max, H - k_max)
    else:
        raise ValueError(axis)
    targets = list(target_range)
    for i in range(0, len(targets), batch_size):
        batch = targets[i:i + batch_size]
        ins = []
        for t in batch:
            if axis == 'z':
                inp = np.stack([vol_cube[t + o, :, :] for o in offsets], axis=0)
            elif axis == 'x':
                inp = np.stack([vol_cube[:, :, t + o] for o in offsets], axis=0)
            else:
                inp = np.stack([vol_cube[:, t + o, :] for o in offsets], axis=0)
            ins.append(inp.astype(np.float32))
        ins = torch.from_numpy(np.stack(ins, axis=0)).to(device)
        pr = G(ins).cpu()
        for j, t in enumerate(batch):
            p = pr[j, 0]
            if axis == 'z':
                out[t] = p
            elif axis == 'x':
                out[:, :, t] = p
            else:
                out[:, t, :] = p
    return out  # torch (D, H, W)


@torch.no_grad()
def reconstruct_axis_parity(G, vol_cube, axis, offsets, device,
                            batch_size=16):
    """Parity variant of `reconstruct_axis`: only ODD-index slices along
    `axis` are replaced by (clamped) model predictions; even (acquired)
    slices and boundary odd slices keep the input volume.

    Used by the strictly-sequential protocol (Table S4) and the k=1
    deployment-parity comparisons.
    """
    D, H, W = vol_cube.shape
    k_max = max(abs(o) for o in offsets)
    out = torch.from_numpy(vol_cube.copy()).float()
    n = {'z': D, 'x': W, 'y': H}[axis]
    targets = [t for t in range(k_max, n - k_max) if t % 2 == 1]
    for i in range(0, len(targets), batch_size):
        batch = targets[i:i + batch_size]
        ins = []
        for t in batch:
            if axis == 'z':
                inp = np.stack([vol_cube[t + o, :, :] for o in offsets], 0)
            elif axis == 'x':
                inp = np.stack([vol_cube[:, :, t + o] for o in offsets], 0)
            else:
                inp = np.stack([vol_cube[:, t + o, :] for o in offsets], 0)
            ins.append(inp.astype(np.float32))
        ins = torch.from_numpy(np.stack(ins, axis=0)).to(device)
        pr = G(ins).cpu()
        for j, t in enumerate(batch):
            p = pr[j, 0].clamp(0.0, 1.0)
            if axis == 'z':
                out[t] = p
            elif axis == 'x':
                out[:, :, t] = p
            else:
                out[:, t, :] = p
    return out


def gtfree_boundary_fill_z(V: torch.Tensor, vol_np: np.ndarray,
                           k_max: int) -> list:
    """GT-free fill of odd z slices OUTSIDE the model coverage
    [k_max, D-k_max): linear interpolation of the +-1 even (acquired)
    neighbours; the last odd slice without a t+1 neighbour copies t-1.
    This is stricter than the idealized parity protocols, which keep GT at
    those boundary odd slices. Modifies `V` in place; returns the filled
    indices."""
    D = V.shape[0]
    filled = []
    for t in range(1, D, 2):
        if k_max <= t < D - k_max:
            continue  # model-filled
        if t + 1 < D:
            V[t] = torch.from_numpy(
                (0.5 * (vol_np[t - 1] + vol_np[t + 1])).astype(np.float32))
        else:
            V[t] = torch.from_numpy(vol_np[t - 1].astype(np.float32))
        filled.append(t)
    return filled


@torch.no_grad()
def sequential_triaxis(G, vol_cube, offsets, device, batch_size=16):
    """Strictly-sequential GT-free tri-axis reconstruction (Table S4).

    Pass 1 (z-fill): parity z pass fills the odd (missing) z slices; odd
    slices outside the offset margin are filled GT-free by linear
    interpolation of the +-1 acquired neighbours. The z pass is deployable
    by construction (odd target + odd offsets => all inputs acquired).
    Pass 2: the x and y passes run ON the z-filled volume, so their inputs
    never contain ground truth at unacquired positions.

    Returns (V_zfill, V_x_seq, V_y_seq) torch tensors.
    """
    k_max = max(abs(o) for o in offsets)
    V_z = reconstruct_axis_parity(G, vol_cube, 'z', offsets, device,
                                  batch_size)
    gtfree_boundary_fill_z(V_z, vol_cube, k_max)
    zfill_np = V_z.numpy().astype(np.float32)
    V_x = reconstruct_axis_parity(G, zfill_np, 'x', offsets, device,
                                  batch_size)
    V_y = reconstruct_axis_parity(G, zfill_np, 'y', offsets, device,
                                  batch_size)
    return V_z, V_x, V_y


def parity_paste(V: torch.Tensor, acquired: torch.Tensor, z0: int) -> torch.Tensor:
    """Deployment-parity protocol: restore acquired even-index z-slices.

    Args:
        V: reconstructed volume (D, H, W)
        acquired: volume holding the acquired data at even global-z indices
                  (in benchmark evaluation this is the GT cube)
        z0: global z index of V[0] (parity is defined on GLOBAL indices)

    Returns:
        copy of V with every slice at even global z replaced by the
        acquired slice. Only odd-z content remains model-synthesized.
    """
    out = V.clone()
    for zl in range(out.shape[0]):
        if (z0 + zl) % 2 == 0:
            out[zl] = acquired[zl]
    return out


# ════════════════════════════════════════════════════════════
# GT-free aggregations
# ════════════════════════════════════════════════════════════

def aggregate_tri_mean(V_z: torch.Tensor, V_x: torch.Tensor, V_y: torch.Tensor):
    return (V_z + V_x + V_y) / 3.0, {'wz': 1/3, 'wx': 1/3, 'wy': 1/3}


def aggregate_tri_median(V_z: torch.Tensor, V_x: torch.Tensor, V_y: torch.Tensor):
    V_stack = torch.stack([V_z, V_x, V_y], dim=0)
    return V_stack.median(dim=0).values, {'wz': None, 'wx': None, 'wy': None,
                                          'note': 'per-voxel median (no scalar weights)'}


def aggregate_tri_consensus(V_z: torch.Tensor, V_x: torch.Tensor, V_y: torch.Tensor,
                            eps: float = 1e-3):
    """Scalar inverse cross-axis disagreement weight.

    For axis a, define mu_other_a = mean of predictions from the other two axes.
    d_a = mean over voxels of |V_a - mu_other_a|. Then w_a is proportional
    to 1 / (d_a + eps).
    """
    mu_other_z = (V_x + V_y) / 2.0
    mu_other_x = (V_z + V_y) / 2.0
    mu_other_y = (V_z + V_x) / 2.0
    d_z = float((V_z - mu_other_z).abs().mean().item())
    d_x = float((V_x - mu_other_x).abs().mean().item())
    d_y = float((V_y - mu_other_y).abs().mean().item())
    wz = 1.0 / (d_z + eps); wx = 1.0 / (d_x + eps); wy = 1.0 / (d_y + eps)
    s = wz + wx + wy
    wz /= s; wx /= s; wy /= s
    V = wz * V_z + wx * V_x + wy * V_y
    return V, {'wz': wz, 'wx': wx, 'wy': wy, 'd_z': d_z, 'd_x': d_x, 'd_y': d_y}


def aggregate_tri_weuler_self(V_z: torch.Tensor, V_x: torch.Tensor, V_y: torch.Tensor,
                              device, eps: float = 1e-3, n_probe: int = 8):
    """GT-free weighted-Euler aggregation.

    For each axis, compute the per-slice Euler characteristic of the
    prediction (no GT). Use the *median across axes* as the consensus anchor
    and weight each axis by 1 / (|Euler_a - median| + eps).
    """
    D = V_z.shape[0]
    idxs = np.linspace(16, D - 16, n_probe).astype(int)

    def ax_euler(V):
        p = torch.stack([V[i] for i in idxs], dim=0).unsqueeze(1).float().to(device)
        return float(euler_number_2d(p).mean().item())

    e_z = ax_euler(V_z); e_x = ax_euler(V_x); e_y = ax_euler(V_y)
    e_med = float(np.median([e_z, e_x, e_y]))
    wz = 1.0 / (abs(e_z - e_med) + eps)
    wx = 1.0 / (abs(e_x - e_med) + eps)
    wy = 1.0 / (abs(e_y - e_med) + eps)
    s = wz + wx + wy
    wz /= s; wx /= s; wy /= s
    V = wz * V_z + wx * V_x + wy * V_y
    return V, {'wz': wz, 'wx': wx, 'wy': wy,
               'e_z': e_z, 'e_x': e_x, 'e_y': e_y, 'e_median': e_med}


def aggregate_tri_voxel_consensus(V_z: torch.Tensor, V_x: torch.Tensor, V_y: torch.Tensor,
                                  tau: float = 0.1):
    """Per-voxel soft weighting by inverse distance to other-axis mean.

    For each voxel v and axis a:
        d_a(v) = |V_a(v) - mu_other_a(v)|
        w_a(v) = exp(-d_a(v) / tau)
    Renormalize so sum_a w_a(v) = 1.
    """
    mu_other_z = (V_x + V_y) / 2.0
    mu_other_x = (V_z + V_y) / 2.0
    mu_other_y = (V_z + V_x) / 2.0
    d_z = (V_z - mu_other_z).abs()
    d_x = (V_x - mu_other_x).abs()
    d_y = (V_y - mu_other_y).abs()
    wz = torch.exp(-d_z / tau)
    wx = torch.exp(-d_x / tau)
    wy = torch.exp(-d_y / tau)
    s = wz + wx + wy + 1e-8
    wz = wz / s; wx = wx / s; wy = wy / s
    V = wz * V_z + wx * V_x + wy * V_y
    return V, {'wz_mean': float(wz.mean().item()),
               'wx_mean': float(wx.mean().item()),
               'wy_mean': float(wy.mean().item()),
               'tau': tau}


def compute_all_gtfree_aggregations(V_z: torch.Tensor, V_x: torch.Tensor,
                                    V_y: torch.Tensor, device):
    """Return dict of {name -> (V_aggregated, info)} for all GT-free variants."""
    out = {}
    out['tri_mean']            = aggregate_tri_mean(V_z, V_x, V_y)
    out['tri_median']          = aggregate_tri_median(V_z, V_x, V_y)
    out['tri_consensus']       = aggregate_tri_consensus(V_z, V_x, V_y)
    out['tri_weuler_self']     = aggregate_tri_weuler_self(V_z, V_x, V_y, device)
    out['tri_voxel_consensus'] = aggregate_tri_voxel_consensus(V_z, V_x, V_y, tau=0.1)
    return out


# ════════════════════════════════════════════════════════════
# Cube-level evaluation
# ════════════════════════════════════════════════════════════

def metrics_from_cube(pred_cube, gt_cube, n_z=24, n_cross=8, device='cpu'):
    """
    Compute SSIM on a batch of z-slices + full 2D morphology on the same batch.
    Also xz/yz SSIM for cross-plane consistency.

    Note: evaluates whatever volume is passed in — for the deployment-parity
    protocol this is the final, parity-pasted volume. The `n_z` z-slices are
    sampled uniformly over the volume interior (both even/acquired and
    odd/synthesized indices), not only the synthesized slices.

    Args:
        pred_cube, gt_cube: torch tensors (D, H, W) in [0, 1]
    """
    D, H, W = gt_cube.shape
    z_idxs = np.linspace(16, D - 16, n_z).astype(int)
    p_stack = torch.stack([pred_cube[zi] for zi in z_idxs], dim=0).unsqueeze(1).float()
    g_stack = torch.stack([gt_cube[zi] for zi in z_idxs], dim=0).unsqueeze(1).float()
    # Move to GPU if available for SSIM/morphology speed
    p = p_stack.to(device); g = g_stack.to(device)
    ssim = float(pytorch_msssim_fn(p, g, data_range=1.0).item())
    mse = F.mse_loss(p, g).item()
    psnr = -10 * np.log10(mse + 1e-10)
    morph = compute_all_morphological_metrics(p, g, max_lag=16)
    H2, W2 = p.shape[-2:]
    area_mpx = (H2 * W2) / 1e6
    # Cross-plane SSIM
    xz_ssims, yz_ssims = [], []
    y_idxs = np.linspace(H // 4, 3 * H // 4, n_cross).astype(int)
    x_idxs = np.linspace(W // 4, 3 * W // 4, n_cross).astype(int)
    for yi in y_idxs:
        pz = pred_cube[:, yi, :].unsqueeze(0).unsqueeze(0).float().to(device)
        gz = gt_cube[:, yi, :].unsqueeze(0).unsqueeze(0).float().to(device)
        if pz.shape[-1] >= 11 and pz.shape[-2] >= 11:
            xz_ssims.append(float(pytorch_msssim_fn(pz, gz, data_range=1.0).item()))
    for xi in x_idxs:
        pz = pred_cube[:, :, xi].unsqueeze(0).unsqueeze(0).float().to(device)
        gz = gt_cube[:, :, xi].unsqueeze(0).unsqueeze(0).float().to(device)
        if pz.shape[-1] >= 11 and pz.shape[-2] >= 11:
            yz_ssims.append(float(pytorch_msssim_fn(pz, gz, data_range=1.0).item()))
    return {
        'ssim_z': ssim, 'psnr_z': psnr,
        'dphi': float(morph['dphi']),
        'dsa': float(morph['dsa']),
        'd_euler': float(morph['d_euler']),
        'd_euler_per_mpx': float(morph['d_euler']) / area_mpx,
        's2_mse': float(morph['s2_mse']),
        'xz_ssim_mean': float(np.mean(xz_ssims)) if xz_ssims else 0.0,
        'yz_ssim_mean': float(np.mean(yz_ssims)) if yz_ssims else 0.0,
        'n_z_samples': len(z_idxs),
        'n_cross_samples': len(xz_ssims),
    }
