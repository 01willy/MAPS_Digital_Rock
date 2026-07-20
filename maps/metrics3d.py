"""
3D pore-morphology metrics for volume-level evaluation.

3D versions of the slice metrics in `maps/metrics.py` (S2, lineal-path,
connected porosity), plus coordination number Z (porespy.networks.snow2),
pore-size-distribution Wasserstein-1, 3D Euler characteristic
(skimage.measure.euler_number), and tortuosity tau (taufactor diffusive +
D3Q19 LBM hydraulic, see `lbm/d3q19.py`).

Input convention (matches `maps/metrics.py`):
    Float volume in [0, 1] with 0=pore, 1=solid.

Binarization: per-image Otsu by default (each of pred and GT gets its own
threshold) — important when GT is hard-binary while pred is continuous from
a sigmoid head.

Master entry: `compute_3d_morphology_metrics(pred_vol, gt_vol, ...)`.

Optional dependencies: porespy (coordination number, PSD), taufactor
(diffusive tortuosity). Functions degrade gracefully (NaN) if missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import wasserstein_distance


# ───────────────────── Otsu + binarization ─────────────────────

def otsu_threshold_np(vol: np.ndarray) -> float:
    """Otsu threshold on a numpy volume in [0, 1]. Stable for empty/uniform inputs."""
    flat = vol.ravel().astype(np.float64)
    hist, _ = np.histogram(flat, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega) + 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k = int(np.argmax(sigma_b2))
    return (k + 0.5) / 256.0


def binarize_per_image_otsu(pred: np.ndarray, gt: np.ndarray):
    """Binarize each volume with its OWN Otsu threshold.

    This matches the convention in `maps.metrics.porosity_hard` (which
    computes Otsu per-image). Important when GT is hard-binary (0/1) while
    pred is continuous from a sigmoid head — using GT's Otsu (~0) would
    over-solidify the continuous pred.

    Returns: pred_solid (bool), gt_solid (bool), (tau_pred, tau_gt)
    """
    tau_p = otsu_threshold_np(pred.astype(np.float32))
    tau_g = otsu_threshold_np(gt.astype(np.float32))
    return (pred > tau_p), (gt > tau_g), (tau_p, tau_g)


def binarize_with_gt_otsu(pred: np.ndarray, gt: np.ndarray,
                          tau: float | None = None):
    """Binarize both volumes with a single GT-derived Otsu threshold.

    For callers who explicitly want shared-threshold binarization
    (binary-vs-binary comparisons).
    """
    if tau is None:
        tau = otsu_threshold_np(gt.astype(np.float32))
    return (pred > tau), (gt > tau), tau


# ───────────────────── S2 (3D, radial) ─────────────────────

def s2_3d_radial(solid: np.ndarray, max_r: int = 64) -> np.ndarray:
    """Two-point correlation S2(r) of the SOLID phase via FFT autocorrelation,
    radial-averaged with periodic-distance bins.

    S2(0) = volume fraction of solid phase = 1 - porosity_pore.
    Returns array of length max_r.
    """
    Z, Y, X = solid.shape
    phase = solid.astype(np.float32)
    f = np.fft.rfftn(phase)
    power = (f * np.conj(f)).real
    ac = np.fft.irfftn(power, s=(Z, Y, X)) / float(Z * Y * X)

    z = np.arange(Z, dtype=np.float32); z = np.minimum(z, Z - z)
    y = np.arange(Y, dtype=np.float32); y = np.minimum(y, Y - y)
    x = np.arange(X, dtype=np.float32); x = np.minimum(x, X - x)
    dz, dy, dx = np.meshgrid(z, y, x, indexing='ij')
    dist = np.sqrt(dz * dz + dy * dy + dx * dx)

    s2 = np.zeros(max_r, dtype=np.float64)
    for r in range(max_r):
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if mask.any():
            s2[r] = ac[mask].mean()
    return s2


def correlation_length(s2: np.ndarray, phi_solid: float) -> float:
    """Correlation length l_c — radius where normalized excess
    (S2(r) - phi^2)/(phi - phi^2) drops below 1/e. Linear interpolation
    between consecutive bins. Returns max_r-1 if it never decays that far.
    """
    var = phi_solid - phi_solid * phi_solid
    if var <= 1e-9:
        return 0.0
    target = phi_solid * phi_solid + var / np.e
    for r in range(1, len(s2)):
        if s2[r] <= target:
            f1, f2 = s2[r - 1], s2[r]
            if f1 == f2:
                return float(r)
            return float(r - 1) + (f1 - target) / (f1 - f2)
    return float(len(s2) - 1)


# ───────────────────── Lineal-path (3D) ─────────────────────

def lineal_path_3d(solid: np.ndarray, max_r: int = 32,
                   phase_pore: bool = True) -> np.ndarray:
    """Lineal-path function L(r) for the pore phase (default), averaged over
    the three Cartesian axes. Non-periodic (truncates the last r positions).
    """
    target = (~solid) if phase_pore else solid
    Lr = np.zeros(max_r, dtype=np.float64)
    Lr[0] = float(target.mean())
    for r in range(1, max_r):
        accum = []
        for axis in (0, 1, 2):
            cur = target
            for k in range(1, r + 1):
                shifted = np.roll(target, shift=-k, axis=axis)
                cur = cur & shifted
            slicer = [slice(None)] * 3
            slicer[axis] = slice(0, target.shape[axis] - r)
            accum.append(float(cur[tuple(slicer)].mean()))
        Lr[r] = float(np.mean(accum))
    return Lr


# ───────────────────── Connected porosity (3D, 26-conn) ─────────────────────

def connected_porosity_3d(solid: np.ndarray) -> float:
    """Fraction of the volume occupied by the LARGEST 26-connected pore component."""
    pore = ~solid
    if pore.sum() == 0:
        return 0.0
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n = ndimage.label(pore, structure=structure)
    if n == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return float(sizes.max()) / float(pore.size)


# ───────────────────── 3D Euler characteristic + specific surface ─────────────────────

def euler_3d(solid: np.ndarray) -> float:
    """3D Euler characteristic of the solid phase (26-connectivity),
    via skimage.measure.euler_number."""
    from skimage.measure import euler_number
    return float(euler_number(solid.astype(bool), connectivity=3))


def specific_surface_3d(solid: np.ndarray) -> float:
    """6-connectivity interface face count per voxel (solid-pore faces)."""
    s = solid.astype(np.uint8)
    faces = 0
    for ax in range(3):
        a = np.take(s, range(s.shape[ax] - 1), axis=ax)
        b = np.take(s, range(1, s.shape[ax]), axis=ax)
        faces += int(np.count_nonzero(a != b))
    return faces / s.size


# ───────────────────── Coordination number Z via snow2 ─────────────────────

def _central_crop(vol: np.ndarray, target_shape) -> np.ndarray:
    """Return central crop of vol to target_shape (no-op if already smaller)."""
    out_slices = []
    for s, t in zip(vol.shape, target_shape):
        if s <= t:
            out_slices.append(slice(0, s))
        else:
            off = (s - t) // 2
            out_slices.append(slice(off, off + t))
    return vol[tuple(out_slices)]


def coordination_z(solid: np.ndarray, sigma: float = 0.4,
                   r_max: int = 4,
                   crop_to: tuple | None = (64, 128, 128)) -> dict:
    """Mean coordination number Z via PoreSpy SNOW network extraction.

    Uses fixed (sigma, r_max) per literature defaults for fair comparison.
    `crop_to` central-crops large cubes to keep snow2 tractable
    (a 128x256x256 cube takes 5+ min; 64x128x128 finishes in ~30 s with
    O(1000) pores). Set crop_to=None to disable.
    """
    try:
        import porespy as ps
    except ImportError as e:
        return {'Z_mean': float('nan'), 'Z_std': float('nan'),
                'n_pores': 0, 'error': f'porespy missing: {e}'}
    if crop_to is not None:
        solid = _central_crop(solid, crop_to)
    pore = ~solid
    if pore.sum() < 100:
        return {'Z_mean': float('nan'), 'Z_std': float('nan'), 'n_pores': 0}
    try:
        net = ps.networks.snow2(pore, sigma=sigma, r_max=r_max,
                                accuracy='standard', boundary_width=0)
    except Exception as e:
        return {'Z_mean': float('nan'), 'Z_std': float('nan'),
                'n_pores': 0, 'error': str(e)}
    net_d = net.network if hasattr(net, 'network') else net
    coords = net_d.get('pore.coords', None)
    if coords is None or len(coords) == 0:
        return {'Z_mean': float('nan'), 'Z_std': float('nan'), 'n_pores': 0}
    n_pores = int(coords.shape[0])
    conns = net_d.get('throat.conns', None)
    if conns is None or len(conns) == 0:
        return {'Z_mean': 0.0, 'Z_std': 0.0, 'n_pores': n_pores, 'n_throats': 0}
    deg = np.zeros(n_pores, dtype=np.int64)
    np.add.at(deg, conns[:, 0], 1)
    np.add.at(deg, conns[:, 1], 1)
    return {
        'Z_mean': float(deg.mean()),
        'Z_std': float(deg.std()),
        'n_pores': n_pores,
        'n_throats': int(conns.shape[0]),
    }


# ───────────────────── PSD Wasserstein-1 ─────────────────────

def psd_wasserstein(solid_pred: np.ndarray, solid_gt: np.ndarray,
                    crop_to: tuple | None = (64, 128, 128)) -> float:
    """Wasserstein-1 distance between pore-size distributions
    (porespy.filters.local_thickness). Distance in voxel units.

    Central-crops to keep local_thickness affordable.
    """
    try:
        import porespy as ps
    except ImportError:
        return float('nan')
    if crop_to is not None:
        solid_pred = _central_crop(solid_pred, crop_to)
        solid_gt = _central_crop(solid_gt, crop_to)
    pore_p, pore_g = ~solid_pred, ~solid_gt
    if pore_p.sum() < 50 or pore_g.sum() < 50:
        return float('nan')
    try:
        sz_p = ps.filters.local_thickness(pore_p).ravel()
        sz_g = ps.filters.local_thickness(pore_g).ravel()
        sz_p = sz_p[sz_p > 0]
        sz_g = sz_g[sz_g > 0]
        if sz_p.size == 0 or sz_g.size == 0:
            return float('nan')
        return float(wasserstein_distance(sz_p, sz_g))
    except Exception:
        return float('nan')


# ───────────────────── Tortuosity via taufactor (diffusive) ─────────────────────

def percolating_pore_mask(pore_bool: np.ndarray, axis: int) -> np.ndarray:
    """Keep only the pore voxels that belong to a 26-connected component
    touching BOTH `axis=0` and `axis=-1` faces. Dead-end pore pockets are
    set to solid (i.e. removed from the pore mask).

    Returns a uint8 mask where 1=percolating pore, 0=solid+isolated pore.

    Tortuosity convention in the digital-rock literature: tau is only
    defined on the percolating phase. Disconnected pockets are excluded
    since they cannot conduct flow/diffusion between the two boundaries
    (Andra et al. 2013; taufactor docs; OpenPNM benchmarks).
    """
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_cc = ndimage.label(pore_bool, structure=structure)
    if n_cc == 0:
        return np.zeros_like(pore_bool, dtype=np.uint8)
    # Labels present on the two boundary faces along `axis`
    take_first = [slice(None)] * 3
    take_last = [slice(None)] * 3
    take_first[axis] = 0
    take_last[axis] = -1
    labels_first = set(np.unique(labeled[tuple(take_first)])) - {0}
    labels_last = set(np.unique(labeled[tuple(take_last)])) - {0}
    percolating_labels = labels_first & labels_last
    if not percolating_labels:
        return np.zeros_like(pore_bool, dtype=np.uint8)
    mask = np.isin(labeled, list(percolating_labels))
    return mask.astype(np.uint8)


def tortuosity_taufactor(solid: np.ndarray, axes=(0, 1, 2),
                         device: str = 'cuda',
                         iter_limit: int = 500,
                         conv_crit: float = 1e-3,
                         percolating_only: bool = True) -> dict:
    """Diffusive tortuosity tau_d via taufactor steady-state Laplacian solver,
    each axis solved independently. taufactor expects pore=1 input.

    `percolating_only=True` (default) pre-filters the pore mask per axis to
    contain only the 26-CC components that span both face=0 and face=-1.
    This is the digital-rock standard convention (Andra et al. 2013,
    taufactor docs). For sandstones it removes <2% of pore voxels
    (essentially noise). For bimodal carbonates (e.g. Estaillades) it
    removes the disconnected micropore pockets that would otherwise cause
    the solver to oscillate indefinitely.

    `iter_limit=500` is a uniform cap; well-behaved cubes converge well
    below this and yield identical tau regardless of the cap.
    """
    try:
        import taufactor as tau_lib
    except ImportError:
        return {f'tau_tf_{n}': float('nan') for n in ('z', 'y', 'x')}
    pore_bool = ~solid
    out = {}
    axis_names = ['z', 'y', 'x']
    for ax in axes:
        name = axis_names[ax]
        if percolating_only:
            pore_u8 = percolating_pore_mask(pore_bool, axis=ax)
            phi_percolating = float(pore_u8.mean())
            out[f'tau_tf_{name}_phi_percolating'] = phi_percolating
            if phi_percolating < 1e-6:
                out[f'tau_tf_{name}'] = float('nan')
                continue
        else:
            pore_u8 = pore_bool.astype(np.uint8)
        # Permute so the solver-axis is axis 0
        order = [ax] + [a for a in range(3) if a != ax]
        img = np.ascontiguousarray(pore_u8.transpose(order))
        # Per-solve wall-clock timeout: taufactor's SOR can occasionally
        # diverge into a non-progressing oscillation on bimodal carbonate
        # cubes. Hard cap prevents runaway solves.
        import signal as _signal

        class _TauTimeout(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _TauTimeout()

        _signal.signal(_signal.SIGALRM, _alarm_handler)
        _signal.alarm(90)  # 90 s per axis-solve
        try:
            s = tau_lib.Solver(img, device=device)
            s.solve(verbose=False, iter_limit=iter_limit, conv_crit=conv_crit)
            tau_val = float(s.tau)
            if not np.isfinite(tau_val):
                tau_val = float('nan')
        except _TauTimeout:
            tau_val = float('nan')
            out[f'tau_tf_{name}_err'] = 'timeout_90s'
        except Exception as e:
            tau_val = float('nan')
            out[f'tau_tf_{name}_err'] = str(e)[:200]
        finally:
            _signal.alarm(0)
        out[f'tau_tf_{name}'] = tau_val
        if device.startswith('cuda'):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return out


# ───────────────────── Tortuosity via D3Q19 LBM (hydraulic) ─────────────────────

def tortuosity_lbm(solid: np.ndarray, n_steps: int = 3000,
                   tau_lbm: float = 1.0, body_force: float = 1e-5,
                   axes=(0, 1, 2), device: str = 'cuda',
                   voxel_size_um: float = 2.25) -> dict:
    """Hydraulic tortuosity from the D3Q19 LBM solver (lbm/d3q19.py).

    tau_h = <|u|>_fluid / <u_along_flow_axis>_fluid

    Also returns LBM permeability for cross-check.
    """
    try:
        from lbm.d3q19 import D3Q19LBM
    except ImportError:
        # Fallback when the repo root is not on sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            from lbm.d3q19 import D3Q19LBM
        except ImportError:
            return {f'tau_lbm_{n}': float('nan') for n in ('z', 'y', 'x')}

    out = {}
    axis_names = ['z', 'y', 'x']
    col_map = {0: 2, 1: 1, 2: 0}  # flow_axis (0=z,1=y,2=x) -> u channel
    for ax in axes:
        name = axis_names[ax]
        try:
            sim = D3Q19LBM(solid_mask=solid, device=device, tau=tau_lbm,
                           body_force=body_force, flow_axis=ax)
            for _ in range(n_steps):
                sim.step()
            _, u = sim.macroscopic()
            fluid_t = (~torch.from_numpy(solid).to(u.device))
            u_mag = torch.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
            u_axis = u[col_map[ax]]
            denom = float(u_axis[fluid_t].abs().mean().cpu().item())
            num = float(u_mag[fluid_t].mean().cpu().item())
            tau_h = num / denom if denom > 1e-30 else float('nan')
            perm = sim.permeability(voxel_size_um=voxel_size_um)
            out[f'tau_lbm_{name}'] = tau_h
            out[f'k_lbm_{name}_mD'] = perm['k_mD']
            del sim
        except Exception as e:
            out[f'tau_lbm_{name}'] = float('nan')
            out[f'k_lbm_{name}_mD'] = float('nan')
            out[f'tau_lbm_{name}_err'] = str(e)[:200]
        finally:
            if device.startswith('cuda'):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    return out


# ───────────────────── Master entry ─────────────────────

def compute_3d_morphology_metrics(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    *,
    max_r_s2: int = 64,
    max_r_lineal: int = 32,
    do_coord: bool = True,
    do_psd: bool = True,
    do_tau_taufactor: bool = True,
    do_tau_lbm: bool = True,
    tau_lbm_steps: int = 3000,
    device: str = 'cuda',
    binarize: str = 'per_image',
) -> dict:
    """Compute all 3D morphology metrics on a (pred, gt) volume pair.

    Inputs are float arrays in [0, 1], shape (Z, Y, X), convention 0=pore, 1=solid.

    `binarize` chooses the binarization convention:
      - 'per_image' (default, matches maps.metrics.porosity_hard):
        Otsu computed independently for pred and gt. Use this when GT is
        hard-binary and pred is continuous (typical for our setup).
      - 'gt_shared': single Otsu threshold from GT applied to both. Use this
        if both volumes have similar histograms (binary-vs-binary).
    """
    if binarize == 'per_image':
        pred_solid, gt_solid, taus = binarize_per_image_otsu(pred_vol, gt_vol)
        tau_th = taus[1]  # report gt threshold for logging
    else:
        pred_solid, gt_solid, tau_th = binarize_with_gt_otsu(pred_vol, gt_vol)
    phi_solid_p = float(pred_solid.mean())
    phi_solid_g = float(gt_solid.mean())
    phi_pore_p = 1.0 - phi_solid_p
    phi_pore_g = 1.0 - phi_solid_g

    out: dict = {
        'otsu_tau': float(tau_th),
        'porosity_pred': phi_pore_p,
        'porosity_gt': phi_pore_g,
        'dphi_3d': abs(phi_pore_p - phi_pore_g),
    }

    # S2 3D (solid-phase autocorrelation)
    s2_p = s2_3d_radial(pred_solid, max_r=max_r_s2)
    s2_g = s2_3d_radial(gt_solid, max_r=max_r_s2)
    out['s2_3d_mse'] = float(np.mean((s2_p[1:] - s2_g[1:]) ** 2))
    out['lc_pred'] = correlation_length(s2_p, phi_solid_p)
    out['lc_gt'] = correlation_length(s2_g, phi_solid_g)
    out['dlc'] = abs(out['lc_pred'] - out['lc_gt'])

    # Lineal-path 3D (pore phase)
    Lp = lineal_path_3d(pred_solid, max_r=max_r_lineal, phase_pore=True)
    Lg = lineal_path_3d(gt_solid, max_r=max_r_lineal, phase_pore=True)
    out['lpath_3d_mse'] = float(np.mean((Lp[1:] - Lg[1:]) ** 2))

    # Connected porosity 3D (26-conn)
    cp_p = connected_porosity_3d(pred_solid)
    cp_g = connected_porosity_3d(gt_solid)
    out['connected_porosity_pred'] = cp_p
    out['connected_porosity_gt'] = cp_g
    out['d_connected_porosity'] = abs(cp_p - cp_g)

    # 3D Euler characteristic + specific surface
    out['euler_3d_pred'] = euler_3d(pred_solid)
    out['euler_3d_gt'] = euler_3d(gt_solid)
    out['d_euler_3d'] = abs(out['euler_3d_pred'] - out['euler_3d_gt'])
    out['ssa_3d_pred'] = specific_surface_3d(pred_solid)
    out['ssa_3d_gt'] = specific_surface_3d(gt_solid)
    out['d_ssa_3d'] = abs(out['ssa_3d_pred'] - out['ssa_3d_gt'])

    if do_coord:
        cz_p = coordination_z(pred_solid)
        cz_g = coordination_z(gt_solid)
        out['Z_mean_pred'] = cz_p.get('Z_mean', float('nan'))
        out['Z_mean_gt'] = cz_g.get('Z_mean', float('nan'))
        out['n_pores_pred'] = cz_p.get('n_pores', 0)
        out['n_pores_gt'] = cz_g.get('n_pores', 0)
        if (np.isfinite(out['Z_mean_pred'])
                and np.isfinite(out['Z_mean_gt'])):
            out['dZ'] = abs(out['Z_mean_pred'] - out['Z_mean_gt'])
        else:
            out['dZ'] = float('nan')

    if do_psd:
        out['psd_wasserstein'] = psd_wasserstein(pred_solid, gt_solid)

    if do_tau_taufactor:
        tp = tortuosity_taufactor(pred_solid, device=device)
        tg = tortuosity_taufactor(gt_solid, device=device)
        for ax in ('z', 'y', 'x'):
            out[f'tau_tf_{ax}_pred'] = tp.get(f'tau_tf_{ax}', float('nan'))
            out[f'tau_tf_{ax}_gt'] = tg.get(f'tau_tf_{ax}', float('nan'))

    if do_tau_lbm:
        tp = tortuosity_lbm(pred_solid, n_steps=tau_lbm_steps, device=device)
        tg = tortuosity_lbm(gt_solid, n_steps=tau_lbm_steps, device=device)
        for ax in ('z', 'y', 'x'):
            out[f'tau_lbm_{ax}_pred'] = tp.get(f'tau_lbm_{ax}', float('nan'))
            out[f'tau_lbm_{ax}_gt'] = tg.get(f'tau_lbm_{ax}', float('nan'))
            out[f'k_lbm_{ax}_pred_mD'] = tp.get(f'k_lbm_{ax}_mD',
                                                float('nan'))
            out[f'k_lbm_{ax}_gt_mD'] = tg.get(f'k_lbm_{ax}_mD',
                                              float('nan'))

    return out


if __name__ == '__main__':
    # Quick self-test with a synthetic cube
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--skip_tau', action='store_true')
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1, size=(64, 96, 96)).astype(np.float32)
    pred = gt + 0.02 * rng.standard_normal(gt.shape).astype(np.float32)
    pred = np.clip(pred, 0, 1)
    m = compute_3d_morphology_metrics(pred, gt,
                                      do_tau_taufactor=not args.skip_tau,
                                      do_tau_lbm=not args.skip_tau,
                                      tau_lbm_steps=500,
                                      device=args.device)
    for k, v in m.items():
        print(f'{k:30s} {v}')
