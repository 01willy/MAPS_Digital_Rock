#!/usr/bin/env python3
"""
LBM permeability solver — D3Q19 BGK in PyTorch.

Single-phase Stokes-flow permeability calculation:
- Lattice: D3Q19, BGK collision
- BC: Bounce-back at solids; body force along the flow axis
      (Stokes regime, e.g. 1e-5 in lattice units). Periodic lateral faces.
- Output: permeability component k_axis from Darcy's law: k = mu <u> / |F|

Validation: `validate_poiseuille()` — parallel-plate channel reproduces the
analytical k = W^2 / 12. Run via `python lbm/poiseuille_validation.py` or
`python lbm/d3q19.py --validate`.

References:
- Andra et al. 2013 (Digital Rock Physics benchmarks)
- Mostaghimi et al. 2013 (D3Q19 standard)
- Pan, Luo, Miller 2006 (BGK / MRT comparison)
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch


# D3Q19 lattice constants
# Velocities (19 directions including rest)
D3Q19_VEL = np.array([
    [0,0,0],            # 0: rest
    [1,0,0],[-1,0,0],   # 1,2: +/-x
    [0,1,0],[0,-1,0],   # 3,4: +/-y
    [0,0,1],[0,0,-1],   # 5,6: +/-z
    [1,1,0],[-1,-1,0],  # 7,8: xy diag
    [1,-1,0],[-1,1,0],  # 9,10
    [1,0,1],[-1,0,-1],  # 11,12: xz
    [1,0,-1],[-1,0,1],  # 13,14
    [0,1,1],[0,-1,-1],  # 15,16: yz
    [0,1,-1],[0,-1,1],  # 17,18
], dtype=np.int64)

D3Q19_W = np.array(
    [1/3] + [1/18]*6 + [1/36]*12, dtype=np.float64)

# Opposite-direction index for bounce-back
D3Q19_OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])


class D3Q19LBM:
    """D3Q19 BGK Lattice Boltzmann solver for Stokes flow permeability."""

    def __init__(self, solid_mask, device='cuda', dtype=torch.float32,
                 tau=1.0, body_force=None, flow_axis=2):
        """
        solid_mask: numpy bool array (Z, Y, X), True = solid
        flow_axis: 0=z, 1=y, 2=x  (cube axis for which we drive flow)
        body_force: scalar in lattice units (e.g. 1e-5 for Stokes regime)
                    Default 1e-5 if None.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.tau = tau
        self.cs2 = 1/3.0  # speed of sound squared (D3Q19)
        self.flow_axis = flow_axis
        self.bf = 1e-5 if body_force is None else body_force

        Z, Y, X = solid_mask.shape
        self.shape = (Z, Y, X)
        self.solid = torch.from_numpy(solid_mask.astype(np.bool_)).to(self.device)
        self.fluid = ~self.solid

        # Initialize distribution f to equilibrium at rest with rho=1
        self.f = torch.zeros((19, Z, Y, X), device=self.device, dtype=self.dtype)
        for i in range(19):
            self.f[i] = D3Q19_W[i]

        # Cache lattice constants
        self.vel = torch.tensor(D3Q19_VEL, device=self.device, dtype=torch.long)  # (19, 3)
        self.w = torch.tensor(D3Q19_W, device=self.device, dtype=self.dtype)
        self.opp = torch.tensor(D3Q19_OPP, device=self.device, dtype=torch.long)

        # Force vector in lattice units (only along flow_axis)
        self.force_vec = torch.zeros(3, device=self.device, dtype=self.dtype)
        # Map flow_axis (0=z,1=y,2=x) to lattice indexing convention used here.
        # We use indices [Z,Y,X] for f tensors but the force vector aligns with
        # vel columns: [vx, vy, vz]. So flow along x=col 0, y=col 1, z=col 2.
        # Map flow_axis -> col: flow_axis 0(z) -> col 2; 1(y) -> col 1; 2(x) -> col 0.
        col_map = {0: 2, 1: 1, 2: 0}
        self.force_vec[col_map[flow_axis]] = self.bf

    def equilibrium(self, rho, u):
        """Compute D3Q19 equilibrium distribution from rho (Z,Y,X) and u (3,Z,Y,X)."""
        # u shape (3, Z, Y, X) where channel order is [ux, uy, uz]
        feq = torch.zeros((19,) + self.shape, device=self.device, dtype=self.dtype)
        u2 = (u * u).sum(0)  # |u|^2
        for i in range(19):
            cu = (self.vel[i, 0].to(self.dtype) * u[0]
                  + self.vel[i, 1].to(self.dtype) * u[1]
                  + self.vel[i, 2].to(self.dtype) * u[2])
            feq[i] = self.w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        return feq

    @torch.no_grad()
    def step(self):
        """One LBM time step: collide, stream, bounce-back at solids."""
        # ── Compute macroscopic rho, u ──
        rho = self.f.sum(0)  # (Z, Y, X)
        # u_alpha = sum_i (vel[i,alpha] * f[i]) / rho
        u = torch.zeros((3,) + self.shape, device=self.device, dtype=self.dtype)
        for i in range(19):
            for a in range(3):
                if self.vel[i, a] != 0:
                    u[a] += self.vel[i, a].to(self.dtype) * self.f[i]
        # Avoid divide-by-zero in solids
        rho_safe = torch.where(self.fluid, rho, torch.ones_like(rho))
        u = u / rho_safe.unsqueeze(0)
        # Apply body force: u_eff = u + tau * F / rho
        u_eff = u + (self.tau / rho_safe.unsqueeze(0)) * self.force_vec.view(3, 1, 1, 1)

        # Collision (BGK): f* = f - (f - feq) / tau
        feq = self.equilibrium(rho, u_eff)
        self.f = self.f - (self.f - feq) / self.tau

        # ── Streaming ──
        # Roll along (Z, Y, X) for each direction.
        # f shape is (19, Z, Y, X) so f[i] axes are (Z, Y, X).
        # Velocity component vx affects the X axis, vy the Y axis, vz the Z axis.
        new_f = torch.empty_like(self.f)
        for i in range(19):
            cx, cy, cz = int(self.vel[i, 0]), int(self.vel[i, 1]), int(self.vel[i, 2])
            new_f[i] = torch.roll(self.f[i], shifts=(cz, cy, cx), dims=(0, 1, 2))
        self.f = new_f

        # ── Bounce-back at solids (after streaming) ──
        # Standard halfway bounce-back implementation:
        #   For each direction i, the post-streaming f[i] in solid sites is
        #   moved to f[opp(i)] in the same site.
        f_solid = self.f.clone()
        for i in range(19):
            self.f[self.opp[i].item(), self.solid] = f_solid[i, self.solid]

    @torch.no_grad()
    def macroscopic(self):
        rho = self.f.sum(0)
        u = torch.zeros((3,) + self.shape, device=self.device, dtype=self.dtype)
        for i in range(19):
            for a in range(3):
                if self.vel[i, a] != 0:
                    u[a] += self.vel[i, a].to(self.dtype) * self.f[i]
        rho_safe = torch.where(self.fluid, rho, torch.ones_like(rho))
        u = u / rho_safe.unsqueeze(0)
        # Zero velocity in solids
        for a in range(3):
            u[a] = torch.where(self.fluid, u[a], torch.zeros_like(u[a]))
        return rho, u

    @torch.no_grad()
    def permeability(self, dx_lu=1.0, mu_lu=None, voxel_size_um=2.25):
        """
        Compute permeability k from steady-state velocity field.
        k_lu = mu_lu * <u_axis> / |F|   (in lattice units squared)
        Convert to physical k via voxel size.
        """
        if mu_lu is None:
            mu_lu = (self.tau - 0.5) / 3.0  # standard BGK kinematic viscosity

        rho, u = self.macroscopic()
        # Mean velocity along flow axis (cube axis), averaged over the WHOLE
        # domain (Darcy superficial-velocity convention)
        col_map = {0: 2, 1: 1, 2: 0}  # flow_axis (z,y,x) -> u channel
        u_axis = u[col_map[self.flow_axis]]
        avg_u_lu = float(u_axis.mean().cpu())

        F_mag = float(self.force_vec.abs().max().cpu())
        # Lattice permeability (in lu^2): k_lu = mu * <u> / F
        k_lu = mu_lu * avg_u_lu / max(F_mag, 1e-30)

        # Physical k in m^2: dx = voxel_size_um * 1e-6 m
        dx_m = voxel_size_um * 1e-6
        k_m2 = k_lu * (dx_m ** 2)
        # mD = milliDarcy ; 1 Darcy = 9.869233e-13 m^2
        k_mD = k_m2 / 9.869233e-13 * 1000.0
        return {
            'avg_u_lu': avg_u_lu,
            'force_lu': F_mag,
            'mu_lu': mu_lu,
            'k_lu': k_lu,
            'k_m2': k_m2,
            'k_mD': k_mD,
            'porosity_total': float(self.fluid.float().mean().cpu()),
        }


def validate_poiseuille(width=24, length=8, n_steps=5000, device='cuda'):
    """
    Parallel-plate Poiseuille flow validation.
    Domain: (Z=length, Y=width, X=2) with solids at y=0 and y=W-1.
    Analytical k = W^2 / 12  (in lu^2) for flow along Z.
    """
    Z, Y, X = length, width, 2
    solid = np.zeros((Z, Y, X), dtype=bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    sim = D3Q19LBM(solid, device=device, tau=1.0, body_force=1e-5, flow_axis=0)
    for _ in range(n_steps):
        sim.step()
    res = sim.permeability(voxel_size_um=1.0)
    fluid_W = Y - 2  # interior fluid width
    k_analytical_lu = (fluid_W ** 2) / 12.0
    res['k_analytical_lu'] = k_analytical_lu
    res['relative_error'] = abs(res['k_lu'] - k_analytical_lu) / k_analytical_lu
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cube_path', help='float32 binary cube file (Z=Y=X=cube_size)')
    ap.add_argument('--cube_size', type=int, default=256)
    ap.add_argument('--n_steps', type=int, default=5000)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--body_force', type=float, default=1e-5)
    ap.add_argument('--flow_axis', type=int, choices=[0,1,2], default=0,
                    help='0=z, 1=y, 2=x')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--validate', action='store_true',
                    help='Run Poiseuille validation only')
    ap.add_argument('--out_json', default=None)
    ap.add_argument('--save_velocity', default=None, metavar='PATH',
                    help='save the steady-state velocity-magnitude field '
                         '|u| (lattice units, zero in solids) as a float32 '
                         '.npy volume -- the field behind the streamline '
                         'renderings')
    ap.add_argument('--voxel_size_um', type=float, default=2.25)
    args = ap.parse_args()

    device = 'cpu' if args.cpu else f'cuda:{args.gpu}'

    if args.validate:
        print(f'[VALIDATE] Poiseuille on {device}...')
        t0 = time.time()
        res = validate_poiseuille(width=24, length=8, n_steps=args.n_steps, device=device)
        t1 = time.time()
        print(json.dumps(res, indent=2, default=str))
        print(f'time: {t1-t0:.1f}s')
        return

    # Load cube
    cube = np.fromfile(args.cube_path, dtype=np.float32).reshape(
        args.cube_size, args.cube_size, args.cube_size)
    # Convention: 1=solid, so solid_mask = (cube > 0.5)
    solid_mask = cube > 0.5
    print(f'[LBM] cube_path={args.cube_path}')
    print(f'  shape={cube.shape}  porosity={1-cube.mean():.4f}')

    sim = D3Q19LBM(solid_mask, device=device, tau=args.tau,
                   body_force=args.body_force, flow_axis=args.flow_axis)
    print(f'  flow_axis={args.flow_axis}  tau={args.tau}  body_force={args.body_force}  device={device}')

    t0 = time.time()
    for step in range(args.n_steps):
        sim.step()
        if (step + 1) % max(1, args.n_steps // 10) == 0:
            res = sim.permeability(voxel_size_um=args.voxel_size_um)
            elapsed = time.time() - t0
            print(f'  [step {step+1:5d}/{args.n_steps}] k_mD={res["k_mD"]:.3f}  '
                  f'<u>={res["avg_u_lu"]:.3e}  elapsed={elapsed:.1f}s')

    res = sim.permeability(voxel_size_um=args.voxel_size_um)
    res['cube_path'] = args.cube_path
    res['n_steps'] = args.n_steps
    res['flow_axis'] = args.flow_axis
    res['voxel_size_um'] = args.voxel_size_um

    if args.save_velocity:
        _rho, u = sim.macroscopic()
        umag = torch.sqrt((u * u).sum(0)).cpu().numpy().astype(np.float32)
        vel_path = Path(args.save_velocity)
        vel_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vel_path, umag)
        print(f'[saved] |u| field -> {vel_path} '
              f'(shape {umag.shape}, max {umag.max():.3e} lu)')

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(res, f, indent=2, default=str)
        print(f'[saved] {args.out_json}')
    print(f'[DONE] k_mD = {res["k_mD"]:.3f}')


if __name__ == '__main__':
    main()
