"""
Shared benchmark-cube selection helpers.

Three cube-picking conventions are used across the paper's evaluations; all
are seeded, so every driver (and every reader re-running the code) gets the
same cube origins for a given volume shape.

1. `get_cubes` (seed 2025): 3 cubes of 128x256x256 inside the test z-slab
   [850, 1000) of a 1000^3 volume, first cube centered, remaining cubes
   rejection-sampled to be at least half-a-cube apart. Used by the BB main
   table, aggregation ablation, parity matrix, k-sweep and morphology
   evaluations. For 1000^3 volumes this yields origins
   (861,372,372) 'center' / (859,739,738) 'rand0' / (864,622,570) 'rand1'.

2. `testonly_origins` (seed 20260616): n cubes of 128^3 anchored at
   z0 = 850 (fully inside the test slab), (y0, x0) drawn from the fixed RNG,
   cube 0 xy-centered. Used by the multi-seed test-only LBM run (Table 3).

3. `find_cube_origins_256` (seed 2025): n cubes of 256^3 anchored at
   z0 = Z - 256 (a 256^3 cube cannot fit the 150-slice test band; the
   campaign uses the deepest slab, which extends into the validation band --
   documented in the paper). Used by the 8-domain LBM campaign (Table S15)
   and the sequential anisotropy sentinel.

4. `multicube_anisotropy_origins` (seed 2025): 3 cubes of 256^3 for the
   multi-cube anisotropy campaign (Table S16): 'center' anchored at
   z0 = Z - 256 and xy-centered; 'rand0'/'rand1' with z0 drawn from
   [val_lo - 100, Z - 256] and random xy.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.data import compute_splits  # noqa: E402


CUBE_ZHW = (128, 256, 256)
SEED_FOR_CUBES = 2025


def get_cubes(vol_shape, n=3, seed=SEED_FOR_CUBES):
    """3 disjoint-ish 128x256x256 cubes inside the test z-slab."""
    Z, Y, X = vol_shape
    Dc, Hc, Wc = CUBE_ZHW
    splits = compute_splits(Z)
    z_lo, z_hi = splits['test']
    Dc = min(Dc, z_hi - z_lo)
    rng = np.random.default_rng(seed)
    z0_c = z_lo + (z_hi - z_lo - Dc) // 2
    y0_c = (Y - Hc) // 2
    x0_c = (X - Wc) // 2
    cubes = [(z0_c, y0_c, x0_c, 'center')]
    while len(cubes) < n:
        z0 = int(rng.integers(z_lo, z_hi - Dc))
        y0 = int(rng.integers(0, Y - Hc))
        x0 = int(rng.integers(0, X - Wc))
        ok = True
        for (zp, yp, xp, _) in cubes:
            if (abs(zp - z0) < Dc // 2 and abs(yp - y0) < Hc // 2
                    and abs(xp - x0) < Wc // 2):
                ok = False
                break
        if ok:
            cubes.append((z0, y0, x0, f'rand{len(cubes) - 1}'))
    return cubes


ORIGIN_SEED_TESTONLY = 20260616  # cube (y0,x0) selection, distinct from model seeds


def testonly_origins(vol_shape, cube_size, n_cubes, splits):
    """n_cubes origins (z0,y0,x0) FULLY inside the test split (Table 3 run).

    z0 anchored at test_lo so the cube z-range = [test_lo, test_lo+cube_size)
    which must be <= test_hi (asserted). (y0,x0) drawn from a fixed RNG so the
    run is reproducible and disjoint-ish across cubes. Cube 0 is xy-centered.
    """
    Z, Y, X = vol_shape
    test_lo, test_hi = splits['test']
    z0 = test_lo
    assert z0 + cube_size <= test_hi, (
        f'cube_size={cube_size} does not fit test band [{test_lo},{test_hi}) '
        f'(need cube_size <= {test_hi - test_lo})')
    rng = np.random.default_rng(ORIGIN_SEED_TESTONLY)
    origins = [(z0, (Y - cube_size) // 2, (X - cube_size) // 2)]
    for _ in range(n_cubes - 1):
        y0 = int(rng.integers(0, Y - cube_size + 1))
        x0 = int(rng.integers(0, X - cube_size + 1))
        origins.append((z0, y0, x0))
    return origins[:n_cubes]


def multicube_anisotropy_origins(vol_shape, n=3, seed=SEED_FOR_CUBES,
                                 cube_size=256):
    """256^3 cube layout of the multi-cube anisotropy campaign (Table S16).

    The test slab (150 z slices) is shallower than the 256-cube z extent,
    so cubes necessarily span the validation+test band (disclosed in the
    paper). 'center' anchors z0 at Z - 256 and centers xy; the random cubes
    draw z0 from [val_lo - 100, Z - 256] and free xy from the fixed RNG.
    Returns [(z0, y0, x0, label), ...].
    """
    Z, Y, X = vol_shape
    splits = compute_splits(Z)
    z_max_valid = Z - cube_size            # 744 for Z=1000
    cubes = [(z_max_valid, (Y - cube_size) // 2, (X - cube_size) // 2,
              'center')]
    rng = np.random.default_rng(seed)
    val_lo = splits['val'][0]              # 700
    z_floor = max(val_lo - 100, 0)         # 600
    for i in range(n - 1):
        z0 = int(rng.integers(z_floor, z_max_valid + 1))
        y0 = int(rng.integers(0, Y - cube_size + 1))
        x0 = int(rng.integers(0, X - cube_size + 1))
        cubes.append((z0, y0, x0, f'rand{i}'))
    return cubes[:n]


def find_cube_origins_256(vol_shape, cube_size=256, n_cubes=1,
                          seed=SEED_FOR_CUBES):
    """256^3 cube origins of the 8-domain LBM campaign (Table S15).

    For 1000^3 volumes with the default seed this reproduces the campaign
    origins cube0=(744,372,372), cube1=(744,333,740), cube2=(744,739,284).
    """
    Z, Y, X = vol_shape
    rng = np.random.default_rng(seed)
    origins = []
    z0_anchor = max(0, Z - cube_size)
    y0c = (Y - cube_size) // 2
    x0c = (X - cube_size) // 2
    origins.append((z0_anchor, y0c, x0c))
    for _ in range(n_cubes - 1):
        y0 = int(rng.integers(0, Y - cube_size + 1))
        x0 = int(rng.integers(0, X - cube_size + 1))
        origins.append((z0_anchor, y0, x0))
    return origins
