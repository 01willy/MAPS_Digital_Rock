"""Oracle aggregation reference (evaluation only).

This module implements the ``tri_weuler_oracle`` reference row of the paper:
an aggregation that weights each axis by its Euler-characteristic agreement
with the ground-truth volume. Because it takes the ground truth as an input,
it serves only as a reference point for evaluating the ground-truth-free
aggregations in ``maps/triaxis.py`` and is not part of the deployable
pipeline.
"""

import numpy as np
import torch

try:
    from .metrics import euler_number_2d
except ImportError:
    from metrics import euler_number_2d


__all__ = ['weighted_euler_aggregation_oracle']


def weighted_euler_aggregation_oracle(V_z, V_x, V_y, gt, device, n_probe=8):
    """Weight each axis by inverse Euler-characteristic error against ``gt``.

    For each axis reconstruction, the per-slice Euler-characteristic error
    against the ground truth is estimated on ``n_probe`` interior slices, and
    the axis weight is the inverse of that error. The result indicates what
    Euler-informed axis weighting could achieve with access to the ground
    truth; deployable variants replace the ground-truth anchor with the
    cross-axis median (see ``aggregate_tri_weuler_self``).

    Args:
        V_z, V_x, V_y: axis-wise reconstructions, torch (D, H, W)
        gt: ground-truth volume, torch (D, H, W)
        device: torch device
        n_probe: number of probe slices for the Euler estimate

    Returns:
        (V_weighted, info_dict)
    """
    D, H, W = V_z.shape

    def ax_ed(V):
        idxs = np.linspace(16, D - 16, n_probe).astype(int)
        p = torch.stack([V[i] for i in idxs], dim=0).unsqueeze(1).float().to(device)
        g = torch.stack([gt[i] for i in idxs], dim=0).unsqueeze(1).float().to(device)
        return float((euler_number_2d(p) - euler_number_2d(g)).abs().mean().item())

    ez, ex, ey = ax_ed(V_z), ax_ed(V_x), ax_ed(V_y)
    wz, wx, wy = 1.0 / (ez + 1e-3), 1.0 / (ex + 1e-3), 1.0 / (ey + 1e-3)
    s = wz + wx + wy
    V_w = (wz * V_z + wx * V_x + wy * V_y) / s
    return V_w, {'ez': ez, 'ex': ex, 'ey': ey,
                 'wz': wz / s, 'wx': wx / s, 'wy': wy / s,
                 'oracle': True}
