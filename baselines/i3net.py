"""I3Net baseline wrapper for digital-rock sparse-slice interpolation.

Appears in Table 1 (BB main comparison, z_only row).

Imports the official I3Net implementation and provides a thin adapter that:
  - Accepts our (B, 6, H, W) input (6 context slices stacked as channels)
  - Outputs (B, 1, H, W) prediction bounded to [0, 1] for the target slice
  - Skips I3Net's `out[:, ::upscale] = x` known-slice re-injection (a medical
    CT-specific behaviour that does not apply to our k=1 sparse setup)

Reference:
  I3Net: Inter-Intra-slice Interpolation Network for Medical Slice Synthesis
  arXiv:2405.02857, IEEE TMI 2024
  Official code: https://github.com/eeeric-code/I3Net

The official code is NOT redistributed here. To run this baseline:

  git clone https://github.com/eeeric-code/I3Net.git _external/I3Net_official

(from the repository root), or set the environment variable
I3NET_OFFICIAL_DIR to an existing clone. This adapter was developed against
upstream commit 30be7ba (2024-09-05). License note: at that commit the
upstream repository carries no LICENSE file, so no license terms can be
assumed; it is referenced here for research comparison only, and users must
obtain the code from the upstream authors and respect their terms.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXT = Path(os.environ.get('I3NET_OFFICIAL_DIR',
                           _REPO_ROOT / '_external' / 'I3Net_official'))


# Defer the official-repo lookup and the heavy einops/DCT imports until
# model build, so this module can be imported without the upstream clone.
def _build_inner_i3net(in_slice=6, out_slice=1, n_feats=64, num_blocks=16,
                       head_num=1, win_num_sqrt=16, window_size=16):
    if not _EXT.exists():
        raise RuntimeError(
            f"I3Net official repo not found at {_EXT}. "
            "Run: git clone https://github.com/eeeric-code/I3Net.git "
            f"{_EXT}  (or set I3NET_OFFICIAL_DIR)"
        )
    if str(_EXT) not in sys.path:
        sys.path.insert(0, str(_EXT))
    from model_zoo.i3net.basic_model import I3Net as _I3NetOfficial
    args = argparse.Namespace()
    args.n_feats = n_feats
    args.kernel_size = 3
    args.num_blocks = num_blocks
    args.res_scale = 1
    args.lr_slice_patch = in_slice
    args.upscale = 1  # not used in our wrapper but kept for shape arithmetic
    args.hr_slice_patch = out_slice
    args.head_num = head_num
    args.win_num_sqrt = win_num_sqrt
    args.window_size = window_size
    return _I3NetOfficial(args)


class I3NetRock(nn.Module):
    """I3Net adapted to our (B, 6, H, W) -> (B, 1, H, W) bounded interface."""

    def __init__(self, in_ch=6, out_ch=1, n_feats=64, num_blocks=16,
                 head_num=1, win_num_sqrt=16, window_size=16):
        super().__init__()
        self.inner = _build_inner_i3net(
            in_slice=in_ch, out_slice=out_ch,
            n_feats=n_feats, num_blocks=num_blocks,
            head_num=head_num, win_num_sqrt=win_num_sqrt,
            window_size=window_size,
        )
        self.in_ch = in_ch
        self.out_ch = out_ch
        # The official forward ends with `out[:, ::upscale] = x`, which for
        # out_ch=1 would overwrite our prediction with an input slice. We
        # therefore re-implement forward below without that final line.

    def forward(self, x):
        """
        Input:  x of shape (B, in_ch, H, W), e.g. (B, 6, 256, 256)
        Output: y of shape (B, out_ch, H, W), bounded to [0, 1].
        """
        # The official model expects (B, H, W, in_slice). Permute.
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()

        # Re-implementation of the relevant parts of inner.forward(x) WITHOUT
        # the input-overwrite at the end (copy of basic_model.py's
        # I3Net.forward up to the final `out[:,::upscale] = x` line).
        inner = self.inner
        x2 = x_bhwc.permute(0, 3, 1, 2).contiguous()       # (B, C, H, W)
        x_head = inner.head(x2)

        res = x_head
        align_list = []
        res = inner.alignment[0](res) + res
        align_list.append(res)

        for idx, layer in enumerate(inner.body):
            res = layer(res)
            if idx in [3, 7]:
                res = inner.alignment[idx // 4 + 1](res) + res
                align_list.append(res)

        res = inner.fuse_align(torch.cat(align_list, 1))
        res = res + x_head
        delta_logit = inner.tail(res)                      # (B, out_ch, H, W)

        # Residual baseline: average of the two closest context slices around
        # the target. OFFSETS_IN6 = [-15, -9, -3, +3, +9, +15], so channels
        # 2 (offset -3) and 3 (offset +3) bracket the target. This gives the
        # linear-interpolation baseline; the network learns the residual.
        # Without this skip the model defaults to sigmoid(0) ~ 0.5 constant
        # output and validation SSIM stalls (observed empirically).
        if self.in_ch == 6:
            baseline = 0.5 * (x[:, 2:3] + x[:, 3:4])       # (B, 1, H, W)
            out = torch.clamp(baseline + 0.1 * delta_logit, 0.0, 1.0)
        else:
            out = torch.sigmoid(delta_logit)
        return out


def build_i3net_rock(**kwargs):
    """Factory matching the model-registry convention."""
    return I3NetRock(**kwargs)


if __name__ == '__main__':
    # Smoke test
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = I3NetRock().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'I3NetRock params: {n_params / 1e6:.2f}M')
    x = torch.randn(2, 6, 256, 256, device=device)
    with torch.no_grad():
        y = model(x)
    print(f'Input  shape: {tuple(x.shape)}')
    print(f'Output shape: {tuple(y.shape)}')
    assert y.shape == (2, 1, 256, 256), f'Unexpected output shape: {y.shape}'
    assert torch.all((y >= 0) & (y <= 1)), 'Output not bounded to [0, 1]'
    print('Smoke test PASSED')
