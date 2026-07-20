"""
MAPS: Multi-Axis 2.5D Slice Interpolation Preserving Petrophysical
Properties in Segmented Digital Rock.

Package modules:
    models    -- UNetG generator, PatchD discriminator, EMA
    losses    -- L1 + SSIM + morphology-preserving + hinge adversarial losses
    data      -- 2.5D slice-neighborhood datasets (offsets +/-{3,9,15})
    metrics   -- 2D slice morphology metrics (porosity, SA, S2, Euler, ...)
    metrics3d -- 3D volume morphology metrics (S2, lineal-path, tortuosity, ...)
    triaxis   -- axis-wise volume reconstruction + GT-free tri-axis aggregation
    oracle_eval -- ground-truth-using oracle aggregation (evaluation reference)
    checkpoint -- checkpoint state_dict extraction + fail-loud loading
"""

__version__ = '1.0.0'
