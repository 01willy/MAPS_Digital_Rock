"""
Learned and classical comparison methods evaluated against MAPS in the paper.

Modules:
    classical        -- B1/B2/B3 linear & cubic interpolation (Section 4.6,
                        Tables S5-S7)
    unet3d           -- 3D U-Net model + fair single-target dataset (b5 / b5-large)
    train_unet3d     -- b5 / b5-large training entry (Tables 1, 3, 4)
    swinunet         -- SwinUNet 2D hybrid CNN+Transformer baseline (Table 1)
    i3net            -- I3Net medical slice-synthesis adapter (Table 1)
    train_i3net      -- I3Net training entry
    diffusion_v1_pixel -- V1 pixel-space conditional DDIM (diffusion comparison)
    diffusion_v2_pixel -- V2 pixel-space DDIM with channel-concat time cond.
    latent_diffusion -- V3 latent diffusion (VAE + latent DDIM; Section 5.7,
                        Table 4)
    eval_diffusion   -- V3 DDIM sampling + cube evaluation (DDIM-step sweep fig.)

All baselines consume the same data pipeline (`maps.data`), the same splits
(train [0,700) / val [700,850) / test [850,1000) along z) and the same
k=1 sparse scenario (even slices acquired, odd slices synthesized) as MAPS.
"""
