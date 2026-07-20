# Third-party notices

This repository (MIT License, see `LICENSE`) depends on or references the
following external software and data. Nothing below is redistributed here.

## Python dependencies (installed separately; see requirements*.txt)
- PyTorch (BSD-3-Clause), NumPy (BSD), scikit-image (BSD-3-Clause),
  pytorch-msssim (MIT), thop (MIT)
- Reproduction extras: pandas, matplotlib, tifffile, porespy, taufactor,
  optuna, einops (each under its own permissive license)

## External code (not included)
- **I3Net** (baseline comparison): official implementation at
  https://github.com/eeeric-code/I3Net (commit 30be7ba). The upstream
  repository carries no LICENSE file, so no license terms can be assumed;
  users must obtain it from the upstream authors and respect their terms.
  This repository ships only an adapter (`baselines/i3net.py`).

## Data (not included)
- Micro-CT volumes are public third-party datasets: Digital Rocks Portal
  project 317 (Neumann et al. 2020; Lucas-Oliveira et al. 2022) and the
  Imperial College London pore-scale collection (Raeini et al. 2017;
  Bijeljic et al. 2013). Downloaded by the user under the providers' terms.
