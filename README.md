# MAPS — Multi-Axis 2.5D Slice Interpolation for Segmented Digital Rock

Reference implementation for the paper

> **MAPS: Multi-Axis 2.5D Slice Interpolation Preserving Petrophysical
> Properties in Segmented Digital Rock**
> Seungwon Baek, Juan Lee, Kyonghee Joo, Honggeun Jo, Suryeom Jo, Yongchae Cho
> *Computers & Geosciences* (submitted, 2026)

MAPS restores sparsely sampled through-plane (axial) slices of segmented
Micro-CT rock volumes to isotropic resolution while preserving the
petrophysical properties downstream analysis measures — porosity, specific
surface area, Euler characteristic, two-point correlation, and the LBM
permeability tensor. A single conditional 2.5D U-Net generator (PatchGAN
adversarial training, morphology-preserving losses) is applied along the
three orthogonal axes at inference and the three reconstructions are fused
by a ground-truth-free aggregation rule.

## Repository layout

```
maps/                 core library
  models.py           UNetG generator (base 80, 24.5 M) + SN-PatchGAN + EMA
  losses.py           L1 + SSIM + soft-Otsu morphology (phi, SA, S2, lineal-path) + hinge GAN
  data.py             2.5D slice-neighborhood datasets (offsets ±{3,9,15}), multi-axis sampling
  triaxis.py          GT-free tri-axis aggregations (tri_mean, tri_weuler_self, ...)
  oracle_eval.py      ground-truth-using oracle aggregation (evaluation reference)
  metrics.py          slice-level metrics (SSIM, dphi, dSA, S2, Euler)
  metrics3d.py        3D morphology metrics (binarized porosity, connectivity, PSD)
train_stage1.py       Stage 1: z-axis GAN pretraining (200 epochs; defaults = Table S9 recipe)
train_stage2.py       Stage 2: multi-axis fine-tune (re-initialized D, EMA, ~90 min)
infer_triaxis.py      tri-axis reconstruction + aggregation (deployment-parity / all-replacement / sequential)
eval_all_metrics.py   unified evaluation entry point
lbm/
  d3q19.py            D3Q19 BGK single-phase Stokes LBM permeability solver (PyTorch)
  poiseuille_validation.py  analytical parallel-plate check (k = W^2/12)
baselines/            learned + classical comparison methods of the paper
  classical.py        B1/B2/B3 linear & cubic interpolation + tri-axis linear
  unet3d.py           3D U-Net model (b5 base=24 ~3.15M / b5-large base=64 ~22.4M)
  train_unet3d.py     b5 / b5-large training (fair single-target, L1, z-axis)
  swinunet.py         SwinUNet 2D hybrid CNN+Transformer (model + training)
  i3net.py            I3Net adapter (official upstream code fetched separately)
  train_i3net.py      I3Net training entry (L1, time-matched)
  diffusion_v1_pixel.py   V1 pixel-space conditional DDIM
  diffusion_v2_pixel.py   V2 pixel-space DDIM (channel-concat time cond.)
  latent_diffusion.py     V3 latent diffusion (VAE + latent DDIM)
  eval_diffusion.py       V1/V2/V3 DDIM sampling + cube evaluation
  train_multik.py         joint multi-k training (per-k vs joint comparison)
analysis/             evaluation drivers behind the paper's tables and figures
  benchmark_cubes.py  seeded test-cube selection shared by all drivers
  inference_tiled.py  tiled 2D/3D inference + parity reconstruction
  benchmark_eval.py   cross-model benchmark eval (b5/SwinUNet/I3Net rows, convergence parity)
  ksweep_eval.py      k-sweep stress evaluation
  morphology3d_eval.py     3D morphology suite (S2, lineal-path, Z, PSD, tau, k)
  aggregation_ablation.py  GT-free aggregation variants + oracle reference
  parity_matrix_eval.py    Stage1 x Stage2 seed-matrix robustness
  sequential_eval.py       idealized vs strictly-sequential tri-axis (k=1)
  anisotropy_eval.py       multi-cube permeability-tensor anisotropy + ratio L1
  lbm_multiseed_eval.py    test-only multi-seed LBM run (+ lbm_multiseed_aggregate.py)
  lbm_8domain_eval.py      8-domain k=1 LBM trace campaign
  slab_stitching_probe.py  contiguous-gap (vertical-stitching) probe
  failure_regression.py    failure-map OLS regression + PvA table + figures
  wilcoxon_stats.py        exact paired Wilcoxon tests on the long CSVs
  channel_ablation_eval.py common-target channel/offset ablation eval
  threshold_sweep.py       binarization-threshold phi(tau) sweep
  pore_size_distribution.py EDT log-binned pore-size distribution f(r)
  hpo_search.py            Optuna TPE (+ ASHA rungs) loss-weight HPO
  compute_cost.py          end-to-end latency + FLOPs benchmark
  gpu_benchmark.py         per-GPU / DDP-scaling benchmark suite
```

Naming note: the aggregation the paper calls `tri_weuler` is implemented as
`tri_weuler_self` (the GT-free Euler-consensus rule). The GT-using reference
row `tri_weuler_oracle` of the paper lives in `maps/oracle_eval.py` and cannot
be used in deployment.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # Python 3.9, PyTorch >= 2.2 (CUDA 11.8/12.1)
```

All training and inference budgets quoted in the paper were measured on a
single NVIDIA RTX 3090 (24 GB); a comparable GPU is sufficient.

The analysis drivers under `analysis/` need a few extra packages; install
them with `pip install -r requirements-reproduce.txt` (see
"Reproducing the paper's results" below).

## Smoke test

A CPU-only self-check of the core pipeline: UNetG forward pass, per-axis
reconstruction, parity paste, tri-mean and Euler-weighted GT-free
aggregation, and the cube-level metric suite, all on a synthetic 64×96×96
binary volume with a randomly initialized generator. No checkpoints, data
files, or GPU required; it finishes in seconds.

```bash
python -m maps.smoke_test --device cpu
```

Expected output — six PASS lines and exit code 0 (parameter count and
shapes are exact; metric values are meaningless under random weights and
may vary slightly across library versions):

```
PASS  [1/6] UNetG forward: in_ch=6 base=80 params=24,495,441 out=(2, 1, 96, 96) range=[0.465,0.497]
PASS  [2/6] reconstruct_axis(z): shape=(64, 96, 96), interior [15,49) replaced, boundary preserved
PASS  [3/6] parity_paste: even-z acquired slices restored, odd-z model output retained
PASS  [4/6] tri_mean aggregation: weights=(0.333, 0.333, 0.333)
PASS  [5/6] tri_weuler_self aggregation: weights=(0.000, 1.000, 0.000) euler=(-696.6, 84.6, 93.3)
PASS  [6/6] metrics_from_cube: all values finite (ssim_z=0.655, dphi=0.1516, d_euler=97.5)
SMOKE TEST OK (3.6s, device=cpu)
```

## Data

All eight Micro-CT volumes are public; none are redistributed here.

| Domain | Source | Download |
|---|---|---|
| BB (Bentheimer-like), Bentheimer 2.25 µm, CastleGate, Parker | Digital Rocks Portal, project 317 (Neumann et al., 2020; Lucas-Oliveira et al., 2022) | https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-317 (DOI [10.17612/F4H1-W124](https://doi.org/10.17612/F4H1-W124)) |
| Doddington, Bentheimer30um (3.00 µm), Ketton, Estaillades | Imperial College London pore-scale collection (Raeini et al., 2017; Bijeljic et al., 2013) | https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/ |

The Digital Rocks Portal has migrated to `digitalporousmedia.org`; the old
`digitalrocksportal.org/projects/317` URL now redirects there. Project 317
carries a single project DOI (above); select each sandstone volume (Bentheimer,
Castlegate, Parker, and the Bentheimer-like sample used here as "BB") from the
project file listing. The Imperial College images are per-sample downloads on
the group page and carry no data DOI (cite Raeini et al., 2017). For **Ketton,
use the Imperial College page above**; do not use DOI `10.17612/P7K09D`, which
points to a different (synchrotron waterflood, DRP-202) dataset.

Prepare each volume as a raw float32 binary file, shape 1000³ (`Z, Y, X` order),
values in [0, 1] with the convention `0 = pore`, `1 = solid` (see
`scripts/prepare_data.py`). The BB training split along z is train `[0, 700)`,
validation `[700, 850)`, test `[850, 1000)`; targets within 15 slices of a slab
boundary are excluded (no inter-slab leakage).

## Training

```bash
# Stage 1 — z-axis GAN pretraining (~3 h on one RTX 3090).
# Writes outputs/stage1/stage1_BB_<MMDD_HHMM>/checkpoints/best.pt
# (best EMA-SSIM checkpoint; <MMDD_HHMM> is the launch timestamp).
python train_stage1.py --volume_path data/BB_1000c_f32.bin \
    --max_epochs 200 --seed 2025 --run_name stage1_BB

# Stage 2 — multi-axis fine-tune from the Stage 1 checkpoint (~90 min).
# Writes outputs/stage2/stage2_multiaxis_pareto4_best_physics_<MMDD_HHMM>/
#   checkpoints/best.pt  (best EMA z-SSIM; also best_dphi.pt).
python train_stage2.py \
    --stage1_ckpt outputs/stage1/stage1_BB_<MMDD_HHMM>/checkpoints/best.pt \
    --preset pareto4 --volume_path data/BB_1000c_f32.bin --seed 2025
```

The `pareto4` preset is the physics-balanced loss configuration used for all
reported results (λ_ssim = 0.349, λ_phi = λ_SA = λ_lp = 0.174,
λ_S2 = 0.0177, λ_adv = 0.131).

`train_stage1.py` defaults follow the same recipe (Table S9): lr_G 4.5e-4,
lr_D 1.9e-4, Adam β1 = 0.5, soft-Otsu temperature 10, GAN warmup 27 epochs
with decay factor 0.77.

The architecture-matched baseline `b4` (2D U-Net, strictly L1-only) of the
paper's primary comparison is trained from the same code by zeroing every
non-L1 loss term (all of which are nonzero by default), under the paper's
1 h wall-clock budget (`--max_seconds 3600`) at lr 3e-4:

```bash
python train_stage1.py --volume_path data/BB_1000c_f32.bin \
    --w_ssim 0 --w_grad 0 --w_phi 0 --w_sa 0 --w_s2 0 --w_lpath 0 \
    --lambda_gan_base 0 \
    --lr_G 3e-4 --max_seconds 3600 --max_epochs 1000 \
    --seed 2025 --run_name b4_l1_only
```

## Inference (tri-axis aggregation)

The command below runs as-is with the shipped checkpoint
(`checkpoints/maps_stage2_s1-2025_s2-2025_best_ssim.pt`; see
`checkpoints/README.md`). To use your own training run, substitute
`outputs/stage2/<run>/checkpoints/best.pt`.

```bash
python infer_triaxis.py \
    --checkpoint checkpoints/maps_stage2_s1-2025_s2-2025_best_ssim.pt \
    --volume_path data/BB_1000c_f32.bin \
    --cube_origin 861 372 372 --cube_size 128 256 256 \
    --agg tri_mean --parity
```

`--parity` (default) is the deployment-parity protocol behind the paper's
reported numbers: acquired even-z planes are restored after aggregation, so
only the missing odd-z planes contain model output. With `--metrics`,
metrics are reported on this final deployed volume (not on an odd-slice
mask), unless a script explicitly states otherwise.

## Evaluation and LBM permeability

```bash
# slice-level metric suite (SSIM, dphi, dSA, S2, Euler)
python eval_all_metrics.py \
    --checkpoint checkpoints/maps_stage2_s1-2025_s2-2025_best_ssim.pt \
    --data data/BB_1000c_f32.bin --split test

# LBM permeability of a reconstructed cube (D3Q19 BGK, 5000 steps)
python lbm/d3q19.py --cube_path outputs/recon_cube.bin --cube_size 256 --flow_axis 0

# analytical solver validation (parallel plates, k = W^2/12)
python lbm/d3q19.py --validate --cpu    # drop --cpu to run on GPU
```

The Poiseuille validation needs no data files and runs on CPU in ~20 s
(width 24, 5000 steps). Expected output — lattice permeability within 2.3%
of the analytical parallel-plate value, as quoted in the paper:

```
[VALIDATE] Poiseuille on cpu...
{
  "k_lu": 41.24047416446842,
  "k_analytical_lu": 40.333333333333336,
  "relative_error": 0.022491094986820184,
  ...
}
```

`--cube_path` expects a headerless raw float32 file. To feed it a
reconstruction saved by `infer_triaxis.py --out` (`.npy` format), convert
first:
`python -c "import numpy as np; np.load('recon.npy').astype(np.float32).tofile('outputs/recon_cube.bin')"`.

## Expected primary result

BB, deployment-parity, `tri_mean`, best-SSIM checkpoint, n = 9
(3 Stage-1 × 3 Stage-2 seeds): porosity error
**Δφ = 0.00134 ± 0.00033** (fixed Stage-1 protocol: 0.00122 ± 0.00026),
about 2× lower than the architecture-matched 2D U-Net baseline, at
pixel SSIM 0.935. See the paper (Tables 1, S2) for the full matrix.

## Reproducing the paper's results

This repository provides scripts and configuration recipes used to
reproduce the reported tables and figures. Learned-method rows require the
corresponding checkpoints or retraining. The map below gives, for each
table/figure, the script and a one-line command sketch; `<vol>` is a
prepared 1000³ volume file and `<ckpt>` a trained checkpoint (for MAPS
rows, `checkpoints/maps_stage2_s1-2025_s2-2025_best_ssim.pt` can be used
directly).

Extra packages for the analysis drivers (not needed for MAPS itself):
`pip install -r requirements-reproduce.txt` (pandas, matplotlib, thop,
porespy, taufactor, optuna, einops, tifffile).

**Checkpoints must be trained first** for any row marked "needs training".
Approximate single-RTX-3090 budgets (the paper's own budgets): MAPS
Stage 1 ≈ 3 h + Stage 2 ≈ 90 min per seed; b4 ≈ 1 h; b5 ≈ 3 h;
b5-large ≈ 3 h; SwinUNet ≈ 90 min; I3Net ≈ 90 min; diffusion V1 ≈ 4 h,
V2 ≈ 24 h, V3 = VAE 6 h + DDIM 18 h. The full multi-seed program of the
paper (3 Stage-1 × 3 Stage-2 MAPS seeds + 3 seeds each of b4/b5) is roughly
1.5 GPU-weeks; the evaluation drivers themselves run in minutes to hours
(the LBM campaigns are the slowest: ~10–20 min per 256³ cube per axis).

| Paper artifact | Script | Command sketch | Needs training first? |
|---|---|---|---|
| Table 1 — BB main comparison (MAPS, b4, b5, SwinUNet, I3Net) | `train_stage1/2.py`, `baselines/*`, `infer_triaxis.py`, `analysis/benchmark_eval.py`, `baselines/classical.py`, `baselines/eval_diffusion.py` | train each method (3 seeds); MAPS/b4: `python infer_triaxis.py --checkpoint <ckpt> --volume_path <vol> --agg tri_mean --parity --metrics` per cube or `analysis/aggregation_ablation.py`; b5/b5-large/SwinUNet/I3Net: `python analysis/benchmark_eval.py --model {unet3d,swinunet,i3net} --checkpoints 2025=<ckpt> … --volume_path <vol>`; classical: `python baselines/classical.py --volume_path <vol>` | yes — all learned rows |
| §5.6 convergence parity (b5-large 3.3× / b5 2.6× on Δφ at MAPS' full 4.5 h budget) | `baselines/train_unet3d.py` + `analysis/benchmark_eval.py` | train with `--max_seconds 16200`, then `python analysis/benchmark_eval.py --model unet3d --base 64 --method_label b5_large --checkpoints 2025=<ckpt> --volume_path <vol>` | yes |
| Table 2 — aggregation ablation (tri_mean … tri_weuler_self, oracle row) | `analysis/aggregation_ablation.py` | `python analysis/aggregation_ablation.py --volume_path <vol> --checkpoints 2025=<ckpt> 2026=<ckpt> 2027=<ckpt> --parity` | yes — 3 MAPS seeds |
| Table S16 — multi-cube anisotropy ratio-L1 (incl. the Ketton 71.88 → 1.58 example cube) | `analysis/anisotropy_eval.py` | per (domain, cube): `python analysis/anisotropy_eval.py --volume_path <vol> --domain Ketton --voxel_um 3.0 --cube center --checkpoint <ckpt> --include_gt`; then `--stage aggregate` | yes — 1 MAPS ckpt |
| Table S4 — idealized vs strictly-sequential tri-axis (k=1) | `analysis/sequential_eval.py` (or `infer_triaxis.py --sequential`) | `python analysis/sequential_eval.py --volume_path <vol> --checkpoint <ckpt>` | yes — 1 MAPS seed |
| Sequential anisotropy diagnostic (~10% of the idealized tensor gain retained) | `analysis/anisotropy_eval.py` | rerun the Ketton cubes with `--protocol sequential --cube_layout campaign256`, then `--stage aggregate` | yes |
| §5.1/§5.5 Wilcoxon significance (exact, p = 0.0039 at n = 9) | `analysis/wilcoxon_stats.py` | `python analysis/wilcoxon_stats.py --csv maps=<agg_ablation csv> b4=<benchmark csv> --pairs maps:tri_mean=b4:tri_mean maps:z_only=maps:tri_mean` | uses the CSVs above |
| Table 3 / Table S10 — multi-seed test-only LBM (k_zz, k_trace; per-axis) | `analysis/lbm_multiseed_eval.py` + `analysis/lbm_multiseed_aggregate.py` | `python analysis/lbm_multiseed_eval.py --volume_path <vol> --domain BB --voxel_um 2.25 --seeds 2025 2026 2027 --maps_ckpts … --b4_ckpts … --b5_ckpts … --out out/lbm; python analysis/lbm_multiseed_aggregate.py --root out/lbm` (repeat per domain) | yes — MAPS/b4/b5 × 3 seeds |
| Table S15 — 8-domain k=1 LBM trace error (GT, linear, tri-linear, MAPS) | `analysis/lbm_8domain_eval.py` | `python analysis/lbm_8domain_eval.py --stage all --volume_path <vol> --domain <name> --voxel_um <um> --checkpoint <ckpt>` per domain, then `--stage aggregate` | yes — 1 MAPS ckpt (zero-shot on 7 domains) |
| Table 4 / Fig. 10 Pareto — compute cost (latency, FLOPs) | `analysis/compute_cost.py` | `python analysis/compute_cost.py --mode latency --cubes 256 --planes1000` and `--mode flops` (needs `thop`) | no (random weights; latency is weight-independent) |
| Fig. 6 — k-sweep crossover (BB + zero-shot domains) | `analysis/ksweep_eval.py` | `python analysis/ksweep_eval.py --volume_path <vol> --ckpt_maps <ckpt> --ckpt_b4 <ckpt> --k_values 1 2 3 5 7` per domain | yes — MAPS + b4 |
| Fig. 11 + Fig. S9 — failure map + predicted-vs-actual | `analysis/failure_regression.py` | `python analysis/failure_regression.py --long_csv <fit csv> --ood_csv <held-out csv> --ood_porosity Bentheimer30um=0.2168 …` (fit CSV from `aggregation_ablation.py` on BB/Bentheimer/Ketton; held-out CSV from the same driver on the 5 remaining domains; the PvA table is generated by the frozen fit) | yes — 3 MAPS seeds, 8 domains evaluated |
| Supplementary Note E contiguous-gap probe (learned 1.6% vs linear 10.2% LBM-trace error at 50% retained) | `analysis/slab_stitching_probe.py` | `python analysis/slab_stitching_probe.py --volume_path <vol> --checkpoint <ckpt> --lbm --lbm_gaps 8 16` | yes — 1 MAPS ckpt |
| §5.4 per-k retrain vs joint multi-k | `baselines/train_multik.py` + `analysis/ksweep_eval.py` | train the joint model (`python baselines/train_multik.py --volume_path <vol> --max_seconds 3600`) and per-k models (`train_stage1.py --offsets <scaled>`), then k-sweep both | yes |
| Table S2 — Stage1 × Stage2 3×3 matrix (reported Δφ = 0.00134 ± 0.00033) | `analysis/parity_matrix_eval.py` | `python analysis/parity_matrix_eval.py --volume_path <vol> --manifest matrix.tsv` (9 checkpoints) | yes — 9 independent MAPS trainings (~9 × 4.5 h) |
| Table S3 — k=1 deployment-parity deep-morphology vs linear | `analysis/morphology3d_eval.py` | `python analysis/morphology3d_eval.py --volume_path <vol> --checkpoint <ckpt> --protocol parity --offsets -5 -3 -1 1 3 5 --methods maps linear_k1_tri` | yes — MAPS seeds |
| Tables S5–S7 — 3D morphology, BB + cross-domain + anisotropy | `analysis/morphology3d_eval.py` | `python analysis/morphology3d_eval.py --volume_path <vol> --domain <name> --checkpoint <ckpt> --methods maps classical_b1 classical_b2 classical_b3` per domain/seed | yes — MAPS (+ b4 for baseline rows) |
| Tables S11–S12 / DDP-scaling figure — GPU scaling & GPU-tier | `analysis/gpu_benchmark.py` | `torchrun --nproc_per_node=N analysis/gpu_benchmark.py --exp exp5 --gpu_name RTX3090_Nx` for N = 1…8; `--exp all` per GPU model | no |
| Diffusion DDIM-step sweep figure | `baselines/eval_diffusion.py` | run once per `--ddim_steps` in {10, 20, 50, 100, 200, 500, 1000} | yes — V3 VAE + DDIM |
| §5.7 V1/V2 diffusion results (~100× Δφ gap) | `baselines/eval_diffusion.py` | `python baselines/eval_diffusion.py --variant v1 --ddim_ckpt <ckpt> --volume_path <vol>` (train via `baselines/diffusion_v{1,2}_pixel.py`) | yes |
| SwinUNet HP sweep (archived; not in compiled supplement) | `baselines/swinunet.py` | re-train with `--base/--num_heads/--window_size` grid | yes |
| Table S8 / Fig. S2 — channel/offset ablation (common-target protocol) | `train_stage1.py` + `analysis/channel_ablation_eval.py` | train each config (recipe in the driver docstring), then `python analysis/channel_ablation_eval.py --volume_path <vol> --runs A6_default=<dir> …` | yes — 1 run per config |
| Fig. S3 — pore-size distributions f(r) | `analysis/pore_size_distribution.py` | `python analysis/pore_size_distribution.py --volume_path <vol> --inputs tri_mean=<recon.npy> --fig <out.png>` (recon cubes from `infer_triaxis.py --out`) | yes — recon cubes |
| Fig. S5 — loss-weight HPO (TPE + ASHA rungs 12/25/50/200) | `analysis/hpo_search.py` | `python analysis/hpo_search.py --volume_path <vol> --mode pareto --n_trials 15` per worker (needs `optuna`) | no (search trains its own trials) |
| Fig. S6 — binarization-threshold sweep φ(τ) | `analysis/threshold_sweep.py` | `python analysis/threshold_sweep.py --volume_path <vol> --domain BB --checkpoint <ckpt>` per domain | yes — 1 MAPS ckpt |
| LBM velocity/streamline field | `lbm/d3q19.py` | add `--save_velocity <out.npy>` to any cube run (the streamline rendering itself is presentation-only) | uses recon cubes |

Reproduction notes:
* Only one trained checkpoint is shipped
  (`checkpoints/maps_stage2_s1-2025_s2-2025_best_ssim.pt`, the fixed-Stage-1
  MAPS model); all other learned-method rows require retraining. With the
  seeds fixed above the training runs are deterministic up to cuDNN/GPU
  nondeterminism, so expect the reported metrics to match within the
  reported ±std, not bit-exactly.
* The seeded cube origins are self-verifying: `analysis/benchmark_cubes.py`
  reproduces the exact origins used for all reported evaluations
  ((861,372,372)/(859,739,738)/(864,622,570) for the 128×256×256 cubes and
  (744,372,372)… for the 256³ LBM cubes on any 1000³ volume).
* `tri_weuler_oracle` (Table 2 reference row) is the only ground-truth-using
  evaluation and lives exclusively in `maps/oracle_eval.py`.
* The I3Net baseline (`baselines/i3net.py`) is a thin adapter around the
  official upstream implementation
  (https://github.com/eeeric-code/I3Net, arXiv:2405.02857 / IEEE TMI 2024),
  which must be cloned separately into `_external/I3Net_official` (or
  pointed to via `I3NET_OFFICIAL_DIR`). No upstream code is redistributed
  here; the upstream repository carries no license file, so users must
  verify the upstream authors' terms before use. Consequently the I3Net
  row of Table 1 is the only result not reproducible from this repository
  alone: it requires the third-party upstream clone and is provided for
  reference comparison. Every other row (MAPS, b4, b5/b5-large, SwinUNet,
  classical, V1/V2/V3 diffusion, LBM, GPU scaling) is fully reproducible
  from the code here.

## Citation

```bibtex
@article{Baek2026MAPS,
  title   = {MAPS: Multi-Axis 2.5D Slice Interpolation Preserving Petrophysical
             Properties in Segmented Digital Rock},
  author  = {Baek, Seungwon and Lee, Juan and Joo, Kyonghee and Jo, Honggeun and Jo, Suryeom and Cho, Yongchae},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {submitted}
}
```

## License

This repository is released under the MIT License. See LICENSE.
