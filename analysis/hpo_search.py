#!/usr/bin/env python3
"""
Loss-weight hyperparameter optimization (Fig. S5).

Optuna TPE search over the 10 training hyperparameters of the paper
(lr_G, lr_D, beta1, w_ssim, morph_scale, w_s2, lambda_gan, gan_warmup,
lambda_decay, soft_temperature), with per-trial normalized objectives
against the linear-baseline morphology errors.

Two modes:
  pareto (default) -- the multi-objective (SSIM up, dphi_norm down,
      dsa_norm down) TPE search that produced the paper's Pareto presets
      (pareto0 / pareto4 / pareto5 in train_stage2.py; the paper's
      pipeline uses pareto4). Optuna does not support pruning for
      multi-objective studies, so trials run the fixed proxy budget
      (--trial_epochs, default 50) with a divergence early stop.
  asha -- single-objective (validation SSIM) TPE with ASHA
      successive-halving (`optuna.pruners.SuccessiveHalvingPruner`),
      reporting at the paper's budget rungs 12 / 25 / 50 / 200 epochs
      (Fig. S5; pruned trials appear faded there).

Requires `pip install optuna` (kept out of requirements.txt; the import
fails with a clear message).

Parallel workers share a journal file:
  CUDA_VISIBLE_DEVICES=0 python analysis/hpo_search.py \\
      --volume_path data/BB_1000c_f32.bin --mode pareto \\
      --n_trials 15 --journal outputs/hpo/journal.log --study hpo_maps &
  CUDA_VISIBLE_DEVICES=1 python analysis/hpo_search.py ... (same journal)
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maps.models import UNetG, PatchD, EMA  # noqa: E402
from maps.losses import CombinedLoss, ssim_value  # noqa: E402
from maps.data import load_volume, compute_splits, create_dataloaders  # noqa: E402
from maps.metrics import porosity_hard, surface_area_hard  # noqa: E402

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import SuccessiveHalvingPruner
except ImportError:  # checked again in main() with a clear error
    optuna = None

# Linear-baseline (k=3) validation errors used to normalize the
# morphology objectives, as in the paper's search.
BASELINE_DPHI = 0.0718
BASELINE_DSA = 0.0068

ASHA_RUNGS = [12, 25, 50, 200]  # Fig. S5 budget rungs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def suggest_params(trial):
    p = {}
    p['lr_G'] = trial.suggest_float('lr_G', 1e-4, 5e-4, log=True)
    p['lr_D'] = trial.suggest_float('lr_D', 1e-4, 5e-4, log=True)
    p['beta1'] = trial.suggest_categorical('beta1', [0.5, 0.7, 0.9])
    p['w_ssim'] = trial.suggest_float('w_ssim', 0.1, 0.5)
    p['morph_scale'] = trial.suggest_float('morph_scale', 0.05, 0.3)
    p['w_s2'] = trial.suggest_float('w_s2', 0.0, 0.1)
    p['lambda_gan'] = trial.suggest_float('lambda_gan', 0.02, 0.15)
    p['gan_warmup'] = trial.suggest_int('gan_warmup', 10, 40)
    p['lambda_decay'] = trial.suggest_float('lambda_decay', 0.5, 1.0)
    p['soft_temperature'] = trial.suggest_categorical(
        'soft_temperature', [10, 20, 50])
    return p


def train_trial(trial, p, device, vol, splits, max_epochs, report_rungs):
    """Train one HPO configuration; returns (best_ssim, dphi, dsa) at the
    best-SSIM epoch. When report_rungs is set (ASHA mode), reports the
    validation SSIM at each rung and honors pruning."""
    set_seed(2025)
    loss_cfg = {
        'w_l1': 1.0, 'w_ssim': p['w_ssim'], 'w_grad': 0.0,
        'w_phi': p['morph_scale'], 'w_sa': p['morph_scale'],
        'w_s2': p['w_s2'], 'w_lpath': p['morph_scale'],
        'soft_temperature': p['soft_temperature'], 'gan_mode': 'hinge',
    }
    cfg_data = {'in_ch': 6, 'patch_size': 256, 'batch_size': 4,
                'num_workers': 2}
    train_loader, val_loader, _ = create_dataloaders(vol, splits, cfg_data,
                                                     axis='z')
    G = UNetG(in_ch=6, base=80).to(device)
    D = PatchD(in_ch=7, base=64).to(device)
    ema = EMA(G, decay=0.999)
    opt_G = torch.optim.Adam(G.parameters(), lr=p['lr_G'],
                             betas=(p['beta1'], 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=p['lr_D'],
                             betas=(p['beta1'], 0.999))
    scaler = GradScaler(enabled=True)
    criterion = CombinedLoss(loss_cfg)

    best = (-1.0, 1.0, 1.0)
    try:
        for epoch in range(1, max_epochs + 1):
            G.train()
            D.train()
            if epoch < p['gan_warmup']:
                lam = p['lambda_gan'] * epoch / p['gan_warmup']
            else:
                prog = ((epoch - p['gan_warmup'])
                        / max(1, max_epochs - p['gan_warmup']))
                lam = max(p['lambda_gan'] * (1.0 - p['lambda_decay'] * prog),
                          p['lambda_gan'] * 0.1)
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt_D.zero_grad(set_to_none=True)
                with autocast():
                    d_loss, _ = criterion.compute_D_loss(D, x, y,
                                                         G(x).detach())
                scaler.scale(d_loss).backward()
                scaler.step(opt_D)
                opt_G.zero_grad(set_to_none=True)
                with autocast():
                    g_loss, _ = criterion.compute_G_loss(G, D, x, y, lam)
                scaler.scale(g_loss).backward()
                scaler.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                scaler.step(opt_G)
                scaler.update()
                ema.update()

            # EMA validation
            ema.store()
            ema.apply()
            G.eval()
            s = dphi = dsa = 0.0
            n = 0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                for xv, yv in val_loader:
                    xv = xv.to(device).float()
                    yv = yv.to(device).float()
                    pred = G(xv)
                    s += ssim_value(pred, yv)
                    dphi += float((porosity_hard(pred)
                                   - porosity_hard(yv)).abs().mean())
                    dsa += float((surface_area_hard(pred)
                                  - surface_area_hard(yv)).abs().mean())
                    n += 1
            ema.restore()
            val_ssim = s / max(n, 1)
            if val_ssim > best[0]:
                best = (val_ssim, dphi / max(n, 1), dsa / max(n, 1))

            if report_rungs and epoch in report_rungs:
                trial.report(val_ssim, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if not report_rungs and epoch >= 20 and val_ssim < 0.75:
                break  # diverged; save the budget
    finally:
        del G, D, ema, opt_G, opt_D, scaler, criterion
        torch.cuda.empty_cache()
    return best


def main():
    ap = argparse.ArgumentParser(
        description='Loss-weight HPO: Optuna TPE (+ ASHA rungs), Fig. S5')
    ap.add_argument('--volume_path', required=True)
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000])
    ap.add_argument('--mode', default='pareto', choices=['pareto', 'asha'])
    ap.add_argument('--n_trials', type=int, default=15,
                    help='trials per worker (paper total: 60)')
    ap.add_argument('--trial_epochs', type=int, default=50,
                    help='proxy budget per trial in pareto mode')
    ap.add_argument('--journal', default='outputs/hpo/journal.log',
                    help='JournalStorage file (parallel-worker safe)')
    ap.add_argument('--study', default='hpo_maps')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    if optuna is None:
        raise SystemExit(
            'analysis/hpo_search.py requires optuna (not part of the core '
            'requirements): pip install optuna')

    device = torch.device(f'cuda:{args.gpu}')
    vol = load_volume(args.volume_path, tuple(args.volume_shape))
    splits = compute_splits(vol.shape[0])

    journal_path = Path(args.journal)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    from optuna.storages import JournalStorage, JournalFileStorage
    storage = JournalStorage(JournalFileStorage(str(journal_path)))

    if args.mode == 'pareto':
        study = optuna.create_study(
            study_name=args.study, storage=storage,
            directions=['maximize', 'minimize', 'minimize'],
            sampler=TPESampler(seed=None, multivariate=True),
            load_if_exists=True)

        def objective(trial):
            p = suggest_params(trial)
            ssim, dphi, dsa = train_trial(trial, p, device, vol, splits,
                                          args.trial_epochs,
                                          report_rungs=None)
            return ssim, dphi / BASELINE_DPHI, dsa / BASELINE_DSA

        study.optimize(objective, n_trials=args.n_trials)
        print(f'\n[HPO] total trials: {len(study.trials)}; Pareto front:')
        for i, t in enumerate(study.best_trials):
            print(f'  [{i}] SSIM={t.values[0]:.4f} '
                  f'dphi_norm={t.values[1]:.4f} dsa_norm={t.values[2]:.4f}')
            print(f'      params: {t.params}')
    else:
        # ASHA successive halving at the Fig. S5 rungs 12/25/50/200
        pruner = SuccessiveHalvingPruner(min_resource=ASHA_RUNGS[0],
                                         reduction_factor=2)
        study = optuna.create_study(
            study_name=args.study, storage=storage, direction='maximize',
            sampler=TPESampler(seed=None, multivariate=True),
            pruner=pruner, load_if_exists=True)

        def objective(trial):
            p = suggest_params(trial)
            ssim, _dphi, _dsa = train_trial(trial, p, device, vol, splits,
                                            ASHA_RUNGS[-1],
                                            report_rungs=set(ASHA_RUNGS))
            return ssim

        study.optimize(objective, n_trials=args.n_trials)
        print(f'\n[HPO] total trials: {len(study.trials)}')
        print(f'  best SSIM={study.best_value:.4f}  '
              f'params: {study.best_params}')


if __name__ == '__main__':
    main()
