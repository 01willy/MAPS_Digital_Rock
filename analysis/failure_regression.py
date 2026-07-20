#!/usr/bin/env python3
"""Failure-map regression -- dphi ~ porosity + A + interaction
(Fig. 11 of the paper + supplement Fig. S9: the deploy/no-deploy failure map
and the predicted-vs-actual regression diagnostic).

NOTE on A (GT-USING at compute time): A = 1 - mean(xz_ssim, yz_ssim) is the
cross-plane SSIM of the RECONSTRUCTION against the dense GROUND TRUTH, so it
requires the dense reference. At deployment it is computed on the small dense
CALIBRATION region (Note H, step 2), not on the target being synthesized. It
indexes reconstruction difficulty, NOT physical rock anisotropy (the most
anisotropic rock, Ketton, has the smallest A).

Inputs:
  --long_csv  long-format results of the MAPS tri_mean cross-domain
              evaluations on the three FITTING domains (BB, Bentheimer,
              Ketton; n=27 = 3 cubes x 3 seeds x 3 domains). Produced by
              analysis/aggregation_ablation.py runs (columns: domain,
              seed, cube, agg, dphi, xz_ssim, yz_ssim; rows with
              agg == tri_mean are used). The OLS fit and its R^2 are
              computed here.
  --ood_csv   (optional, one or more) long CSVs of the same evaluation on
              the five HELD-OUT domains (Bentheimer30um, CastleGate,
              Doddington, Estaillades, Parker). When given, the FROZEN fit
              is applied to every (fit + held-out) cube and the
              predicted_vs_actual table is generated in --out_dir; the two
              figures are then drawn from it. Held-out domains need a
              measured GT porosity, either directly (--ood_porosity
              Domain=phi) or from the full volume (--ood_volume
              Domain=path, porosity = 1 - mean(vol > 0.5)).
  --pva_csv   (optional) pre-computed predicted_vs_actual table (columns:
              domain, split, porosity, anisotropy_proxy, pred_dphi,
              actual_dphi). Overrides --ood_csv generation.

A = 1 - mean(xz_ssim, yz_ssim) is the cross-plane reconstruction gap (see
the NOTE above; GT-using, calibration-region at deploy). The per-domain porosity values used in the fit are
approximate dataset constants (BB 0.20, Bentheimer 0.24, Ketton 0.15), as
stated in the paper.

Usage:
  python analysis/failure_regression.py \\
      --long_csv outputs/analysis/agg_ablation_fitdomains.csv \\
      --ood_csv outputs/analysis/agg_ablation_ood.csv \\
      --ood_porosity Bentheimer30um=0.2168 CastleGate=0.2487 \\
                     Doddington=0.1958 Estaillades=0.1273 Parker=0.1365 \\
      --out_dir outputs/analysis/failure --fig_dir outputs/figures
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Journal figure style: serif + STIX mathtext, ~8 pt labels at single-column
# width, thin spines, muted colorblind-safe colors.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# Approximate per-domain porosity constants used by the paper's fit
POROSITY = {'BB': 0.20, 'Bentheimer': 0.24, 'Ketton': 0.15}

FIT = ['BB', 'Bentheimer', 'Ketton']
VALID = ['Bentheimer30um', 'CastleGate', 'Doddington']
BREACH = ['Estaillades', 'Parker']
PHI_LO, PHI_HI, A_HI = 0.15, 0.25, 0.15   # fitted property envelope box


def main():
    ap = argparse.ArgumentParser(description='Failure-map regression '
                                             '(Figs. 12-13)')
    ap.add_argument('--long_csv', type=str, required=True)
    ap.add_argument('--ood_csv', type=str, nargs='+', default=None,
                    help='held-out-domain long CSVs; generates the '
                         'predicted_vs_actual table with the frozen fit')
    ap.add_argument('--ood_porosity', type=str, nargs='+', default=None,
                    help='Domain=phi pairs (measured full-volume GT '
                         'porosity of held-out domains)')
    ap.add_argument('--ood_volume', type=str, nargs='+', default=None,
                    help='Domain=path pairs; porosity measured as '
                         '1 - mean(volume > 0.5)')
    ap.add_argument('--volume_shape', nargs=3, type=int,
                    default=[1000, 1000, 1000],
                    help='shape of the --ood_volume files')
    ap.add_argument('--pva_csv', type=str, default=None,
                    help='pre-computed predicted_vs_actual table '
                         '(overrides --ood_csv generation)')
    ap.add_argument('--method_name', type=str, default='maps',
                    help='method column value to select (if the long CSV '
                         'contains multiple methods)')
    ap.add_argument('--out_dir', type=str, default='outputs/analysis/failure')
    ap.add_argument('--fig_dir', type=str, default='outputs/figures')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    def select_trimean(frame):
        s = frame[frame['agg'] == 'tri_mean'].copy()
        if 'method' in s.columns and args.method_name:
            s = s[s['method'].isin([args.method_name, 'ours', 'MAPS'])].copy()
        if 'criterion' in s.columns:
            s = s[s['criterion'] == 'best_ssim'].copy()
        # Anisotropy proxy: cross-plane SSIM gap
        s['anisotropy_proxy'] = (1.0 - s[['xz_ssim',
                                          'yz_ssim']].mean(axis=1)).abs()
        return s

    df = pd.read_csv(args.long_csv)
    sub = select_trimean(df)
    sub['porosity'] = sub['domain'].map(POROSITY)
    sub = sub.dropna(subset=['porosity'])

    # OLS regression with intercept: dphi ~ porosity + aniso + interaction
    from numpy.linalg import lstsq
    X = np.column_stack([
        np.ones(len(sub)),
        sub['porosity'].values,
        sub['anisotropy_proxy'].values,
        (sub['porosity'].values * sub['anisotropy_proxy'].values),
    ])
    y = sub['dphi'].values
    coef, residuals, rank, sv = lstsq(X, y, rcond=None)
    pred = X @ coef
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2_fit = 1 - sse / sst

    coef_names = ['intercept', 'porosity', 'anisotropy_proxy',
                  'porosity:anisotropy']
    out_csv = out_dir / 'failure_regression.csv'
    pd.DataFrame({'term': coef_names, 'coefficient': coef}).to_csv(
        out_csv, index=False)

    md = [
        '# Failure regression -- dphi ~ porosity + anisotropy_proxy '
        '+ interaction',
        '',
        f'_OLS on MAPS tri_mean fitting-domain evaluations (n={len(sub)})._',
        '_anisotropy_proxy = 1 - mean(xz_ssim, yz_ssim)._',
        '_porosity = approximate per-domain constants '
        '(BB 0.20, Bentheimer 0.24, Ketton 0.15)._',
        '',
        f'**Model R^2** = {r2_fit:.4f}', '',
        '| Term | Coefficient |',
        '|---|---:|',
    ]
    for name, c in zip(coef_names, coef):
        md.append(f'| `{name}` | {c:+.5f} |')
    md.append('')
    out_md = out_dir / 'failure_regression.md'
    out_md.write_text('\n'.join(md))
    print(f'[saved] {out_csv}')
    print(f'[saved] {out_md}  (fit R^2 = {r2_fit:.4f})')

    # ------------------------------------------- predicted-vs-actual table
    # The FROZEN fit applied to every domain at cube level. Either read a
    # pre-computed table (--pva_csv) or generate it from held-out-domain
    # long CSVs (--ood_csv + measured GT porosities).
    pva = None
    if args.pva_csv and Path(args.pva_csv).exists():
        pva = pd.read_csv(args.pva_csv)
    elif args.ood_csv:
        ood_porosity = {}
        for spec in (args.ood_porosity or []):
            dom, phi = spec.split('=')
            ood_porosity[dom] = float(phi)
        for spec in (args.ood_volume or []):
            dom, path = spec.split('=', 1)
            vol = np.memmap(path, dtype=np.float32, mode='r',
                            shape=tuple(args.volume_shape))
            solid = 0.0
            for z in range(0, vol.shape[0], 100):  # chunked full-volume mean
                solid += float((np.asarray(vol[z:z + 100]) > 0.5).mean()
                               * min(100, vol.shape[0] - z))
            ood_porosity[dom] = 1.0 - solid / vol.shape[0]
            print(f'[porosity] {dom}: {ood_porosity[dom]:.4f} '
                  f'(measured from {path})')
        ood = pd.concat([pd.read_csv(p) for p in args.ood_csv],
                        ignore_index=True)
        ood = select_trimean(ood)
        ood['porosity'] = ood['domain'].map(ood_porosity)
        missing = sorted(ood[ood['porosity'].isna()]['domain'].unique())
        if missing:
            raise SystemExit(f'missing GT porosity for held-out domains '
                             f'{missing}; pass --ood_porosity or '
                             f'--ood_volume')
        X_ood = np.column_stack([
            np.ones(len(ood)), ood['porosity'].values,
            ood['anisotropy_proxy'].values,
            ood['porosity'].values * ood['anisotropy_proxy'].values])
        ood['pred_dphi'] = X_ood @ coef
        fit_part = sub.copy()
        fit_part['pred_dphi'] = pred
        fit_part['split'] = 'in_fit'
        ood['split'] = 'OOD'
        cols = ['domain', 'split', 'seed', 'cube', 'porosity',
                'anisotropy_proxy', 'dphi', 'pred_dphi']
        cols_fit = [c for c in cols if c in fit_part.columns]
        cols_ood = [c for c in cols if c in ood.columns]
        pva = pd.concat([fit_part[cols_fit], ood[cols_ood]],
                        ignore_index=True)
        pva = pva.rename(columns={'dphi': 'actual_dphi'})
        pva['residual'] = pva['actual_dphi'] - pva['pred_dphi']
        pva_path = out_dir / 'predicted_vs_actual.csv'
        pva.to_csv(pva_path, index=False)
        print(f'[saved] {pva_path} ({len(pva)} rows: '
              f'{int((pva["split"] == "in_fit").sum())} in-fit + '
              f'{int((pva["split"] == "OOD").sum())} held-out)')

    # ---------------------------------------------------------- figures
    # Both figures are drawn from the predicted-vs-actual table (the frozen
    # regression applied to every domain at cube level), which makes them
    # internally consistent and covers all eight domains:
    #   fit        : BB, Bentheimer, Ketton
    #   validation : Bentheimer30um, CastleGate, Doddington (in-envelope)
    #   breaching  : Estaillades, Parker (out-of-envelope)
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    if pva is None:
        print('[skip figures] no predicted-vs-actual table '
              '(--pva_csv or --ood_csv)')
        return

    def group(d):
        return ('fit' if d in FIT
                else ('validation' if d in VALID else 'breaching'))
    pva['group'] = pva['domain'].map(group)

    # goodness-of-fit (frozen regression, actual vs predicted)
    def r2(a, p):
        a = np.asarray(a)
        p = np.asarray(p)
        ss_res = float(np.sum((a - p) ** 2))
        ss_tot = float(np.sum((a - a.mean()) ** 2))
        return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    infit = pva[pva['split'] == 'in_fit']
    oodall = pva[pva['split'] != 'in_fit']
    r2_in = r2(infit['actual_dphi'], infit['pred_dphi'])
    r2_ood = r2(oodall['actual_dphi'], oodall['pred_dphi'])

    dmean = pva.groupby('domain').agg(
        A=('anisotropy_proxy', 'mean'), phi=('porosity', 'mean'),
        dphi=('actual_dphi', 'mean'), group=('group', 'first')).reset_index()
    vmin = float(pva['actual_dphi'].min())
    vmax = float(pva['actual_dphi'].max())
    MARK = {'fit': 'o', 'validation': 's', 'breaching': '^'}
    SIZE = {'fit': 42, 'validation': 42, 'breaching': 52}

    # ===== MAIN figure: deploy/no-deploy boundary in (A, phi) space =====
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    xmin, xmax = 0.0, float(dmean['A'].max()) * 1.13
    ymin = float(dmean['phi'].min()) - 0.022
    ymax = float(dmean['phi'].max()) + 0.030
    ax.add_patch(Rectangle((xmin, PHI_LO), A_HI - xmin, PHI_HI - PHI_LO,
                           facecolor='0.55', alpha=0.10,
                           edgecolor='0.35', lw=0.8, ls=(0, (4, 2)),
                           zorder=1))
    ax.text(0.004, PHI_HI - 0.004, 'zero-shot envelope', ha='left', va='top',
            fontsize=7, color='0.30', style='italic', zorder=6)
    sc = None
    for grp, g in dmean.groupby('group'):
        sc = ax.scatter(g['A'], g['phi'], c=g['dphi'], cmap='viridis',
                        vmin=vmin, vmax=vmax, marker=MARK[grp], s=SIZE[grp],
                        edgecolor='0.15', linewidth=0.6, zorder=5)
    lab_off = {'BB': (6, -2.5, 'left'), 'Bentheimer': (-6, -1.5, 'right'),
               'Ketton': (6, 4, 'left'), 'Bentheimer30um': (-6, 1.5, 'right'),
               'CastleGate': (6, 1.5, 'left'), 'Doddington': (0, -10, 'center'),
               'Estaillades': (6, -2.5, 'left'), 'Parker': (0, 6, 'center')}
    for _, r in dmean.iterrows():
        dx, dy, ha = lab_off.get(r['domain'], (6, 6, 'left'))
        ax.annotate(r['domain'], xy=(r['A'], r['phi']), xytext=(dx, dy),
                    textcoords='offset points', fontsize=7, color='0.15',
                    ha=ha, va='center', zorder=6)
    ax.set_xlabel('Anisotropy proxy $A = 1-\\overline{\\mathrm{SSIM}}_{xz,yz}$')
    ax.set_ylabel('Porosity $\\phi$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label('$\\Delta\\phi$ $(\\times 10^{-3})$', fontsize=7.5)
    cb.set_ticks([0.002, 0.004, 0.006, 0.008])
    cb.set_ticklabels(['2', '4', '6', '8'])
    cb.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    legend_el = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.75',
               markeredgecolor='0.15', markeredgewidth=0.6, markersize=5.5,
               label='fit domains'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='0.75',
               markeredgecolor='0.15', markeredgewidth=0.6, markersize=5.5,
               label='held-out, in-envelope'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='0.75',
               markeredgecolor='0.15', markeredgewidth=0.6, markersize=6,
               label='held-out, out-of-envelope')]
    ax.legend(handles=legend_el, loc='lower left', frameon=False,
              handletextpad=0.3, borderaxespad=0.3, labelspacing=0.35)
    fig.tight_layout(pad=0.3)
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'fig_failure_map_regression.{ext}')
    plt.close(fig)
    print(f'[saved] {fig_dir}/fig_failure_map_regression.{{png,pdf}}')

    # ===== SUPPLEMENT figure: predicted vs actual =====
    fig2, ax2 = plt.subplots(figsize=(3.3, 3.2))
    GCOL = {'fit': '#808080', 'validation': '#0072B2', 'breaching': '#D55E00'}
    GLAB = {'fit': 'fit (in-sample)', 'validation': 'zero-shot, in-envelope',
            'breaching': 'zero-shot, out-of-envelope'}
    lim_hi = float(max(pva['actual_dphi'].max(),
                       pva['pred_dphi'].max())) * 1.08
    ax2.plot([0, lim_hi], [0, lim_hi], ls='--', color='0.45', lw=0.8,
             zorder=1, label='$y=x$')
    for grp in ['fit', 'validation', 'breaching']:
        g = pva[pva['group'] == grp]
        ax2.scatter(g['pred_dphi'], g['actual_dphi'], c=GCOL[grp],
                    marker=MARK[grp], s=24 if grp == 'breaching' else 20,
                    edgecolor='0.15', linewidth=0.4, alpha=0.85, zorder=4,
                    label=GLAB[grp])
    for d in BREACH:
        g = pva[pva['domain'] == d]
        if len(g) == 0:
            continue
        ax2.annotate(d, xy=(float(g['pred_dphi'].max()),
                            float(g['actual_dphi'].mean())),
                     xytext=(6, 0), textcoords='offset points', fontsize=7,
                     color='#D55E00', va='center', ha='left')
    ax2.set_xlabel('Predicted $\\Delta\\phi$ $(\\times 10^{-3})$, '
                   'frozen regression')
    ax2.set_ylabel('Actual $\\Delta\\phi$ $(\\times 10^{-3})$')
    ax2.set_xlim(0, lim_hi)
    ax2.set_ylim(0, lim_hi)
    tick_v = [0.000, 0.002, 0.004, 0.006, 0.008]
    tick_l = ['0', '2', '4', '6', '8']
    ax2.set_xticks(tick_v)
    ax2.set_xticklabels(tick_l)
    ax2.set_yticks(tick_v)
    ax2.set_yticklabels(tick_l)
    ax2.text(0.97, 0.36,
             f'in-sample $R^2={r2_in:.2f}$\nzero-shot $R^2={r2_ood:.2f}$',
             transform=ax2.transAxes, va='bottom', ha='right', fontsize=7.5,
             linespacing=1.4)
    ax2.legend(loc='lower right', frameon=False, handletextpad=0.3,
               borderaxespad=0.3, labelspacing=0.35)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_aspect('equal', adjustable='box')
    fig2.tight_layout(pad=0.3)
    for ext in ('png', 'pdf'):
        fig2.savefig(fig_dir / f'fig_failure_predicted_vs_actual.{ext}')
    plt.close(fig2)
    print(f'[saved] {fig_dir}/fig_failure_predicted_vs_actual.{{png,pdf}}  '
          f'(R2_in={r2_in:.3f}, R2_ood={r2_ood:.3f})')


if __name__ == '__main__':
    main()
