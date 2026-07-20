#!/usr/bin/env python3
"""
Paired Wilcoxon signed-rank tests behind the paper's significance claims
(exact two-sided p-values; p = 0.0039 is the minimum attainable at n = 9).

Reads one or more long-format result CSVs (as emitted by
`analysis/aggregation_ablation.py` and `analysis/benchmark_eval.py`) and,
for each requested pair of (method, aggregation) arms, pairs the rows by
(seed, cube) within each domain and runs scipy's exact small-sample
Wilcoxon test (zero_method='wilcox'); the normal-approximation p-value is
reported alongside for reference (at n = 9 the approximation is
anti-conservative).

CSV requirements: columns seed, cube, agg and the metric columns
(dphi / dsa / ssim). A `domain` column is used when present; a `method`
column is used when present, otherwise assign a method label per file
with the `label=path` form of --csv.

Paper pairs (Sections 5.1 and 5.5; protocol Section 4.8): maps vs b4 at
tri_mean on dphi/dsa/ssim, the
z_only -> tri_mean aggregation effect on dphi, and maps vs the no-GAN
ablation on dphi/dsa.

Usage:
  python analysis/wilcoxon_stats.py \\
      --csv maps=outputs/analysis/agg_ablation_BB.csv \\
            b4=outputs/analysis/benchmark_b4.csv \\
      --pairs maps:tri_mean=b4:tri_mean maps:z_only=maps:tri_mean \\
      --metrics dphi dsa ssim \\
      --out_csv outputs/analysis/wilcoxon_pairwise.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def load_rows(specs):
    rows = []
    for spec in specs:
        if '=' in spec and not Path(spec).exists():
            label, path = spec.split('=', 1)
        else:
            label, path = None, spec
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                r = dict(r)
                if label is not None:
                    r['method'] = label
                r.setdefault('method', 'unknown')
                r.setdefault('domain', 'all')
                rows.append(r)
    return rows


def parse_pair(spec):
    """'methodA:aggA=methodB:aggB' -> ((mA, aA), (mB, aB))."""
    a, b = spec.split('=')
    ma, aa = a.split(':')
    mb, ab = b.split(':')
    return (ma, aa), (mb, ab)


def paired_values(rows, arm, metric, domain, protocol):
    method, agg = arm
    out = {}
    for r in rows:
        if r['method'] != method or r['agg'] != agg:
            continue
        if r['domain'] != domain:
            continue
        if protocol and r.get('protocol') and r['protocol'] != protocol:
            continue
        try:
            v = float(r[metric])
        except (KeyError, TypeError, ValueError):
            continue
        out[(str(r.get('seed', '')), str(r.get('cube', '')))] = v
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Exact paired Wilcoxon tests (Sections 5.1 and 5.5)')
    ap.add_argument('--csv', nargs='+', required=True,
                    help='long CSVs; optional method label as label=path')
    ap.add_argument('--pairs', nargs='+', required=True,
                    help='pairs methodA:aggA=methodB:aggB')
    ap.add_argument('--metrics', nargs='+', default=['dphi', 'dsa', 'ssim'])
    ap.add_argument('--domains', nargs='+', default=None,
                    help='default: every domain present in the CSVs')
    ap.add_argument('--protocol', default='parity',
                    help="filter on the protocol column when present "
                         "(default parity; '' disables the filter)")
    ap.add_argument('--out_csv',
                    default='outputs/analysis/wilcoxon_pairwise.csv')
    args = ap.parse_args()

    from scipy.stats import wilcoxon  # deferred: clearer error if missing

    rows = load_rows(args.csv)
    domains = args.domains or sorted({r['domain'] for r in rows})
    protocol = args.protocol or None

    results = []
    for pair_spec in args.pairs:
        arm_a, arm_b = parse_pair(pair_spec)
        label = f'{arm_a[0]}:{arm_a[1]} vs {arm_b[0]}:{arm_b[1]}'
        for metric in args.metrics:
            for domain in domains:
                va = paired_values(rows, arm_a, metric, domain, protocol)
                vb = paired_values(rows, arm_b, metric, domain, protocol)
                keys = sorted(set(va) & set(vb))
                diffs = np.array([va[k] - vb[k] for k in keys])
                if len(diffs) < 2:
                    results.append(dict(pair=label, metric=metric,
                                        domain=domain, n=len(diffs),
                                        note='insufficient pairs'))
                    continue
                try:
                    _, p_exact = wilcoxon(diffs, method='exact',
                                          zero_method='wilcox')
                except ValueError as e:
                    results.append(dict(pair=label, metric=metric,
                                        domain=domain, n=len(diffs),
                                        note=str(e)))
                    continue
                try:
                    _, p_approx = wilcoxon(diffs, method='approx',
                                           zero_method='wilcox')
                except Exception:
                    p_approx = float('nan')
                a_vals = np.array([va[k] for k in keys])
                b_vals = np.array([vb[k] for k in keys])
                sig = ('***' if p_exact < 0.001 else
                       '**' if p_exact < 0.01 else
                       '*' if p_exact < 0.05 else
                       'marginal' if p_exact < 0.1 else 'n.s.')
                results.append(dict(
                    pair=label, metric=metric, domain=domain, n=len(diffs),
                    mean_a=float(a_vals.mean()), std_a=float(a_vals.std()),
                    mean_b=float(b_vals.mean()), std_b=float(b_vals.std()),
                    mean_diff=float(diffs.mean()),
                    p_exact=float(p_exact), p_approx=float(p_approx),
                    note=sig))
                print(f'{label:45s} {metric:5s} {domain:12s} n={len(diffs):2d} '
                      f'mean_diff={diffs.mean():+.5f} '
                      f'p_exact={p_exact:.4f} ({sig})  '
                      f'p_approx={p_approx:.4f}')

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ['pair', 'metric', 'domain', 'n', 'mean_a', 'std_a', 'mean_b',
            'std_b', 'mean_diff', 'p_exact', 'p_approx', 'note']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f'[saved] {out_csv} ({len(results)} rows)')


if __name__ == '__main__':
    main()
