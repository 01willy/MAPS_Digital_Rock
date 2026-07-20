#!/usr/bin/env python3
"""
Aggregate the TEST-ONLY multi-seed k=1 LBM run into long + summary CSVs
(Table 3 numbers; per-axis rows for Table S10 come from the per-cube JSONs'
`k_per_axis_mD` fields).

Run after analysis/lbm_multiseed_eval.py finishes (or at any time -- it
reads whatever per-cube *_lbm.json exist so far under <root>/lbm/).

Primary metric: k_zz error % vs GT (interpolation-axis permeability).
Also reports k_trace error %. Per (domain, method): mean +/- std over
seeds x cubes.

Usage:
  python analysis/lbm_multiseed_aggregate.py --root outputs/analysis/lbm_testonly
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description='Aggregate multi-seed LBM run '
                                             '(Table 3)')
    ap.add_argument('--root', type=str, required=True,
                    help='Output root of analysis/lbm_multiseed_eval.py')
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for p in sorted((root / 'lbm').glob('**/*_lbm.json')):
        d = json.load(open(p))
        rows.append(dict(domain=d['domain'], method=d['method'],
                         seed=d['seed'], cube=d['cube'], phi=d.get('phi'),
                         gt_phi=d.get('gt_phi'),
                         k_zz_mD=d['k_zz_mD'], k_trace_mD=d['k_trace_mD'],
                         porosity=d.get('porosity_total')))
    if not rows:
        print('No LBM JSONs yet.')
        return
    df = pd.DataFrame(rows)
    long_p = root / 'lbm_testonly_long.csv'
    df.to_csv(long_p, index=False)
    print(f'[WROTE] {long_p} ({len(df)} rows)')

    # GT per (domain, cube) -- seed-independent.
    gt = (df[df.method == 'GT']
          .groupby(['domain', 'cube'])[['k_zz_mD', 'k_trace_mD']].first())

    recs = []
    for (dom, meth), g in df[df.method != 'GT'].groupby(['domain', 'method']):
        zz_err, tr_err = [], []
        for _, r in g.iterrows():
            kz = gt.loc[(dom, r.cube), 'k_zz_mD']
            kt = gt.loc[(dom, r.cube), 'k_trace_mD']
            if kz and kz > 0:
                zz_err.append(abs(r.k_zz_mD - kz) / kz * 100)
            if kt and kt > 0:
                tr_err.append(abs(r.k_trace_mD - kt) / kt * 100)
        recs.append(dict(domain=dom, method=meth, n=len(g),
                         k_zz_err_pct_mean=np.mean(zz_err),
                         k_zz_err_pct_std=np.std(zz_err),
                         k_trace_err_pct_mean=np.mean(tr_err),
                         k_trace_err_pct_std=np.std(tr_err)))
    summ = pd.DataFrame(recs).sort_values(['domain', 'k_zz_err_pct_mean'])
    summ_p = root / 'lbm_testonly_summary.csv'
    summ.to_csv(summ_p, index=False)
    print(f'[WROTE] {summ_p}\n')
    print('HEADLINE = k_zz_err_pct (interpolation-axis permeability error '
          'vs GT)')
    print(summ.to_string(index=False, float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()
