#!/usr/bin/env python3
"""Catalog coverage@20 and exposure Gini@20 on the training-active catalog (Section 7.2, Table A.2).

  python analysis/coverage_gini.py [--runs runs] [--out results]

For every final run, the saved top-20 lists give the exposure count v_i of each
training-active item (items with at least one training interaction; zero-exposure
items included).  Coverage@K = share of the catalog recommended at least once;
Gini@K = sum_j (2j - n - 1) v_(j) / (n sum_j v_(j)) over ascending counts.  Both are
computed per seed and then averaged (standard deviations use the population convention).
nALRP@K is recomputed from the same lists as a consistency check.

Writes coverage_gini_per_seed.csv and coverage_gini.csv (every method: seed means and
standard deviations) and exposure_summary_k20.csv (Table A.2: base, PDA, TIDE and STFR).
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import main_runs, read_recs, stale_signal, display, CELL_ORDER

SUMMARY_ARMS = ['base', 'PDA', 'TIDE', 'STFR']


def gini(x):
    x = np.sort(x.astype(np.float64))
    n, tot = x.size, x.sum()
    if tot == 0:
        return 0.0
    j = np.arange(1, n + 1)
    return float(((2 * j - n - 1) * x).sum() / (n * tot))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    cache = {}
    for r in main_runs(a.runs):
        if r['data'] not in cache:
            cache[r['data']] = stale_signal(r['data'])
        d, c, n, active = cache[r['data']]
        for k in (20,):
            recs = read_recs(r['run_dir'], k)
            if recs is None:
                continue
            x = np.bincount(np.concatenate([np.asarray(rr) for _, rr, _ in recs]), minlength=n)[:n][active]
            rows.append(dict(data=r['data'], backbone=r['backbone'], arm=r['arm'], seed=r['seed'], k=k,
                             catalog_size=int(active.sum()), n_users=len(recs),
                             coverage=float((x > 0).mean()), gini=gini(x),
                             nalrp=float(np.mean([d[rr].mean() for _, rr, _ in recs])),
                             recall=float(np.mean([len(set(rr) & set(g)) / len(g) for _, rr, g in recs]))))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('no list dumps found under %s (run eval_ckpt with --dump_recs)' % a.runs)
    df.to_csv(os.path.join(a.out, 'coverage_gini_per_seed.csv'), index=False)
    s = df.groupby(['data', 'backbone', 'arm', 'k'], sort=False).agg(
        n_seeds=('seed', 'count'), catalog_size=('catalog_size', 'first'),
        coverage=('coverage', 'mean'), coverage_sd=('coverage', lambda v: v.std(ddof=0)),
        gini=('gini', 'mean'), gini_sd=('gini', lambda v: v.std(ddof=0)),
        nalrp=('nalrp', 'mean'), recall=('recall', 'mean')).reset_index()
    s.to_csv(os.path.join(a.out, 'coverage_gini.csv'), index=False)

    # Table A.2: base, PDA, TIDE and STFR (seed means and standard deviations)
    for kk in (20,):
        g = s[(s.k == kk) & s.arm.isin(SUMMARY_ARMS)]
        out = g.pivot_table(index=['data', 'backbone'], columns='arm', sort=False,
                            values=['coverage', 'coverage_sd', 'gini', 'gini_sd'])
        out = out.reindex(columns=[(v, m) for v in ('coverage', 'coverage_sd', 'gini', 'gini_sd') for m in SUMMARY_ARMS
                                   if (v, m) in out.columns])
        out = out.reindex([c for c in CELL_ORDER if c in out.index] + [c for c in out.index if c not in CELL_ORDER])
        missing = [(d, b, m) for (d, b) in out.index for m in SUMMARY_ARMS
                   if ('coverage', m) not in out.columns or pd.isna(out.loc[(d, b), ('coverage', m)])]
        for d, b, m in missing:
            print('[check] %s/%s: no top-%d lists of %s' % (d, b, kk, m))
        flat = out.copy()
        flat.columns = ['%s_%s' % c for c in flat.columns]
        flat.insert(0, 'k', kk)
        flat.reset_index().to_csv(os.path.join(a.out, 'exposure_summary_k%d.csv' % kk), index=False)
        print('\n== Table A.2: coverage@%d / exposure Gini@%d (seed means) ==' % (kk, kk))
        print(out[['coverage', 'gini']].map(lambda v: display(v, 3)).to_string())
    pd.set_option('display.width', 220)
    for k in (20,):
        for col in ('coverage', 'gini', 'nalrp'):
            t = s[s.k == k].pivot_table(index=['data', 'backbone'], columns='arm', values=col, sort=False)
            print('\n== %s@%d (training-active catalog, seed means) ==' % (col, k))
            print(t.round(4).to_string())
    print('\n[coverage/gini] -> %s/coverage_gini.csv, coverage_gini_per_seed.csv, exposure_summary_k20.csv (Table A.2)' % a.out)


if __name__ == '__main__':
    main()
