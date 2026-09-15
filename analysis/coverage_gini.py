#!/usr/bin/env python3
"""Catalog coverage@20 and exposure Gini@20 on the training-active catalog (Section 7.2, Table E.4).

  python analysis/coverage_gini.py [--runs runs] [--out results]

For every final run, the saved top-20 lists give the exposure count v_i of each
training-active item (items with at least one training interaction; zero-exposure
items included).  Coverage@K = share of the catalog recommended at least once;
Gini@K = sum_j (2j - n - 1) v_(j) / (n sum_j v_(j)) over ascending counts.  Both are
computed per seed and then averaged (standard deviations use the population convention).
nALRP@K is recomputed from the same lists as a consistency check.
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import final_runs, read_recs, stale_signal, test_metrics, BASELINES


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
    for r in final_runs(a.runs):
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

    # Table E.4: base, the non-STFR method with the highest mean Recall@20 in the setting, and STFR
    for kk in (20,):
        out = []
        for (data, bb), g in s[s.k == kk].groupby(['data', 'backbone'], sort=False):
            g = g.set_index('arm')
            if 'STFR' not in g.index or 'base' not in g.index:
                continue
            riv = [b for b in BASELINES if b in g.index]
            if not riv:
                continue
            rival = max(riv, key=lambda b: g.loc[b, 'recall'])
            out.append(dict(data=data, backbone=bb, k=kk, rival=rival,
                            coverage_base=g.loc['base', 'coverage'], coverage_rival=g.loc[rival, 'coverage'], coverage_STFR=g.loc['STFR', 'coverage'],
                            gini_base=g.loc['base', 'gini'], gini_rival=g.loc[rival, 'gini'], gini_STFR=g.loc['STFR', 'gini']))
        pd.DataFrame(out).to_csv(os.path.join(a.out, 'exposure_summary_k%d.csv' % kk), index=False)
    pd.set_option('display.width', 220)
    for k in (20,):
        for col in ('coverage', 'gini', 'nalrp'):
            t = s[s.k == k].pivot_table(index=['data', 'backbone'], columns='arm', values=col, sort=False)
            print('\n== %s@%d (training-active catalog, seed means) ==' % (col, k))
            print(t.round(4).to_string())
    print('\n[coverage/gini] -> %s/coverage_gini.csv, coverage_gini_per_seed.csv, exposure_summary_k20.csv (Table E.4)' % a.out)


if __name__ == '__main__':
    main()
