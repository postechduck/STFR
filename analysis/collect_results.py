#!/usr/bin/env python3
"""Collect test results of every final run into tables (Tables 4-7 of the manuscript).

  python analysis/collect_results.py [--runs runs] [--out results]

Writes
  per_seed.csv       one row per final run
  summary.csv        seed means / standard deviations per arm
  main_table.csv     per setting: Recall@20, NDCG@20 and nALRP@20 of the main arms,
                     the manuscript's mean rank (Tables 4-5: per backbone, over the
                     Recall@20 and NDCG@20 columns of its datasets -- 6 columns for MF and
                     LightGCN, 4 for SimGCL; unrounded three-seed means; ties receive the
                     average rank only when the unrounded means are exactly equal; the
                     non-personalized fresh prior is not an arm and is never ranked)
                     and STFR's improvement over the strongest baseline per column
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import long_table, summarize, MAIN_ARMS, BASELINES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = long_table(a.runs)
    if df.empty:
        raise SystemExit('no test results under %s' % a.runs)
    df.to_csv(os.path.join(a.out, 'per_seed.csv'), index=False)
    s = summarize(df)
    s.to_csv(os.path.join(a.out, 'summary.csv'), index=False)

    # main table: main arms only
    m = s[s.arm.isin(MAIN_ARMS)]
    if m.empty:
        print('[collect] no main-arm runs; per_seed.csv and summary.csv written')
        return
    wide = m.pivot_table(index=['data', 'backbone', 'arm'], columns='k',
                         values=['recall', 'ndcg', 'nalrp'], sort=False)
    wide.columns = ['%s@%d' % (c[0], c[1]) for c in wide.columns]
    wide = wide.reset_index()
    cols = [c for c in ['recall@20', 'ndcg@20', 'nalrp@20'] if c in wide.columns]
    acc_cols = [c for c in cols if not c.startswith('nalrp')]
    wide['arm'] = pd.Categorical(wide['arm'], MAIN_ARMS)
    wide = wide.sort_values(['backbone', 'data', 'arm']).reset_index(drop=True)

    # ranks within (data, backbone, column) on the UNROUNDED seed means, averaged per
    # backbone across datasets.  Only the @20 accuracy columns enter the mean rank
    # (the manuscript reports @20); 'average' assigns tied ranks only to exactly equal
    # unrounded values (displayed ties at four decimals are not ties).
    rank_src = [c for c in ('recall@20', 'ndcg@20') if c in acc_cols]
    for c in rank_src:
        wide['rank_' + c] = wide.groupby(['data', 'backbone'])[c].rank(ascending=False, method='average')
    rank_cols = ['rank_' + c for c in rank_src]
    mean_rank = wide.groupby(['backbone', 'arm'], observed=True)[rank_cols].mean().mean(axis=1).rename('mean_rank')
    n_cols = wide.groupby(['backbone', 'arm'], observed=True)[rank_cols].count().sum(axis=1).rename('mean_rank_n_cols')
    wide = wide.merge(mean_rank.reset_index(), on=['backbone', 'arm'], how='left')
    wide = wide.merge(n_cols.reset_index(), on=['backbone', 'arm'], how='left')
    exact_ties = wide[wide.duplicated(['data', 'backbone'] , keep=False)].groupby(['data', 'backbone'])[rank_src].apply(
        lambda g: {c: sorted(g[c][g[c].duplicated(keep=False)].unique().tolist()) for c in rank_src if g[c].duplicated().any()})
    exact_ties = {k: v for k, v in exact_ties.items() if v}
    if exact_ties:
        print('[collect] exactly equal unrounded means (average rank applied):', exact_ties)

    # STFR improvement over the strongest baseline per column (unrounded means)
    rows = []
    for (d, b), g in wide.groupby(['data', 'backbone'], sort=False):
        if 'STFR' not in g.arm.values:
            continue
        st = g[g.arm == 'STFR'].iloc[0]
        bl = g[g.arm.isin(BASELINES)]
        if bl.empty:
            continue
        r = dict(data=d, backbone=b)
        for c in acc_cols:
            best = bl.loc[bl[c].idxmax()]
            r['best_baseline_' + c] = best['arm']
            r['improv_pct_' + c] = 100.0 * (st[c] / best[c] - 1.0)
        rows.append(r)
    imp = pd.DataFrame(rows)
    wide.to_csv(os.path.join(a.out, 'main_table.csv'), index=False)
    imp.to_csv(os.path.join(a.out, 'improvement_vs_strongest_baseline.csv'), index=False)

    pd.set_option('display.width', 220)
    for (d, b), g in wide.groupby(['data', 'backbone'], sort=False):
        print('\n== %s / %s ==' % (d, b))
        print(g[['arm'] + cols + ['mean_rank']].to_string(index=False, float_format=lambda v: '%.4f' % v))
    if not imp.empty:
        print('\n== STFR vs strongest baseline (%) ==')
        print(imp.to_string(index=False, float_format=lambda v: '%+.1f' % v))
    print('\n[collect] -> %s/{per_seed,summary,main_table,improvement_vs_strongest_baseline}.csv' % a.out)


if __name__ == '__main__':
    main()
