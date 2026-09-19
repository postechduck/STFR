#!/usr/bin/env python3
"""Collect test results of every final run into tables (Tables 4-7 of the manuscript).

  python analysis/collect_results.py [--runs runs] [--out results]

Writes
  per_seed.csv       one row per final run
  summary.csv        seed means / standard deviations per arm (unrounded)
  main_table.csv     per setting: Recall@20, NDCG@20 and nALRP@20 of the main arms (unrounded
                     three-seed means), the displayed four-decimal accuracy values, their
                     column ranks and the manuscript's mean rank
  improvement_vs_strongest_baseline.csv   STFR vs the strongest baseline per column

Mean rank (Tables 4-5): the three seeds are averaged first; each accuracy column is then
ranked within its setting on the displayed four-decimal mean (common.display, the same
string the table prints), equal displayed values sharing the average rank.  A method's
mean rank averages its column ranks per backbone: 6 columns for MF and LightGCN, 4 for
SimGCL.  The non-personalized fresh prior and nALRP never enter it.  Improvements,
significance tests and hyperparameter selection use the unrounded records.
"""
import argparse
import os

import pandas as pd

from common import long_table, summarize, display, display_ranks, check_records, MAIN_ARMS, BASELINES, RANK_COLUMNS


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

    # column ranks on the displayed values, mean rank per backbone (see the module docstring)
    rank_src = [c for c in ('recall@20', 'ndcg@20') if c in acc_cols]
    wide, mean_rank = display_ranks(wide, rank_src, ['data', 'backbone'])
    wide = wide.merge(mean_rank, on=['backbone', 'arm'], how='left')
    for msg in check_records(df[df.arm.isin(MAIN_ARMS)]):
        print('[check]', msg)
    for _, r in mean_rank.iterrows():
        if r['mean_rank_n_cols'] != RANK_COLUMNS.get(r['backbone'], r['mean_rank_n_cols']):
            print('[check] %s/%s: mean rank over %d columns (the manuscript uses %d)' % (
                r['backbone'], r['arm'], r['mean_rank_n_cols'], RANK_COLUMNS[r['backbone']]))
    for (d, b), g in wide.groupby(['data', 'backbone'], sort=False):
        for c in rank_src:
            tied = g[g['disp_' + c].duplicated(keep=False)]
            if not tied.empty:
                print('[collect] %s/%s %s: equal displayed values share the average rank: %s' % (
                    d, b, c, ', '.join('%s %s' % (x, y) for x, y in zip(tied['arm'], tied['disp_' + c]))))

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
        shown = g[['arm']].copy()
        for c in cols:
            shown[c] = g['disp_' + c] if 'disp_' + c in g else g[c].map(display)
        shown['mean_rank'] = g['mean_rank'].map(lambda v: '%.2f' % v)
        print(shown.to_string(index=False))
    if not imp.empty:
        print('\n== STFR vs strongest baseline (%) ==')
        print(imp.to_string(index=False, float_format=lambda v: '%+.1f' % v))
    print('\n[collect] -> %s/{per_seed,summary,main_table,improvement_vs_strongest_baseline}.csv' % a.out)


if __name__ == '__main__':
    main()
