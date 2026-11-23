#!/usr/bin/env python3
"""Collect test results of every final run of the nine settings (summary.csv is read by
plot_pareto.py, popularity_signals.py and train_vs_rec_popularity.py).

  python analysis/collect_results.py [--runs runs] [--out results]

Writes
  per_seed.csv       one row per final run (STFR (forecast) serving runs included)
  summary.csv        seed means / standard deviations per arm (unrounded)
  main_table.csv     per setting: Recall@20, NDCG@20 and nALRP@20 of the nine methods and of
                     STFR (forecast) (unrounded three-seed means), the displayed four-decimal
                     accuracy values, their column ranks and the mean rank per backbone
  mean_rank.csv      mean rank per backbone and over the three backbones
  improvement_vs_strongest_baseline.csv   STFR and STFR (forecast) vs the strongest compared
                     method of each setting and accuracy column

Mean rank: the three seeds are averaged first; each accuracy column is then
ranked within its setting on the unrounded means (--rank_on displayed ranks the printed
four-decimal values instead, equal displayed values sharing the average rank; the two rules
differ only in columns where two methods print the same value).  A method's mean rank
averages its column ranks per backbone: 6 columns (three datasets x Recall@20, NDCG@20) for
every backbone; the overall mean rank averages the three backbones.  The nine
methods are ranked among themselves.  STFR (forecast) is ranked against the eight other
methods with STFR removed (it never changes another method's rank).  The non-personalized
fresh ranking and nALRP never enter the ranks.  Improvements, significance tests and
hyperparameter selection use the unrounded records.
"""
import argparse
import os

import pandas as pd

from common import (long_table, summarize, display, display_ranks, check_records, MAIN_ARMS, BASELINES,
                    VARIANT_ARMS, RANK_COLUMNS, label)


def ranks_for(wide, arms, rank_src, on):
    """column ranks and mean ranks of `arms` ranked among themselves"""
    sub = wide[wide.arm.isin(arms)].copy()
    sub['arm'] = sub['arm'].astype(str)
    return display_ranks(sub, rank_src, ['data', 'backbone'], on=on)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    p.add_argument('--rank_on', default='unrounded', choices=('unrounded', 'displayed'))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = long_table(a.runs)
    if df.empty:
        raise SystemExit('no test results under %s' % a.runs)
    df.to_csv(os.path.join(a.out, 'per_seed.csv'), index=False)
    s = summarize(df)
    s.to_csv(os.path.join(a.out, 'summary.csv'), index=False)

    arms_all = MAIN_ARMS + VARIANT_ARMS
    m = s[s.arm.isin(arms_all)]
    if m.empty:
        print('[collect] no main-arm runs; per_seed.csv and summary.csv written')
        return
    wide = m.pivot_table(index=['data', 'backbone', 'arm'], columns='k',
                         values=['recall', 'ndcg', 'nalrp'], sort=False)
    wide.columns = ['%s@%d' % (c[0], c[1]) for c in wide.columns]
    wide = wide.reset_index()
    cols = [c for c in ['recall@20', 'ndcg@20', 'nalrp@20'] if c in wide.columns]
    acc_cols = [c for c in cols if not c.startswith('nalrp')]
    wide['arm'] = pd.Categorical(wide['arm'], arms_all)
    wide = wide.sort_values(['backbone', 'data', 'arm']).reset_index(drop=True)
    rank_src = [c for c in ('recall@20', 'ndcg@20') if c in acc_cols]

    # the nine methods ranked among themselves
    ranked, mean_rank = ranks_for(wide, MAIN_ARMS, rank_src, a.rank_on)
    mean_rank['ranked_against'] = 'nine methods'
    rank_tabs = [ranked]
    mrs = [mean_rank]
    # each serve-only variant ranked in place of STFR
    for v in VARIANT_ARMS:
        if v not in set(wide['arm'].astype(str)):
            continue
        rv, mv = ranks_for(wide, [x for x in MAIN_ARMS if x != 'STFR'] + [v], rank_src, a.rank_on)
        rank_tabs.append(rv[rv.arm == v])
        mrs.append(mv[mv.arm == v].assign(ranked_against='eight other methods, STFR removed'))
    ranked = pd.concat(rank_tabs, ignore_index=True)
    mean_rank = pd.concat(mrs, ignore_index=True)
    overall = mean_rank.groupby('arm', sort=False).agg(mean_rank=('mean_rank', 'mean'),
                                                     n_backbones=('backbone', 'count')).reset_index()
    overall['backbone'] = 'all'
    mean_rank_out = pd.concat([mean_rank, overall], ignore_index=True)
    mean_rank_out.to_csv(os.path.join(a.out, 'mean_rank.csv'), index=False)
    ranked = ranked.merge(mean_rank[['backbone', 'arm', 'mean_rank']], on=['backbone', 'arm'], how='left')
    ranked['arm'] = pd.Categorical(ranked['arm'], arms_all)
    ranked = ranked.sort_values(['backbone', 'data', 'arm']).reset_index(drop=True)

    for msg in check_records(df[df.arm.isin(arms_all)], arms=arms_all):
        print('[check]', msg)
    for _, r in mean_rank.iterrows():
        if r['mean_rank_n_cols'] != RANK_COLUMNS.get(r['backbone'], r['mean_rank_n_cols']):
            print('[check] %s/%s: mean rank over %d columns (the manuscript uses %d)' % (
                r['backbone'], r['arm'], r['mean_rank_n_cols'], RANK_COLUMNS[r['backbone']]))
    main9 = ranked[ranked.arm.isin(MAIN_ARMS)]
    for (d, b), g in main9.groupby(['data', 'backbone'], sort=False, observed=True):
        for c in rank_src:
            tied = g[g['disp_' + c].duplicated(keep=False)]
            if not tied.empty:
                print('[collect] %s/%s %s: equal displayed values (ranked on %s means): %s' % (
                    d, b, c, a.rank_on, ', '.join('%s %s' % (x, y) for x, y in zip(tied['arm'], tied['disp_' + c]))))

    # STFR and STFR (forecast) vs the strongest compared method per column (unrounded means)
    rows = []
    for (d, b), g in ranked.groupby(['data', 'backbone'], sort=False, observed=True):
        bl = g[g.arm.isin(BASELINES)]
        for focal in ['STFR'] + VARIANT_ARMS:
            if focal not in set(g.arm.astype(str)) or bl.empty:
                continue
            st = g[g.arm == focal].iloc[0]
            r = dict(data=d, backbone=b, focal=focal)
            for c in acc_cols:
                best = bl.loc[bl[c].idxmax()]
                r['strongest_' + c] = best['arm']
                r['improv_pct_' + c] = 100.0 * (st[c] / best[c] - 1.0)
            rows.append(r)
    imp = pd.DataFrame(rows)
    ranked.to_csv(os.path.join(a.out, 'main_table.csv'), index=False)
    imp.to_csv(os.path.join(a.out, 'improvement_vs_strongest_baseline.csv'), index=False)

    pd.set_option('display.width', 220)
    for (d, b), g in ranked.groupby(['data', 'backbone'], sort=False, observed=True):
        print('\n== %s / %s ==' % (d, b))
        shown = pd.DataFrame({'method': [label(x) for x in g['arm']]})
        for c in cols:
            shown[c] = (g['disp_' + c] if 'disp_' + c in g else g[c].map(display)).values
        shown['mean_rank'] = g['mean_rank'].map(lambda v: '%.2f' % v).values
        print(shown.to_string(index=False))
    print('\n== mean rank (overall = mean of the three backbones) ==')
    print(overall[['arm', 'mean_rank']].assign(arm=lambda x: x.arm.map(label))
          .to_string(index=False, float_format=lambda v: '%.2f' % v))
    if not imp.empty:
        print('\n== STFR / STFR (forecast) vs the strongest compared method (%) ==')
        print(imp.to_string(index=False, float_format=lambda v: '%+.1f' % v))
    print('\n[collect] -> %s/{per_seed,summary,main_table,mean_rank,improvement_vs_strongest_baseline}.csv' % a.out)


if __name__ == '__main__':
    main()
