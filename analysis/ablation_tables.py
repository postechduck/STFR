#!/usr/bin/env python3
"""Section 7.3 tables (Tables 9-11) from the ablation arms of scripts/run_ablations_vg.sh.

  python analysis/ablation_tables.py [--runs runs] [--out results] [--data Amazon-VG]

  ablation_components.csv   base, SSNS only, fresh channel only, STFR (shared gain), STFR
  ablation_recency.csv      recent-N and recent-N + fresh: the window N with the highest
                            mean validation Recall@20 (three seeds) per backbone and arm,
                            with its test Recall@20; ablation_recency_all.csv lists every N
  ablation_samplers.csv     DNS / AUC-NS / FairNeg replacing SSNS vs STFR
All values are three-seed test means (Recall@20, nALRP@20).
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from common import final_runs, test_metrics, val_best


def collect(runs, data):
    rows = []
    for a in final_runs(runs, data=data):
        m = test_metrics(a['run_dir'], 20)
        v = val_best(a['run_dir'])
        if m is None:
            continue
        rows.append(dict(backbone=a['backbone'], arm=a['arm'], seed=a['seed'], recall=m['recall'],
                         ndcg=m['ndcg'], nalrp=m['nalrp'], val_recall=v['recall'] if v else np.nan))
    return pd.DataFrame(rows)


def means(df, arms):
    g = df[df.arm.isin(arms)].groupby(['backbone', 'arm'], sort=False).agg(
        n=('seed', 'count'), recall=('recall', 'mean'), nalrp=('nalrp', 'mean'), ndcg=('ndcg', 'mean'),
        val_recall=('val_recall', 'mean')).reset_index()
    g['arm'] = pd.Categorical(g['arm'], arms)
    return g.sort_values(['backbone', 'arm'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    p.add_argument('--data', default='Amazon-VG')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = collect(a.runs, a.data)
    if df.empty:
        raise SystemExit('no runs for %s under %s' % (a.data, a.runs))
    pd.set_option('display.width', 200)

    comp = means(df, ['base', 'SSNS_only', 'fresh_only', 'shared_gain', 'STFR'])
    comp.to_csv(os.path.join(a.out, 'ablation_components.csv'), index=False)
    print('== component ablation (%s) ==' % a.data)
    print(comp[['backbone', 'arm', 'n', 'recall', 'nalrp']].to_string(index=False, float_format=lambda v: '%.4f' % v))

    rec = df[df.arm.str.match(r'^recent\d+(_fresh)?$')].copy()
    if not rec.empty:
        rec['family'] = np.where(rec.arm.str.endswith('_fresh'), 'recent+fresh', 'recent')
        rec['N'] = rec.arm.str.extract(r'recent(\d+)').astype(int)
        allN = rec.groupby(['backbone', 'family', 'N']).agg(
            n=('seed', 'count'), val_recall=('val_recall', 'mean'), recall=('recall', 'mean'),
            nalrp=('nalrp', 'mean')).reset_index().sort_values(['backbone', 'family', 'N'])
        allN.to_csv(os.path.join(a.out, 'ablation_recency_all.csv'), index=False)
        sel = allN.loc[allN.groupby(['backbone', 'family'])['val_recall'].idxmax()]
        ref = means(df, ['STFR'])[['backbone', 'recall', 'nalrp']].rename(columns={'recall': 'stfr_recall', 'nalrp': 'stfr_nalrp'})
        sel = sel.merge(ref, on='backbone', how='left')
        sel['deficit_vs_stfr_pct'] = 100.0 * (sel['recall'] / sel['stfr_recall'] - 1.0)
        sel.to_csv(os.path.join(a.out, 'ablation_recency.csv'), index=False)
        print('\n== recency-restricted training: window selected on validation ==')
        print(sel[['backbone', 'family', 'N', 'val_recall', 'recall', 'stfr_recall', 'deficit_vs_stfr_pct']]
              .to_string(index=False, float_format=lambda v: '%.4f' % v))

    sw = means(df, ['swap_DNS', 'swap_AUCNS', 'swap_FairNeg', 'STFR'])
    if len(sw) > len(df[df.arm == 'STFR'].backbone.unique()):
        sw.to_csv(os.path.join(a.out, 'ablation_samplers.csv'), index=False)
        print('\n== sampler replacement ==')
        print(sw[['backbone', 'arm', 'n', 'recall', 'nalrp']].to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('\n[ablations] -> %s/ablation_*.csv' % a.out)


if __name__ == '__main__':
    main()
