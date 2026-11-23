#!/usr/bin/env python3
"""Seed means of the Section 7.4 control arms and the recency-window selection on validation.

  python analysis/ablation_summary.py [--runs runs] [--out results] [--data Amazon-VG]

Reads every final run of one dataset (the arms of scripts/run_ablations_vg.sh next to the
main-comparison arms of the same settings) and writes

  ablation_means.csv        per backbone and arm: number of seeds and three-seed means of the
                            test Recall@20, NDCG@20 and nALRP@20 and of the best validation
                            Recall@20 of each run (every arm found, in the order of the runs)
  ablation_recency_all.csv  recency-restricted training (arms recent<N>, the unmodified backbone,
                            and recent<N>_fresh, the fresh channel only): every window N
  ablation_recency.csv      the window N with the highest three-seed mean validation Recall@20
                            per backbone and family, with its test means and the relative
                            Recall@20 difference from STFR of the same backbone

Seed standard deviations of the test metrics are in results/summary.csv of
analysis/collect_results.py, which reads the same final runs.
"""
import argparse
import os

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


def means(df):
    return df.groupby(['backbone', 'arm'], sort=False).agg(
        n=('seed', 'count'), recall=('recall', 'mean'), nalrp=('nalrp', 'mean'), ndcg=('ndcg', 'mean'),
        val_recall=('val_recall', 'mean')).reset_index()


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

    allm = means(df)
    allm.to_csv(os.path.join(a.out, 'ablation_means.csv'), index=False)
    print('== three-seed means of every arm (%s) ==' % a.data)
    print(allm[['backbone', 'arm', 'n', 'recall', 'ndcg', 'nalrp', 'val_recall']]
          .to_string(index=False, float_format=lambda v: '%.4f' % v))

    rec = df[df.arm.str.match(r'^recent\d+(_fresh)?$')].copy()
    if not rec.empty:
        rec['family'] = np.where(rec.arm.str.endswith('_fresh'), 'recent+fresh', 'recent')
        rec['N'] = rec.arm.str.extract(r'recent(\d+)').astype(int)
        allN = rec.groupby(['backbone', 'family', 'N']).agg(
            n=('seed', 'count'), val_recall=('val_recall', 'mean'), recall=('recall', 'mean'),
            nalrp=('nalrp', 'mean')).reset_index().sort_values(['backbone', 'family', 'N'])
        allN.to_csv(os.path.join(a.out, 'ablation_recency_all.csv'), index=False)
        sel = allN.loc[allN.groupby(['backbone', 'family'])['val_recall'].idxmax()]
        ref = allm[allm.arm == 'STFR'][['backbone', 'recall', 'nalrp']].rename(
            columns={'recall': 'stfr_recall', 'nalrp': 'stfr_nalrp'})
        sel = sel.merge(ref, on='backbone', how='left')
        sel['deficit_vs_stfr_pct'] = 100.0 * (sel['recall'] / sel['stfr_recall'] - 1.0)
        sel.to_csv(os.path.join(a.out, 'ablation_recency.csv'), index=False)
        print('\n== recency-restricted training: window selected on validation ==')
        print(sel[['backbone', 'family', 'N', 'val_recall', 'recall', 'stfr_recall', 'deficit_vs_stfr_pct']]
              .to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('\n[ablations] -> %s/ablation_means.csv, ablation_recency_all.csv, ablation_recency.csv' % a.out)


if __name__ == '__main__':
    main()
