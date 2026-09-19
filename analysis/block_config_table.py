#!/usr/bin/env python3
"""Section 7.4 (Figure 4): base vs STFR across temporal block configurations on Amazon-VG.

  python analysis/block_config_table.py [--runs runs] [--out results]

Reads the finals of data variants Amazon-VG_b15, Amazon-VG_b30, Amazon-VG (L=60) and
Amazon-VG_b120 and writes block_configurations.csv (three-seed test means, recall seed
standard deviation, selected (f, alpha), relative Recall gain and nALRP difference of
STFR over base within each configuration).
"""
import argparse
import json
import os
import re

import pandas as pd

from common import final_runs, test_metrics

VARIANTS = [('Amazon-VG_b15', 15), ('Amazon-VG_b30', 30), ('Amazon-VG', 60), ('Amazon-VG_b120', 120)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for data, L in VARIANTS:
        for r in final_runs(a.runs, data=data):
            if r['arm'] not in ('base', 'STFR'):
                continue
            for k in (20,):
                m = test_metrics(r['run_dir'], k)
                if m:
                    rows.append(dict(L_days=L, data=data, backbone=r['backbone'], method=r['arm'], seed=r['seed'], k=k,
                                     recall=m['recall'], ndcg=m['ndcg'], nalrp=m['nalrp'],
                                     f=r['config'].get('ssns_frac'), alpha=r['config'].get('ssns_alpha'),
                                     simgcl_lambda=r.get('simgcl_lambda')))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('no block-configuration runs under %s' % a.runs)
    g = df.groupby(['L_days', 'backbone', 'method', 'k'], sort=True).agg(
        n=('seed', 'count'), recall=('recall', 'mean'), recall_sd=('recall', lambda v: v.std(ddof=0)),
        ndcg=('ndcg', 'mean'), nalrp=('nalrp', 'mean'), f=('f', 'first'), alpha=('alpha', 'first'),
        simgcl_lambda=('simgcl_lambda', 'first')).reset_index()
    out = []
    for (L, bb, k), gg in g.groupby(['L_days', 'backbone', 'k']):
        b = gg[gg.method == 'base']
        s = gg[gg.method == 'STFR']
        for _, r in gg.iterrows():
            d = r.to_dict()
            if not b.empty and not s.empty and r.method == 'STFR':
                d['recall_gain_pct'] = 100.0 * (s.recall.iloc[0] / b.recall.iloc[0] - 1.0)
                d['nalrp_diff'] = s.nalrp.iloc[0] - b.nalrp.iloc[0]
            out.append(d)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(a.out, 'block_configurations.csv'), index=False)
    pd.set_option('display.width', 220)
    print(res.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('\n[block configurations] -> %s/block_configurations.csv' % a.out)


if __name__ == '__main__':
    main()
