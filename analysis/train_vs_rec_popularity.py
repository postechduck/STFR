#!/usr/bin/env python3
"""Average stale popularity of training interactions vs base recommendations (Section 4.2, Table 2).

  python analysis/train_vs_rec_popularity.py [--datasets ...] [--out results]

mu_train = sum_i c_i d_i / sum_i c_i weights every training interaction equally; the
recommendation side is the base models' nALRP@20 (three-seed means from results/summary.csv).
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import stale_signal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default='Amazon-VG,Amazon-Movies,Douban-movie')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ_path = os.path.join(a.out, 'summary.csv')
    summ = pd.read_csv(summ_path) if os.path.exists(summ_path) else None
    rows = []
    for ds in a.datasets.split(','):
        d, c, n, active = stale_signal(ds)
        r = dict(dataset=ds, mu_train=float((c * d).sum() / c.sum()), n_train_interactions=int(c.sum()),
                 catalog_size=n, training_active_items=int(active.sum()))
        if summ is not None:
            for bb in ('MF', 'LightGCN', 'SimGCL'):
                q = summ[(summ.data == ds) & (summ.backbone == bb) & (summ.arm == 'base') & (summ.k == 20)]
                if len(q):
                    r['base_%s_nalrp@20' % bb] = float(q.nalrp.iloc[0])
                    r['delta_%s' % bb] = float(q.nalrp.iloc[0]) - r['mu_train']
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'train_vs_rec_popularity.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('\n[train vs recommendation popularity] -> %s/train_vs_rec_popularity.csv' % a.out)


if __name__ == '__main__':
    main()
