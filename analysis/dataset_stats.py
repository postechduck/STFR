#!/usr/bin/env python3
"""Dataset statistics (Table 3) and per-block interaction counts of the split figure (Figure 2).

  python analysis/dataset_stats.py [--datasets Amazon-VG,Amazon-Movies,Douban-movie] [--out results]

Reads data/<dataset>/ (written by prep/prep_split.py).  Window rows before the target
filters are recomputed from data_raw/<dataset>/raw_interactions.csv when present.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import ROOT

DAY = 86400


def date(ts):
    return pd.Timestamp(int(ts), unit='s').strftime('%Y-%m-%d')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default='Amazon-VG,Amazon-Movies,Douban-movie')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, blocks = [], []
    for ds in a.datasets.split(','):
        dp = os.path.join(ROOT, 'data', ds)
        m = json.load(open(os.path.join(dp, 'dataset_meta.json')))
        tr = pd.read_csv(os.path.join(dp, 'train_data.csv'), sep='\t')
        va = pd.read_csv(os.path.join(dp, 'val_data.csv'), sep='\t')
        te = pd.read_csv(os.path.join(dp, 'test_data.csv'), sep='\t')
        at = np.load(os.path.join(dp, 'all_times.npy'))
        W = m['block_days'] * DAY
        t_star = m['t_star']
        rows.append(dict(dataset=ds, users=m['user_num'], items=m['item_num'], train_interactions=len(tr),
                         density=len(tr) / (m['user_num'] * m['item_num']), training_blocks=t_star,
                         val_targets=len(va), test_targets=len(te), val_users=va.user.nunique(), test_users=te.user.nunique(),
                         training_active_users=tr.user.nunique(), training_active_items=tr.item.nunique(),
                         train_start=date(tr.timestamp.min()), train_end=date(tr.timestamp.max()),
                         eval_block_start=date(at[t_star] - W), val_test_cut=date(m.get('val_cut', at[t_star] - W + m['val_days'] * DAY)),
                         eval_block_end=date(at[t_star] - 1)))
        cnt = np.bincount(tr['split_idx'].values.astype(int), minlength=t_star)
        for k in range(t_star):
            blocks.append(dict(dataset=ds, block_index=k, block_start_date=date(at[k] - W), block_end_date=date(at[k] - 1),
                               n_interactions=int(cnt[k]), n_rows_prefilter=int(cnt[k]), role='train'))
        raw = os.path.join(ROOT, 'data_raw', ds, 'raw_interactions.csv')
        val_cut = m.get('val_cut', at[t_star] - W + m['val_days'] * DAY)
        vraw, traw = len(va), len(te)
        if os.path.exists(raw):
            full = pd.read_csv(raw, sep='\t', usecols=['user', 'item', 'timestamp'])
            full['timestamp'] = full['timestamp'].astype(np.int64)
            if m.get('dedupe', True):
                full = full.sort_values('timestamp', kind='stable').drop_duplicates(['user', 'item'], keep='first')
            win = full[(full.timestamp > at[t_star] - W) & (full.timestamp <= at[t_star])]
            vraw = int((win.timestamp < val_cut).sum())
            traw = int((win.timestamp >= val_cut).sum())
        blocks.append(dict(dataset=ds, block_index=t_star, block_start_date=date(at[t_star] - W), block_end_date=date(val_cut - 1),
                           n_interactions=len(va), n_rows_prefilter=vraw, role='val'))
        blocks.append(dict(dataset=ds, block_index=t_star, block_start_date=date(val_cut), block_end_date=date(at[t_star] - 1),
                           n_interactions=len(te), n_rows_prefilter=traw, role='test'))
    T = pd.DataFrame(rows)
    B = pd.DataFrame(blocks)
    T.to_csv(os.path.join(a.out, 'dataset_table.csv'), index=False)
    B.to_csv(os.path.join(a.out, 'split_blocks.csv'), index=False)
    pd.set_option('display.width', 200)
    print(T.T.to_string())
    print('\n[dataset stats] -> %s/dataset_table.csv, split_blocks.csv' % a.out)


if __name__ == '__main__':
    main()
