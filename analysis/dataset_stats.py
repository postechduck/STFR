#!/usr/bin/env python3
"""Dataset statistics (Table 3) and per-block interaction counts of the split figure (Figure 2).

  python analysis/dataset_stats.py [--datasets Amazon-VG,Amazon-Movies,Douban-movie] [--out results]

Reads data/<dataset>/ (written by prep/prep_split.py) and, for the Figure 2 rows of the
evaluation halves and of the blocks after the evaluation block, the full interaction
file data_raw/<dataset>/raw_interactions.csv:

  role val / test   n_interactions = evaluation targets, n_rows_prefilter = interactions of
                    the window before the user / item target filters (the height drawn)
  role after_eval   60-day blocks after the evaluation block.  They are counted for the
                    figure only; training, model selection and evaluation never read them.

Without the full interaction file these counts cannot be computed: the rows are written
with empty n_rows_prefilter and no after_eval rows, and plot_split.py refuses to draw.
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
        vraw = traw = None
        after = []
        if os.path.exists(raw):
            full = pd.read_csv(raw, sep='\t', usecols=['user', 'item', 'timestamp'])
            full['timestamp'] = full['timestamp'].astype(np.int64)
            if m.get('dedupe', True):
                full = full.sort_values('timestamp', kind='stable').drop_duplicates(['user', 'item'], keep='first')
            ts = full.timestamp.values
            win = ts[(ts > at[t_star] - W) & (ts <= at[t_star])]
            vraw = int((win < val_cut).sum())
            traw = int((win >= val_cut).sum())
            k, edge = t_star + 1, int(at[t_star])
            while edge < ts.max():
                after.append((k, edge, int(((ts > edge) & (ts <= edge + W)).sum())))
                k, edge = k + 1, edge + W
        else:
            print('[dataset stats] %s: %s not found -- pre-filter window counts and the blocks after the '
                  'evaluation block are left empty (Figure 2 needs them)' % (ds, raw))
        blocks.append(dict(dataset=ds, block_index=t_star, block_start_date=date(at[t_star] - W), block_end_date=date(val_cut - 1),
                           n_interactions=len(va), n_rows_prefilter=vraw, role='val'))
        blocks.append(dict(dataset=ds, block_index=t_star, block_start_date=date(val_cut), block_end_date=date(at[t_star] - 1),
                           n_interactions=len(te), n_rows_prefilter=traw, role='test'))
        for k, edge, n in after:
            blocks.append(dict(dataset=ds, block_index=k, block_start_date=date(edge), block_end_date=date(edge + W - 1),
                               n_interactions=n, n_rows_prefilter=n, role='after_eval'))
    T = pd.DataFrame(rows)
    B = pd.DataFrame(blocks)
    B['n_rows_prefilter'] = B['n_rows_prefilter'].astype('Int64')
    T.to_csv(os.path.join(a.out, 'dataset_table.csv'), index=False)
    B.to_csv(os.path.join(a.out, 'split_blocks.csv'), index=False)
    pd.set_option('display.width', 200)
    print(T.T.to_string())
    print('\n[dataset stats] -> %s/dataset_table.csv, split_blocks.csv' % a.out)


if __name__ == '__main__':
    main()
