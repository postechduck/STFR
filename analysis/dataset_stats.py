#!/usr/bin/env python3
"""Dataset statistics and per-block interaction counts of the chronological split (plot_split.py).

  python analysis/dataset_stats.py [--datasets Amazon-VG,Amazon-Movies,Douban-movie] [--out results]

Reads data/<dataset>/ (written by prep/prep_split.py) and, for the split-figure rows of the
evaluation halves and of the blocks after the evaluation block, the full interaction
file data_raw/<dataset>/raw_interactions.csv:

  role val / test   n_interactions = evaluation targets, n_rows_prefilter = interactions of
                    the window before the user / item target filters (the height drawn)
  role after_eval   60-day blocks after the evaluation block.  They are counted for the
                    figure only; training, model selection and evaluation never read them.

Without the full interaction file these counts cannot be computed: the rows are written
with empty n_rows_prefilter and no after_eval rows, and plot_split.py refuses to draw.

dataset_table.csv also gives the local calendar dates and day counts of the validation and
test windows: blocks are (left edge, right edge] and timestamps have daily
resolution at local midnight, so with a 30-day validation setting the validation window
covers 29 days and the test window 31 days.
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


def local_offset(ts):
    """Timestamps have daily resolution at local midnight (Amazon UTC, Douban-Movie UTC+8):
    the offset (seconds) that maps every timestamp to a UTC midnight."""
    r = np.unique(np.asarray(ts, dtype=np.int64) % DAY)
    if len(r) != 1:
        return None
    return int((DAY - r[0]) % DAY)


def calendar_windows(eval_start, val_cut, eval_end, off):
    """Local calendar dates of the two windows.  Blocks are (left edge, right edge] and the
    edges fall on local midnights, so the evaluation block holds the days after its start
    date up to and including its end date; validation is the days before the cut date and
    test the days from the cut date to the end date."""
    day = lambda t: pd.Timestamp(int(t) + off, unit='s').normalize()
    v0, v1 = day(eval_start) + pd.Timedelta(days=1), day(val_cut) - pd.Timedelta(days=1)
    t0, t1 = day(val_cut), day(eval_end)
    f = lambda x: x.strftime('%Y-%m-%d')
    return dict(val_first_day=f(v0), val_last_day=f(v1), val_calendar_days=(v1 - v0).days + 1,
                test_first_day=f(t0), test_last_day=f(t1), test_calendar_days=(t1 - t0).days + 1,
                local_utc_offset_hours=off / 3600.0)


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
        off = local_offset(np.concatenate([tr.timestamp.values, va.timestamp.values, te.timestamp.values]))
        loc = (lambda t: date(int(t) + off)) if off is not None else date
        nu, ni = tr.user.nunique(), tr.item.nunique()
        rows.append(dict(dataset=ds, users=m['user_num'], items=m['item_num'], train_interactions=len(tr),
                         training_active_users=nu, training_active_items=ni,
                         training_density=len(tr) / (nu * ni), training_blocks=t_star,
                         val_users=va.user.nunique(), val_targets=len(va), test_users=te.user.nunique(), test_targets=len(te),
                         first_training_day=loc(tr.timestamp.min()), last_training_day=loc(tr.timestamp.max()),
                         eval_block_start_utc=date(at[t_star] - W),
                         val_test_cut_utc=date(m.get('val_cut', at[t_star] - W + m['val_days'] * DAY)),
                         eval_block_end_utc=date(at[t_star])))
        if off is not None:
            rows[-1].update(calendar_windows(at[t_star] - W, m.get('val_cut', at[t_star] - W + m['val_days'] * DAY),
                                             at[t_star], off))
            # the windows hold exactly the interactions dated inside them
            dv = (va.timestamp.values + off) // DAY
            dt = (te.timestamp.values + off) // DAY
            to_day = lambda x: (pd.Timestamp(x) - pd.Timestamp('1970-01-01')).days
            assert dv.min() >= to_day(rows[-1]['val_first_day']) and dv.max() <= to_day(rows[-1]['val_last_day'])
            assert dt.min() >= to_day(rows[-1]['test_first_day']) and dt.max() <= to_day(rows[-1]['test_last_day'])
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
                  'evaluation block are left empty (plot_split.py needs them)' % (ds, raw))
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
