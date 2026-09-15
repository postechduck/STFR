"""Chronological block split and popularity tables (the manuscript's evaluation protocol).

  python prep/prep_split.py --dataset Amazon-VG
  python prep/prep_split.py --dataset Amazon-VG --block_days 120 --eval_block 46 --out_name Amazon-VG_b120

Protocol
  1. One interaction per (user, item): the first one is kept (first-interaction prediction).
  2. The timeline is cut into fixed-width blocks of --block_days days, starting at the
     earliest interaction.
  3. Evaluation block t* = the most active block among those preceded by at least
     --min_past (default one half) of all interactions.  --eval_block forces the index
     instead (block-configuration study: same evaluation start for every block length).
  4. Inside t*: the first --val_days days are validation, the rest is test.  Training =
     all interactions before t*; blocks after t* are discarded.
  5. Validation targets need training history; test targets need training or
     validation history; targets are restricted to items with training interactions.
     Users are ranked against the full catalog with their observed history masked.
  6. Popularity tables use training interactions only.

Input   data_raw/<dataset>/raw_interactions.csv   (tab-separated: user item rating timestamp; dense ids)
Output  data/<out_name>/  (split files, interaction lists, popularity tables, dataset_meta.json)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DAY = 86400


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, help='name of data_raw/<dataset>')
    p.add_argument('--out_name', default=None, help='name of data/<out_name> (default: dataset)')
    p.add_argument('--block_days', type=int, default=60)
    p.add_argument('--val_days', type=int, default=None, help='default: block_days // 2')
    p.add_argument('--min_past', type=float, default=0.5,
                   help='minimum share of interactions before the evaluation block')
    p.add_argument('--eval_block', type=int, default=-1, help='force the evaluation block index')
    p.add_argument('--pop_quantile', type=float, default=0.8,
                   help='within-block high-popularity quantile (CausalEPP user sensitivity)')
    p.add_argument('--no_dedupe', action='store_true', help='keep repeated (user, item) pairs')
    return p.parse_args()


def write_lines(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def write_history_list(df, path):
    """uid \\t n \\t item ... (chronological)."""
    lines = []
    for uid, g in df.sort_values('timestamp').groupby('user', sort=True):
        items = g['item'].astype(int).tolist()
        lines.append('%d\t%d\t%s' % (int(uid), len(items), '\t'.join(map(str, items))))
    write_lines(path, lines)


def write_target_list(df, path):
    """uid \\t item ... (chronological)."""
    lines = []
    for uid, g in df.sort_values('timestamp').groupby('user', sort=True):
        lines.append('%d\t%s' % (int(uid), '\t'.join(map(str, g['item'].astype(int).tolist()))))
    write_lines(path, lines)


def main():
    a = parse()
    val_days = a.val_days if a.val_days is not None else a.block_days // 2
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'data_raw', a.dataset, 'raw_interactions.csv')
    out_name = a.out_name or a.dataset
    out = os.path.join(root, 'data', out_name)
    os.makedirs(out, exist_ok=True)
    W = a.block_days * DAY
    print('[prep] %s -> %s  block_days=%d val_days=%d' % (a.dataset, out, a.block_days, val_days))

    full = pd.read_csv(src, sep='\t')
    full['timestamp'] = full['timestamp'].astype(np.int64)
    item_num = int(full['item'].max()) + 1
    user_num = int(full['user'].max()) + 1
    print('[prep] interactions=%d item_num=%d user_num=%d' % (len(full), item_num, user_num))
    if not a.no_dedupe:
        n0 = len(full)
        full = full.sort_values('timestamp', kind='stable') \
                   .drop_duplicates(subset=['user', 'item'], keep='first').reset_index(drop=True)
        print('[prep] first interaction per (user, item): %d -> %d rows' % (n0, len(full)))

    # fixed-width block grid; all_times[k] is the right edge of block k
    t0, t1 = int(full['timestamp'].min()), int(full['timestamp'].max())
    N = int(np.ceil((t1 - t0 + 1) / W))
    all_times = (t0 + (np.arange(N) + 1) * W).astype(np.int64)
    block = np.clip(np.searchsorted(all_times, full['timestamp'].values), 0, N - 1)
    counts = np.bincount(block, minlength=N)
    past_frac = np.concatenate([[0.0], np.cumsum(counts)[:-1] / counts.sum()])
    if a.eval_block >= 0:
        t_star, rule = a.eval_block, 'forced'
    else:
        eligible = past_frac >= a.min_past
        if not eligible.any():
            raise SystemExit('no block is preceded by %.0f%% of the interactions' % (100 * a.min_past))
        t_star, rule = int(np.where(eligible, counts, -1).argmax()), 'most_active_with_past_share'
    print('[prep] blocks=%d evaluation block t*=%d (%d rows, %.1f%% of interactions before it, rule=%s)'
          % (N, t_star, counts[t_star], 100 * past_frac[t_star], rule))

    eval_df = full[block == t_star].copy()
    train_df = full[block < t_star].copy()
    train_df['split_idx'] = block[block < t_star]
    eval_start = int(all_times[t_star] - W)
    val_cut = eval_start + val_days * DAY
    val_df = eval_df[eval_df['timestamp'] < val_cut].copy()
    test_df = eval_df[eval_df['timestamp'] >= val_cut].copy()
    val_df['split_idx'] = t_star
    test_df['split_idx'] = t_star
    print('[prep] discarding %d rows after the evaluation block' % int((block > t_star).sum()))

    # target eligibility: users with observed history, items with training interactions
    train_users = set(train_df['user'].unique())
    val_df = val_df[val_df['user'].isin(train_users)].copy()
    trainval_users = train_users | set(val_df['user'].unique())
    test_df = test_df[test_df['user'].isin(trainval_users)].copy()
    train_items = set(train_df['item'].unique())
    vu0, tu0 = val_df['user'].nunique(), test_df['user'].nunique()
    val_df = val_df[val_df['item'].isin(train_items)].copy()
    test_df = test_df[test_df['item'].isin(train_items)].copy()
    n_val_users_empty = vu0 - val_df['user'].nunique()
    n_test_users_empty = tu0 - test_df['user'].nunique()
    print('[prep] train=%d val=%d test=%d (val users=%d, test users=%d)' % (
        len(train_df), len(val_df), len(test_df), val_df['user'].nunique(), test_df['user'].nunique()))

    cols = ['user', 'item', 'rating', 'timestamp', 'split_idx']
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    for d, nm in [(train_df, 'train_data.csv'), (val_df, 'val_data.csv'),
                  (test_df, 'test_data.csv'), (trainval_df, 'trainval_data.csv')]:
        d[cols].to_csv(os.path.join(out, nm), sep='\t', index=False)
    write_history_list(train_df, os.path.join(out, 'train_list.txt'))
    write_history_list(trainval_df, os.path.join(out, 'trainval_list.txt'))
    write_target_list(val_df, os.path.join(out, 'val_list.txt'))
    write_target_list(test_df, os.path.join(out, 'test_list.txt'))

    # invariants: targets are unique per user and disjoint from the masked history
    for nm, edf, hist in (('val', val_df, train_df), ('test', test_df, trainval_df)):
        assert int(edf.duplicated(['user', 'item']).sum()) == 0, nm
        overlap = len(edf.merge(hist[['user', 'item']].drop_duplicates(), on=['user', 'item'], how='inner'))
        assert overlap == 0, nm

    # ---- popularity tables (training interactions only) ----
    np.save(os.path.join(out, 'all_times.npy'), all_times)
    item_cum = train_df['item'].value_counts().reindex(range(item_num)).fillna(0).astype(int)
    item_cum.name = 'count'
    item_cum.to_frame().to_csv(os.path.join(out, 'item_frequency.csv'), index=False)
    d = np.log1p(item_cum.values.astype(np.float64))
    np.save(os.path.join(out, 'DICE_popularity.npy'), (d - d.min()) / (d.max() - d.min() + 1e-12))

    train_block = train_df['split_idx'].values.astype(int)
    item_ft, user_ft, user_hi_ft = {}, {}, {}
    for k in range(N):
        sub = train_df[train_block == k]
        item_ft[k] = sub['item'].value_counts()
        user_ft[k] = sub['user'].value_counts()
        if len(sub):
            ifreq = sub['item'].value_counts()
            high_items = ifreq[ifreq > ifreq.quantile(a.pop_quantile)].index
            user_hi_ft[k] = sub[sub['item'].isin(high_items)]['user'].value_counts()
        else:
            user_hi_ft[k] = pd.Series(dtype=float)
    item_ft = pd.DataFrame(item_ft).reindex(range(item_num)).sort_index()
    user_ft = pd.DataFrame(user_ft).reindex(range(user_num)).sort_index()
    user_hi_ft = pd.DataFrame(user_hi_ft).reindex(range(user_num)).sort_index()
    item_ft.to_csv(os.path.join(out, 'item_frequency_all_times.csv'), index_label='item')
    user_ft.to_csv(os.path.join(out, 'user_frequency_all_times.csv'), index_label='user')
    user_hi_ft.to_csv(os.path.join(out, 'user_frequency_high_popularity_all_times.csv'), index_label='user')

    # PDA per-block popularity: Laplace smoothing, then min-max within the block
    raw_cnt = item_ft.fillna(0).values.astype(np.float64)
    pda = np.zeros_like(raw_cnt)
    for k in range(raw_cnt.shape[1]):
        col = (raw_cnt[:, k] + 1.0) / (raw_cnt[:, k].sum() + item_num)
        pda[:, k] = (col - col.min()) / (col.max() - col.min() + 1e-12)
    pd.DataFrame(pda, columns=[str(k) for k in range(raw_cnt.shape[1])]) \
        .to_csv(os.path.join(out, 'PDA_popularity.csv'), sep='\t', index=False)

    # item_interactions.csv (TIDE / CausalEPP time-decayed popularity): "idx,count,ts1,ts2,..."
    ts_by_item = {i: [] for i in range(item_num)}
    tr_sorted = train_df.sort_values('timestamp')
    for it, ts in zip(tr_sorted['item'].astype(int).values, tr_sorted['timestamp'].astype(int).values):
        ts_by_item[it].append(ts)
    lines = ['%d,%d%s' % (i, len(ts_by_item[i]), (',' + ','.join(map(str, ts_by_item[i]))) if ts_by_item[i] else '')
             for i in range(item_num)]
    write_lines(os.path.join(out, 'item_interactions.csv'), lines)

    meta = dict(dataset=out_name, source=a.dataset, block_days=a.block_days, val_days=val_days,
                n_blocks=N, t_star=t_star, anchor_block=max(t_star - 1, 0), split_rule=rule,
                min_past=a.min_past, dedupe=not a.no_dedupe, item_num=item_num, user_num=user_num,
                n_train=len(train_df), n_val=len(val_df), n_test=len(test_df),
                n_val_users=int(val_df['user'].nunique()), n_test_users=int(test_df['user'].nunique()),
                n_val_users_empty=int(n_val_users_empty), n_test_users_empty=int(n_test_users_empty),
                eval_start=eval_start, val_cut=val_cut, eval_end=int(all_times[t_star]))
    with open(os.path.join(out, 'dataset_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('[prep] meta:', json.dumps(meta))
    print('[prep] done.')


if __name__ == '__main__':
    main()
