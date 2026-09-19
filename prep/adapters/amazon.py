"""Amazon (Video Games / Movies & TV) -> data_raw/<dataset>/raw_interactions.csv

Three sources are supported.

1. Files of the Amazon Review Data (2018) release (Ni et al., 2019;
   https://nijianmo.github.io/amazon/).  The manuscript's inputs are
     Amazon-VG      Video_Games_5.json.gz   5-core review file, 497,577 reviews
     Amazon-Movies  Movies_and_TV.csv       ratings-only file, 8,765,568 ratings
   followed by the manuscript's preprocessing (docs/data_and_evaluation.md): the inherited
   sequential 5-core filter, then dense ids in the sorted order of the original
   reviewerID / asin strings (written to user_ids.csv / item_ids.csv).  Expected output:
   496,904 interactions / 55,144 users / 17,286 items and 3,408,612 / 297,377 / 59,925.

     python prep/adapters/amazon.py --dataset Amazon-VG --src_reviews Video_Games_5.json.gz
     python prep/adapters/amazon.py --dataset Amazon-Movies --src_reviews Movies_and_TV.csv

   A .json / .json.gz file is read as review records (reviewerID, asin, overall,
   unixReviewTime); a csv uses --csv_columns (default: the release's item,user,rating,timestamp).

2. Already preprocessed interaction files with dense ids (tab-separated, columns user
   item rating timestamp, possibly split into several files); they are merged as they are.

     python prep/adapters/amazon.py --dataset Amazon-VG --src_files DIR/train_data.csv DIR/val_data.csv DIR/test_data.csv

3. Any other ratings csv (user,item,rating,timestamp), filtered to an iterative k-core
   after keeping one interaction per (user, item).  This does not reproduce the
   manuscript's ids or item set.

     python prep/adapters/amazon.py --dataset Amazon-VG --src_raw ratings_Video_Games.csv --core 5
"""
import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, help='output name, e.g. Amazon-VG or Amazon-Movies')
p.add_argument('--src_reviews', default=None, help='Amazon Review Data (2018) review file (.json / .json.gz) or ratings-only csv')
p.add_argument('--csv_columns', default='item,user,rating,timestamp', help='column order of a --src_reviews csv (no header)')
p.add_argument('--src_files', nargs='*', default=None, help='preprocessed dense-id interaction files (tab-separated)')
p.add_argument('--src_raw', default=None, help='raw review csv (user,item,rating,timestamp)')
p.add_argument('--core', type=int, default=5, help='k-core threshold for --src_reviews / --src_raw')
a = p.parse_args()

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(root, 'data_raw', a.dataset)
os.makedirs(out, exist_ok=True)



def read_reviews(path):
    if path.endswith(('.json', '.json.gz')):
        df = pd.read_json(path, lines=True, dtype={'reviewerID': str, 'asin': str})
        df = df.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating', 'unixReviewTime': 'timestamp'})
    else:
        df = pd.read_csv(path, header=None, names=a.csv_columns.split(','), dtype={'user': str, 'item': str})
    df = df[['user', 'item', 'rating', 'timestamp']].copy()
    df['rating'] = df['rating'].astype(float)
    df['timestamp'] = df['timestamp'].astype(np.int64)
    return df


def inherited_core_filter(df, core):
    """The 5-core filter of the code base the experiments started from, applied to the review
    rows as they are (repeated user-item reviews count).  Each round keeps the users with at
    least `core` reviews of still-kept items and then the items whose review count among
    the previous round's users is at least `core`, until a round removes nothing."""
    u, i = df['user'].values, df['item'].values
    keep_u = np.ones(len(df), bool)       # row's user kept
    keep_i = np.ones(len(df), bool)       # row's item kept
    item_rows = np.ones(len(df), bool)    # rows counted for an item: users kept before this round's user step
    while True:
        cnt_u = pd.Series(u[keep_u & keep_i]).value_counts()
        new_u = keep_u & pd.Series(u).map(cnt_u).fillna(0).values.astype(int).__ge__(core)
        cnt_i = pd.Series(i[item_rows & keep_i]).value_counts()
        new_i = keep_i & pd.Series(i).map(cnt_i).fillna(0).values.astype(int).__ge__(core)
        done = (new_u == keep_u).all() and (new_i == keep_i).all()
        item_rows, keep_u, keep_i = new_u.copy(), new_u, new_i
        if done:
            return df[keep_u & keep_i]


if a.src_reviews:
    df = inherited_core_filter(read_reviews(a.src_reviews), a.core)
    users, items = np.unique(df['user'].values.astype(str)), np.unique(df['item'].values.astype(str))
    full = pd.DataFrame({'user': np.searchsorted(users, df['user'].values.astype(str)),
                         'item': np.searchsorted(items, df['item'].values.astype(str)),
                         'rating': df['rating'].values, 'timestamp': df['timestamp'].values})
    full = full.sort_values(['timestamp', 'user', 'item', 'rating'], kind='stable').reset_index(drop=True)
    pd.DataFrame({'id': np.arange(len(users)), 'reviewerID': users}).to_csv(os.path.join(out, 'user_ids.csv'), index=False)
    pd.DataFrame({'id': np.arange(len(items)), 'asin': items}).to_csv(os.path.join(out, 'item_ids.csv'), index=False)
    print('[amazon] %d-core: %d reviews, %d users, %d items (id maps: user_ids.csv, item_ids.csv)' % (
        a.core, len(full), len(users), len(items)))
elif a.src_files:
    parts = [pd.read_csv(f, sep='\t') for f in a.src_files]
    full = pd.concat(parts, ignore_index=True)[['user', 'item', 'rating', 'timestamp']]
elif a.src_raw:
    df = pd.read_csv(a.src_raw, header=None, names=['user', 'item', 'rating', 'timestamp'])
    if str(df.iloc[0]['timestamp']).lower() == 'timestamp':   # header present
        df = df.iloc[1:]
    df['timestamp'] = df['timestamp'].astype(np.int64)
    df['rating'] = df['rating'].astype(float)
    n0 = len(df)
    df = df.sort_values('timestamp', kind='stable').drop_duplicates(['user', 'item'], keep='first')
    print('[amazon] one interaction per (user, item): %d -> %d' % (n0, len(df)))
    while True:  # iterative k-core
        uc = df['user'].value_counts()
        ic = df['item'].value_counts()
        keep = df['user'].isin(uc[uc >= a.core].index) & df['item'].isin(ic[ic >= a.core].index)
        if keep.all():
            break
        df = df[keep]
    uid = {u: i for i, u in enumerate(sorted(df['user'].unique()))}
    iid = {m: i for i, m in enumerate(sorted(df['item'].unique()))}
    full = pd.DataFrame({'user': df['user'].map(uid).astype(int), 'item': df['item'].map(iid).astype(int),
                         'rating': df['rating'], 'timestamp': df['timestamp']})
    print('[amazon] after %d-core: %d interactions, %d users, %d items' % (
        a.core, len(full), full.user.nunique(), full.item.nunique()))
else:
    raise SystemExit('give --src_reviews, --src_files or --src_raw')

full.to_csv(os.path.join(out, 'raw_interactions.csv'), sep='\t', index=False)
print('[amazon] %s: %d interactions -> %s' % (a.dataset, len(full), out))
