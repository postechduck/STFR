"""Amazon (Video Games / Movies & TV) -> data_raw/<dataset>/raw_interactions.csv

Two sources are supported.

1. Preprocessed 5-core interaction files with dense ids (the files used in the
   manuscript; tab-separated with columns user item rating timestamp, possibly
   split into train/val/test files).  All files are merged into the full
   interaction set; the chronological split is re-derived by prep/prep_split.py.

     python prep/adapters/amazon.py --dataset Amazon-VG --src_files DIR/train_data.csv DIR/val_data.csv DIR/test_data.csv

2. A raw review dump (csv with columns user, item, rating, timestamp in unix
   seconds; e.g. the per-category ratings-only files of the Amazon review data),
   filtered to 5-core and re-indexed densely.

     python prep/adapters/amazon.py --dataset Amazon-VG --src_raw ratings_Video_Games.csv --core 5
"""
import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, help='output name, e.g. Amazon-VG or Amazon-Movies')
p.add_argument('--src_files', nargs='*', default=None, help='preprocessed dense-id interaction files (tab-separated)')
p.add_argument('--src_raw', default=None, help='raw review csv (user,item,rating,timestamp)')
p.add_argument('--core', type=int, default=5, help='k-core threshold for --src_raw')
a = p.parse_args()

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(root, 'data_raw', a.dataset)
os.makedirs(out, exist_ok=True)

if a.src_files:
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
    raise SystemExit('give --src_files or --src_raw')

full.to_csv(os.path.join(out, 'raw_interactions.csv'), sep='\t', index=False)
print('[amazon] %s: %d interactions -> %s' % (a.dataset, len(full), out))
