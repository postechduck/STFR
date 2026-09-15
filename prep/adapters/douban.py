"""Douban-Movie -> data_raw/Douban-movie/raw_interactions.csv

Source: the socialRec DoubanMovie dump (github DeepGraphLearning/RecommenderSystems,
Douban.tar.gz -> movie/douban_movie.tsv with columns UserId ItemId Rating Timestamp).
Preprocessing: keep interactions after 2010-01-01, then iterative 10-core filtering
(48,799 users / 26,813 items), dense re-indexing of users and items.

  python prep/adapters/douban.py --tsv data_raw/Douban/movie/douban_movie.tsv
"""
import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--tsv', required=True, help='douban_movie.tsv of the socialRec dump')
p.add_argument('--user_core', type=int, default=10)
p.add_argument('--item_core', type=int, default=10)
p.add_argument('--no_cut2010', action='store_true', help='keep interactions before 2010-01-01')
a = p.parse_args()

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(root, 'data_raw', 'Douban-movie')
os.makedirs(out, exist_ok=True)

df = pd.read_csv(a.tsv, sep='\t')
df.columns = ['user_key', 'item_key', 'rating', 'timestamp']
df['timestamp'] = df['timestamp'].astype(np.int64)
if not a.no_cut2010:
    cut = int(pd.Timestamp('2010-01-01').timestamp())
    df = df[df['timestamp'] > cut]
    print('[douban] after the 2010 cut: %d interactions' % len(df))

while True:  # iterative k-core
    uc = df['user_key'].value_counts()
    ic = df['item_key'].value_counts()
    keep = df['user_key'].isin(uc[uc >= a.user_core].index) & df['item_key'].isin(ic[ic >= a.item_core].index)
    if keep.all():
        break
    df = df[keep]
print('[douban] after %d/%d-core: %d interactions, %d users, %d items'
      % (a.user_core, a.item_core, len(df), df.user_key.nunique(), df.item_key.nunique()))

uid = {u: i for i, u in enumerate(sorted(df['user_key'].unique()))}
iid = {m: i for i, m in enumerate(sorted(df['item_key'].unique()))}
pd.DataFrame({'user': df['user_key'].map(uid).astype(int),
              'item': df['item_key'].map(iid).astype(int),
              'rating': df['rating'].astype(float),
              'timestamp': df['timestamp'].astype(np.int64)}) \
    .to_csv(os.path.join(out, 'raw_interactions.csv'), sep='\t', index=False)
print('[douban] done ->', out)
