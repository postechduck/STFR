#!/usr/bin/env python3
"""Non-personalized cumulative vs previous-block popularity rankings (Section 4.1, Table 1).

  python analysis/popularity_signals.py [--datasets Amazon-VG,Amazon-Movies,Douban-movie] [--out results]

Every user receives the same item scores -- cumulative training counts (stale) or the
counts of the last completed training block t*-1 (fresh) -- and is evaluated with the
test protocol of stfr/evaluate.py (train+validation history masked, targets restricted
to the training-active catalog).  Popularity persistence rho is the Spearman correlation
between cumulative training counts and test-window counts over items with at least one
training interaction.  Base-model rows are read from results/summary.csv when present.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT


def load(ds):
    root = os.path.join(ROOT, 'data', ds)
    meta = json.load(open(os.path.join(root, 'dataset_meta.json')))
    t, n = meta['t_star'], meta['item_num']
    f = pd.read_csv(os.path.join(root, 'item_frequency_all_times.csv'), index_col=0).fillna(0)
    freq = np.zeros((n, f.shape[1]))
    freq[:f.shape[0], :] = f.values
    test = pd.read_csv(os.path.join(root, 'test_data.csv'), sep='\t')
    tc = np.bincount(test['item'].values.astype(np.int64), minlength=n)[:n]
    cnt = pd.read_csv(os.path.join(root, 'item_frequency.csv'))['count'].values
    universe = set(np.flatnonzero(cnt > 0).tolist())
    seen = {}
    with open(os.path.join(root, 'trainval_list.txt')) as fh:
        for l in fh:
            p = l.strip().split('\t')
            if len(p) >= 2:
                seen[int(p[0])] = np.array([int(x) for x in p[2:]], dtype=np.int64)
    targets = {}
    with open(os.path.join(root, 'test_list.txt')) as fh:
        for l in fh:
            p = l.strip().split('\t')
            if len(p) < 2:
                continue
            uid = int(p[0])
            hist = set(seen.get(uid, ()).tolist())
            keep, s = [], set()
            for it in map(int, p[1:]):
                if it in s or it in hist or it not in universe:
                    s.add(it)
                    continue
                s.add(it)
                keep.append(it)
            if keep:
                targets[uid] = keep
    stale = freq[:, :t].sum(1)
    fresh = freq[:, t - 1]
    return n, stale, fresh, tc, cnt, seen, targets


def mostpop(pop, seen, targets, users, K, batch=2048):
    pop = pop.astype(np.float32)
    log2inv = 1.0 / np.log2(np.arange(2, K + 2))
    idcg_cum = np.cumsum(log2inv)
    rec, ndcg = [], []
    for b0 in range(0, len(users), batch):
        ub = users[b0:b0 + batch]
        rate = np.tile(pop, (len(ub), 1))
        for i, u in enumerate(ub):
            s = seen.get(u)
            if s is not None and s.size:
                rate[i, s] = -np.inf
        top = np.argpartition(rate, -K, axis=1)[:, -K:]
        for i, u in enumerate(ub):
            row = top[i][np.argsort(rate[i, top[i]])[::-1]]
            tset = set(targets[u])
            hit, dcg = 0, 0.0
            for r, it in enumerate(row, start=1):
                if it in tset:
                    hit += 1
                    dcg += log2inv[r - 1]
            rec.append(hit / len(targets[u]))
            ndcg.append(dcg / idcg_cum[min(K, len(targets[u])) - 1])
    return float(np.mean(rec)), float(np.mean(ndcg))


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
        n, stale, fresh, tc, cnt, seen, targets = load(ds)
        users = sorted(targets)
        active = cnt > 0
        rho = stats.spearmanr(stale[active], tc[active]).correlation
        r = dict(dataset=ds, n_users=len(users), rho_persistence=float(rho))
        for K in (20,):
            rs, ns = mostpop(stale, seen, targets, users, K)
            rf, nf = mostpop(fresh, seen, targets, users, K)
            r.update({'stale_recall@%d' % K: rs, 'stale_ndcg@%d' % K: ns, 'fresh_recall@%d' % K: rf,
                      'fresh_ndcg@%d' % K: nf, 'fresh_over_stale_recall@%d' % K: rf / rs if rs else np.nan})
            if summ is not None:
                for bb in ('MF', 'LightGCN'):
                    q = summ[(summ.data == ds) & (summ.backbone == bb) & (summ.arm == 'base') & (summ.k == K)]
                    if len(q):
                        r['base_%s_recall@%d' % (bb, K)] = float(q.recall.iloc[0])
        rows.append(r)
        print('[done] %s: rho=%.3f' % (ds, rho), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'popularity_signals.csv'), index=False)
    pd.set_option('display.width', 220)
    print(df.T.to_string())
    print('\n[popularity signals] -> %s/popularity_signals.csv' % a.out)


if __name__ == '__main__':
    main()
