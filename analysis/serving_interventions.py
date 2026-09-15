#!/usr/bin/env python3
"""Separate serving interventions on base checkpoints (Table E.2).

  python analysis/serving_interventions.py [--runs runs] [--out results] [--k 20]

Norm removal rescales every item vector to unit length (direction kept); direction
removal projects out the common popularity direction (normalized mean direction of the
top-1% stale items) and restores each vector's original length.  Test users are re-ranked
with the base scoring (inner product, observed history masked) under each intervention;
delta nALRP = original - intervened.  Embeddings are reconstructed as at serving (MF: item
table; LightGCN: mean of layers 0..3; SimGCL: mean of layers 1..3).  cos_pop and
nrm_ratio follow analysis/embedding_geometry.py and are computed on the SAME item
representation that the interventions act on.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import torch

from common import final_runs, read_recs, stale_signal, ROOT
from embedding_geometry import build_graph, embeddings, geometry


def seen_ledger(data):
    seen = {}
    with open(os.path.join(ROOT, 'data', data, 'trainval_list.txt')) as f:
        for line in f:
            p = line.split()
            if p:
                seen[int(p[0])] = np.array([int(x) for x in p[2:]], dtype=np.int64)
    return seen


def topk_nalrp(U, E, users, seen, d, k, batch=2048):
    tot = 0.0
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        sc = U[ub] @ E.T
        for r, u in enumerate(ub):
            sv = sc[r]
            m = seen.get(int(u))
            if m is not None and len(m):
                sv[m] = -np.inf
            tot += d[np.argpartition(-sv, k)[:k]].mean()
    return tot / len(users)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    p.add_argument('--k', type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    graphs, cache = {}, {}
    for r in final_runs(a.runs):
        if r['arm'] != 'base':
            continue
        recs = read_recs(r['run_dir'], a.k)
        ck = os.path.join(r['run_dir'], 'best.pth')
        if recs is None or not os.path.exists(ck):
            continue
        data, bb = r['data'], r['backbone']
        if data not in cache:
            d, c, n, _ = stale_signal(data)
            import json
            m = json.load(open(os.path.join(ROOT, 'data', data, 'dataset_meta.json')))
            cache[data] = (d, c, n, m['user_num'], seen_ledger(data))
        d, counts, n_item, n_user, seen = cache[data]
        Ah = None
        if bb != 'MF':
            if data not in graphs:
                graphs[data] = build_graph(data, n_user, n_item)
            Ah = graphs[data]
        sd = torch.load(ck, map_location='cpu')
        U, E = embeddings(sd, bb, n_user, Ah, include_ego=(bb != 'SimGCL'))
        geo = geometry(E, counts)
        users = np.array([u for u, _, _ in recs])
        nrm = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        Eh = E / nrm
        top = np.argsort(-counts)[:max(20, int(n_item * 0.01))]
        v = Eh[top].mean(0)
        v /= np.linalg.norm(v) + 1e-12
        resid = E - np.outer(E @ v, v)
        E_dir = resid / (np.linalg.norm(resid, axis=1, keepdims=True) + 1e-12) * nrm
        full = topk_nalrp(U, E, users, seen, d, a.k)
        rows.append(dict(data=data, backbone=bb, seed=r['seed'], n_users=len(users),
                         nalrp_lists=float(np.mean([d[rr].mean() for _, rr, _ in recs])),
                         nalrp_full=full, nalrp_norm_removed=topk_nalrp(U, Eh, users, seen, d, a.k),
                         nalrp_dir_removed=topk_nalrp(U, E_dir, users, seen, d, a.k),
                         cos_pop=geo['cos_pop'], nrm_ratio=geo['nrm_ratio']))
        rows[-1]['delta_norm'] = rows[-1]['nalrp_full'] - rows[-1]['nalrp_norm_removed']
        rows[-1]['delta_dir'] = rows[-1]['nalrp_full'] - rows[-1]['nalrp_dir_removed']
        print('[done] %s/%s s%d' % (data, bb, r['seed']), flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('no base runs with list dumps found')
    df.to_csv(os.path.join(a.out, 'serving_interventions_per_seed.csv'), index=False)
    s = df.groupby(['data', 'backbone'], sort=False).agg(
        n=('seed', 'count'), cos_pop=('cos_pop', 'mean'), nrm_ratio=('nrm_ratio', 'mean'),
        nalrp_full=('nalrp_full', 'mean'), delta_dir=('delta_dir', 'mean'), delta_norm=('delta_norm', 'mean'),
        reconstruction_gap=('nalrp_full', lambda v: float(np.mean(np.abs(v.values - df.loc[v.index, 'nalrp_lists'].values))))).reset_index()
    s.to_csv(os.path.join(a.out, 'serving_interventions.csv'), index=False)
    pd.set_option('display.width', 200)
    print(s.to_string(index=False, float_format=lambda v: '%.4f' % v))
    if len(s) >= 3:
        pr = stats.pearsonr(s.nrm_ratio, s.delta_norm)
        sr = stats.spearmanr(s.nrm_ratio, s.delta_norm)
        print('\nnrm_ratio vs delta_norm across settings: Pearson %.3f (p=%.3f), Spearman %.3f' % (pr[0], pr[1], sr.correlation))
    print('\n[serving interventions] -> %s/serving_interventions.csv' % a.out)


if __name__ == '__main__':
    main()
