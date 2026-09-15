#!/usr/bin/env python3
"""Embedding geometry of the top-1% most popular items (Section 7.5, Tables 12-13).

  python analysis/embedding_geometry.py [--runs runs] [--out results] [--arms base,PDA,TIDE,STFR]

For every final checkpoint: cos_pop = mean pairwise cosine over 4,000 random pairs of
items in the top 1% by cumulative training count (cos_rand: pairs from the whole
catalog); nrm_ratio = median norm of that group / median norm of the catalog;
r_dir = correlation between log(1 + c_i) and the projection of the unit vector on the
group's mean direction; r_nrm = correlation between log(1 + c_i) and the norm.

Item embeddings are the ones the serving score uses (the manuscript's convention):
MF the learned item table; LightGCN the mean of layers 0..3 (layer 0 = the learned
table before propagation, layer l = l graph propagations); SimGCL the mean of layers
1..3, as in its reference encoder.  Graph embeddings are reconstructed from the
training graph without SimGCL's training-time noise.  --simgcl_include_layer0 is a
legacy option (mean of layers 0..3 for SimGCL as well) that is NOT the manuscript's
convention.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from common import final_runs, ROOT

N_LAYERS = 3


def build_graph(data, n_user, n_item):
    rows, cols = [], []
    with open(os.path.join(ROOT, 'data', data, 'train_list.txt')) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            u = int(p[0])
            for it in p[2:]:
                rows.append(u)
                cols.append(int(it))
    n = n_user + n_item
    A = sp.coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols) + n_user)), shape=(n, n))
    A = (A + A.T).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    return (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()


def embeddings(sd, backbone, n_user, Ah, include_ego):
    """(user, item) embeddings as float64 arrays."""
    if backbone == 'MF':
        return sd['embed_user.weight'].numpy().astype(np.float64), sd['embed_item.weight'].numpy().astype(np.float64)
    e0 = np.vstack([sd['embed_user_0.weight'].numpy(), sd['embed_item_0.weight'].numpy()]).astype(np.float64)
    embs = [e0] if include_ego else []
    cur = e0
    for _ in range(N_LAYERS):
        cur = Ah @ cur
        embs.append(cur)
    out = np.mean(np.stack(embs, 1), 1)
    return out[:n_user], out[n_user:]


def geometry(E, counts, seed=0):
    rng = np.random.default_rng(seed)
    nrm = np.linalg.norm(E, axis=1) + 1e-12
    Eh = E / nrm[:, None]
    n = E.shape[0]
    top = np.argsort(-counts)[:max(20, int(n * 0.01))]

    def pair_cos(idx, n_pairs=4000):
        a = Eh[rng.choice(idx, n_pairs)]
        b = Eh[rng.choice(idx, n_pairs)]
        keep = (a != b).any(1)
        return float((a[keep] * b[keep]).sum(1).mean())

    v = Eh[top].mean(0)
    v /= np.linalg.norm(v) + 1e-12
    return dict(cos_pop=pair_cos(top), cos_rand=pair_cos(np.arange(n)),
                r_dir=float(np.corrcoef(np.log1p(counts), Eh @ v)[0, 1]),
                r_nrm=float(np.corrcoef(np.log1p(counts), nrm)[0, 1]),
                nrm_ratio=float(np.median(nrm[top]) / np.median(nrm)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    p.add_argument('--arms', default='base,PDA,TIDE,STFR')
    p.add_argument('--simgcl_include_layer0', action='store_true',
                   help='legacy: average layers 0..3 for SimGCL too (not the manuscript convention)')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arms = a.arms.split(',')
    rows = []
    graphs, meta = {}, {}
    for r in final_runs(a.runs):
        if r['arm'] not in arms:
            continue
        ck = os.path.join(r['run_dir'], 'best.pth')
        if not os.path.exists(ck):
            continue
        data = r['data']
        if data not in meta:
            m = json.load(open(os.path.join(ROOT, 'data', data, 'dataset_meta.json')))
            c = pd.read_csv(os.path.join(ROOT, 'data', data, 'item_frequency.csv'))['count'].values.astype(np.float64)
            c = np.concatenate([c, np.zeros(max(0, m['item_num'] - len(c)))])[:m['item_num']]
            meta[data] = (m['user_num'], m['item_num'], c)
        n_user, n_item, counts = meta[data]
        Ah = None
        if r['backbone'] != 'MF':
            if data not in graphs:
                graphs[data] = build_graph(data, n_user, n_item)
            Ah = graphs[data]
        sd = torch.load(ck, map_location='cpu')
        include_ego = (r['backbone'] != 'SimGCL') or a.simgcl_include_layer0
        _, E = embeddings(sd, r['backbone'], n_user, Ah, include_ego)
        rows.append(dict(data=data, backbone=r['backbone'], arm=r['arm'], seed=r['seed'], **geometry(E, counts)))
        print('[done] %s/%s %s s%d' % (data, r['backbone'], r['arm'], r['seed']), flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('no checkpoints found')
    df.to_csv(os.path.join(a.out, 'embedding_geometry_per_seed.csv'), index=False)
    s = df.groupby(['data', 'backbone', 'arm'], sort=False).agg(
        n=('seed', 'count'), cos_pop=('cos_pop', 'mean'), cos_rand=('cos_rand', 'mean'),
        nrm_ratio=('nrm_ratio', 'mean'), r_dir=('r_dir', 'mean'), r_nrm=('r_nrm', 'mean')).reset_index()
    s.to_csv(os.path.join(a.out, 'embedding_geometry.csv'), index=False)
    pd.set_option('display.width', 200)
    print(s.to_string(index=False, float_format=lambda v: '%.3f' % v))
    print('\n[geometry] -> %s/embedding_geometry.csv' % a.out)


if __name__ == '__main__':
    main()
