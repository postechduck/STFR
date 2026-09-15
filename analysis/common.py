"""Shared helpers of the analysis scripts (run layout of scripts/run_cell.py)."""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_ARMS = ['base', 'IPS', 'DICE', 'DDC', 'PDA', 'TIDE', 'CausalEPP', 'STFR']
BASELINES = ['IPS', 'DICE', 'DDC', 'PDA', 'TIDE', 'CausalEPP']
CELL_ORDER = [('Amazon-VG', 'MF'), ('Amazon-VG', 'LightGCN'), ('Amazon-VG', 'SimGCL'),
              ('Amazon-Movies', 'MF'), ('Amazon-Movies', 'LightGCN'), ('Amazon-Movies', 'SimGCL'),
              ('Douban-movie', 'MF'), ('Douban-movie', 'LightGCN')]
EPS = 1e-12


def read_json(path):
    with open(path) as f:
        return json.load(f)


def final_runs(runs_root, data=None, backbone=None):
    """Every final run (arm.json written by run_cell.py) as a list of dicts."""
    out = []
    for p in sorted(glob.glob(os.path.join(runs_root, '*', '*', 'final', '*', 'arm.json'))):
        a = read_json(p)
        if data and a['data'] != data:
            continue
        if backbone and a['backbone'] != backbone:
            continue
        a['run_dir'] = os.path.dirname(p)
        out.append(a)
    return out


def test_metrics(run_dir, k):
    p = os.path.join(run_dir, 'test_k%d.json' % k)
    return read_json(p) if os.path.exists(p) else None


def val_best(run_dir):
    p = os.path.join(run_dir, 'metrics.json')
    return read_json(p)['best'] if os.path.exists(p) else None


def long_table(runs_root):
    """One row per final run x cutoff: data, backbone, arm, method, seed, k, recall, ndcg, nalrp."""
    rows = []
    for a in final_runs(runs_root):
        for k in (20,):
            m = test_metrics(a['run_dir'], k)
            if m is None:
                continue
            rows.append(dict(data=a['data'], dataset=a['dataset'], backbone=a['backbone'], arm=a['arm'],
                             method=a['method'], seed=a['seed'], k=k, recall=m['recall'], ndcg=m['ndcg'],
                             nalrp=m['nalrp'], n_users=m['n_users'], config=json.dumps(a['config'], sort_keys=True),
                             run_dir=a['run_dir']))
    return pd.DataFrame(rows)


def summarize(df):
    """Seed means and standard deviations (population convention) per data/backbone/arm/k."""
    g = df.groupby(['data', 'dataset', 'backbone', 'arm', 'method', 'k'], sort=False)
    s = g.agg(n_seeds=('seed', 'count'), recall=('recall', 'mean'), recall_sd=('recall', lambda v: v.std(ddof=0)),
              ndcg=('ndcg', 'mean'), ndcg_sd=('ndcg', lambda v: v.std(ddof=0)),
              nalrp=('nalrp', 'mean'), nalrp_sd=('nalrp', lambda v: v.std(ddof=0)),
              config=('config', 'first')).reset_index()
    return s


def stale_signal(data_name):
    """(d_i, c_i, item_num, training-active mask) of a data directory."""
    dp = os.path.join(ROOT, 'data', data_name)
    n = read_json(os.path.join(dp, 'dataset_meta.json'))['item_num']
    c = pd.read_csv(os.path.join(dp, 'item_frequency.csv'))['count'].values.astype(np.float64)
    c = np.concatenate([c, np.zeros(max(0, n - len(c)))])[:n]
    lg = np.log1p(c)
    d = (lg - lg.min()) / (lg.max() - lg.min() + EPS)
    return d, c, n, c > 0


def read_recs(run_dir, k, split='test', source_k=20):
    """[(user, top-k list, targets)] from the top-<source_k> list dump (first k entries)."""
    p = os.path.join(run_dir, '%s_recs_k%d.txt' % (split, source_k))
    if not os.path.exists(p):
        return None
    out = []
    with open(p) as f:
        for line in f:
            u, recs, gt = line.rstrip('\n').split('\t')
            r = [int(x) for x in recs.split()][:k]
            g = [int(x) for x in gt.split()]
            if r and g:
                out.append((int(u), r, g))
    return out


def per_user_from_ranks(run_dir, k, split='test'):
    """{user: (recall@k, ndcg@k)} from the rank dump (rank of every target item)."""
    p = os.path.join(run_dir, '%s_ranks.npz' % split)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    users, ranks = z['user'], z['rank'].astype(np.int64)
    order = np.argsort(users, kind='stable')
    users, ranks = users[order], ranks[order]
    out = {}
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(users)]
    log2inv = 1.0 / np.log2(np.arange(2, k + 2))
    idcg_cum = np.cumsum(log2inv)
    for s, e in zip(starts, ends):
        rs = ranks[s:e]
        n = len(rs)
        hits = rs[rs <= k]
        rec = len(hits) / n
        dcg = float(log2inv[hits - 1].sum())
        ndcg = dcg / idcg_cum[min(n, k) - 1]
        out[int(users[s])] = (rec, ndcg)
    return out


def per_user_from_lists(run_dir, k, split='test'):
    """{user: (recall@k, ndcg@k)} from the saved top-K lists (first k entries of the list dump)."""
    recs = read_recs(run_dir, k, split)
    if recs is None:
        return None
    log2inv = 1.0 / np.log2(np.arange(2, k + 2))
    idcg_cum = np.cumsum(log2inv)
    out = {}
    for u, rr, g in recs:
        gs = set(g)
        hits = [i for i, it in enumerate(rr) if it in gs]
        dcg = float(log2inv[hits].sum()) if hits else 0.0
        out[u] = (len(hits) / len(g), dcg / idcg_cum[min(len(g), len(rr)) - 1])
    return out


def per_user_accuracy(run_dir, k, split='test'):
    """Per-user Recall@k / NDCG@k from the list dump, falling back to the rank dump."""
    out = per_user_from_lists(run_dir, k, split)
    return out if out is not None else per_user_from_ranks(run_dir, k, split)


def seed_average(per_seed):
    """{user: mean value} over the users present in every seed."""
    common = set(per_seed[0])
    for d in per_seed[1:]:
        common &= set(d)
    return {u: float(np.mean([d[u] for d in per_seed])) for u in common}


def cell_label(data, backbone):
    return '%s/%s' % (data, backbone)
