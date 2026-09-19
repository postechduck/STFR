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
SEEDS = (20, 21, 22)
DISPLAY_DECIMALS = 4                     # accuracy columns of Tables 4-5
RANK_COLUMNS = {'MF': 6, 'LightGCN': 6, 'SimGCL': 4}   # accuracy columns entering the mean rank


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


def main_runs(runs_root):
    """Final runs of the eight main settings (CELL_ORDER).  The block-configuration variants
    (data Amazon-VG_b<L>) share the run layout and are read by block_config_table.py only."""
    return [a for a in final_runs(runs_root) if (a['data'], a['backbone']) in CELL_ORDER]


def test_metrics(run_dir, k):
    p = os.path.join(run_dir, 'test_k%d.json' % k)
    return read_json(p) if os.path.exists(p) else None


def val_best(run_dir):
    p = os.path.join(run_dir, 'metrics.json')
    return read_json(p)['best'] if os.path.exists(p) else None


def long_table(runs_root):
    """One row per final run x cutoff: data, backbone, arm, method, seed, k, recall, ndcg, nalrp."""
    rows = []
    for a in main_runs(runs_root):
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


def display(v, decimals=DISPLAY_DECIMALS):
    """Printed form of a table value.  The tables and the rank keys both use this string, so
    a rank never distinguishes two values that the table shows as equal."""
    return '%.*f' % (decimals, float(v))


def display_ranks(table, value_cols, group_cols, arm_col='arm', backbone_col='backbone'):
    """Column ranks and mean rank on the displayed values (Tables 4-5).

    `table` holds one row per setting x method with the unrounded three-seed means in
    `value_cols`.  Every column is ranked within its setting on display(mean) (rank 1 =
    highest; equal displayed values share the average rank), and the mean rank of a method
    on a backbone averages its column ranks over the backbone's settings.  Seed values are
    never rounded before averaging.  Returns (table with disp_/rank_ columns, mean ranks)."""
    t = table.copy()
    for c in value_cols:
        t['disp_' + c] = t[c].map(display)
        t['rank_' + c] = t.groupby(group_cols, sort=False)['disp_' + c].transform(
            lambda v: v.astype(float).rank(ascending=False, method='average'))
    rank_cols = ['rank_' + c for c in value_cols]
    g = t.groupby([backbone_col, arm_col], observed=True, sort=False)[rank_cols]
    mr = pd.DataFrame(dict(mean_rank=g.sum().sum(axis=1) / g.count().sum(axis=1),
                           mean_rank_n_cols=g.count().sum(axis=1))).reset_index()
    return t, mr


def check_records(df, arm_col='arm', arms=None, seeds=SEEDS, keys=('data', 'backbone')):
    """Missing / duplicated seeds and missing methods of a per-seed table, as a list of messages."""
    arms = list(arms or MAIN_ARMS)
    msgs = []
    for key, g in df.groupby(list(keys), sort=False):
        for arm in arms:
            sd = sorted(g[g[arm_col] == arm]['seed'].tolist())
            if not sd:
                msgs.append('%s: method %s is missing' % ('/'.join(map(str, key)), arm))
            elif sd != sorted(seeds):
                msgs.append('%s: %s has seeds %s (expected %s)' % ('/'.join(map(str, key)), arm, sd, list(seeds)))
    return msgs


def stale_signal(data_name):
    """(d_i, c_i, item_num, training-active mask) of a data directory."""
    dp = os.path.join(ROOT, 'data', data_name)
    n = read_json(os.path.join(dp, 'dataset_meta.json'))['item_num']
    c = pd.read_csv(os.path.join(dp, 'item_frequency.csv'))['count'].values.astype(np.float64)
    c = np.concatenate([c, np.zeros(max(0, n - len(c)))])[:n]
    lg = np.log1p(c)
    d = (lg - lg.min()) / (lg.max() - lg.min() + EPS)
    return d, c, n, c > 0


def list_dump_path(run_dir, split='test', source_k=20):
    return os.path.join(run_dir, '%s_recs_k%d.txt' % (split, source_k))


def read_recs(run_dir, k, split='test', source_k=20, required=False):
    """[(user, top-k list, targets)] from the top-<source_k> list dump (first k entries).

    Returns None when the dump does not exist (SystemExit with the command that writes it
    when `required`).  A stored list shorter than k is an error: it is never used as a
    top-k list."""
    p = list_dump_path(run_dir, split, source_k)
    if not os.path.exists(p):
        if required:
            raise SystemExit('missing top-%d list dump: %s\n  write it from the saved checkpoint with\n'
                             '  python -m stfr.eval_ckpt --run_dir %s --split %s --topk %d --dump_recs'
                             % (source_k, p, run_dir, split, source_k))
        return None
    if k > source_k:
        raise ValueError('top-%d lists cannot be read from a top-%d dump' % (k, source_k))
    out = []
    with open(p) as f:
        for line in f:
            u, recs, gt = line.rstrip('\n').split('\t')
            r = [int(x) for x in recs.split()]
            g = [int(x) for x in gt.split()]
            if len(r) < k:
                raise SystemExit('%s: user %s has %d stored items (< %d); re-run eval_ckpt with --topk %d --dump_recs'
                                 % (p, u, len(r), k, source_k))
            if g:
                out.append((int(u), r[:k], g))
    return out


def per_user_from_lists(run_dir, k, split='test', required=False):
    """{user: (recall@k, ndcg@k)} from the saved top-K lists (first k entries of the list dump)."""
    recs = read_recs(run_dir, k, split, required=required)
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


def seed_average(per_seed):
    """{user: mean value} over the users present in every seed."""
    common = set(per_seed[0])
    for d in per_seed[1:]:
        common &= set(d)
    return {u: float(np.mean([d[u] for d in per_seed])) for u in common}


def cell_label(data, backbone):
    return '%s/%s' % (data, backbone)
