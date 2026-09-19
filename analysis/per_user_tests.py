#!/usr/bin/env python3
"""Paired accuracy tests (Table A.1) and the mean-popularity gap Calib@20 (Table 8).

  python analysis/per_user_tests.py [--runs runs] [--out results] [--focal STFR]

Both tables are computed from the saved top-20 lists and targets of the three final runs
of every method (<run>/test_recs_k20.txt, written by eval_ckpt --dump_recs).  A missing
list dump stops the script with the command that writes it; nothing is substituted.

Table A.1 (per_user_tests.csv): for every setting and metric (Recall@20, NDCG@20) the
opponent is the compared method with the highest test mean of that metric.  Each user's
metric is averaged over the three seeds (users present in every seed) and compared with a
two-sided paired t-test.  Unrounded per-user values;
no multiple-comparison adjustment.

Table 8 (calib_table.csv, calib_per_seed.csv): Calib@20(u) = |mean d(top-20) - mean d(targets)|
averaged over users within a seed and then over the three seeds, for STFR, PDA, TIDE and
base in every setting.  The per-seed file also holds the other compared methods.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import main_runs, test_metrics, per_user_from_lists, read_recs, seed_average, stale_signal, display, BASELINES, CELL_ORDER, SEEDS

CALIB_ARMS = ['STFR', 'PDA', 'TIDE', 'base']


def paired(a, b, lower_better=False):
    users = sorted(set(a) & set(b))
    x = np.array([a[u] for u in users])
    y = np.array([b[u] for u in users])
    diff = (y - x) if lower_better else (x - y)
    t = stats.ttest_rel(x, y)
    return dict(n=len(users), focal_mean=float(x.mean()), opponent_mean=float(y.mean()),
                delta=float((x - y).mean()), delta_pct=100.0 * float((x - y).mean()) / float(y.mean()) if y.mean() else np.nan,
                focal_win_frac=float((diff > 0).mean()), p_t=float(t.pvalue))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    p.add_argument('--focal', default='STFR')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = main_runs(a.runs)
    cells = sorted({(r['data'], r['backbone']) for r in runs})
    cells.sort(key=lambda c: CELL_ORDER.index(c) if c in CELL_ORDER else len(CELL_ORDER))
    acc_rows, cal_rows = [], []
    for data, bb in cells:
        cell = [r for r in runs if r['data'] == data and r['backbone'] == bb]
        arms = {}
        for r in cell:
            arms.setdefault(r['arm'], []).append(r)
        if a.focal not in arms:
            continue
        # test means per arm and cutoff (from the evaluation outputs)
        mean = {}
        for arm, rs in arms.items():
            for k in (20,):
                ms = [test_metrics(r['run_dir'], k) for r in rs]
                if all(ms):
                    mean[(arm, k)] = dict(recall=np.mean([m['recall'] for m in ms]), ndcg=np.mean([m['ndcg'] for m in ms]))
        rivals = [b for b in BASELINES if b in arms]
        pu_cache = {}

        def per_user(arm, k):
            if (arm, k) not in pu_cache:
                ds = [per_user_from_lists(r['run_dir'], k, required=True)
                      for r in sorted(arms[arm], key=lambda r: r['seed'])]
                pu_cache[(arm, k)] = (seed_average([{u: v[0] for u, v in d.items()} for d in ds]),
                                      seed_average([{u: v[1] for u, v in d.items()} for d in ds]))
            return pu_cache[(arm, k)]

        for k in (20,):
            for mi, metric in enumerate(('recall', 'ndcg')):
                cands = [(mean[(b, k)][metric], b) for b in rivals if (b, k) in mean]
                if not cands:
                    continue
                opp = max(cands)[1]
                f, o = per_user(a.focal, k), per_user(opp, k)
                res = paired(f[mi], o[mi])
                acc_rows.append(dict(data=data, backbone=bb, metric='%s@%d' % (metric, k), opponent=opp, **res))
        # Table 8: Calib@20 per seed (user mean) for every method of the setting
        d = stale_signal(data)[0]
        for arm, rs in arms.items():
            for r in rs:
                recs = read_recs(r['run_dir'], 20, required=arm in CALIB_ARMS)
                if recs is None:
                    continue
                gap = [abs(float(d[rr].mean()) - float(d[g].mean())) for _, rr, g in recs]
                cal_rows.append(dict(data=data, backbone=bb, arm=arm, seed=r['seed'], n_users=len(gap),
                                     calib20=float(np.mean(gap))))
        print('[done] %s/%s' % (data, bb), flush=True)

    pd.set_option('display.width', 220)
    if acc_rows:
        acc = pd.DataFrame(acc_rows)
        acc.to_csv(os.path.join(a.out, 'per_user_tests.csv'), index=False)
        print('\n== accuracy: %s vs strongest baseline (paired t over users, three-seed user means) ==' % a.focal)
        print(acc[['data', 'backbone', 'metric', 'opponent', 'n', 'focal_mean', 'opponent_mean', 'delta', 'delta_pct', 'p_t']]
              .to_string(index=False, float_format=lambda v: '%.5f' % v))
    if cal_rows:
        cal = pd.DataFrame(cal_rows)
        cal.to_csv(os.path.join(a.out, 'calib_per_seed.csv'), index=False)
        for (data, bb, arm), g in cal[cal.arm.isin(CALIB_ARMS)].groupby(['data', 'backbone', 'arm']):
            if sorted(g.seed) != sorted(SEEDS):
                print('[check] %s/%s %s: Calib over seeds %s (expected %s)' % (data, bb, arm, sorted(g.seed), list(SEEDS)))
        tab = cal[cal.arm.isin(CALIB_ARMS)].pivot_table(index=['data', 'backbone'], columns='arm', values='calib20',
                                                        aggfunc='mean', sort=False)
        tab = tab[[c for c in CALIB_ARMS if c in tab.columns]]
        tab.reset_index().to_csv(os.path.join(a.out, 'calib_table.csv'), index=False)
        print('\n== Table 8: Calib@20 (user mean per seed, three-seed mean; lower is better) ==')
        print(tab.map(lambda v: display(v, 3)).to_string())
    print('\n[per-user tests] -> %s/per_user_tests.csv (Table A.1), calib_table.csv, calib_per_seed.csv (Table 8)' % a.out)


if __name__ == '__main__':
    main()
