#!/usr/bin/env python3
"""Paired user-level comparisons of STFR with its strongest baseline (Table 8; Table E.1).

  python analysis/per_user_tests.py [--runs runs] [--out results] [--focal STFR]

Accuracy (per_user_tests.csv): for every setting and metric (Recall@20, NDCG@20),
the opponent is the compared method with the highest test mean.  Each user's metric is
computed from the saved top-K lists (the rank dump is used when no list dump exists),
averaged over the three seeds (users present in every seed), and compared with a
two-sided paired t-test (Wilcoxon signed-rank p also reported).

Mean-popularity gap and nALRP (per_user_calib.csv): Calib@20 = |mean d(top-20) - mean d(targets)|
and nALRP@20 per user from the saved top-K lists, opponent = strongest Recall@20 baseline;
positive deltas favour the focal method (lower is better for both).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import final_runs, test_metrics, per_user_accuracy, per_user_from_ranks, read_recs, seed_average, stale_signal, BASELINES


def paired(a, b, lower_better=False):
    users = sorted(set(a) & set(b))
    x = np.array([a[u] for u in users])
    y = np.array([b[u] for u in users])
    diff = (y - x) if lower_better else (x - y)
    t = stats.ttest_rel(x, y)
    try:
        pw = float(stats.wilcoxon(x, y, zero_method='wilcox').pvalue)
    except ValueError:
        pw = np.nan
    return dict(n=len(users), focal_mean=float(x.mean()), opponent_mean=float(y.mean()),
                delta=float((x - y).mean()), delta_pct=100.0 * float((x - y).mean()) / float(y.mean()) if y.mean() else np.nan,
                focal_win_frac=float((diff > 0).mean()), p_t=float(t.pvalue), p_wilcoxon=pw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results')
    p.add_argument('--focal', default='STFR')
    p.add_argument('--from_ranks', action='store_true',
                   help='per-user accuracy from the rank dumps instead of the saved lists (differs only in ties)')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = final_runs(a.runs)
    cells = sorted({(r['data'], r['backbone']) for r in runs})
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
                fn = per_user_from_ranks if a.from_ranks else per_user_accuracy
                ds = [fn(r['run_dir'], k) for r in sorted(arms[arm], key=lambda r: r['seed'])]
                if any(d is None for d in ds):
                    pu_cache[(arm, k)] = None
                else:
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
                if f is None or o is None:
                    continue
                res = paired(f[mi], o[mi])
                acc_rows.append(dict(data=data, backbone=bb, metric='%s@%d' % (metric, k), opponent=opp, **res))
        # Calib@20 / nALRP@20 from the saved lists, opponent = strongest Recall@20 baseline
        cands = [(mean[(b, 20)]['recall'], b) for b in rivals if (b, 20) in mean]
        if cands:
            opp = max(cands)[1]
            d, _, n, _ = stale_signal(data)

            def lists(arm):
                per = []
                for r in sorted(arms[arm], key=lambda r: r['seed']):
                    recs = read_recs(r['run_dir'], 20)
                    if recs is None:
                        return None
                    per.append({u: (float(d[rr].mean()), abs(float(d[rr].mean()) - float(d[g].mean())))
                                for u, rr, g in recs})
                return (seed_average([{u: v[0] for u, v in p.items()} for p in per]),
                        seed_average([{u: v[1] for u, v in p.items()} for p in per]))
            f, o = lists(a.focal), lists(opp)
            if f is not None and o is not None:
                for i, name in enumerate(('nalrp@20', 'calib@20')):
                    cal_rows.append(dict(data=data, backbone=bb, metric=name, opponent=opp,
                                         **paired(f[i], o[i], lower_better=True)))
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
        cal.to_csv(os.path.join(a.out, 'per_user_calib.csv'), index=False)
        print('\n== nALRP@20 / Calib@20 (lower is better; delta = focal - opponent) ==')
        print(cal[['data', 'backbone', 'metric', 'opponent', 'n', 'focal_mean', 'opponent_mean', 'delta', 'focal_win_frac', 'p_t']]
              .to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('\n[per-user tests] -> %s/per_user_tests.csv, per_user_calib.csv' % a.out)


if __name__ == '__main__':
    main()
