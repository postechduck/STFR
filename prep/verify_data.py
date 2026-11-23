#!/usr/bin/env python3
"""Compare a preprocessed dataset with the splits used in the manuscript.

  python prep/verify_data.py --dataset Amazon-VG [--data_dir Amazon-VG]
  python prep/verify_data.py --dataset Amazon-VG --write_reference     (maintainers)

data/reference/checksums.json holds, per dataset, SHA-256 digests of the manuscript's
files and order-independent digests of their content:

  content digest of an interaction file   sorted (user, item, timestamp[, split_idx]) rows
  content digest of a target / mask list  per user, the sorted set of items
  content digest of a per-block table     its training-block columns 0 .. t*-1 (the
                                          manuscript's files carry two extra trailing
                                          analysis columns that no method reads)

The content digests identify the split: which interactions are training, validation and
test, the evaluation targets, the masks and the popularity counts.  The file digests
additionally depend on the order of rows that share a timestamp, which the manuscript's
input inherited from its source files and which prep/adapters/amazon.py does not
reproduce (it writes rows in (timestamp, user, item) order).  Equal content digests with
different file digests therefore mean: same split, different row order within equal
timestamps (this changes mini-batch composition, not the data).
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, 'data', 'reference', 'checksums.json')
INTERACTIONS = ['train_data.csv', 'val_data.csv', 'test_data.csv', 'trainval_data.csv']
LISTS = {'train_list.txt': 2, 'trainval_list.txt': 2, 'val_list.txt': 1, 'test_list.txt': 1}   # first item column
TABLES = {'item_frequency.csv': ',', 'item_frequency_all_times.csv': ',', 'PDA_popularity.csv': '\t'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def rows_digest(df, cols):
    a = df[cols].astype(np.int64).values
    a = a[np.lexsort(a.T[::-1])]
    return sha(np.ascontiguousarray(a).tobytes())


def list_digest(path, first):
    h = hashlib.sha256()
    lines = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) > first:
                lines[int(p[0])] = sorted(set(int(x) for x in p[first:]))
    for u in sorted(lines):
        h.update(('%d:%s\n' % (u, ' '.join(map(str, lines[u])))).encode())
    return h.hexdigest()


def digests(dataset, data_dir):
    dp = os.path.join(ROOT, 'data', data_dir)
    out = dict(files={}, content={})
    raw = os.path.join(ROOT, 'data_raw', dataset, 'raw_interactions.csv')
    if os.path.exists(raw):
        out['files']['raw_interactions.csv'] = file_digest(raw)
        out['content']['raw_interactions.csv'] = rows_digest(pd.read_csv(raw, sep='\t'), ['user', 'item', 'timestamp'])
    for nm in INTERACTIONS:
        p = os.path.join(dp, nm)
        out['files'][nm] = file_digest(p)
        out['content'][nm] = rows_digest(pd.read_csv(p, sep='\t'), ['user', 'item', 'timestamp', 'split_idx'])
    for nm, first in LISTS.items():
        p = os.path.join(dp, nm)
        out['files'][nm] = file_digest(p)
        out['content'][nm] = list_digest(p, first)
    t_star = json.load(open(os.path.join(dp, 'dataset_meta.json')))['t_star']
    for nm, sep in TABLES.items():
        p = os.path.join(dp, nm)
        out['files'][nm] = file_digest(p)
        t = pd.read_csv(p, sep=sep)
        cols = [c for c in t.columns if not c.isdigit() or int(c) < t_star]
        out['content'][nm] = sha(np.ascontiguousarray(t[cols].fillna(0).values.astype(np.float64)).tobytes())
    out['files']['all_times.npy'] = file_digest(os.path.join(dp, 'all_times.npy'))
    out['content']['all_times.npy'] = sha(np.load(os.path.join(dp, 'all_times.npy')).astype(np.int64).tobytes())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_dir', default=None, help='directory under data/ (default: the dataset name)')
    p.add_argument('--write_reference', action='store_true')
    a = p.parse_args()
    got = digests(a.dataset, a.data_dir or a.dataset)
    ref = json.load(open(REF)) if os.path.exists(REF) else {}
    if a.write_reference:
        ref[a.dataset] = got
        with open(REF, 'w') as f:
            json.dump(ref, f, indent=1, sort_keys=True)
        print('[verify] reference written for %s' % a.dataset)
        return
    if a.dataset not in ref:
        raise SystemExit('no reference digests for %s' % a.dataset)
    exp, bad = ref[a.dataset], 0
    print('%-32s %-10s %s' % ('file', 'content', 'file bytes'))
    for nm in sorted(set(exp['files']) | set(exp['content'])):
        c = exp['content'].get(nm)
        f = exp['files'].get(nm)
        cs = '-' if c is None else ('missing' if nm not in got['content'] else 'same' if got['content'][nm] == c else 'DIFFERENT')
        fs = '-' if f is None else ('missing' if nm not in got['files'] else 'same' if got['files'][nm] == f else 'different')
        bad += cs == 'DIFFERENT'
        print('%-32s %-10s %s' % (nm, cs, fs))
    print('[verify] %s: %s' % (a.dataset, 'content identical to the manuscript split' if not bad
                               else '%d item(s) differ from the manuscript split' % bad))
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
