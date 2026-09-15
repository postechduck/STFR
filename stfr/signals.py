"""Popularity signals read from a preprocessed data directory.

Every signal is computed from training interactions only.  The two STFR
signals follow the manuscript: the stale signal d_i (Eq. 1) is the min-max
normalized log cumulative count over the catalog, and the fresh signal P_i^t
(Eq. 2) is the min-max normalized log count within block t over the items
active in that block (zero for inactive items).
"""
import os
import sys

import numpy as np
import pandas as pd

EPS = 1e-12
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fit_len(arr, n):
    """Pad with zeros / truncate along axis 0 to length n."""
    if arr.shape[0] < n:
        pad = np.zeros((n - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)
    return arr[:n]


def cumulative_counts(data_path, item_num):
    """c_i: cumulative training interaction count per item (float64, length item_num)."""
    c = pd.read_csv(os.path.join(data_path, 'item_frequency.csv'))['count'].values.astype(np.float64)
    return fit_len(c, item_num)


def stale_signal(counts):
    """d_i = minmax(log(1 + c_i)) over the catalog (Eq. 1)."""
    lg = np.log1p(counts)
    return (lg - lg.min()) / (lg.max() - lg.min() + EPS)


def block_count_table(data_path, item_num):
    """c_i^t: per-block training counts, shape [item_num, n_blocks] (float64, zeros for missing)."""
    tab = pd.read_csv(os.path.join(data_path, 'item_frequency_all_times.csv'), index_col=0).fillna(0)
    return fit_len(tab.values.astype(np.float64), item_num)


def fresh_signal(block_counts, min_count=1):
    """P_i^t (Eq. 2): within-block min-max normalized log count over active items; 0 if inactive."""
    P = np.zeros_like(block_counts)
    for t in range(block_counts.shape[1]):
        col = block_counts[:, t]
        active = col >= min_count
        if active.sum() > 0:
            lg = np.log1p(col[active])
            P[active, t] = (lg - lg.min()) / (lg.max() - lg.min() + EPS)
    return P


def all_times(data_path):
    """Right edges of the fixed-width time blocks (unix seconds)."""
    return np.load(os.path.join(data_path, 'all_times.npy'))


def dice_popularity(data_path):
    """Normalized log cumulative popularity used by IPS and DICE (written by prep)."""
    return np.load(os.path.join(data_path, 'DICE_popularity.npy'))


def pda_table(data_path):
    """PDA per-block popularity table (Laplace-smoothed, per-block min-max), columns = block index."""
    return pd.read_csv(os.path.join(data_path, 'PDA_popularity.csv'), sep='\t')


def robust_scale(x):
    med = np.median(x)
    q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
    return (x - med) / (q3 - q1)


def causalepp_tables(data_path, smooth):
    """Item quality frequencies, per-block item popularity and user popularity sensitivity."""
    freq = pd.read_csv(os.path.join(data_path, 'item_frequency.csv'))['count'].values
    item_quality_freq = np.clip(robust_scale(freq), 1e-3, 1.0)

    item_ft = pd.read_csv(os.path.join(data_path, 'item_frequency_all_times.csv'), index_col=0).fillna(0) + 1
    user_ft = pd.read_csv(os.path.join(data_path, 'user_frequency_all_times.csv'), index_col=0).fillna(0) + 1
    user_hi = pd.read_csv(os.path.join(data_path, 'user_frequency_high_popularity_all_times.csv'), index_col=0)

    # s_u^t: share of a user's block interactions that fall on high-popularity items,
    # smoothed, then min-max normalized over the whole user x block table.
    sens = user_hi.fillna(0) / (smooth + user_ft.fillna(0) + 1)
    sens = sens.values.astype(np.float64)
    sens = (sens - sens.min()) / (sens.max() - sens.min())
    sens = np.clip(sens, 1e-3, 1.0)

    # p_i^t: per-block item popularity, min-max normalized within each block.
    vals = item_ft.values.astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        item_norm = (vals - vals.min(axis=0, keepdims=True)) / \
            (vals.max(axis=0, keepdims=True) - vals.min(axis=0, keepdims=True))
    return dict(item_quality_freq=item_quality_freq,
                item_pop_norm_np=np.asarray(item_norm, dtype=np.float64),
                user_sens_np=np.asarray(sens, dtype=np.float64))


class DecayedPopularity:
    """Time-decayed item popularity of TIDE (C++ module in cppcode/, built on first use).

    For item i with training interaction times t_1 <= t_2 <= ..., the popularity
    just after t_j is pop_j = (pop_{j-1} + 1) exp(-(t_j - t_{j-1}) / tau_i) with
    pop_1 = 0, and at an arbitrary time t it is (pop_j + 1) exp(-(t - t_j) / tau_i)
    for the last interaction t_j <= t (pop_j + 1 exactly at t = t_j; 0 before t_1).
    """

    def __init__(self, data_path, tau):
        import cppimport
        cpp_dir = os.path.join(REPO_ROOT, 'cppcode')
        if cpp_dir not in sys.path:
            sys.path.insert(0, cpp_dir)
        self.mod = cppimport.imp('stfr_popularity')
        self.mod.load_popularity(np.ascontiguousarray(tau, dtype=np.float64),
                                 os.path.join(data_path, 'item_interactions.csv'))

    def __call__(self, items, timestamps):
        return self.mod.popularity(np.ascontiguousarray(items, dtype=np.float64),
                                   np.ascontiguousarray(timestamps, dtype=np.float64))
