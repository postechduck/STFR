"""BPR training triplets with the SSNS negative sampler.

For every positive (u, i, t) the dataset draws ``num_neg`` negatives.  A
negative starts as a uniform draw from the catalog that is redrawn while it
collides with the user's training history.  With probability ``ssns_frac`` it
is then replaced by a draw proportional to c_j^alpha (cumulative training
counts); a stale-proportional draw that lands in the user's history keeps the
uniform candidate instead (Appendix S.3 of the manuscript).
"""
import numpy as np
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, features, item_num, num_neg=4, ssns_frac=0.0, ssns_alpha=1.0,
                 dice_sampling=False, dice_margin=40.0, dice_pool=40, ext_sampler=None):
        super().__init__()
        self.features = features
        self.num_item = item_num
        self.num_neg = num_neg
        self.ssns_frac = float(ssns_frac)
        self.ext_sampler = ext_sampler
        if self.ssns_frac > 0:
            cnt = features['item'].value_counts()
            probs = np.zeros(item_num)
            probs[cnt.index.values.astype(int)] = cnt.values.astype(np.float64)
            if ssns_alpha != 1.0:
                probs = probs ** ssns_alpha
            self.ssns_cdf = np.cumsum(probs / probs.sum())
        # (user, item) keys of the training positives, sorted for collision tests
        self.pos_keys = np.sort(features['user'].values.astype(np.int64) * item_num
                                + features['item'].values.astype(np.int64))
        # DICE: popularity-conditioned negative pools (official sampler semantics)
        self.dice_sampling = dice_sampling
        if dice_sampling:
            cnt = np.zeros(item_num)
            vc = features['item'].value_counts()
            cnt[vc.index.values.astype(int)] = vc.values.astype(np.float64)
            self.dice_cnt = cnt
            self.dice_order = np.argsort(cnt, kind='stable')
            self.dice_sorted_cnt = cnt[self.dice_order]
            self.dice_margin = float(dice_margin)
            self.dice_pool = int(dice_pool)

    def _uniform_negatives(self, users_rep):
        n = len(users_rep)
        neg = np.random.randint(0, self.num_item, size=n, dtype=np.int64)
        for _ in range(16):
            keys = users_rep * self.num_item + neg
            idx = np.searchsorted(self.pos_keys, keys)
            bad = (idx < len(self.pos_keys)) & \
                  (self.pos_keys[np.minimum(idx, len(self.pos_keys) - 1)] == keys)
            if not bad.any():
                break
            neg[bad] = np.random.randint(0, self.num_item, size=int(bad.sum()), dtype=np.int64)
        return neg

    def _dice_negatives(self, users_rep, items_pos_rep):
        # popular pool: cnt(j) > cnt(pos) + margin; unpopular pool: cnt(j) < cnt(pos) / 2
        n_items = self.num_item
        srt, order = self.dice_sorted_cnt, self.dice_order
        cnt_pos = self.dice_cnt[items_pos_rep]
        hi = np.searchsorted(srt, cnt_pos + self.dice_margin, side='right')
        lo = np.searchsorted(srt, cnt_pos / 2.0, side='left')
        has_pop = (n_items - hi) >= self.dice_pool
        has_unpop = lo >= self.dice_pool
        use_pop = np.where(has_pop & has_unpop, np.random.random(len(users_rep)) < 0.5, has_pop)
        neg = self._uniform_negatives(users_rep)
        ok = has_pop | has_unpop
        r = np.random.random(len(users_rep))
        pop_idx = (hi + r * (n_items - hi)).astype(np.int64)
        unpop_idx = (r * np.maximum(lo, 1)).astype(np.int64)
        drawn = np.where(use_pop, order[np.minimum(pop_idx, n_items - 1)],
                         order[np.minimum(unpop_idx, n_items - 1)])
        keys = users_rep * n_items + drawn
        idx = np.searchsorted(self.pos_keys, keys)
        clean = ~((idx < len(self.pos_keys)) &
                  (self.pos_keys[np.minimum(idx, len(self.pos_keys) - 1)] == keys))
        take = ok & clean
        neg[take] = drawn[take]
        return neg

    def ng_sample(self):
        """Draw this epoch's negatives (called once per epoch before iterating)."""
        users_rep = np.array(self.features['user']).repeat(self.num_neg).astype(np.int64)
        if self.dice_sampling:
            items_pos_rep = np.array(self.features['item']).repeat(self.num_neg).astype(np.int64)
            item_negative = self._dice_negatives(users_rep, items_pos_rep).astype(np.float64)
        else:
            item_negative = self._uniform_negatives(users_rep).astype(np.float64)

        if self.ssns_frac > 0:
            mask = np.random.random(item_negative.shape[0]) < self.ssns_frac
            n_rep = int(mask.sum())
            cand = np.searchsorted(self.ssns_cdf, np.random.random(n_rep)).astype(np.int64)
            keys = users_rep[mask] * self.num_item + cand
            ok = np.searchsorted(self.pos_keys, keys)
            ok = ~((ok < len(self.pos_keys)) &
                   (self.pos_keys[np.minimum(ok, len(self.pos_keys) - 1)] == keys))
            idx = np.where(mask)[0][ok]
            item_negative[idx] = cand[ok]

        if self.ext_sampler is not None:
            item_negative = self.ext_sampler.resample(
                item_negative.astype(np.int64), users_rep).astype(np.float64)

        cols = [np.array(self.features['user']).repeat(self.num_neg).astype(np.float64),
                np.array(self.features['item']).repeat(self.num_neg).astype(np.float64),
                item_negative,
                np.array(self.features['timestamp']).repeat(self.num_neg).astype(np.float64),
                np.array(self.features['split_idx']).repeat(self.num_neg).astype(np.float64)]
        self.features_fill = np.stack(cols, axis=1)

    def __len__(self):
        return self.num_neg * len(self.features)

    def __getitem__(self, idx):
        row = self.features_fill[idx]
        return row[0], row[1], row[2], row[3], row[4]
