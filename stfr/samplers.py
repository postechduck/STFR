"""Negative samplers used as controls in the sampler-replacement comparison.

``fairneg`` replaces a share of the uniform negatives after they are
drawn (``ExtNegSampler.resample``); ``dns`` and ``aucns`` are batch-level
dynamic samplers that score candidates with the current model and are applied
inside the training loop.

  fairneg  FairNeg (Chen et al., WWW'23): group sampling probabilities updated
           each epoch from per-group positive losses; groups are popularity
           deciles here, within-group uniform
  dns      dynamic negative sampling (Zhang et al., SIGIR'13): M uniform
           candidates, the highest-scored one is the negative
  aucns    AUC-NS (Liu et al., 2023), following the authors' released code:
           M candidates scored with the current model, posterior "true negative"
           probability from the empirical score CDF and a popularity prior,
           candidate with the largest AUC gain selected
"""
import numpy as np
import torch


def _csr_from_pairs(rows, cols, n_rows):
    order = np.argsort(rows, kind='stable')
    r, c = rows[order], cols[order]
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.add.at(indptr, r + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, c


def _collide(pos_keys, num_item, users, cands):
    keys = users * num_item + cands
    idx = np.searchsorted(pos_keys, keys)
    idx_c = np.minimum(idx, len(pos_keys) - 1)
    return (idx < len(pos_keys)) & (pos_keys[idx_c] == keys)


class ExtNegSampler:
    def __init__(self, features, num_item, mode, opt):
        self.mode = mode
        self.num_item = num_item
        self.frac = float(opt.neg_sampler_frac)
        self.model = None
        users = features['user'].values.astype(np.int64)
        items = features['item'].values.astype(np.int64)
        self.num_user = int(users.max()) + 1
        cnt = np.zeros(num_item, dtype=np.float64)
        np.add.at(cnt, items, 1.0)
        self.cnt = cnt
        self.pos_keys = np.sort(users * num_item + items)
        self._epoch = 0
        if mode == 'fairneg':
            G = int(opt.fairneg_groups)
            self.outer_lr = float(opt.fairneg_lr)
            order = np.argsort(cnt, kind='stable')
            gid = np.zeros(num_item, dtype=np.int64)
            gid[order] = np.arange(num_item) * G // num_item
            self.group_of = gid
            self.G = G
            self.group_size = np.bincount(gid, minlength=G).astype(np.float64)
            self.gprob = np.full(G, 1.0 / G)
            self.momentum = None
            self.pairs_u = users
            self.pairs_i = items
            self._refresh_fair_cdf()
        else:
            raise ValueError('unknown sampler %s' % mode)

    def set_model(self, model):
        self.model = model

    def _emb(self):
        with torch.no_grad():
            eu, ei = self.model.get_emb(is_training=False)
        return eu.detach(), ei.detach()

    def _refresh_fair_cdf(self):
        p = self.gprob[self.group_of] / self.group_size[self.group_of]
        self.cdf = np.cumsum(p / p.sum())

    def _fairneg_outer_step(self):
        eu, ei = self._emb()
        dev = eu.device
        losses = np.zeros(self.G)
        counts = np.zeros(self.G)
        B = 262144
        for s in range(0, len(self.pairs_u), B):
            u = torch.from_numpy(self.pairs_u[s:s + B]).long().to(dev)
            i = torch.from_numpy(self.pairs_i[s:s + B]).long().to(dev)
            sc = (eu[u] * ei[i]).sum(-1)
            l = -torch.log(torch.sigmoid(sc).clamp_min(1e-8))
            g = self.group_of[self.pairs_i[s:s + B]]
            np.add.at(losses, g, l.cpu().numpy())
            np.add.at(counts, g, 1.0)
        counts[counts == 0] = 1.0
        gl = losses / counts
        grads = gl - gl.mean()
        if self.momentum is None:
            buf = grads
        else:
            buf = self.momentum * 0.9 + grads * 0.1
        self.momentum = grads
        p = np.clip(self.gprob - buf * self.outer_lr, 0.0, 1.0)
        if p.sum() <= 0:
            p = np.full(self.G, 1.0 / self.G)
        self.gprob = p / p.sum()
        self._refresh_fair_cdf()

    def resample(self, flat, users_rep):
        """flat: (n,) uniform negatives; users_rep: (n,) users. Returns the modified array."""
        self._epoch += 1
        n = flat.shape[0]
        mask = np.random.random(n) < self.frac
        sel = np.where(mask)[0]
        if len(sel) == 0:
            return flat
        u = users_rep[sel]
        if self.mode == 'fairneg' and self.model is not None:
            self._fairneg_outer_step()
        cand = np.searchsorted(self.cdf, np.random.random(len(sel))).astype(np.int64)
        cand = np.minimum(cand, self.num_item - 1)
        ok = ~_collide(self.pos_keys, self.num_item, u, cand)
        flat[sel[ok]] = cand[ok]
        return flat


class DNSDynamic:
    def __init__(self, features, num_item, opt):
        self.num_item = num_item
        self.M = int(opt.dns_m)
        users = features['user'].values.astype(np.int64)
        items = features['item'].values.astype(np.int64)
        self.pos_keys = np.sort(users * num_item + items)

    @torch.no_grad()
    def sample_batch(self, model, user_t, item_j_t):
        eu, ei = model.get_emb(is_training=False)
        eu, ei = eu.detach(), ei.detach()
        dev = eu.device
        u_all = user_t.cpu().numpy().astype(np.int64)
        cand = np.random.randint(0, self.num_item, size=(len(u_all), self.M))
        for _ in range(2):
            coll = _collide(self.pos_keys, self.num_item, np.repeat(u_all, self.M),
                            cand.reshape(-1)).reshape(len(u_all), self.M)
            if not coll.any():
                break
            cand[coll] = np.random.randint(0, self.num_item, size=int(coll.sum()))
        chosen_parts = []
        C = 4096
        for s in range(0, len(u_all), C):
            ut = user_t[s:s + C]
            cand_t = torch.from_numpy(cand[s:s + C]).long().to(dev)
            s_cand = (eu[ut].unsqueeze(1) * ei[cand_t]).sum(-1)
            best = s_cand.argmax(1)
            chosen_parts.append(cand_t.gather(1, best.unsqueeze(1)).squeeze(1))
        chosen = torch.cat(chosen_parts)
        coll = _collide(self.pos_keys, self.num_item, u_all, chosen.cpu().numpy())
        if coll.any():
            chosen = torch.where(torch.from_numpy(coll).to(dev), item_j_t, chosen)
        return chosen


class AUCNSDynamic:
    def __init__(self, features, num_item, opt):
        self.num_item = num_item
        self.M = int(opt.aucns_m)
        self.N = int(opt.aucns_n)
        self.alpha = float(opt.aucns_alpha)
        self.beta = float(opt.aucns_beta)
        self.gamma = float(opt.aucns_gamma)
        users = features['user'].values.astype(np.int64)
        items = features['item'].values.astype(np.int64)
        num_user = int(users.max()) + 1
        cnt = np.zeros(num_item, dtype=np.float64)
        np.add.at(cnt, items, 1.0)
        prior = cnt / cnt.sum()
        self.prior_beta = torch.from_numpy(prior ** self.beta).float()
        self.pos_keys = np.sort(users * num_item + items)
        self.u_indptr, self.u_items = _csr_from_pairs(users, items, num_user)
        self.deg_u = np.diff(self.u_indptr)

    def _draw_noninteracted(self, users, k):
        c = np.random.randint(0, self.num_item, size=(len(users), k))
        for _ in range(2):
            coll = _collide(self.pos_keys, self.num_item, np.repeat(users, k),
                            c.reshape(-1)).reshape(len(users), k)
            if not coll.any():
                break
            c[coll] = np.random.randint(0, self.num_item, size=int(coll.sum()))
        return c

    @torch.no_grad()
    def sample_batch(self, model, user_t, item_j_t):
        eu, ei = model.get_emb(is_training=False)
        eu, ei = eu.detach(), ei.detach()
        dev = eu.device
        if self.prior_beta.device != dev:
            self.prior_beta = self.prior_beta.to(dev)
        u_all = user_t.cpu().numpy().astype(np.int64)
        chosen_parts = []
        C = 1024
        for s in range(0, len(u_all), C):
            u_np = u_all[s:s + C]
            b = len(u_np)
            ut = user_t[s:s + C]
            scores = eu[ut] @ ei.T
            rpos = np.random.randint(0, np.maximum(self.deg_u[u_np], 1), size=(self.N, b))
            pos = self.u_items[self.u_indptr[u_np][None, :] + rpos].T
            negs = self._draw_noninteracted(u_np, self.N)
            cand = self._draw_noninteracted(u_np, self.M)
            pos_t = torch.from_numpy(pos).long().to(dev)
            neg_t = torch.from_numpy(negs).long().to(dev)
            cand_t = torch.from_numpy(cand).long().to(dev)
            s_pos = scores.gather(1, pos_t)
            s_neg = scores.gather(1, neg_t)
            s_cand = scores.gather(1, cand_t)
            info_p = (1 - torch.sigmoid(s_pos.unsqueeze(1) - s_cand.unsqueeze(2))).mean(2)
            info_n = (1 - torch.sigmoid(s_cand.unsqueeze(2) - s_neg.unsqueeze(1))).mean(2)
            degu = torch.from_numpy(self.deg_u[u_np]).float().to(dev)
            info_plus = info_p * degu.unsqueeze(1)
            info_minus = self.gamma * info_n * (self.num_item - degu).unsqueeze(1)
            F_n = (scores.unsqueeze(1) <= s_cand.unsqueeze(2)).float().sum(2) / (self.num_item + 1)
            p_fn = self.prior_beta[cand_t]
            negdist = (2 * (1 - F_n) * self.alpha + 2 * F_n * (1 - self.alpha)) * p_fn
            posdist = (2 * (1 - F_n) * (1 - self.alpha) + 2 * F_n * self.alpha) * (1 - p_fn)
            unbias = negdist / (negdist + posdist).clamp_min(1e-12)
            auc_gain = unbias * info_plus - (1 - unbias) * info_minus
            best = auc_gain.argmax(1)
            chosen_parts.append(cand_t.gather(1, best.unsqueeze(1)).squeeze(1))
        chosen = torch.cat(chosen_parts)
        coll = _collide(self.pos_keys, self.num_item, u_all, chosen.cpu().numpy())
        if coll.any():
            chosen = torch.where(torch.from_numpy(coll).to(dev), item_j_t, chosen)
        return chosen
