"""Full-catalog ranking evaluation: Recall@K, NDCG@K and nALRP@K.

Every evaluated user is ranked against the whole catalog with the observed
history masked (training history for validation, training plus validation
history for test).  Targets are the user's interactions inside the evaluation
window that are not in the masked history and belong to the training-active
catalog; users without a remaining target are not evaluated.  nALRP@K is the
mean stale signal d_i (Eq. 1) over the top-K lists (Eq. 3).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from stfr.config import opt
from stfr import signals

USER_BATCH = 8192 * 4


def _elu1(x):
    return np.maximum(0, x) + np.minimum(0, np.exp(x) - 1) + 1


def _softplus_np(x):
    out = x.copy()
    out[x < 20] = np.log(1 + np.exp(x[x < 20]))
    return out


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.item_num = opt.item_num
        self.best = dict(recall=0.0, ndcg=0.0, nalrp=0.0, epoch=-1)

        # seen-item mask ledger (uid \t n \t items...)
        self.masked = {}
        with open(opt.mask_list) as f:
            for line in f:
                p = line.strip('\n').split('\t')
                if len(p) < 2:
                    continue
                self.masked[int(p[0])] = [int(x) for x in p[2:]]
        # evaluation targets (uid \t items...)
        self.targets = {}
        with open(opt.eval_list) as f:
            for line in f:
                p = line.strip('\n').split('\t')
                if len(p) < 2:
                    continue
                self.targets[int(p[0])] = [int(x) for x in p[1:]]

        counts = signals.cumulative_counts(opt.data_path, self.item_num)
        self.d = signals.stale_signal(counts)
        universe = set(np.flatnonzero(counts > 0).tolist())
        # keep unique targets outside the masked history and inside the training-active catalog
        n_drop = 0
        for uid in list(self.targets):
            hist = set(self.masked.get(uid, ()))
            seen, keep = set(), []
            for it in self.targets[uid]:
                if it in seen or it in hist or it not in universe:
                    seen.add(it)
                    continue
                seen.add(it)
                keep.append(it)
            if keep:
                self.targets[uid] = keep
            else:
                del self.targets[uid]
                n_drop += 1
        self.users = list(self.targets.keys())
        if opt.eval_users and opt.eval_users < len(self.users):
            self.users = list(np.random.RandomState(12345).choice(
                np.array(self.users), opt.eval_users, replace=False))
        self.n_users_dropped = n_drop

    # ------------------------------------------------------------ method-specific serving inputs
    def _serving_inputs(self):
        m = opt.method
        model = self.model
        inp = {}
        if m == 'PDA':
            tab = signals.pda_table(opt.data_path)
            prev, last = str(opt.t_star - 2), str(opt.t_star - 1)
            pop = tab[last] + opt.pda_alpha * (tab[last] - tab[prev])
            inp['pop'] = np.clip(pop.values, 1e-9, 1.0)
        elif m == 'TIDE':
            ts = model.max_time if opt.val else self._test_serving_time()
            pop = model.decayed(np.arange(self.item_num), np.ones(self.item_num) * ts)
            q = F.softplus(model.q).cpu().detach().numpy()
            b = F.softplus(model.b).cpu().detach().numpy()
            inp['pop'] = q + b * pop
        elif m == 'CausalEPP':
            if opt.val:
                ts = float(model.all_times[min(opt.anchor_block, len(model.all_times) - 1)])
            else:
                ts = self._test_serving_time()
            pop = model.decayed(np.arange(self.item_num), np.ones(self.item_num) * ts)
            q = F.softplus(model.q).cpu().detach().numpy()
            b = F.softplus(model.b).cpu().detach().numpy()
            inp['quality'] = q * model.item_quality_freq
            inp['conformity'] = b * pop
        elif m == 'STFR':
            P = model.fresh_np[:, opt.anchor_block]
            if opt.shared_gain:
                inp['boost'] = F.softplus(model.gain).item() * P
            else:
                w = F.softplus(model.gain).detach().cpu().numpy()
                inp['boost'] = w * P
        elif m == 'DDC':
            inp['chan_pop'] = (model.ddc_ei @ model.ddc_epop).detach().cpu().numpy()
            inp['unorm'] = model.ddc_unorm.detach().cpu().numpy()
            inp['alpha'] = model.ddc_alpha.detach().cpu().numpy()
            inp['beta'] = model.ddc_beta.detach().cpu().numpy()
            inp['epre'] = model.ddc_epre_dirs.detach().cpu().numpy()
            inp['ei'] = model.ddc_ei.detach().cpu().numpy()
        return inp

    @staticmethod
    def _test_serving_time():
        # end of the observed input at test time: the validation/test cut inside the evaluation block
        at = signals.all_times(opt.data_path)
        return float(at[opt.t_star] - (opt.block_days - opt.val_days) * 86400)

    def _score_batch(self, rate_batch, ub, inp):
        m = opt.method
        if m == 'PDA':
            return _elu1(rate_batch) * (inp['pop'] ** opt.pda_gamma)
        if m == 'TIDE':
            return _softplus_np(rate_batch) * np.tanh(inp['pop'])
        if m == 'CausalEPP':
            model = self.model
            anchor = opt.anchor_block
            stp = model.item_pop_norm_np[:, anchor].reshape(1, -1)
            sens = model.user_sens_np[ub, anchor].reshape(-1, 1)
            align = np.exp(-opt.cepp_alpha * np.abs(sens - stp))
            popularity = np.array(inp['quality'] + align * inp['conformity'].reshape(1, -1))
            return _softplus_np(rate_batch) * np.tanh(popularity)
        if m == 'STFR':
            return rate_batch + inp['boost'].reshape(1, -1)
        if m == 'DDC':
            a = (inp['alpha'][ub] * inp['unorm'][ub]).reshape(-1, 1)
            rate_batch = rate_batch + a * inp['chan_pop'].reshape(1, -1)
            b = (inp['beta'][ub] * inp['unorm'][ub]).reshape(-1, 1)
            return rate_batch + b * (inp['epre'][ub] @ inp['ei'].T)
        return rate_batch

    # ------------------------------------------------------------ one evaluation pass
    @torch.no_grad()
    def run(self, epoch=0, tag='val', dump_ranks=False, dump_recs=False):
        k = opt.topk
        inp = self._serving_inputs()
        embed_user, embed_item = self.model.get_emb(is_training=False)
        emb_i = embed_item.detach().cpu().numpy()
        log2inv = 1.0 / np.log2(np.arange(2, k + 2))
        idcg_cum = np.cumsum(log2inv)

        rec_sum = ndcg_sum = 0.0
        pop_sum, pop_cnt = 0.0, 0
        n_users = 0
        ranks_u, ranks_i, ranks_r = [], [], []
        recs_lines = []
        users = self.users
        for start in range(0, len(users), USER_BATCH):
            ub = np.asarray(users[start:start + USER_BATCH])
            emb_u = embed_user[torch.as_tensor(ub, device=self.device)].detach().cpu().numpy()
            rate_batch = np.asarray(emb_u) @ np.asarray(emb_i.T)
            rate_batch = np.array(self._score_batch(np.array(rate_batch), ub, inp))
            for r, u in enumerate(ub):
                masked = self.masked.get(int(u))
                if masked:
                    rate_batch[r][masked] = -np.inf
            if dump_ranks:
                for r, u in enumerate(ub):
                    tg = np.asarray(self.targets[int(u)], dtype=np.int64)
                    s = rate_batch[r]
                    rk = (s[None, :] > s[tg][:, None]).sum(axis=1) + 1
                    ranks_u.append(np.full(tg.size, int(u), dtype=np.int64))
                    ranks_i.append(tg)
                    ranks_r.append(rk.astype(np.int32))
            topk_idx = np.argpartition(-rate_batch, k - 1, axis=1)[:, :k]
            pop_sum += float(self.d[topk_idx].sum())
            pop_cnt += int(topk_idx.size)
            pred = np.argpartition(rate_batch, -k)[:, -k:]
            for r, u in enumerate(ub):
                top = pred[r, np.argsort(rate_batch[r, pred[r]])][::-1].astype(int)
                targets = self.targets[int(u)]
                tset = set(targets)
                hit, dcg = 0, 0.0
                for rank_i, it in enumerate(top, start=1):
                    if it in tset:
                        hit += 1
                        dcg += log2inv[rank_i - 1]
                rec_sum += hit / len(targets)
                ndcg_sum += dcg / idcg_cum[min(k, len(targets)) - 1]
                n_users += 1
                if dump_recs:
                    recs_lines.append('%d\t%s\t%s\n' % (int(u), ' '.join(map(str, top.tolist())),
                                                        ' '.join(map(str, targets))))
        res = dict(recall=rec_sum / n_users, ndcg=ndcg_sum / n_users,
                   nalrp=(pop_sum / pop_cnt) if pop_cnt else 0.0, n_users=n_users, k=k, epoch=epoch)
        if dump_ranks and ranks_u:
            np.savez_compressed(os.path.join(opt.run_dir, '%s_ranks.npz' % tag),
                                user=np.concatenate(ranks_u), item=np.concatenate(ranks_i),
                                rank=np.concatenate(ranks_r))
        if dump_recs:
            with open(os.path.join(opt.run_dir, '%s_recs_k%d.txt' % (tag, k)), 'w') as f:
                f.writelines(recs_lines)
        return res

    def update_best(self, res):
        """Checkpoint selection on validation Recall@K alone (strict improvement)."""
        if float(res['recall']) > float(self.best['recall']):
            self.best = dict(recall=res['recall'], ndcg=res['ndcg'], nalrp=res['nalrp'], epoch=res['epoch'])
            return True
        return False
