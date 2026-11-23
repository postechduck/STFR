"""Backbones (MF, LightGCN, SimGCL) and scoring methods.

Training scores per method (m = e_u . e_i is the backbone matching score):

  base       m
  STFR       m + softplus(w_i) * P_i^t            fresh channel, block of the interaction
  PDA        (ELU(m) + 1) * pop_i(t)^gamma
  TIDE       softplus(m) * tanh(softplus(q_i) + softplus(b_i) * decayed_pop_i(t))
  CausalEPP  softplus(m) * tanh(softplus(q_i) * freq_i
                                + softplus(b_i) * decayed_pop_i(t) * exp(-alpha |p_i^t - s_u^t|))
  IPS        BPR weighted by capped inverse popularity (batch mean, weights normalized to mean 1)
  DICE       interest / conformity embedding halves with case-based BPR terms (batch means)
  DDC        frozen base embeddings; the positive item is scored with
             e_u + beta_u ||e_u|| e_pre_u and the negative with e_u + alpha_u ||e_u|| e_pop
             (official loss combination b_a); serving adds both offsets to every item
  TPAB       m + p_u . g_{group(c_i^t)}   popularity tower over K popularity groups, with a
             popularity-bootstrapping BPR term weighted by tpab_lambda
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from stfr.config import opt
from stfr import signals


class Model(nn.Module):
    def __init__(self, user_num, item_num, emb_dim, min_time, max_time, device):
        super().__init__()
        self.n_users = user_num
        self.m_items = item_num
        self.embed_size = emb_dim
        self.device = device
        self.min_time = min_time
        self.max_time = max_time
        m = opt.method
        dp = opt.data_path

        # ---- TIDE / CausalEPP: item quality q_i, conformity b_i, time-decayed popularity ----
        if m in ('TIDE', 'CausalEPP'):
            self.q = nn.Parameter(torch.ones(item_num) * opt.tide_q)
            self.b = nn.Parameter(torch.ones(item_num) * opt.tide_b)
            self.tau = torch.ones(item_num) * opt.tide_tau
            self.decayed = signals.DecayedPopularity(dp, self.tau.numpy())
        if m == 'CausalEPP':
            t = signals.causalepp_tables(dp, opt.cepp_smooth, opt.cepp_window)
            self.item_quality_freq = t['item_quality_freq']
            self.item_pop_norm_np = t['item_pop_norm_np']
            self.user_sens_np = t['user_sens_np']
            self.all_times = signals.all_times(dp)
            self.last_block = min(len(self.all_times), self.item_pop_norm_np.shape[1]) - 1

        # ---- STFR: fresh signal table and the learnable fresh gains ----
        if m == 'STFR':
            self.all_times = signals.all_times(dp)
            self.last_block = len(self.all_times) - 1
            self.fresh_np = signals.fresh_signal(signals.block_count_table(dp, item_num))
            g0 = float(np.log(np.expm1(1.0)))          # softplus(g0) = 1: neutral start
            if opt.shared_gain:
                self.gain = nn.Parameter(torch.tensor(g0))
            else:
                self.gain = nn.Parameter(torch.ones(item_num) * g0)

        if m == 'PDA':
            tab = signals.pda_table(dp)
            self.pda_nper = tab.shape[1]
            self.pda_array = tab.values.reshape(-1)
        if m in ('IPS', 'DICE'):
            self.dice_pop = signals.dice_popularity(dp)

        # ---- backbone ----
        if opt.backbone in ('LightGCN', 'SimGCL'):
            self.get_graph()
            self.embed_user_0 = nn.Embedding(user_num, emb_dim)
            self.embed_item_0 = nn.Embedding(item_num, emb_dim)
            nn.init.normal_(self.embed_user_0.weight, std=0.1)
            nn.init.normal_(self.embed_item_0.weight, std=0.1)
            self.get_emb()
        else:
            self.embed_user = nn.Embedding(user_num, emb_dim)
            self.embed_item = nn.Embedding(item_num, emb_dim)
            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.normal_(self.embed_item.weight, std=0.01)

        if m == 'DDC':
            self._setup_ddc(device)
        if m == 'TPAB':
            self._setup_tpab(user_num, item_num, emb_dim, device)

    # ------------------------------------------------------------------ DDC
    def _setup_ddc(self, device):
        """Frozen base model, popularity direction e_pop and per-user preference direction e_pre.

        Follows the official reproduction command (loss combination b_a, e_pre sort mode y_uio):
        e_pop as in the DDC paper's Eq. 4 (top/bottom rho of the training-active items by training count);
        e_pre_u = normalized mean of the unit embeddings of the user's training items with the
        highest frozen base score e_u . e_i (top max(1, int(n_u * k)), stable sort)."""
        ckpt = opt.ddc_ckpt
        if not (ckpt and os.path.exists(ckpt)):
            raise FileNotFoundError('DDC needs --ddc_ckpt (frozen base checkpoint), got %r' % ckpt)
        self.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        with torch.no_grad():
            eu, ei = self.get_emb(is_training=False)
            self.ddc_eu = eu.detach().to(device)
            self.ddc_ei = ei.detach().to(device)
        for p in self.parameters():
            p.requires_grad_(False)
        counts = signals.cumulative_counts(opt.data_path, self.m_items)
        seen = counts > 0
        rho = opt.ddc_epop_rho
        hi = np.where(seen & (counts >= np.quantile(counts[seen], 1.0 - rho)))[0]
        lo = np.where(seen & (counts <= np.quantile(counts[seen], rho)))[0]
        v = self.ddc_ei[torch.as_tensor(hi, device=device)].mean(0) - \
            self.ddc_ei[torch.as_tensor(lo, device=device)].mean(0)
        self.ddc_epop = F.normalize(v, p=2, dim=0)
        self.ddc_unorm = self.ddc_eu.norm(dim=1)
        self.ddc_alpha = nn.Parameter(torch.empty(self.n_users).uniform_(-0.5, 0.5))
        self.ddc_beta = nn.Parameter(torch.empty(self.n_users).uniform_(-0.5, 0.5))
        ei_n = F.normalize(self.ddc_ei, p=2, dim=1)
        epre = torch.zeros(self.n_users, self.embed_size, device=device)
        train = pd.read_csv(opt.train_data, sep='\t')
        for u, g in train.groupby('user'):
            items = g['item'].values.astype(int)
            items = items[(items >= 0) & (items < self.m_items)]
            if len(items) == 0:
                continue
            k = max(1, int(len(items) * opt.ddc_topk))
            it = torch.as_tensor(items, device=device)
            score = (self.ddc_ei[it] @ self.ddc_eu[int(u)]).cpu().numpy()
            top = items[np.argsort(-score, kind='stable')][:k]
            v = ei_n[torch.as_tensor(top, device=device)].mean(0)
            epre[int(u)] = F.normalize(v, p=2, dim=0)
        self.ddc_epre_dirs = epre
        self.ddc_info = 'e_pop rho=%.2f (head %d / tail %d of %d training-active items), e_pre top share %.2f' % (
            rho, len(hi), len(lo), int(seen.sum()), opt.ddc_topk)

    # ------------------------------------------------------------------ TPAB
    def _setup_tpab(self, user_num, item_num, emb_dim, device):
        """Popularity groups of TPAB (official dataloader): per-block absolute training counts,
        K groups with thresholds pmax^(g/K), g = 1..K (pmax over the training blocks); an item is
        in the first group whose threshold is not below its count, else in group K-1.  Serving
        uses the predicted next-block count c_{t*-1} + predict * (c_{t*-1} - c_{t*-2}) for
        validation and test alike."""
        tr = pd.read_csv(opt.train_data, sep='\t', usecols=['item', 'split_idx'])
        t_star = opt.t_star
        n_blk = max(int(tr['split_idx'].max()) + 1, t_star)
        cnt = np.zeros((item_num, n_blk), dtype=np.float64)
        np.add.at(cnt, (tr['item'].values.astype(np.int64), tr['split_idx'].values.astype(np.int64)), 1.0)
        K = int(opt.tpab_k)
        pmax = float(cnt.max())
        self.tpab_pthres = pmax ** (np.arange(1, K + 1, dtype=np.float64) / K)

        def group(pop):
            return np.minimum(np.searchsorted(self.tpab_pthres, pop, side='left'), K - 1)

        self.tpab_n_blk = n_blk
        self.tpab_group_tab = torch.from_numpy(group(cnt).astype(np.int64)).to(device)
        c_last, c_prev = cnt[:, t_star - 1], cnt[:, t_star - 2]
        pred = c_last + opt.tpab_predict * (c_last - c_prev)
        self.tpab_serve_group = torch.from_numpy(group(pred).astype(np.int64)).to(device)
        # popularity tower: per-user popularity embedding and per-group embedding, initialized
        # like the backbone's preference embeddings
        std = 0.01 if opt.backbone == 'MF' else 0.1
        self.tpab_user_pop = nn.Embedding(user_num, emb_dim)
        self.tpab_group_emb = nn.Embedding(K, emb_dim)
        nn.init.normal_(self.tpab_user_pop.weight, std=std)
        nn.init.normal_(self.tpab_group_emb.weight, std=std)
        self.tpab_boot = None
        self.tpab_info = 'K=%d pmax=%.0f blocks=%d predict=%g serving groups used=%d' % (
            K, pmax, n_blk, opt.tpab_predict, len(np.unique(self.tpab_serve_group.cpu().numpy())))

    # ------------------------------------------------------------------ graph backbones
    def get_graph(self):
        cache = opt.graph_cache
        if os.path.exists(cache):
            z = np.load(cache)
            gidx = torch.from_numpy(z['index'])
            gval = torch.from_numpy(z['value'])
        else:
            train_data = pd.read_csv(opt.train_data, sep='\t')
            if opt.recent_blocks > 0:
                tmax = int(train_data['split_idx'].max())
                train_data = train_data[train_data['split_idx'] > tmax - opt.recent_blocks]
            user_dim = torch.LongTensor(train_data['user'].values)
            item_dim = torch.LongTensor(train_data['item'].values)
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            N = self.n_users + self.m_items
            # symmetric normalization D^-1/2 A D^-1/2 built sparsely
            g = torch.sparse_coo_tensor(index, data.float(), (N, N)).coalesce()
            gidx, gval = g.indices(), g.values()
            deg = torch.zeros(N).scatter_add_(0, gidx[0], gval)
            deg[deg == 0.] = 1.
            dinv_sqrt = deg.pow(-0.5)
            gval = gval * dinv_sqrt[gidx[0]] * dinv_sqrt[gidx[1]]
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, index=gidx.numpy(), value=gval.numpy())
        N = self.n_users + self.m_items
        self.Graph = torch.sparse_coo_tensor(gidx, gval, (N, N)).coalesce().to(self.device)

    def get_emb(self, is_training=True):
        if opt.backbone == 'MF':
            return self.embed_user.weight, self.embed_item.weight
        all_emb = torch.cat([self.embed_user_0.weight, self.embed_item_0.weight]).to(self.device)
        # LightGCN averages layers 0..L; SimGCL (following its reference encoder) averages 1..L
        embs = [] if opt.backbone == 'SimGCL' else [all_emb]
        for _ in range(opt.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        light_out = torch.mean(torch.stack(embs, dim=1), dim=1)
        self.embed_user, self.embed_item = torch.split(light_out, [self.n_users, self.m_items])
        return self.embed_user, self.embed_item

    def simgcl_perturbed_emb(self):
        all_emb = torch.cat([self.embed_user_0.weight, self.embed_item_0.weight]).to(self.device)
        embs = []
        for _ in range(opt.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            noise = torch.rand_like(all_emb)
            all_emb = all_emb + torch.sign(all_emb) * F.normalize(noise, dim=-1) * opt.simgcl_eps
            embs.append(all_emb)
        out = torch.mean(torch.stack(embs, dim=1), dim=1)
        return torch.split(out, [self.n_users, self.m_items])

    @staticmethod
    def _info_nce(view1, view2, temperature):
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        pos_score = (view1 @ view2.T) / temperature
        return -torch.diag(F.log_softmax(pos_score, dim=1)).mean()

    def simgcl_cl_loss(self, user, item_i):
        u_idx = torch.unique(user)
        i_idx = torch.unique(item_i)
        u1, i1 = self.simgcl_perturbed_emb()
        u2, i2 = self.simgcl_perturbed_emb()
        return self._info_nce(u1[u_idx], u2[u_idx], opt.simgcl_tau) \
            + self._info_nce(i1[i_idx], i2[i_idx], opt.simgcl_tau)

    def reg_loss(self, user, item_i, item_j):
        if opt.backbone in ('LightGCN', 'SimGCL'):
            eu0 = self.embed_user_0.weight[user]
            ei0 = self.embed_item_0.weight[item_i]
            ej0 = self.embed_item_0.weight[item_j]
        else:
            eu0 = self.embed_user.weight[user]
            ei0 = self.embed_item.weight[item_i]
            ej0 = self.embed_item.weight[item_j]
        return (1 / 2) * (eu0.norm(2).pow(2) + ei0.norm(2).pow(2) + ej0.norm(2).pow(2)) / float(len(user))

    # ------------------------------------------------------------------ training score
    def forward(self, user, item_i, item_j, timestamp, split_idx):
        embed_user, embed_item = self.get_emb()
        user_embedding = embed_user[user]
        item_i_embedding = embed_item[item_i]
        item_j_embedding = embed_item[item_j]
        reg_loss = self.reg_loss(user, item_i, item_j)
        m = opt.method

        if m == 'base':
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            return prediction_i, prediction_j, reg_loss

        if m == 'STFR':
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            time_block = np.clip(np.searchsorted(self.all_times, timestamp), 0, self.last_block)
            P_i = torch.from_numpy(self.fresh_np[item_i_np, time_block]).to(self.device)
            P_j = torch.from_numpy(self.fresh_np[item_j_np, time_block]).to(self.device)
            if opt.shared_gain:
                g = F.softplus(self.gain)
                fresh_i = g * P_i
                fresh_j = g * P_j
            else:
                fresh_i = F.softplus(self.gain[item_i]) * P_i
                fresh_j = F.softplus(self.gain[item_j]) * P_j
            m_i = (user_embedding * item_i_embedding).sum(dim=-1)
            m_j = (user_embedding * item_j_embedding).sum(dim=-1)
            return m_i + fresh_i, m_j + fresh_j, reg_loss

        if m == 'DDC':
            # loss combination b_a (DDC paper, Eq. 13): the positive is scored along the user's
            # preference direction (beta), the negative along the popularity direction (alpha)
            eu = self.ddc_eu[user]
            ei = self.ddc_ei[item_i]
            ej = self.ddc_ei[item_j]
            unorm = self.ddc_unorm[user]
            prediction_i = (eu * ei).sum(dim=-1) + \
                self.ddc_beta[user] * unorm * (self.ddc_epre_dirs[user] * ei).sum(dim=-1)
            prediction_j = (eu * ej).sum(dim=-1) + \
                self.ddc_alpha[user] * unorm * (ej @ self.ddc_epop)
            reg = (self.ddc_alpha[user] ** 2).mean() + (self.ddc_beta[user] ** 2).mean()
            return prediction_i, prediction_j, reg

        if m == 'TPAB':
            # negatives take their popularity group at the positive's block
            blk = torch.from_numpy(np.clip(split_idx, 0, self.tpab_n_blk - 1).astype(np.int64)).to(self.device)
            pu = self.tpab_user_pop(user)
            pgi = self.tpab_group_emb(self.tpab_group_tab[item_i, blk])
            pgj = self.tpab_group_emb(self.tpab_group_tab[item_j, blk])
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1) + (pu * pgi).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1) + (pu * pgj).sum(dim=-1)
            reg_pop = (1 / 2) * (pu.norm(2).pow(2) + pgi.norm(2).pow(2) + pgj.norm(2).pow(2)) / float(len(user))
            # popularity bootstrapping (official switch_concat + bpr_loss): user [pop, pref]
            # against items [pop permuted within the batch, pref], with its own L2 term
            perm = torch.randperm(len(user), device=self.device)
            u_cat = torch.cat([pu, user_embedding], dim=1)
            i_cat = torch.cat([pgi[perm], item_i_embedding], dim=1)
            j_cat = torch.cat([pgj[perm], item_j_embedding], dim=1)
            boot = torch.mean(F.softplus((u_cat * j_cat).sum(dim=-1) - (u_cat * i_cat).sum(dim=-1)))
            self.tpab_boot = boot + opt.reg * (reg_loss + reg_pop)
            return prediction_i, prediction_j, reg_loss + reg_pop

        if m == 'IPS':
            item_i_np = item_i.cpu().numpy().astype(int)
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            IPS_c = torch.from_numpy(np.minimum(1 / self.dice_pop[item_i_np], opt.ips_lambda)
                                     ).to(self.device) / opt.ips_lambda
            # batch mean of the weighted BPR term, weights normalized to batch mean 1, so that the
            # loss has the scale of the other methods' mean BPR and the L2 / SimGCL terms keep
            # their meaning (the lineage code summed over the batch)
            w = IPS_c / IPS_c.mean()
            ips_loss = -1 * (w * (prediction_i - prediction_j).sigmoid().log()).mean() \
                + opt.reg * reg_loss
            return ips_loss

        if m == 'DICE':
            return self._dice_forward(user, item_i, item_j, user_embedding,
                                      item_i_embedding, item_j_embedding, reg_loss)

        if m == 'PDA':
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            per = np.clip(split_idx, 0, self.pda_nper - 1)
            pop_i = torch.from_numpy(self.pda_array[(item_i_np * self.pda_nper + per).astype(int)]).to(self.device)
            pop_j = torch.from_numpy(self.pda_array[(item_j_np * self.pda_nper + per).astype(int)]).to(self.device)
            pref_i = F.elu((user_embedding * item_i_embedding).sum(dim=-1)) + 1
            pref_j = F.elu((user_embedding * item_j_embedding).sum(dim=-1)) + 1
            return pref_i * (pop_i ** opt.pda_gamma), pref_j * (pop_j ** opt.pda_gamma), reg_loss

        if m == 'TIDE':
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            dec_i = torch.from_numpy(self.decayed(item_i_np, timestamp)).to(self.device)
            dec_j = torch.from_numpy(self.decayed(item_j_np, timestamp)).to(self.device)
            popularity_i = F.softplus(self.q[item_i]) + F.softplus(self.b[item_i]) * dec_i
            popularity_j = F.softplus(self.q[item_j]) + F.softplus(self.b[item_j]) * dec_j
            prediction_i = F.softplus((user_embedding * item_i_embedding).sum(dim=-1)) * torch.tanh(popularity_i)
            prediction_j = F.softplus((user_embedding * item_j_embedding).sum(dim=-1)) * torch.tanh(popularity_j)
            return prediction_i, prediction_j, reg_loss

        if m == 'CausalEPP':
            user_np = user.cpu().numpy()
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            time_block = np.clip(np.searchsorted(self.all_times, timestamp), 0, self.last_block)
            frequency_i = torch.from_numpy(self.item_quality_freq[item_i_np]).to(self.device)
            frequency_j = torch.from_numpy(self.item_quality_freq[item_j_np]).to(self.device)
            stp_i = torch.from_numpy(self.item_pop_norm_np[item_i_np, time_block]).to(self.device)
            stp_j = torch.from_numpy(self.item_pop_norm_np[item_j_np, time_block]).to(self.device)
            dec_i = torch.from_numpy(self.decayed(item_i_np, timestamp)).to(self.device)
            dec_j = torch.from_numpy(self.decayed(item_j_np, timestamp)).to(self.device)
            sens_u = torch.from_numpy(self.user_sens_np[user_np, time_block]).to(self.device)
            align_i = torch.exp(-opt.cepp_alpha * torch.abs(stp_i - sens_u))
            align_j = torch.exp(-opt.cepp_alpha * torch.abs(stp_j - sens_u))
            popularity_i = F.softplus(self.q[item_i]) * frequency_i + F.softplus(self.b[item_i]) * dec_i * align_i
            popularity_j = F.softplus(self.q[item_j]) * frequency_j + F.softplus(self.b[item_j]) * dec_j * align_j
            prediction_i = F.softplus((user_embedding * item_i_embedding).sum(dim=-1)) * torch.tanh(popularity_i)
            prediction_j = F.softplus((user_embedding * item_j_embedding).sum(dim=-1)) * torch.tanh(popularity_j)
            return prediction_i, prediction_j, reg_loss

        raise ValueError('unknown method %s' % m)

    def _dice_forward(self, user, item_i, item_j, user_embedding, item_i_embedding, item_j_embedding, reg_loss):
        DICE_size = int(self.embed_size / 2)
        prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
        prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
        loss_click = -(prediction_i - prediction_j).sigmoid().log().sum()

        def half(idx, lo, hi, table):
            if opt.backbone == 'MF':
                return table(idx)[:, lo:hi]
            return table[idx][:, lo:hi]

        eu_tab = self.embed_user
        ei_tab = self.embed_item
        user_embedding_1 = half(user.unique(), 0, DICE_size, eu_tab)
        item_i_embedding_1 = half(item_i.unique(), 0, DICE_size, ei_tab)
        item_j_embedding_1 = half(item_j.unique(), 0, DICE_size, ei_tab)
        user_embedding_2 = half(user.unique(), DICE_size, None, eu_tab)
        item_i_embedding_2 = half(item_i.unique(), DICE_size, None, ei_tab)
        item_j_embedding_2 = half(item_j.unique(), DICE_size, None, ei_tab)

        # discrepancy: maximize the mean-L1 distance between the interest and conformity halves
        def md(a, b):
            return (a - b).abs().mean()
        dist = md(user_embedding_1, user_embedding_2) + md(item_i_embedding_1, item_i_embedding_2) \
            + md(item_j_embedding_1, item_j_embedding_2)
        loss_discrepancy = -dist

        item_i_np = item_i.cpu().numpy().astype(int)
        item_j_np = item_j.cpu().numpy().astype(int)
        pop_relation = self.dice_pop[item_i_np] > self.dice_pop[item_j_np]
        user_O1, user_O2 = user[pop_relation], user[~pop_relation]
        item_i_O1, item_j_O1 = item_i[pop_relation], item_j[pop_relation]
        item_i_O2, item_j_O2 = item_i[~pop_relation], item_j[~pop_relation]

        # interest: the less popular positive over a more popular negative (O2)
        ue = half(user_O2, 0, DICE_size, eu_tab)
        ie = half(item_i_O2, 0, DICE_size, ei_tab)
        je = half(item_j_O2, 0, DICE_size, ei_tab)
        loss_interest = -((ue * ie).sum(dim=-1) - (ue * je).sum(dim=-1)).sigmoid().log().sum()
        # conformity, unpopular-negative case (O1): positive above negative
        ue = half(user_O1, DICE_size, None, eu_tab)
        ie = half(item_i_O1, DICE_size, None, ei_tab)
        je = half(item_j_O1, DICE_size, None, ei_tab)
        loss_popularity_1 = -((ue * ie).sum(dim=-1) - (ue * je).sum(dim=-1)).sigmoid().log().sum()
        # conformity, popular-negative case (O2): negative above positive
        ue = half(user_O2, DICE_size, None, eu_tab)
        ie = half(item_i_O2, DICE_size, None, ei_tab)
        je = half(item_j_O2, DICE_size, None, ei_tab)
        loss_popularity_2 = -((ue * je).sum(dim=-1) - (ue * ie).sum(dim=-1)).sigmoid().log().sum()
        # every BPR term is a mean over the full batch (official mask_bpr_loss: the mask zeroes
        # the other subset), so the discrepancy, L2 and SimGCL terms keep their relative weight
        n = float(len(user))
        return (loss_click / n, loss_interest / n, loss_popularity_1 / n, loss_popularity_2 / n,
                loss_discrepancy, reg_loss)
