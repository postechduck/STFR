"""Run settings shared by training and evaluation.

The global ``opt`` object is populated once by an entry point
(``python -m stfr.train`` or ``python -m stfr.eval_ckpt``) from command-line
arguments and from the dataset metadata written by ``prep/prep_split.py``.
Defaults reproduce the manuscript protocol: 64-d embeddings, batch size 8192,
Adam, dataset/backbone-specific learning rates and L2 weights, validation every
three epochs, and early stopping after 15 epochs without a validation
Recall@20 improvement.
"""
import argparse
import json
import os
import time

BACKBONES = ('MF', 'LightGCN', 'SimGCL')
METHODS = ('base', 'STFR', 'IPS', 'DICE', 'DDC', 'PDA', 'TIDE', 'CausalEPP')
NEG_SAMPLERS = ('none', 'dns', 'aucns', 'fairneg')

# Shared backbone settings (all datasets).
BACKBONE_HP = {
    'MF': dict(lr=0.001, reg=0.001),
    'LightGCN': dict(lr=0.01, reg=0.01),
    'SimGCL': dict(lr=0.01, reg=0.01),
}

# TIDE / CausalEPP: initial values of the item quality (q) and conformity (b)
# parameters and their learning rates, per dataset family and backbone type.
TIDE_INIT = {
    ('Amazon', 'MF'): dict(q=-1.0, b=-3.0, lr_q=0.001, lr_b=0.0001),
    ('Amazon', 'graph'): dict(q=0.0, b=-3.0, lr_q=0.01, lr_b=0.01),
    ('Douban', 'MF'): dict(q=-1.0, b=-5.0, lr_q=0.1, lr_b=0.0001),
    ('Douban', 'graph'): dict(q=-2.0, b=-5.0, lr_q=0.0001, lr_b=0.00001),
}
# IPS propensity cap defaults (the search grid overrides them).
IPS_LAMBDA_DEFAULT = {('Amazon', 'MF'): 30.0, ('Amazon', 'graph'): 30.0,
                      ('Douban', 'MF'): 10.0, ('Douban', 'graph'): 30.0}


def build_parser():
    p = argparse.ArgumentParser(description='STFR: training and evaluation')
    g = p.add_argument_group('run')
    g.add_argument('--dataset', required=True,
                   help='dataset name; hyperparameter defaults key on it (Amazon-VG, Amazon-Movies, Douban-movie)')
    g.add_argument('--data_dir', default=None,
                   help='read data/<data_dir> instead of data/<dataset> (block-configuration variants)')
    g.add_argument('--backbone', default='MF', choices=BACKBONES)
    g.add_argument('--method', default='STFR', choices=METHODS)
    g.add_argument('--seed', type=int, default=20)
    g.add_argument('--run_dir', default=None, help='output directory (log, checkpoint, metrics)')
    g.add_argument('--device', default='auto', help="'auto', 'cpu' or 'cuda'")

    g = p.add_argument_group('training')
    g.add_argument('--epochs', type=int, default=200)
    g.add_argument('--eval_every', type=int, default=3)
    g.add_argument('--patience', type=int, default=15,
                   help='stop after this many epochs without a validation Recall@K gain (0 = off)')
    g.add_argument('--topk', type=int, default=20, help='cutoff K of the reported metrics')
    g.add_argument('--eval_users', type=int, default=0,
                   help='evaluate a fixed random subset of users during training (0 = all users)')
    g.add_argument('--recent_blocks', type=int, default=0,
                   help='train only on the most recent N blocks (recency-restricted control; 0 = full history)')
    g.add_argument('--num_neg', type=int, default=4, help='negatives per positive')

    g = p.add_argument_group('backbone')
    g.add_argument('--emb_dim', type=int, default=64)
    g.add_argument('--batch_size', type=int, default=8192)
    g.add_argument('--lr', type=float, default=None, help='default: per backbone (see BACKBONE_HP)')
    g.add_argument('--reg', type=float, default=None, help='L2 weight; default: per backbone')
    g.add_argument('--n_layers', type=int, default=3, help='propagation layers (LightGCN, SimGCL)')
    g.add_argument('--simgcl_lambda', type=float, default=0.5, help='contrastive weight (selected per dataset)')
    g.add_argument('--simgcl_eps', type=float, default=0.1)
    g.add_argument('--simgcl_tau', type=float, default=0.2)

    g = p.add_argument_group('STFR')
    g.add_argument('--ssns_frac', type=float, default=0.0, help='SSNS mixture share f (0 = uniform negatives)')
    g.add_argument('--ssns_alpha', type=float, default=1.0, help='SSNS count exponent alpha')
    g.add_argument('--shared_gain', action='store_true',
                   help='one learnable fresh gain shared by all items instead of item-specific gains (ablation)')
    g.add_argument('--lr_gain', type=float, default=0.01, help='learning rate of the fresh gains')

    g = p.add_argument_group('negative-sampler controls (replace SSNS)')
    g.add_argument('--neg_sampler', default='none', choices=NEG_SAMPLERS)
    g.add_argument('--neg_sampler_frac', type=float, default=1.0,
                   help='share of uniform negatives replaced by the sampler (fairneg)')
    g.add_argument('--dns_m', type=int, default=5, help='DNS candidates per positive')
    g.add_argument('--aucns_alpha', type=float, default=0.75)
    g.add_argument('--aucns_beta', type=float, default=0.01)
    g.add_argument('--aucns_gamma', type=float, default=0.006)
    g.add_argument('--aucns_m', type=int, default=5)
    g.add_argument('--aucns_n', type=int, default=10)
    g.add_argument('--fairneg_groups', type=int, default=10)
    g.add_argument('--fairneg_lr', type=float, default=0.2)

    g = p.add_argument_group('compared methods')
    g.add_argument('--ips_lambda', type=float, default=None, help='IPS propensity cap')
    g.add_argument('--dice_alpha', type=float, default=0.1, help='DICE interest/conformity loss weight')
    g.add_argument('--dice_margin', type=float, default=40.0)
    g.add_argument('--dice_pool', type=int, default=40)
    g.add_argument('--pda_gamma', type=float, default=0.1, help='PDA popularity exponent (training)')
    g.add_argument('--pda_alpha', type=float, default=1.25, help='PDA extrapolation weight (serving)')
    g.add_argument('--tide_q', type=float, default=None)
    g.add_argument('--tide_b', type=float, default=None)
    g.add_argument('--tide_lr_q', type=float, default=None)
    g.add_argument('--tide_lr_b', type=float, default=None)
    g.add_argument('--tide_tau', type=float, default=1e7, help='popularity decay time scale (seconds)')
    g.add_argument('--cepp_smooth', type=float, default=256.0, help='CausalEPP sensitivity smoothing')
    g.add_argument('--cepp_alpha', type=float, default=0.5, help='CausalEPP alignment sharpness')
    g.add_argument('--ddc_ckpt', default='', help='frozen base checkpoint (same backbone and seed)')
    g.add_argument('--ddc_topk', type=float, default=0.2, help='top share of a user\'s most popular items forming e_pre')
    g.add_argument('--ddc_lr', type=float, default=0.01)
    g.add_argument('--ddc_reg', type=float, default=1e-4)
    g.add_argument('--ddc_head_q', type=float, default=0.95)
    g.add_argument('--ddc_tail_q', type=float, default=0.50)

    g = p.add_argument_group('evaluation')
    g.add_argument('--split', default='val', choices=('val', 'test'))
    g.add_argument('--load_ckpt', default=None, help='checkpoint to evaluate (eval_ckpt)')
    g.add_argument('--rank_dump', action='store_true', help='save the rank of every target item')
    g.add_argument('--dump_recs', action='store_true', help='save every user\'s top-K list and targets')
    return p


class Settings:
    """Attribute bag holding the parsed arguments and derived paths."""

    def configure(self, args, mode='train'):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.mode = mode
        self.test_only = (mode == 'eval')
        self.val = (self.split == 'val')

        self.data_name = self.data_dir or self.dataset
        self.data_path = os.path.join('data', self.data_name)
        with open(os.path.join(self.data_path, 'dataset_meta.json')) as f:
            self.meta = json.load(f)
        self.item_num = int(self.meta['item_num'])
        self.user_num = int(self.meta['user_num'])
        self.t_star = int(self.meta['t_star'])
        self.block_days = int(self.meta['block_days'])
        self.val_days = int(self.meta['val_days'])
        # fresh anchor for validation and test alike: the last completed training block
        self.anchor_block = self.t_star - 1

        fam = 'Douban' if self.dataset.startswith('Douban') else 'Amazon'
        bb = 'MF' if self.backbone == 'MF' else 'graph'
        hp = BACKBONE_HP[self.backbone]
        if self.lr is None:
            self.lr = hp['lr']
        if self.reg is None:
            self.reg = hp['reg']
        if self.ips_lambda is None:
            self.ips_lambda = IPS_LAMBDA_DEFAULT[(fam, bb)]
        ti = TIDE_INIT[(fam, bb)]
        for k in ('q', 'b', 'lr_q', 'lr_b'):
            if getattr(self, 'tide_' + k) is None:
                setattr(self, 'tide_' + k, ti[k])

        # model inputs are training data for validation and test alike; only the
        # seen-item mask grows with time (train for validation, train+val for test)
        dp = self.data_path
        self.train_data = os.path.join(dp, 'train_data.csv')
        self.train_list = os.path.join(dp, 'train_list.txt')
        self.mask_list = os.path.join(dp, 'train_list.txt' if self.val else 'trainval_list.txt')
        self.eval_list = os.path.join(dp, 'val_list.txt' if self.val else 'test_list.txt')
        sfx = '_t%d_n%d' % (self.t_star, int(self.meta['n_train']))
        if self.recent_blocks:
            sfx += '_r%d' % self.recent_blocks
        self.graph_cache = os.path.join(dp, 'graph_cache', 'lightgcn%s.npz' % sfx)

        if self.run_dir is None:
            stamp = time.strftime('%Y%m%d-%H%M%S')
            self.run_dir = os.path.join('runs', self.data_name, self.backbone, self.method,
                                        's%d_%s' % (self.seed, stamp))
        os.makedirs(self.run_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.run_dir, 'best.pth')
        if mode == 'train':
            self.log_path = os.path.join(self.run_dir, 'train.log')
        else:
            self.log_path = os.path.join(self.run_dir, '%s_k%d.log' % (self.split, self.topk))
        return self

    def as_dict(self):
        skip = {'meta'}
        return {k: v for k, v in vars(self).items() if k not in skip}

    def save(self, path=None):
        path = path or os.path.join(self.run_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(self.as_dict(), f, indent=2, sort_keys=True)

    def log(self, msg, stdout=True):
        with open(self.log_path, 'a') as f:
            f.write(msg + '\n')
        if stdout:
            print(msg, flush=True)


def args_from_config(path, overrides=None):
    """Rebuild an argparse namespace from a saved config.json (eval_ckpt)."""
    with open(path) as f:
        saved = json.load(f)
    ns = build_parser().parse_args(['--dataset', saved['dataset']])
    for k in vars(ns):
        if k in saved:
            setattr(ns, k, saved[k])
    for k, v in (overrides or {}).items():
        setattr(ns, k, v)
    return ns


opt = Settings()
