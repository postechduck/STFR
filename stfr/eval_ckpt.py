"""Evaluate a saved checkpoint on the validation or test split at a cutoff K.

  python -m stfr.eval_ckpt --run_dir runs/.../STFR_f0.7a0.75_s20 --split test --topk 20 --rank_dump --dump_recs

The model settings are read from <run_dir>/config.json; only the evaluation
options are overridden.  Outputs in run_dir: <split>_k<K>.log, <split>_k<K>.json
and, on request, <split>_ranks.npz (rank of every target item) and
<split>_recs_k<K>.txt (per-user top-K list and targets).
"""
import argparse
import json
import os

import numpy as np
import torch

from stfr.config import opt, args_from_config
from stfr.evaluate import Evaluator
from stfr.models import Model
from stfr.train import set_seed, pick_device, load_training_frame


def main(argv=None):
    p = argparse.ArgumentParser(description='evaluate a checkpoint')
    p.add_argument('--run_dir', required=True, help='directory written by stfr.train')
    p.add_argument('--split', default='test', choices=('val', 'test'))
    p.add_argument('--topk', type=int, default=20)
    p.add_argument('--ckpt', default=None, help='checkpoint path (default <run_dir>/best.pth)')
    p.add_argument('--rank_dump', action='store_true')
    p.add_argument('--dump_recs', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--eval_users', type=int, default=0, help='0 = every user of the split')
    a = p.parse_args(argv)

    overrides = dict(run_dir=a.run_dir, split=a.split, topk=a.topk, device=a.device,
                     eval_users=a.eval_users, rank_dump=a.rank_dump, dump_recs=a.dump_recs,
                     load_ckpt=a.ckpt or os.path.join(a.run_dir, 'best.pth'))
    args = args_from_config(os.path.join(a.run_dir, 'config.json'), overrides)
    opt.configure(args, mode='eval')
    if os.path.exists(opt.log_path):
        os.remove(opt.log_path)
    opt.log('evaluate %s split=%s K=%d ckpt=%s' % (opt.run_dir, opt.split, opt.topk, opt.load_ckpt))

    set_seed(opt.seed)
    train_data = load_training_frame()
    min_time, max_time = train_data['timestamp'].min(), train_data['timestamp'].max()
    device = pick_device()
    model = Model(opt.user_num, opt.item_num, opt.emb_dim, min_time, max_time, device)
    model.to(device)
    model.load_state_dict(torch.load(opt.load_ckpt, map_location=device))
    model.eval()
    evaluator = Evaluator(model, device)
    res = evaluator.run(0, tag=opt.split, dump_ranks=opt.rank_dump, dump_recs=opt.dump_recs)
    opt.log('[%s] recall@%d=%.5f ndcg@%d=%.5f nalrp@%d=%.5f n_users=%d' % (
        opt.split, opt.topk, res['recall'], opt.topk, res['ndcg'], opt.topk, res['nalrp'], res['n_users']))
    out = dict(split=opt.split, k=opt.topk, recall=res['recall'], ndcg=res['ndcg'], nalrp=res['nalrp'],
               n_users=res['n_users'], checkpoint=opt.load_ckpt, dataset=opt.dataset, data=opt.data_name,
               backbone=opt.backbone, method=opt.method, seed=opt.seed)
    with open(os.path.join(opt.run_dir, '%s_k%d.json' % (opt.split, opt.topk)), 'w') as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == '__main__':
    main()
