"""Train one configuration with checkpoint selection on validation Recall@K.

Example (STFR on Amazon-VG with MF, the manuscript's selected setting):

  python -m stfr.train --dataset Amazon-VG --backbone MF --method STFR \
      --ssns_frac 0.7 --ssns_alpha 0.75 --seed 20 --run_dir runs/Amazon-VG/MF/STFR_f0.7a0.75_s20

Outputs in --run_dir: config.json, train.log, best.pth (best validation
checkpoint), metrics.json (validation curve and the selected epoch).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from stfr.config import opt, build_parser
from stfr.data import TripletDataset
from stfr.evaluate import Evaluator
from stfr.models import Model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pick_device():
    if opt.device == 'auto':
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.device(opt.device)


def build_optimizer(model):
    if opt.method == 'DDC':
        return optim.Adam([{'params': model.ddc_alpha, 'lr': opt.ddc_lr},
                           {'params': model.ddc_beta, 'lr': opt.ddc_lr}])
    if opt.backbone == 'MF':
        params = [{'params': model.embed_user.weight, 'lr': opt.lr},
                  {'params': model.embed_item.weight, 'lr': opt.lr}]
    else:
        params = [{'params': model.embed_user_0.weight, 'lr': opt.lr},
                  {'params': model.embed_item_0.weight, 'lr': opt.lr}]
    if opt.method in ('TIDE', 'CausalEPP'):
        params.append({'params': model.q, 'lr': opt.tide_lr_q})
        params.append({'params': model.b, 'lr': opt.tide_lr_b})
    if opt.method == 'STFR':
        params.append({'params': model.gain, 'lr': opt.lr_gain})
    return optim.Adam(params)


def quality_loss(f_i, f_j, q_i, q_j):
    sgn = torch.sign(f_i - f_j)
    return -torch.log(torch.sigmoid(sgn * (q_i - q_j)) + 1e-6)


def load_training_frame():
    train_data = pd.read_csv(opt.train_data, sep='\t')
    if opt.recent_blocks > 0:
        tmax = int(train_data['split_idx'].max())
        train_data = train_data[train_data['split_idx'] > tmax - opt.recent_blocks].reset_index(drop=True)
        opt.log('[recent] training restricted to blocks %d..%d (%d rows)'
                % (tmax - opt.recent_blocks + 1, tmax, len(train_data)))
    return train_data


def main(argv=None):
    args = build_parser().parse_args(argv)
    opt.configure(args, mode='train')
    opt.save()
    opt.log('dataset=%s data=%s backbone=%s method=%s seed=%d' % (
        opt.dataset, opt.data_name, opt.backbone, opt.method, opt.seed))
    opt.log('lr=%g reg=%g batch_size=%d emb_dim=%d epochs=%d eval_every=%d patience=%d topk=%d' % (
        opt.lr, opt.reg, opt.batch_size, opt.emb_dim, opt.epochs, opt.eval_every, opt.patience, opt.topk))
    if opt.method == 'STFR':
        opt.log('STFR: ssns_frac=%g ssns_alpha=%g shared_gain=%s' % (opt.ssns_frac, opt.ssns_alpha, opt.shared_gain))
    elif opt.ssns_frac > 0:
        opt.log('SSNS sampler on %s: frac=%g alpha=%g' % (opt.method, opt.ssns_frac, opt.ssns_alpha))
    if opt.backbone == 'SimGCL':
        opt.log('SimGCL: lambda=%g eps=%g tau=%g' % (opt.simgcl_lambda, opt.simgcl_eps, opt.simgcl_tau))

    set_seed(opt.seed)
    train_data = load_training_frame()
    min_time = train_data['timestamp'].min()
    max_time = train_data['timestamp'].max()

    ext_sampler = None
    if opt.neg_sampler in ('pns', 'fairneg'):
        from stfr.samplers import ExtNegSampler
        assert opt.ssns_frac == 0, '--neg_sampler replaces SSNS; set --ssns_frac 0'
        ext_sampler = ExtNegSampler(train_data, opt.item_num, opt.neg_sampler, opt)
    dataset = TripletDataset(train_data, opt.item_num, opt.num_neg, opt.ssns_frac, opt.ssns_alpha,
                             dice_sampling=(opt.method == 'DICE'), dice_margin=opt.dice_margin,
                             dice_pool=opt.dice_pool, ext_sampler=ext_sampler)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    device = pick_device()
    opt.log('device: %s' % device)
    model = Model(opt.user_num, opt.item_num, opt.emb_dim, min_time, max_time, device)
    model.to(device)
    optimizer = build_optimizer(model)
    evaluator = Evaluator(model, device)
    if ext_sampler is not None:
        ext_sampler.set_model(model)
    dyn_sampler = None
    if opt.neg_sampler == 'aucns':
        from stfr.samplers import AUCNSDynamic
        dyn_sampler = AUCNSDynamic(train_data, opt.item_num, opt)
    elif opt.neg_sampler == 'dns':
        from stfr.samplers import DNSDynamic
        dyn_sampler = DNSDynamic(train_data, opt.item_num, opt)

    curve = []
    dice_alpha = opt.dice_alpha / 0.9
    opt.log('training start (%d evaluated users)' % len(evaluator.users))
    epoch = -1
    for epoch in range(opt.epochs):
        dice_alpha *= 0.9
        if epoch > 0 and opt.method == 'DICE':
            dataset.dice_margin *= 0.9
        model.train()
        t0 = time.time()
        dataset.ng_sample()
        loss_sum = 0.0
        for user, item_i, item_j, timestamp, split_idx in loader:
            user = user.to(device).long()
            item_i = item_i.to(device).long()
            item_j = item_j.to(device).long()
            if dyn_sampler is not None:
                item_j = dyn_sampler.sample_batch(model, user, item_j)
            timestamp = timestamp.cpu().numpy()
            split_idx = split_idx.cpu().numpy()
            optimizer.zero_grad()
            if opt.method == 'IPS':
                loss = model(user, item_i, item_j, timestamp, split_idx)
            elif opt.method == 'DICE':
                loss_click, loss_interest, loss_pop_1, loss_pop_2, loss_disc, reg_loss = \
                    model(user, item_i, item_j, timestamp, split_idx)
                loss = loss_click + dice_alpha * (loss_interest + loss_pop_1 + loss_pop_2) + 0.01 * loss_disc
                loss += opt.reg * reg_loss
            else:
                prediction_i, prediction_j, reg_loss = model(user, item_i, item_j, timestamp, split_idx)
                loss = torch.mean(torch.nn.functional.softplus(prediction_j - prediction_i))
                if opt.method == 'CausalEPP':
                    frequency_i = torch.from_numpy(model.item_quality_freq[item_i.cpu().numpy()]).to(device)
                    frequency_j = torch.from_numpy(model.item_quality_freq[item_j.cpu().numpy()]).to(device)
                    loss += torch.mean(quality_loss(frequency_i, frequency_j, model.q[item_i], model.q[item_j]))
                loss += (opt.ddc_reg if opt.method == 'DDC' else opt.reg) * reg_loss
            if opt.backbone == 'SimGCL':
                loss = loss + opt.simgcl_lambda * model.simgcl_cl_loss(user, item_i)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        elapsed = time.time() - t0

        if epoch % opt.eval_every == 0 or epoch == opt.epochs - 1:
            model.eval()
            res = evaluator.run(epoch, tag='val')
            improved = evaluator.update_best(res)
            curve.append(res)
            opt.log('[val] epoch=%03d recall@%d=%.5f ndcg@%d=%.5f nalrp@%d=%.5f%s' % (
                epoch, opt.topk, res['recall'], opt.topk, res['ndcg'], opt.topk, res['nalrp'],
                '  (best, checkpoint saved)' if improved else ''))
            if improved:
                torch.save(model.state_dict(), opt.ckpt_path)
        opt.log('epoch %03d loss=%.6f time=%.1fs' % (epoch, loss_sum, elapsed))

        if opt.patience > 0 and (epoch - evaluator.best['epoch']) >= opt.patience:
            opt.log('early stop at epoch %d' % epoch)
            break

    best = evaluator.best
    opt.log('best epoch %03d: val recall@%d=%.5f ndcg@%d=%.5f nalrp@%d=%.5f' % (
        best['epoch'], opt.topk, best['recall'], opt.topk, best['ndcg'], opt.topk, best['nalrp']))
    with open(os.path.join(opt.run_dir, 'metrics.json'), 'w') as f:
        json.dump(dict(best=best, val_curve=curve, k=opt.topk, epochs_run=epoch + 1,
                       n_val_users=len(evaluator.users), checkpoint=opt.ckpt_path), f, indent=2)
    return best


if __name__ == '__main__':
    main()
