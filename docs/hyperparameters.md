# Hyperparameters

All values below are `python -m stfr.train` arguments; the search ranges are in
`configs/grids.json`.  The protocol that selects a value from each range is in
[training_and_selection.md](training_and_selection.md).  The selected values themselves
are not distributed: running the protocol writes them to `runs/<data>/<backbone>/selection.json`
and `lambda.json`.  "Source" states where a fixed value comes from: the original paper, the
official code (default or README command), the TIDE / CausalEPP code lineage the pipeline
started from, or our choice.

## Fixed settings and search ranges

| Method | Hyperparameter | Values | Argument |
| --- | --- | --- | --- |
| All backbones (fixed) | Embedding dimension; optimizer; batch size; negatives per positive | 64; Adam; 8,192 triplets; 4 | `--emb_dim`; –; `--batch_size`; `--num_neg` |
| MF (fixed) | Learning rate and L2 coefficient | 0.001 | `--lr`, `--reg` |
| LightGCN, SimGCL (fixed) | Learning rate and L2 coefficient; propagation layers | 0.01; 3 | `--lr`, `--reg`; `--n_layers` |
| SimGCL (fixed) | Noise magnitude; temperature | 0.1; 0.2 | `--simgcl_eps`; `--simgcl_tau` |
| SimGCL (backbone) | Contrastive weight | 0.02, 0.05, 0.1, 0.5, 2 | `--simgcl_lambda` |
| IPS | Weight cap | 10, 30, 50 | `--ips_lambda` |
| DICE | Weight of the interest and conformity losses | 0.05, 0.1, 0.2 | `--dice_alpha` |
| DDC | Share of top-scored items that define the preference direction | 0.1, 0.3, 0.5, 0.7 | `--ddc_topk` |
| PDA | Popularity exponent; forecast coefficient | 0.05, 0.1, 0.25; 0.05, 0.1, 0.2, 0.5 | `--pda_gamma`; `--pda_alpha` |
| TIDE | Initial quality and conformity parameters (q, b); decay constant (days) | code default, (0, −3), (−1, −3), (−1, −5); 35, 116, 347 | `--tide_q`, `--tide_b`; `--tide_tau` (seconds) |
| CausalEPP | Same as TIDE | Same as TIDE | as TIDE |
| TPAB | Number of popularity groups; bootstrapping weight | 10, 20, 30, 40; 0.5, 1.0, 1.5 | `--tpab_k`; `--tpab_lambda` |
| STFR | SSNS fraction f; exponent α | 0.3, 0.5, 0.7, 0.9; 0.5, 0.75, 1.0 | `--ssns_frac`; `--ssns_alpha` |
| STFR (forecast) | Forecast coefficient η | 0, 0.05, 0.1, 0.2, 0.5 | `--fresh_forecast` (serving only) |

Notes on the ranges:

* **TIDE / CausalEPP decay constant.** The code takes the constant in seconds, because
  interaction timestamps are unix seconds: `--tide_tau` 3e6, 1e7 and 3e7 seconds are 34.7,
  115.7 and 347.2 days, which the manuscript rounds to 35, 116 and 347 days.
* **TIDE / CausalEPP (q, b).** "code default" is the initial value of the code lineage for the
  dataset family and backbone type (`TIDE_INIT` in `stfr/config.py`): (−1, −3) for MF and
  (0, −3) for LightGCN / SimGCL on the Amazon datasets, (−1, −5) for MF and (−2, −5) for
  LightGCN / SimGCL on Douban-Movie.  Where the default coincides with an explicit point,
  the two candidates are the same configuration.
* **CausalEPP.** Only (q, b) and the decay constant are searched.  The sensitivity smoothing
  and the alignment sharpness are fixed at the official configuration, `--cepp_smooth 256`
  and `--cepp_alpha 0.5`; the quality-ordering loss weight at the paper's `--cepp_lambda 0.2`;
  the popularity window at the paper's six months, `--cepp_window 3` (three 60-day blocks).
* **DDC.** For each user, the share of the user's training items, ranked by the frozen
  unmodified backbone's score, whose mean direction defines the user's preference
  direction (official `--epre_sort_mode y_uio`, `--epre_topk`).
* **PDA.** The popularity exponent is trained; the forecast coefficient only changes serving,
  so every trained checkpoint is served with each coefficient (see the selection protocol).
* **STFR (forecast).** η = 0 is STFR itself; the coefficient only changes serving (Eq. 8).

## Shared backbone settings

| Setting | MF | LightGCN | SimGCL | Source |
| --- | --- | --- | --- | --- |
| embedding dimension (`--emb_dim`) | 64 | 64 | 64 | lineage |
| learning rate (`--lr`, Adam, no schedule) | 0.001 | 0.01 | 0.01 | lineage |
| L2 weight (`--reg`) | 0.001 | 0.01 | 0.01 | lineage |
| batch size, triplets (`--batch_size`) | 8,192 | 8,192 | 8,192 | lineage / ours |
| propagation layers (`--n_layers`) | – | 3 | 3 | lineage |
| embedding initialization | normal, std 0.01 | normal, std 0.1 | normal, std 0.1 | lineage |
| negatives per positive (`--num_neg`) | 4 | 4 | 4 | ours (DICE official 4) |
| SimGCL noise / InfoNCE temperature (`--simgcl_eps`, `--simgcl_tau`) | – | – | 0.1 / 0.2 | paper, SELFRec |
| SimGCL contrastive weight (`--simgcl_lambda`) | – | – | searched on the unmodified backbone, shared by all methods of the setting | ours (SELFRec default 0.5 is in the range) |
| validation during training | every 3 epochs (`--eval_every`), early stopping after 15 epochs without a validation Recall@20 gain (`--patience`) | same | same | ours |

The L2 term is 0.5 (||U||^2 + ||I_pos||^2 + ||I_neg||^2) / B over the batch's layer-0
embeddings (not Adam weight decay).  Negatives are uniform over the item id space,
redrawn every epoch and rejected (up to 16 tries) when they are training positives of the
user; DICE and STFR change the sampler as listed below.  The main loss of every method is a
batch mean: softplus(s_j - s_i) for most methods; the weighted BPR term of IPS and every BPR
term of DICE are batch means as well (see the IPS and DICE rows).  The main setting uses
60-day blocks and the last completed training block (t* - 1) as the fresh and serving
anchor for validation and test alike.

## Fixed settings per method

| Method | Fixed setting | Value | Source |
| --- | --- | --- | --- |
| IPS | weight of the positive item | min(1/p_i, lambda) / lambda, normalized to batch mean 1 | lineage weight; normalization ours |
| IPS | loss | batch mean of the weighted BPR term + L2 (the lineage code summed over the batch) | ours |
| IPS | propensity p_i | min-max normalized log(1 + training count), 0..1 | ours (lineage: count / max) |
| DICE | interest / conformity split | first / second 32 dimensions (of the propagated embeddings on graph backbones) | lineage |
| DICE | loss | click BPR + w (interest BPR on O2 + conformity BPR on O1 + reversed conformity BPR on O2) + 0.01 x (-mean L1 discrepancy) + L2; every BPR term is a mean over the full batch | official structure and reduction (`mask_bpr_loss`) |
| DICE | weight decay of w | x 0.9 per epoch (epoch 0 = the searched weight) | official (`--adaptive`, loss_decay 0.9) |
| DICE | discrepancy | mean L1 distance, weight 0.01 | official weight; L1 is an official option (official default: dCor) |
| DICE | negative sampler | popular pool count(j) > count(pos) + margin, unpopular pool count(j) < count(pos)/2; margin 40, pool 40, margin x 0.9 per epoch from epoch 1 (`--dice_margin`, `--dice_pool`) | official DICESampler |
| TIDE | score | softplus(m) tanh(softplus(q_i) + softplus(b_i) pop_i(t)), time-decayed popularity exp(-(t - t_k)/tau) | lineage |
| TIDE | lr of q / b (`--tide_lr_q`, `--tide_lr_b`) | Amazon MF 0.001 / 0.0001; Amazon LightGCN, SimGCL 0.01 / 0.01; Douban-Movie MF 0.1 / 0.0001; Douban-Movie LightGCN, SimGCL 0.0001 / 0.00001; no L2 on q, b | lineage |
| TIDE | serving | full mode: q + b pop(t_serve) | lineage (TIDE's click-prediction mode) |
| CausalEPP | quality-ordering loss weight (`--cepp_lambda`) | 0.2 | paper (CausalEPP, Section 5.1) |
| CausalEPP | quality-ordering loss | mean -log sigmoid(sign(f_i - f_j)(q_i - q_j)) with f = robust-scaled cumulative count | official, index of f_j corrected (see implementation.md) |
| CausalEPP | popularity window (`--cepp_window`) | three 60-day blocks (six months), top-20% threshold recomputed per window | paper |
| CausalEPP | sensitivity smoothing (`--cepp_smooth`), alignment sharpness (`--cepp_alpha`) | 256, 0.5 | official configuration |
| CausalEPP | inference-time intervention | none | ours |
| CausalEPP | learning rates of q, b | as TIDE | lineage |
| DDC | frozen backbone (`--ddc_ckpt`) | unmodified-backbone final of the same backbone and seed (search: seed 20); all backbone parameters frozen | official two-stage structure |
| DDC | loss combination | b_a: positive scored with beta_u ||e_u|| (e_pre_u . e_i), negative with alpha_u ||e_u|| (e_pop . e_j); batch-mean BPR | official README command |
| DDC | e_pre (`--ddc_topk` = k) | per user, training items ranked by the frozen score e_u . e_i (stable sort), top max(1, int(n_u k)), mean of unit vectors, renormalized | official README (`y_uio`) |
| DDC | e_pop (`--ddc_epop_rho`) | centroid of the top 5% minus centroid of the bottom 5% of training-active items by training count (ties at both cuts included), normalized | DDC paper, Eq. 4 (rho example 0.05) |
| DDC | alpha_u, beta_u initialization | uniform(-0.5, 0.5) per user | official |
| DDC | regularization (`--ddc_reg`) | 1e-4 x (mean alpha^2 + mean beta^2) over the batch | official default |
| DDC | learning rate (`--ddc_lr`) | 0.01 (only alpha and beta are trained) | ours (official: unspecified) |
| DDC | serving | full: base score + alpha term + beta term | official default |
| PDA | training score | (ELU(m) + 1) pop_i(block)^gamma | official |
| PDA | popularity table | per-block Laplace (count + 1)/(total + items), per-block min-max | official `pop_pre.py` |
| PDA | serving | clip(pop(t*-1) + alpha (pop(t*-1) - pop(t*-2)), 1e-9, 1)^gamma | official (exponent = training gamma) |
| PDA | training-time validation of the exponent search | served at `--pda_alpha 0.1` | ours |
| TPAB | score | m + p_u . g[group(c_i^t)], c_i^t = absolute training count of item i in block t | official PureMF_TPAB / LightGCN_TPAB |
| TPAB | popularity groups | K groups with thresholds pmax^(g/K), g = 1..K, pmax over the training blocks; first group whose threshold is not below the count, else K-1 | official dataloader (pmax there also scans the evaluation stages) |
| TPAB | stages | our 60-day training blocks; negatives take their group at the positive's block | ours (official period 8) |
| TPAB | bootstrapping loss | BPR of user [p_u, e_u] against items [p_g permuted within the batch, e_i], plus its own L2, weighted by lambda | official |
| TPAB | serving extrapolation (`--tpab_predict`) | 0.2: group of c_{t*-1} + 0.2 (c_{t*-1} - c_{t*-2}), validation and test alike | official default (not tuned by the authors) |
| TPAB | popularity tower | not propagated (layer_p 0); normal initialization with the backbone's std; backbone lr; L2 as the backbone embeddings | official default / ours |
| STFR | training score | m + softplus(w_i) P_i^t (block of the interaction), additive (Eq. 4) | ours |
| STFR | fresh signal | per block, min-max of log(1 + count) over the items active in the block; 0 for inactive items (Eq. 2) | ours |
| STFR | fresh gains | softplus(w_i) = 1 at start, lr 0.01 (`--lr_gain`), no L2 or other regularizer | ours |
| STFR | serving | m + softplus(w_i) P_i^{t*-1} (Eq. 7) | ours |
| STFR | SSNS | each uniform negative replaced with probability f by an item drawn proportionally to (cumulative training count)^alpha; a draw that is a training positive of the user keeps the uniform negative (Eq. 5) | ours |
| STFR (forecast) | serving | m + softplus(w_i) clip(P^{t*-1} + eta (P^{t*-1} - P^{t*-2}), 0, 1) from the trained STFR checkpoint (Eq. 8); validation and test use the same two training blocks | ours (form of PDA's forecast) |

## Controls of Section 7.4 (`configs/ablations_vg.json`)

* SSNS only: `--method base --ssns_frac f --ssns_alpha alpha` with the setting's selected STFR (f, alpha).
* Fresh channel only: `--method STFR --ssns_frac 0`.
* STFR (shared gain): the selected STFR configuration with `--shared_gain`.
* Training on recent history only: `--recent_blocks N`, N in {1, 2, 3, 6, 12, 24}, for
  the unmodified backbone and for the fresh channel only; N is selected on the three-seed
  mean validation Recall@20.
* Replacing SSNS with other negative samplers: `--neg_sampler dns` (`--dns_m` in {2, 5, 10}),
  `--neg_sampler aucns` (`--aucns_gamma` in {0.002, 0.006, 0.018}; alpha 0.75, beta 0.01,
  M 5, N 10) and `--neg_sampler fairneg` (10 popularity-decile groups, outer learning rate
  0.2), each with `--ssns_frac 0` and `--neg_sampler_frac` equal to the selected f.  The DNS
  candidate count and the AUC-NS rate are selected on validation (seed 20, 30-epoch cap).
  DNS and AUC-NS are batch-level dynamic samplers and replace every negative.

## Block configurations of Section 7.5

For L = 15, 30 and 120 days, the unmodified backbone and STFR are run with the main
protocol on each configuration's own split: STFR's (f, alpha) is searched over the range
above (seed 20, 30-epoch cap) and the SimGCL contrastive weight is searched on the
unmodified backbone over the range above.  L = 60 is the main comparison.
