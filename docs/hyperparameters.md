# Hyperparameters

All values below are `python -m stfr.train` arguments; the grids are in
`configs/grids.json` and the selected configurations in `configs/selected_paper.json`.

## Shared backbone settings

All backbones use 64-dimensional embeddings (`--emb_dim`), batch size 8,192
(`--batch_size`), Adam, four negatives per positive (`--num_neg`).  MF uses learning
rate 0.001 and L2 weight 0.001; LightGCN and SimGCL use three propagation layers
(`--n_layers`), learning rate 0.01 and L2 weight 0.01 (`--lr`, `--reg`).  SimGCL uses
noise magnitude 0.1 (`--simgcl_eps`) and temperature 0.2 (`--simgcl_tau`); its
contrastive weight (`--simgcl_lambda`) is selected per dataset on the base model from
{0.02, 0.05, 0.1, 0.5, 2}: 0.1 on Amazon-VG and 0.05 on Amazon-Movies.

The main setting uses 60-day blocks and the last completed training block as the fresh
anchor.

## Search spaces

| Method | Grid | Fixed |
| --- | --- | --- |
| base | none | shared backbone settings |
| IPS | `--ips_lambda` in {10, 30, 50} | capped inverse-propensity weights from the normalized log popularity |
| DICE | `--dice_alpha` in {0.05, 0.1, 0.2} | margin sampler with margin 40 and pool 40 (`--dice_margin`, `--dice_pool`), 0.9 decay per epoch of the loss weight and of the margin, mean-L1 discrepancy |
| PDA | `--pda_gamma` in {0.05, 0.1, 0.25} at training x `--pda_alpha` in {0.05, 0.1, 0.2} at serving | Laplace-smoothed per-block popularity with per-block min-max normalization, linear extrapolation from the last two training blocks |
| TIDE | (q, b) in {default, (0, -3), (-1, -3), (-1, -5)} x `--tide_tau` in {3e6, 1e7, 3e7} seconds | default (q, b) = (-1, -3) for MF on Amazon, (-1, -5) for MF on Douban, (0, -3) for LightGCN and SimGCL on Amazon, (-2, -5) for LightGCN on Douban; learning rates of q and b per dataset and backbone in `stfr/config.py` (`--tide_q`, `--tide_b`, `--tide_lr_q`, `--tide_lr_b`) |
| CausalEPP | `--cepp_smooth` in {64, 256, 1024} x `--cepp_alpha` in {0.25, 0.5, 1.0} | quality and conformity parameters initialized as TIDE's |
| DDC | `--ddc_topk` in {0.1, 0.3, 0.5, 0.7} (share of a user's most popular training items forming the preference direction) | frozen base checkpoint of the same seed (`--ddc_ckpt`), lr 0.01, regularization 1e-4, head quantile .95, tail quantile .50 |
| STFR | `--ssns_frac` in {0.3, 0.5, 0.7, 0.9} x `--ssns_alpha` in {0.5, 0.75, 1.0} | additive training score, per-item fresh gains initialized at softplus(w_i) = 1 with learning rate 0.01 (`--lr_gain`), no additional regularizer |

## Selected configurations (validation Recall@20)

| Setting | IPS lambda | DICE alpha | PDA (gamma, alpha) | TIDE (q, b, tau) | CausalEPP (smoothing, alpha) | DDC top share | STFR (f, alpha) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VG / MF | 10 | .05 | (.25, .2) | (default, 3e6) | (1024, 1.0) | .3 | (.7, .75) |
| VG / LightGCN | 50 | .05 | (.25, .2) | (-1, -5, 3e6) | (256, 1.0) | .7 | (.7, 1.0) |
| VG / SimGCL | 30 | .05 | (.25, .2) | (-1, -3, 3e6) | (64, .25) | .3 | (.7, .75) |
| Movies / MF | 30 | .2 | (.25, .2) | (-1, -5, 3e6) | (1024, .25) | .7 | (.3, .5) |
| Movies / LightGCN | 50 | .1 | (.25, .2) | (-1, -5, 3e6) | (256, .25) | .7 | (.3, .5) |
| Movies / SimGCL | 10 | .05 | (.25, .2) | (-1, -5, 3e6) | (64, 1.0) | .7 | (.5, .5) |
| Douban / MF | 10 | .2 | (.1, .2) | (default, 1e7) | (256, 1.0) | .7 | (.9, .5) |
| Douban / LightGCN | 10 | .2 | (.25, .1) | (default, 3e6) | (256, .25) | .7 | (.3, .75) |

Block-configuration study (Amazon-VG, STFR (f, alpha); SimGCL weight in parentheses):
L=15: MF (.7, 1.0), LightGCN (.7, .75), SimGCL (.7, 1.0) (0.05); L=30: (.7, 1.0), (.7, 1.0),
(.9, 1.0) (0.1); L=120: (.5, 1.0), (.7, 1.0), (.9, 1.0) (0.05).

## Controls of Section 7.3 (`configs/ablations_vg.json`)

* SSNS only: `--method base --ssns_frac f --ssns_alpha alpha` with the selected STFR (f, alpha).
* Fresh channel only: `--method STFR --ssns_frac 0`.
* Shared gain: the selected STFR configuration with `--shared_gain`.
* Recency-restricted training: `--recent_blocks N`, N in {1, 2, 3, 6, 12, 24}, for base and
  for the fresh channel only; N is selected on validation Recall@20 (three-seed mean).
* Sampler replacement: `--neg_sampler dns` (`--dns_m` in {2, 5, 10}), `--neg_sampler aucns`
  (`--aucns_gamma` in {.002, .006, .018}; alpha .75, beta .01, M 5, N 10) and
  `--neg_sampler fairneg` (10 popularity-decile groups, outer learning rate .2), each with
  `--ssns_frac 0` and `--neg_sampler_frac` equal to the selected f.  DNS and AUC-NS are
  batch-level dynamic samplers and replace every negative.
