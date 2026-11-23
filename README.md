# STFR: Stale Taxation and Fresh Refund

Code for *Taxing stale popularity, refunding fresh demand: A temporal popularity
framework for collaborative filtering*.

STFR combines a stale-suppressing negative sampler (SSNS, the tax) with a jointly learned,
item-specific fresh channel (the refund).  SSNS draws a share of the negatives in
proportion to cumulative interaction counts; the fresh channel adds popularity within the
interaction's time block to the score.  The backbone and the fresh gains are trained
jointly with BPR, with the backbone's regularization and auxiliary loss where applicable.

## Model

The training score is (Eq. 4)

$$s(u,i,t)=e_u^\top e_i+\operatorname{softplus}(w_i)P_i^t .$$

Here $P_i^t$ is the log-transformed block interaction count, min-max normalized over
the items active in block $t$; inactive items receive zero (Eq. 2).  The learned gain is
item-specific, shared across users, and non-negative.  At serving, the channel uses the
last completed training block for validation and test alike (Eq. 7); neither evaluation
window contributes to the popularity features or to parameter training.

For cumulative training counts $c_i$, SSNS uses the catalog-level mixture (Eq. 5)

$$q_{f,\alpha}(i)=\frac{1-f}{M}+f\frac{c_i^\alpha}{\sum_k c_k^\alpha}.$$

The sampler uses raw counts raised to a power, not the normalized log-popularity
signal.  Negatives exclude the user's training history: a count-weighted draw that
falls in that history falls back to the uniform candidate (this is not repeated
rejection sampling from the weighted component; see
[docs/implementation.md](docs/implementation.md)), and the uniform candidate itself is
always outside the history.  The manuscript's analysis of the BPR loss concerns a reduced
model and is not a convergence or causal-debiasing guarantee for the full personalized
system.

**STFR (forecast)** is a serving variant (Eq. 8): the trained STFR model is served with a
one-step linear forecast of the fresh signal from the last two training blocks,
$\mathrm{clip}(P_i^{t^*-1}+\eta(P_i^{t^*-1}-P_i^{t^*-2}),0,1)$, with $\eta$ selected on
validation.  No model is retrained, and $\eta=0$ is STFR.

## Evaluation scope

Three datasets (Amazon-VG, Amazon-Movies, Douban-Movie) and three backbones (MF,
LightGCN, SimGCL) give nine settings.  STFR is compared with the unmodified backbone,
three removal-oriented methods (IPS, DICE, DDC) and four injection-oriented methods (PDA,
TIDE, CausalEPP, TPAB), and STFR (forecast) is reported alongside it.  Every method is
selected by validation Recall@20 under one protocol and evaluated at K = 20 over seeds 20,
21 and 22.

The chronological split uses 60-day blocks.  The evaluation block is split into a
validation window and a test window; because blocks are left-open intervals and
timestamps have daily resolution, the two windows cover 29 and 31 calendar days (exact
dates in [docs/data_and_evaluation.md](docs/data_and_evaluation.md)).  Training ends before
validation, and popularity is computed from training interactions only.  The main
comparison is supplemented by component ablations, a shared fresh gain, training on recent
history only, replacing SSNS with other negative samplers, temporal block configurations on
Amazon-VG, and embedding analyses.  The additional block configurations
change the evaluation windows as well as the fresh statistics; they are not an isolated
block-length experiment on a fixed test set.

Reported metrics: Recall, NDCG, normalized average log recommendation popularity
(nALRP, the mean stale popularity of the recommendations), the per-user mean-popularity
gap (Calib), catalog coverage and exposure Gini, with the definitions in
[docs/metrics_and_experiments.md](docs/metrics_and_experiments.md).  Lower nALRP does not by
itself imply broader coverage or lower Gini.

## Repository

```
stfr/          training and evaluation package (config, signals, data, samplers, models, evaluate, train, eval_ckpt)
prep/          dataset adapters and the chronological block split
scripts/       protocol driver (run_cell.py) and the experiment scripts of the manuscript
analysis/      measurements, summaries and plots from saved run outputs (no training)
configs/       search ranges (grids.json) and the Amazon-VG control arms (ablations_vg.json)
cppcode/       time-decayed popularity of TIDE / CausalEPP (C++, built on first use)
docs/          data and evaluation, training and selection, hyperparameters, implementation, metrics
data/          preprocessed datasets (created by prep/; data/reference/ holds the manuscript's split metadata)
reproduction/  per-seed records of the manuscript's runs and the re-aggregation of the main comparison
```

## Installation

```bash
pip install -r requirements.txt      # torch, numpy, pandas, scipy, matplotlib; cppimport + pybind11 for TIDE / CausalEPP
```

Tested with Python 3.13, PyTorch 2.10 (CUDA 12.8), NumPy 2.5, pandas 3.0, SciPy 1.17
on an NVIDIA H100 NVL; the code also runs on CPU (`--device cpu`).  Run every command from
the repository root.

## Data preparation

```bash
# 1. raw interactions -> data_raw/<dataset>/raw_interactions.csv (user item rating timestamp, dense ids)
python prep/adapters/douban.py --tsv data_raw/Douban/movie/douban_movie.tsv
python prep/adapters/amazon.py --dataset Amazon-VG --src_reviews Video_Games_5.json.gz        # Amazon Review Data (2018), 5-core reviews
python prep/adapters/amazon.py --dataset Amazon-Movies --src_reviews Movies_and_TV.csv       # Amazon Review Data (2018), ratings only
# 2. chronological split, masks, targets and popularity tables -> data/<dataset>/
python prep/prep_split.py --dataset Amazon-VG
python prep/prep_split.py --dataset Amazon-Movies
python prep/prep_split.py --dataset Douban-movie
# 3. compare with the manuscript's split (content digests in data/reference/checksums.json)
python prep/verify_data.py --dataset Amazon-VG
```

Sources and versions, the preprocessing order, the dense-id mapping, the split rule,
target filtering and the exact split boundaries are documented in
[docs/data_and_evaluation.md](docs/data_and_evaluation.md);
`data/reference/<dataset>.json` gives the expected metadata of each split.

## Quick start

The hyperparameter values below are examples from the search ranges, not the selected
configurations; the selected values are produced by the protocol of `scripts/run_cell.py`
(written to `runs/<data>/<backbone>/selection.json`) and are not distributed.

```bash
# STFR on Amazon-VG with MF, then test evaluation at K=20
bash scripts/train_eval.sh runs/demo/vg_mf_stfr_s20 --dataset Amazon-VG --backbone MF --method STFR \
    --ssns_frac 0.5 --ssns_alpha 0.75 --seed 20
# STFR (forecast): serve the same checkpoint with a forecast coefficient (no retraining)
python -m stfr.eval_ckpt --run_dir runs/demo/vg_mf_stfr_s20 --split val --fresh_forecast 0.1

# other backbones (SimGCL takes the contrastive weight selected on the unmodified backbone)
python -m stfr.train --dataset Amazon-VG --backbone LightGCN --method STFR --ssns_frac 0.5 --ssns_alpha 0.75
python -m stfr.eval_ckpt --run_dir <run_dir> --split test --topk 20 --dump_recs

# compared methods (example values)
python -m stfr.train --dataset Amazon-VG --backbone MF --method PDA --pda_gamma 0.1 --pda_alpha 0.2
python -m stfr.train --dataset Amazon-VG --backbone MF --method TIDE --tide_tau 30000000
python -m stfr.train --dataset Amazon-VG --backbone MF --method DDC --ddc_topk 0.3 --ddc_ckpt <unmodified-backbone run>/best.pth
python -m stfr.train --dataset Amazon-VG --backbone MF --method TPAB --tpab_k 20 --tpab_lambda 1.0
```

`python -m stfr.train --help` lists every argument.  A run directory contains
`config.json`, `train.log`, `best.pth` (best validation checkpoint), `metrics.json`
and, after `eval_ckpt`, `test_k<K>.json` and the top-K list dump used by the analysis scripts
(`--rank_dump` additionally writes the optional rank dump).

## Reproducing the manuscript

The repository contains no trained weights, checkpoints or selected hyperparameter values.
The selected values can be chosen again from the search ranges (`configs/grids.json`) with
the validation protocol (highest validation Recall@20), which `scripts/run_cell.py` runs.
Run the experiments first; the analysis scripts read the checkpoints, test metrics and
top-20 list dumps that these runs write (the embedding analysis reads the checkpoints;
coverage, exposure Gini, Calib@20 and the paired tests read the list dumps), and no
analysis script trains a model.

```bash
bash scripts/run_main.sh            # main comparison, nine settings: search, three-seed finals, STFR (forecast), test
bash scripts/run_ablations_vg.sh    # Section 7.4 controls on Amazon-VG; needs the search of run_main.sh
bash scripts/run_block_configs.sh   # Section 7.5: block configurations (L = 15, 30, 120 days)
```

Experiments and analyses, with the commands that produce them (outputs in `results/` and
`figures/`):

| Experiment or analysis | Command |
| --- | --- |
| Main comparison: Recall@20, NDCG@20, nALRP@20, mean rank | `python analysis/collect_results.py` (after `run_main.sh`) |
| Recall@20 vs nALRP@20 per setting (plot) | `python analysis/plot_pareto.py` (after `collect_results.py`) |
| Paired accuracy tests, Calib@20 | `python analysis/per_user_tests.py` |
| Coverage and exposure Gini | `python analysis/coverage_gini.py` |
| Embedding analysis (directional concentration, norm ratio) | `python analysis/embedding_geometry.py` |
| Popularity signals (non-personalized stale / fresh rankings, popularity persistence) | `python analysis/popularity_signals.py` |
| Training vs recommendation stale popularity | `python analysis/train_vs_rec_popularity.py` (after `collect_results.py`) |
| Dataset statistics and chronological split | `python analysis/dataset_stats.py; python analysis/plot_split.py` |
| Component ablation, training on recent history only, replacing SSNS with other negative samplers (Amazon-VG) | `python analysis/ablation_summary.py` (after `run_ablations_vg.sh`) |
| Temporal block configurations on Amazon-VG (Section 7.5) | `python analysis/block_config_table.py; python analysis/plot_block_configs.py` (after `run_block_configs.sh`) |
| Fixed settings and search ranges | `configs/grids.json`, [docs/hyperparameters.md](docs/hyperparameters.md) |

The per-user analyses read the saved top-20 lists only; a missing list dump stops the
script with the `eval_ckpt` command that writes it from the saved checkpoint (and
`scripts/run_cell.py` re-evaluates a final run whose list dump is missing).  The split plot
additionally needs the full interaction file `data_raw/<dataset>/raw_interactions.csv`
for the pre-filter window counts and the blocks after the evaluation block, which are
counted for that plot only.

`scripts/run_cell.py` implements the protocol for one dataset x backbone setting (SimGCL
contrastive-weight selection, validation search of every method with a 30-epoch cap,
selection by validation Recall@20, three-seed finals with a 200-epoch cap and early
stopping, the STFR (forecast) serving selection, test evaluation); it is resume-safe and
accepts `--jobs N` for concurrent runs.
[docs/training_and_selection.md](docs/training_and_selection.md) describes the protocol and
the run layout, [docs/hyperparameters.md](docs/hyperparameters.md) the fixed settings and
search ranges, and [docs/metrics_and_experiments.md](docs/metrics_and_experiments.md) the
metrics and the experiments with their commands and outputs.

## Results

The per-seed test records of the manuscript's runs are in
[reproduction/](reproduction/README.md): every final run of the main comparison, the
Amazon-VG control experiments and the block configurations.  `python
reproduction/rebuild_ranks.py` computes the three-seed means and mean ranks of the main
comparison from them.  No result is restated here.

## Implementation notes

The pipeline descends from the TIDE authors' code (which also provides the LightGCN
backbone and the MF-IPS and DICE baselines) through the CausalEPP code.  CausalEPP follows
the authors' code with the settings of the original paper (quality-ordering loss weight 0.2,
six-month popularity window) and a corrected item index in its quality-ordering loss; IPS
and DICE use batch-mean BPR losses (DICE as in its official release); PDA was rebuilt on the
official popularity definition (forecast coefficient searched up to the official 0.5); DDC
reimplements the official two-stage procedure with its reproduction command (loss
combination b_a, preference direction from the frozen backbone's score); TPAB reimplements
the official code on this pipeline's 60-day blocks; the SimGCL encoder and InfoNCE follow
the SELFRec implementation; and AUC-NS follows the authors' released code.  Sources,
deviations, what was verified and the recorded environment are in
[docs/implementation.md](docs/implementation.md).  The time-decayed popularity of TIDE /
CausalEPP is a small C++ module compiled on first use (`cppcode/`); no other component
needs it.

## License

MIT (see `LICENSE`).  Datasets remain subject to the terms of their original providers.
