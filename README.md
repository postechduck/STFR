# STFR: Stale Taxation and Fresh Refund

Code for *Taxing stale popularity, refunding fresh demand: A temporal popularity
framework for collaborative filtering*.

STFR combines a stale-suppressing negative sampler (SSNS) with a jointly learned,
item-level fresh channel.  SSNS adjusts negative sampling using cumulative interaction
counts; the fresh channel incorporates popularity within the interaction's time block.
The backbone and the fresh gains are trained jointly with BPR, with the backbone's
regularization and auxiliary loss where applicable.

## Model

The training score is

$$s(u,i,t)=e_u^\top e_i+\operatorname{softplus}(w_i)P_i^t .$$

Here $P_i^t$ is the log-transformed block interaction count, min-max normalized over
the items active in block $t$; inactive items receive zero.  The learned gain is
item-specific, shared across users, and non-negative.  At serving, the channel uses the
last completed training block for validation and test alike; neither window
contributes to the popularity features or to parameter training.

For cumulative training counts $c_i$, SSNS uses the catalog-level mixture

$$q_{f,\alpha}(i)=\frac{1-f}{M}+f\frac{c_i^\alpha}{\sum_k c_k^\alpha}.$$

The sampler uses raw counts raised to a power, not the normalized log-popularity
signal.  Negatives exclude the user's training history: a count-weighted draw that
falls in that history falls back to the uniform candidate (this is not repeated
rejection sampling from the weighted component; see
[docs/implementation.md](docs/implementation.md)).  The manuscript's simplified BPR
analysis concerns a reduced model and is not a convergence or causal-debiasing
guarantee for the full personalized system.

## Evaluation scope

The main evaluation covers Amazon Video Games, Amazon Movies and Douban with MF and
LightGCN, and the two Amazon datasets with SimGCL: eight dataset-backbone settings.
Compared methods are base, IPS, DICE, DDC, PDA, TIDE (full serving) and CausalEPP.
Models and checkpoints are selected by validation Recall@20; selected checkpoints are
evaluated at K=20 over seeds 20, 21 and 22.

The chronological split uses 60-day blocks; the first 30 days of the selected
evaluation block are validation and the last 30 days test.  Training ends before
validation.  Popularity is computed from training interactions only.  The main
comparison is supplemented by component ablations, a shared fresh gain,
recency-restricted training, sampler replacements, temporal block configurations on
Amazon-VG, and embedding analyses.  The additional block configurations change the
evaluation windows as well as the fresh statistics; they are not an isolated
block-length experiment on a fixed test set.

Reported metrics: Recall, NDCG, normalized average log recommendation popularity
(nALRP), coverage, exposure Gini, and the per-user mean-popularity gap (Calib), with the
definitions and cutoffs in [docs/metrics_and_experiments.md](docs/metrics_and_experiments.md).
Lower nALRP does not by itself imply broader coverage or lower Gini.

## Repository

```
stfr/        training and evaluation package (config, signals, data, samplers, models, evaluate, train, eval_ckpt)
prep/        dataset adapters and the chronological block split
scripts/     protocol driver (run_cell.py) and the experiment scripts of the manuscript
analysis/    tables and figures from saved outputs (no training)
configs/     search grids, selected configurations, control arms
cppcode/     time-decayed popularity of TIDE / CausalEPP (C++, built on first use)
docs/        data and evaluation, training and selection, hyperparameters, implementation, metrics
data/        preprocessed datasets (created by prep/; data/reference/ holds the manuscript's split metadata)
```

## Installation

```bash
pip install -r requirements.txt      # torch, numpy, pandas, scipy, matplotlib; cppimport + pybind11 for TIDE / CausalEPP
```

Tested with Python 3.13, PyTorch 2.10 (CUDA 12.8), NumPy 2.5, pandas 3.0, SciPy 1.17
on one NVIDIA H100; the code also runs on CPU (`--device cpu`).  Run every command from
the repository root.

## Data preparation

```bash
# 1. raw interactions -> data_raw/<dataset>/raw_interactions.csv (user item rating timestamp, dense ids)
python prep/adapters/douban.py --tsv data_raw/Douban/movie/douban_movie.tsv
python prep/adapters/amazon.py --dataset Amazon-VG --src_files <dir>/train_data.csv <dir>/val_data.csv <dir>/test_data.csv
# 2. chronological split, masks, targets and popularity tables -> data/<dataset>/
python prep/prep_split.py --dataset Amazon-VG
python prep/prep_split.py --dataset Amazon-Movies
python prep/prep_split.py --dataset Douban-movie
```

Sources, preprocessing, the split rule, target filtering and the exact split
boundaries are documented in [docs/data_and_evaluation.md](docs/data_and_evaluation.md);
`data/reference/<dataset>.json` gives the expected metadata of each split.

## Quick start

```bash
# STFR on Amazon-VG with MF (selected configuration), then test evaluation at K=20
bash scripts/train_eval.sh runs/demo/vg_mf_stfr_s20 --dataset Amazon-VG --backbone MF --method STFR \
    --ssns_frac 0.7 --ssns_alpha 0.75 --seed 20

# the same with LightGCN / SimGCL (SimGCL takes the dataset's contrastive weight)
python -m stfr.train --dataset Amazon-VG --backbone SimGCL --simgcl_lambda 0.1 --method STFR --ssns_frac 0.7 --ssns_alpha 0.75
python -m stfr.eval_ckpt --run_dir <run_dir> --split test --topk 20 --rank_dump --dump_recs

# compared methods (arguments of their selected configurations: configs/selected_paper.json)
python -m stfr.train --dataset Amazon-VG --backbone MF --method PDA --pda_gamma 0.25 --pda_alpha 0.2
python -m stfr.train --dataset Amazon-VG --backbone MF --method TIDE --tide_tau 3000000
python -m stfr.train --dataset Amazon-VG --backbone MF --method DDC --ddc_topk 0.3 --ddc_ckpt <base run>/best.pth
```

`python -m stfr.train --help` lists every argument.  A run directory contains
`config.json`, `train.log`, `best.pth` (best validation checkpoint), `metrics.json`
and, after `eval_ckpt`, `test_k<K>.json`, the rank dump and the top-K list dump used by
the analysis scripts.

## Reproducing the manuscript

```bash
bash scripts/run_main.sh                                   # main comparison: search, three-seed finals, test (8 settings)
SELECTED=configs/selected_paper.json bash scripts/run_main.sh   # same, skipping the search
bash scripts/run_ablations_vg.sh                           # Section 7.3 controls on Amazon-VG (Tables 9-11)
bash scripts/run_block_configs.sh                          # Section 7.4, Figure 4 and Table E.3: block configurations (L = 15, 30, 120 days)

python analysis/collect_results.py                         # Tables 4-7 (results/main_table.csv; mean rank over the @20 accuracy columns)
python analysis/ablation_tables.py                         # Tables 9-11 (components, recency window N, sampler swaps)
python analysis/per_user_tests.py                          # Table 8 (Calib@20) and Table E.1 (paired tests)
python analysis/embedding_geometry.py                      # Tables 12-13 (embedding geometry; serving representation by default)
python analysis/serving_interventions.py                   # Table E.2 (serving interventions)
python analysis/block_config_table.py                      # Table E.3 (block configurations)
python analysis/coverage_gini.py                           # Table E.4 (coverage, exposure Gini at @20)
python analysis/popularity_signals.py; python analysis/train_vs_rec_popularity.py; python analysis/dataset_stats.py   # Tables 1-3
python analysis/plot_pareto.py; python analysis/plot_block_configs.py; python analysis/plot_split.py             # Figures 2, 3, 4
```

`scripts/run_cell.py` implements the protocol for one dataset x backbone setting
(SimGCL weight selection, validation search of every method with a 30-epoch cap,
selection by validation Recall@20, three-seed finals with a 200-epoch cap and early
stopping, test evaluation); it is resume-safe and accepts `--jobs N` for concurrent
runs.  [docs/training_and_selection.md](docs/training_and_selection.md) describes the
stages and the run layout, [docs/hyperparameters.md](docs/hyperparameters.md) the grids
and selected values, and [docs/metrics_and_experiments.md](docs/metrics_and_experiments.md)
maps every table and figure to its script and output file.

## Results (test @20, three-seed means)

STFR against the compared method with the highest mean Recall@20 in each setting
(full tables: Tables 4-7 of the manuscript, `results/main_table.csv`).

| Setting | STFR R@20 | N@20 | nALRP@20 | Strongest baseline | its R@20 | its N@20 | its nALRP@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VG / MF | .1102 | .0538 | .595 | PDA | .1048 | .0505 | .694 |
| VG / LightGCN | .1216 | .0620 | .619 | TIDE | .0896 | .0457 | .755 |
| VG / SimGCL | .1134 | .0539 | .691 | PDA | .1052 | .0505 | .745 |
| Movies / MF | .0837 | .0407 | .718 | TIDE | .0741 | .0389 | .725 |
| Movies / LightGCN | .0808 | .0394 | .747 | TIDE | .0661 | .0282 | .788 |
| Movies / SimGCL | .0774 | .0366 | .738 | PDA | .0762 | .0368 | .811 |
| Douban / MF | .1090 | .0733 | .708 | TIDE | .1144 | .0775 | .710 |
| Douban / LightGCN | .1200 | .0756 | .699 | TIDE | .0786 | .0600 | .833 |

Mean rank over the @20 accuracy columns (Recall@20 and NDCG@20 of the backbone's
datasets, unrounded three-seed means, exact ties only): STFR 1.33 on MF, 1.00 on
LightGCN, 1.25 on SimGCL.  STFR has the highest mean Recall@20 in seven of the eight
settings (Douban / MF: TIDE) and lower nALRP@20 than base, PDA and TIDE in all eight.
Per-user paired tests, Calib@20, coverage and exposure Gini are produced by the analysis
scripts listed above.  The three-seed records behind these numbers, the ranks, the
ablation / recency / sampler runs of Tables 9-11 and the selection sweeps are in
[reproduction/](reproduction/README.md).

## Implementation notes

The pipeline descends from the TIDE authors' code (which also provides the LightGCN
backbone and the MF-IPS and DICE baselines); CausalEPP follows the authors' code, PDA
was rebuilt on the official popularity definition, DDC is a port of the official
two-stage procedure, the SimGCL encoder and InfoNCE follow the SELFRec implementation,
and AUC-NS follows the authors' released code.  Details, deviations and the recorded
environment are in [docs/implementation.md](docs/implementation.md).  Geometry
diagnostics use the serving representation of each backbone (MF item table, LightGCN
mean of layers 0-3, SimGCL mean of layers 1-3).  The time-decayed
popularity of TIDE / CausalEPP is a small C++ module compiled on first use
(`cppcode/`); no other component needs it.

## License

MIT (see `LICENSE`).  Datasets remain subject to the terms of their original providers.
