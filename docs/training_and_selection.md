# Training and model selection

## Protocol (`scripts/run_cell.py`)

Every method follows the same procedure in every dataset x backbone setting:

1. **Search.** Every candidate configuration of the method's grid
   (`configs/grids.json`) is trained with seed 20 for at most 30 epochs, validated every
   3 epochs, and stopped after 15 epochs without a validation Recall@20 improvement.
   The candidate's score is its best validation Recall@20.  On Douban, training-time
   validation uses a fixed random subset of 4,000 users; every candidate checkpoint is
   re-evaluated on the full validation set before selection.  PDA trains its three
   values of gamma and serves each with the three values of alpha (nine candidates).
2. **Selection.** The candidate with the highest validation Recall@20 (first in grid
   order on ties).
3. **Finals.** The selected configuration is trained with seeds 20, 21 and 22 for at most
   200 epochs (validation every 3 epochs, patience 15).  A seed-20 search run that early
   stopping ended before the 30-epoch cap is reused as that seed's final run.  DDC finals
   use the base checkpoint of the same seed as the frozen backbone.
4. **Test.** The checkpoint with the best validation Recall@20 of each final run is
   evaluated on the test split at K=20; test data never influence selection.
5. **SimGCL.** The contrastive weight is selected once per dataset on the base model
   (grid {0.02, 0.05, 0.1, 0.5, 2}; same search budget) and shared by every method.

Reported numbers are three-seed means.  Run directories:

```
runs/<data>/<backbone>/lamsel/L<lambda>/          SimGCL weight selection
runs/<data>/<backbone>/sweep/<arm>_<config>/      search candidates (serve_<alpha>/ for PDA)
runs/<data>/<backbone>/final/<arm>_<config>_s<seed>/
    config.json  train.log  best.pth  metrics.json  arm.json
    test_k20.{log,json}  test_ranks.npz  test_recs_k20.txt
runs/<data>/<backbone>/selection.json, lambda.json, summary.csv
```

`test_ranks.npz` stores the rank of every target item; `test_recs_k20.txt` stores each
user's top-20 list and targets (per-user accuracy, Calib, coverage, exposure Gini, nALRP
recomputation).

## Single runs

```bash
python -m stfr.train --dataset Amazon-VG --backbone LightGCN --method STFR \
    --ssns_frac 0.7 --ssns_alpha 1.0 --seed 20 --run_dir runs/demo/vg_lgcn_stfr_s20
python -m stfr.eval_ckpt --run_dir runs/demo/vg_lgcn_stfr_s20 --split test --topk 20 --rank_dump --dump_recs
```

`bash scripts/train_eval.sh <run_dir> <train arguments>` chains the two commands.
`python -m stfr.train --help` lists every argument; unspecified learning rates,
regularization weights and TIDE / CausalEPP initial values take the dataset- and
backbone-specific defaults of `stfr/config.py`.

## Statistical reporting

`analysis/per_user_tests.py` averages each user's metric over the three seeds (users
evaluated in every seed) and compares STFR with the compared method that has the
highest test mean for the metric and cutoff at hand, using an unadjusted two-sided
paired t-test (Wilcoxon signed-rank p-values are also written).  The opponent choice is
descriptive and separate from validation-based selection.  Non-significance does not
establish equivalence.

## Compute

The manuscript's runs used one NVIDIA H100 NVL.  Approximate per-epoch times of the
base model with training-time validation on the same GPU: Amazon-VG a few seconds (MF)
to tens of seconds (SimGCL); Amazon-Movies about one to a few minutes; Douban several
minutes (SimGCL was not run on Douban).  CPU execution works (`--device cpu`) and is
practical for Amazon-VG.
