# Training and model selection

## Protocol (`scripts/run_cell.py`, manuscript Section 7.1.3)

Every method follows the same procedure in every dataset x backbone setting:

1. **Fixed settings.** Datasets, chronological split and backbone settings are fixed and not
   tuned ([hyperparameters.md](hyperparameters.md)).  The only exception is the SimGCL
   contrastive weight, which is selected on validation with the unmodified backbone (arm
   `base`, same search budget as below) and shared by all methods of the setting.
2. **Search.** Every candidate of the method's search range (`configs/grids.json`) is
   trained with one seed (20) for at most 30 epochs, validated every 3 epochs, and stopped
   after 15 epochs without a validation Recall@20 improvement.  The candidate with the
   highest validation Recall@20 is selected (the first in grid order on ties).  PDA trains
   its three popularity exponents (training-time validation served at forecast coefficient
   0.1) and serves each checkpoint with the four forecast coefficients (twelve candidates).
3. **Finals.** The selected configuration is trained with three seeds (20, 21, 22) for at
   most 200 epochs, with early stopping on validation Recall@20 (validation every 3 epochs,
   patience 15).  DDC finals use the unmodified-backbone final of the same seed as the
   frozen backbone.
4. **Test.** For each final run, the checkpoint with the highest validation Recall@20 is
   evaluated once on the test window at K = 20.  Test data never influence selection.
5. **STFR (forecast).** The forecast coefficient η is selected for each setting from
   {0, 0.05, 0.1, 0.2, 0.5}.  Each of the three final STFR models is served on the full
   validation set with every η (no model is retrained), and the selection uses the mean
   validation Recall@20 of the three models.  Candidates whose relative improvement over
   η = 0 is within 0.1 percentage points of the best are treated as ties, and the smallest η
   among them is chosen.  The test window is then evaluated once with the selected η.

**Douban-Movie.** Validation during training (the curve, early stopping and checkpoint
selection of every run) uses a fixed sample of 4,000 users (seed 12345) to limit cost.
The candidates of each method are then compared on the full validation set: every
candidate's best checkpoint is re-evaluated on it (PDA's served candidates are evaluated on
it directly).  The two exceptions are the SimGCL contrastive weight and the STFR
configuration with SimGCL, which are selected on the 4,000-user sample
(`"select_on_training_subset": ["SimGCL"]` of the STFR arm in `configs/grids.json`).  On
Amazon-VG and Amazon-Movies, validation always uses all validation users.

**Reuse of search runs.** A seed-20 search run that early stopping ended before the 30-epoch
cap is a complete seed-20 final (the same training run with a longer cap would have stopped
at the same epoch), so `run_cell.py` reuses it, as the manuscript's runs did.  PDA and STFR
finals are always trained anew (`"reuse_sweep": false`): PDA's search checkpoints were
selected under forecast coefficient 0.1, and the manuscript's STFR finals were all trained
anew.

**Losses.** Every method's BPR loss is a batch mean, including IPS (weights normalized to
batch mean 1) and DICE (every BPR term divided by the batch size, as in the official
`mask_bpr_loss`); the manuscript's IPS and DICE results were obtained with these batch-mean
losses.

## Statistical reporting

Reported results are three-seed means.  `analysis/per_user_tests.py` computes each user's
Recall@20 and NDCG@20 from the saved top-20 lists, averages them over the three seeds
(users evaluated in every seed) and compares STFR and STFR (forecast) with the strongest
competing method using a two-sided paired t-test over users.  The strongest competing
method is the compared method with the highest test mean for the setting and metric; this
choice is descriptive and separate from the validation-based selection.  Significance is
assessed at p < .05 with no multiple-comparison adjustment; non-significance does not
establish equivalence.

## Block configurations (Section 7.5)

`scripts/run_block_configs.sh` prepares the Amazon-VG splits with L = 15, 30 and 120 days
(same training interactions and evaluation-block start as the main split) and runs the
unmodified backbone and STFR on each with the protocol above: the STFR (f, alpha) search
uses seed 20 and at most 30 epochs, and the SimGCL contrastive weight is selected on the
unmodified backbone.  L = 60 is the main comparison and reuses its runs.  The validation
and test windows cover 6/9, 14/16, 29/31 and 59/61 calendar days for L = 15, 30, 60 and 120
(settings `--val_days` = L/2; see [data_and_evaluation.md](data_and_evaluation.md)).

## Run layout

```
runs/<data>/<backbone>/lamsel/L<lambda>/              SimGCL contrastive-weight selection
runs/<data>/<backbone>/sweep/<arm>_<config>/          search candidates (serve_<alpha>/ for PDA)
runs/<data>/<backbone>/serve/STFR_forecast/...        validation serving of every STFR final with each eta
runs/<data>/<backbone>/final/<arm>_<config>_s<seed>/
    config.json  train.log  best.pth  metrics.json  arm.json
    test_k20.{log,json}  test_recs_k20.txt
runs/<data>/<backbone>/selection.json, lambda.json, summary.csv
```

`selection.json` records, per arm, the selected configuration and the validation score of
every candidate; for `STFR_forecast` the mean validation Recall@20 and the relative
improvement of every η.  A `final/STFR_forecast_...` directory is a serving run of the STFR
final of the same seed (same checkpoint, `fresh_forecast` set in its `config.json`).

`test_recs_k20.txt` stores each user's top-20 list and targets; per-user accuracy, Calib,
coverage and exposure Gini are computed from it, and a final run whose list dump is
missing is evaluated again from its checkpoint when `run_cell.py` is re-run.
`test_ranks.npz` (strict rank of every target item) is written only with `--rank_dump`; no
table reads it.  Accuracy, nALRP and the list dump of one evaluation use the same top-K set.

## Single runs

```bash
# example values from the search ranges, not a selected configuration
python -m stfr.train --dataset Amazon-VG --backbone LightGCN --method STFR \
    --ssns_frac 0.5 --ssns_alpha 0.75 --seed 20 --run_dir runs/demo/vg_lgcn_stfr_s20
python -m stfr.eval_ckpt --run_dir runs/demo/vg_lgcn_stfr_s20 --split test --topk 20 --dump_recs
# STFR (forecast): the same checkpoint served with a forecast coefficient (no retraining)
python -m stfr.eval_ckpt --run_dir runs/demo/vg_lgcn_stfr_s20 --split val --fresh_forecast 0.1
```

`bash scripts/train_eval.sh <run_dir> <train arguments>` chains the first two commands.
`python -m stfr.train --help` lists every argument; unspecified learning rates,
regularization weights and TIDE / CausalEPP initial values take the dataset- and
backbone-specific defaults of `stfr/config.py`.

## Compute

The manuscript's runs used an NVIDIA H100 NVL.  Approximate per-epoch times of the
unmodified backbone with training-time validation on the same GPU: Amazon-VG a few seconds
(MF) to tens of seconds (SimGCL); Amazon-Movies about one to a few minutes; Douban-Movie
several minutes (SimGCL several minutes per epoch with three concurrent runs).  CPU
execution works (`--device cpu`) and is practical for Amazon-VG.
