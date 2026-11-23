# Reproduction records

Per-seed test records of the manuscript's runs, exported from the original experiment logs
(parsed again from the logs, five recorded decimals), and the script that re-aggregates the
main comparison from them.  Only per-seed rows are stored; seed means, ranks and differences
are computed from them.  Full training logs, list dumps, checkpoints and trained weights
are not distributed.  Re-aggregating these records is not the same as re-running training
and evaluation with the public code, which is what `scripts/` does; the analysis scripts in
`analysis/` compute everything else from the outputs of those runs.

What the records do not contain: run identifiers, run settings and the selected
hyperparameter values of any method (the search ranges are in `configs/grids.json`), and
the validation scores of the search candidates.  The only selected value that can be
read from the records is the recency window N of the training on recent history only, which
the manuscript reports (the window with the highest three-seed mean validation Recall@20).

## Records

| Experiment | File | Content |
| --- | --- | --- |
| Main comparison | `main_per_seed_k20.csv` | Recall@20, NDCG@20 and nALRP@20 of every final run: 9 settings x (9 methods + STFR (forecast)) x seeds 20/21/22 = 270 rows |
| Component ablation (Amazon-VG) | `components_vg_per_seed_k20.csv` | unmodified backbone, SSNS only, fresh channel only, STFR (shared gain), STFR |
| Training on recent history only (Amazon-VG) | `recency_vg_per_seed_k20.csv` | every window N with its validation Recall@20 (best epoch of each run) and test results; the reported N is the one with the highest three-seed mean validation Recall@20 |
| Replacing SSNS with other negative samplers (Amazon-VG) | `samplers_vg_per_seed_k20.csv` | SSNS replaced by DNS, AUC-NS or FairNeg (the validation-selected sampler settings) and SSNS (STFR) |
| Temporal block configurations (Amazon-VG) | `block_configurations_vg_per_seed_k20.csv` | L = 15, 30, 60, 120 days: unmodified backbone and STFR per seed (L = 60 is the main comparison) |

```bash
python reproduction/rebuild_ranks.py     # main_per_seed_k20.csv -> results/: means, column ranks, mean ranks (no model is run)
```

checks that every setting has the nine methods and STFR (forecast) with seeds 20, 21 and 22
exactly once (270 rows) and that the mean rank averages six columns per backbone, and
writes the three-seed means (unrounded), the column ranks and the mean ranks per backbone
and overall to `results/` (`--out`): `main_means_k20.csv`, `main_column_ranks_k20.csv`,
`main_mean_rank_k20.csv` and `MEAN_RANK_K20.md`.  Ranks are computed on the unrounded three-seed means, as in the
manuscript (`--rank_on displayed` ranks the printed four-decimal values instead; see
`docs/metrics_and_experiments.md`).  The nine methods are ranked among themselves; STFR
(forecast) is ranked against the other eight with STFR removed.

Displayed values are `'%.4f'` of the floating-point mean of the recorded values (`display`
in `analysis/common.py`).  Five accuracy means fall exactly on a rounding boundary of the
five-decimal records (Amazon-VG / MF TPAB N@20 0.03745, Amazon-Movies / MF DICE R@20
0.03285, Amazon-Movies / LightGCN STFR (forecast) N@20 0.04235, Douban-Movie / LightGCN
unmodified backbone N@20 0.01985, Douban-Movie / SimGCL DDC N@20 0.03895); the
floating-point mean decides the side, which prints .0375 / .0328 / .0423 / .0198 / .0390 as
in the manuscript.  No rank depends on these cells.

Conventions of the records: `method` uses the names of `analysis/common.py` (`base` = the
unmodified backbone, `STFR_forecast` = STFR (forecast)); datasets are Amazon-VG,
Amazon-Movies and Douban-movie (Douban-Movie in the manuscript).  Every arm of the Amazon-VG
control experiments uses the manuscript's STFR definition (min-max fresh signal, 200-epoch
cap, patience 15, validation every 3 epochs, seeds 20/21/22); SSNS only and STFR (shared
gain) inherit the setting's selected (f, alpha).  The checkpoints of the main comparison are
the best-validation checkpoints of each run's completed training attempt.
