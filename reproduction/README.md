# Reproduction records

Lightweight records that tie the manuscript's reported numbers to the original three-seed
runs.  The files in this directory are exports of the original experiment logs (parsed,
unrounded, five recorded decimals); the files in `tables/` are the outputs of this
repository's `analysis/` scripts on the original runs' saved outputs (test metrics, top-20
list dumps, checkpoints).  Full training logs, list dumps and checkpoints are not
distributed; the `log` / `tag` columns name the original run identifiers so that a reported
value can be traced to one record.  Re-aggregating these records is not the same as
re-running training and evaluation with the public code, which is what `scripts/` does.

## Records of the original logs

| File | Content |
| --- | --- |
| `per_seed_k20.csv` | Recall@20 / NDCG@20 of every final run (8 settings x 8 methods x seeds 20/21/22 = 192 rows) |
| `means_unrounded_k20.csv` | three-seed means (unrounded) per setting and method, with the seed count |
| `column_ranks_k20.csv` | unrounded mean, displayed four-decimal value and rank of every method in every (setting, metric) column |
| `mean_rank_by_backbone.csv`, `MEAN_RANK_K20.md` | mean rank per backbone (Tables 4-5): 6 columns for MF / LightGCN, 4 for SimGCL |
| `table9_ablation_per_seed.csv` | Table 9 (VG components): per-seed Recall@20 / NDCG@20 / nALRP@20 and the run setting of every variant |
| `table10_recency_candidates_val_test.csv` | Table 10: every candidate window N with validation Recall@20 (selection criterion), test means, and the selected N |
| `table10_recency_per_seed_k20.csv` | Table 10: per-seed records of every N |
| `table11_sampler_sweep_val.csv` | Table 11: validation sweep of the sampler hyperparameters and the selected value |
| `table11_sampler_per_seed_k20.csv` | Table 11: per-seed test records of the selected sampler configurations |
| `stfr_selection_sweep_12pt.csv` | STFR's 12-point (f, alpha) validation sweep in every setting and the selected point |

```bash
python reproduction/rebuild_ranks.py     # per_seed_k20.csv -> means, column ranks, mean rank (no model is run)
```

checks that every setting has the eight methods with seeds 20, 21 and 22 exactly once
(192 rows) and that the mean rank averages 6 / 6 / 4 columns, and rewrites the three rank
files.  Mean rank: ranks are computed from the displayed four-decimal three-seed mean
values, with average ranks assigned to ties (seed values are never rounded before
averaging; the non-personalized fresh prior and nALRP are not ranked).  Two columns hold a
displayed tie (base and CausalEPP, NDCG@20 of VG / SimGCL and Movies / SimGCL), which gives
base 4.75 and CausalEPP 4.25 on SimGCL; STFR is 1.33 / 1.00 / 1.25 on MF / LightGCN /
SimGCL.  Improvements, tests and selection use the unrounded records.

Displayed values are `'%.4f'` of the mean of the recorded values (`display` in
`analysis/common.py`).  Four accuracy means fall exactly on a rounding boundary of the
five-decimal records (VG / LightGCN DICE N@20 0.02785, Movies / LightGCN TIDE N@20 0.02815,
Movies / SimGCL IPS R@20 0.05175, Douban / MF IPS R@20 0.02975); this rule prints
.0278 / .0281 / .0517 / .0297, as in the manuscript (all 128 accuracy cells agree).  At full
precision (recomputed from the list dumps) the TIDE mean is 0.028149.  No rank depends on
these four cells.

## Outputs of the analysis scripts on the original runs (`tables/`)

| File | Manuscript | Script |
| --- | --- | --- |
| `table1_popularity_signals.csv` | Table 1 | `analysis/popularity_signals.py` |
| `table2_train_vs_rec_popularity.csv` | Table 2 | `analysis/train_vs_rec_popularity.py` |
| `table3_datasets.csv`, `figure2_split_blocks.csv` | Table 3, Figure 2 (training blocks, pre-filter evaluation halves, blocks after the evaluation block) | `analysis/dataset_stats.py` |
| `nalrp_per_seed_k20.csv` | Tables 6-7: nALRP@20 of every final run (192 rows, with the original log) | `analysis/collect_results.py` |
| `table8_calib_k20.csv`, `calib_per_seed_k20.csv` | Table 8: Calib@20 of STFR, PDA, TIDE and base; per-seed values of every method | `analysis/per_user_tests.py` |
| `tableA1_paired_tests_k20.csv` | Table A.1: paired accuracy tests from the saved top-20 lists | `analysis/per_user_tests.py` |
| `tableA2_coverage_gini_k20.csv`, `coverage_gini_all_methods_k20.csv`, `coverage_gini_per_seed_k20.csv` | Table A.2 (base, PDA, TIDE, STFR); seed means and standard deviations of every method; per-seed values | `analysis/coverage_gini.py` |
| `tables12_13_embedding_geometry.csv`, `tables12_13_embedding_geometry_per_seed.csv` | Tables 12-13 (serving representation of each backbone) | `analysis/embedding_geometry.py` |
| `figure4_block_configurations_k20.csv`, `figure4_block_configurations_per_seed_k20.csv` | Section 7.4, Figure 4 (@20) | `analysis/block_config_table.py` |

The list dumps behind Tables 8, A.1 and A.2 were written on the GPU of the original runs.
Re-evaluating a checkpoint on another device can order tied scores differently (on CPU the
top-20 sets of 146 of 21,342 users change for Movies / SimGCL PDA, the setting with many
tied scores, and of no user in the other settings that were checked); the aggregate
metrics agree to the recorded five decimals.

Conventions of the records: every arm of Tables 9-11 uses the manuscript's STFR definition (min-max fresh
signal, no explicit stale slot, 200-epoch cap, patience 15, validation every 3 epochs,
seeds 20/21/22); `SSNS only` and `STFR (shared gain)` inherit the selected (f, alpha) of
the setting, the recency window N and the sampler hyperparameters are selected on
validation Recall@20 (Douban settings select STFR's (f, alpha) on the full validation
split after a 4,000-user sweep).

Correspondence between the records' original run identifiers and this repository:

| original identifier | this repository |
| --- | --- |
| `v3_<data>_<backbone>_base_s<seed>` | arm `base` |
| `v3_<data>_<backbone>_final_<method>_<slug>_s<seed>` | arms `IPS`, `DICE`, `DDC`, `PDA`, `TIDE`, `CausalEPP` (selected configurations in `configs/selected_paper.json`) |
| `v3nob_final_<data>_<backbone>_SFDmmNoB_f<f>a<alpha>_s<seed>` | arm `STFR` (`--ssns_frac f --ssns_alpha alpha`) |
| `v3nobABL_..._Negonly` / `..._Fonly` | `SSNS_only` / `fresh_only` (`configs/ablations_vg.json`) |
| `v3wgN_VG_<backbone>_s<seed>` | `shared_gain` (`--shared_gain`) |
| `v3nobREC_..._recentN<N>` / `..._recentFN<N>` | `recent<N>` / `recent<N>_fresh` (`--recent_blocks N`) |
| `tuned_v3ax2N_<backbone>_swap_<dns|aucns|fairneg>_<value>` | `swap_DNS` / `swap_AUCNS` / `swap_FairNeg` |

Historical names in the original run identifiers: `SFDmmNoB` / `SFD-noB champion` = STFR;
`ARP` / `rec_pop` = nALRP (normalized log-count scale; the record columns are named
`nalrp`); `noB` = no explicit stale slot.  The identifiers are kept for traceability.
