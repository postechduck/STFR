# Reproduction records

Lightweight records that tie the manuscript's reported numbers to the original three-seed
runs.  They are exports of the original experiment logs (parsed, unrounded); nothing here
is produced by re-running the public code, and the public code never reads these files
(every analysis script recomputes its tables from `runs/`).  Full training logs and
checkpoints are not distributed; the `log`/`tag` columns name the original run
identifiers so that a reported value can be traced to one record.

| File | Content |
| --- | --- |
| `per_seed_k20.csv` | Recall@20 / NDCG@20 of every final run (8 settings x 8 methods x seeds 20/21/22 = 192 rows) |
| `means_unrounded_k20.csv` | three-seed means (unrounded) per setting and method, with the seed count |
| `column_ranks_k20.csv` | rank of every method in every (setting, metric) column |
| `mean_rank_by_backbone.csv` | mean rank per backbone (Tables 4-5): 6 columns for MF / LightGCN, 4 for SimGCL |
| `table9_ablation_per_seed.csv` | Table 9 (VG components): per-seed Recall@20 / NDCG@20 / nALRP@20 and the run setting of every variant |
| `table10_recency_candidates_val_test.csv` | Table 10: every candidate window N with validation Recall@20 (selection criterion), test means, and the selected N |
| `table10_recency_per_seed_k20.csv` | Table 10: per-seed records of every N |
| `table11_sampler_sweep_val.csv` | Table 11: validation sweep of the sampler hyperparameters and the selected value |
| `table11_sampler_per_seed_k20.csv` | Table 11: per-seed test records of the selected sampler configurations |
| `stfr_selection_sweep_12pt.csv` | STFR's 12-point (f, alpha) validation sweep in every setting and the selected point |
| `RANK_AUDIT_K20.md`, `AUDIT_TABLES_9_10_11.md` | audit notes (Korean) written when these records were checked against the manuscript |

Conventions of the records: mean rank uses the unrounded means and averages ranks only
for exactly equal values (none occurred); the non-personalized fresh prior is not
ranked.  Every arm of Tables 9-11 uses the manuscript's STFR definition (min-max fresh
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

Historical names in the records: `SFDmmNoB` / `SFD-noB champion` = STFR; `ARP` /
`rec_pop` = nALRP (normalized log-count scale); `noB` = no explicit stale slot.
