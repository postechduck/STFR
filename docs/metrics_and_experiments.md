# Metrics and reported experiments

## Popularity scale

With c_i the cumulative training count of item i and the catalog I,

    d_i = (log(1 + c_i) - min_j log(1 + c_j)) / (max_j log(1 + c_j) - min_j log(1 + c_j) + eps)

(Eq. 1, eps = 1e-12).  The fresh signal P_i^t applies the same transformation to the
block-t count within the items active in block t (zero for inactive items, Eq. 2).
For evaluated users U and top-K lists R_u^K,

    nALRP@K = 1 / (|U| K) sum_u sum_{i in R_u^K} d_i        (Eq. 3)

is the average stale popularity of the recommendations (K = 20).  The training
reference of Table 2 is mu_train = sum_i c_i d_i / sum_i c_i (every training interaction
weighted equally).

## Accuracy and mean-popularity gap

Recall@K and NDCG@K use binary relevance and full-catalog ranking with the masks and
targets of `docs/data_and_evaluation.md`; NDCG is normalized by the ideal DCG of
min(K, number of targets); K = 20.  Test metrics are computed on the
validation-selected checkpoint.

    Calib@K(u) = | mean_{i in R_u^K} d_i - mean_{i in GT_u} d_i |

with GT_u the user's test-window targets; Calib@20 is averaged over users.

## Mean rank (Tables 4-5)

Within each dataset-backbone setting, the eight compared methods (base, IPS, DICE, DDC,
PDA, TIDE, CausalEPP, STFR) are ranked on the unrounded three-seed mean of Recall@20 and
of NDCG@20 (rank 1 = highest).  The mean rank of a method on a backbone averages these
column ranks over the backbone's datasets: six columns for MF and LightGCN (three
datasets x two metrics), four for SimGCL (two datasets x two metrics).  Ranks are
computed on the unrounded means, so equal values at four displayed decimals are not
ties; the average rank is used only when the unrounded means are exactly equal (none
occurred in the reported runs).  The non-personalized fresh prior is not ranked.
`analysis/collect_results.py` writes the ranks and the mean rank into `results/main_table.csv`.

## Coverage and exposure Gini

Over the training-active catalog I+ = {i : c_i > 0} (n = 14,189 / 54,774 / 23,425 for
VG / Movies / Douban), v_i counts the users whose top-K list contains i (zero-exposure
items included):

    Coverage@K = |{i in I+ : v_i > 0}| / n
    Gini@K     = sum_j (2j - n - 1) v_(j) / (n sum_j v_(j))     (ascending v_(j))

Both are computed per seed and averaged; standard deviations use the population
convention.  Lower nALRP does not imply broader coverage or lower Gini.

## Manuscript to code map

| Manuscript | Output | Command |
| --- | --- | --- |
| Table 1 (non-personalized popularity signals) | `results/popularity_signals.csv` | `python analysis/popularity_signals.py` |
| Table 2 (training vs recommendation popularity) | `results/train_vs_rec_popularity.csv` | `python analysis/train_vs_rec_popularity.py` |
| Table 3, Figure 2 (datasets, split) | `results/dataset_table.csv`, `results/split_blocks.csv`, `figures/split_blocks.pdf` | `python analysis/dataset_stats.py`, `python analysis/plot_split.py` |
| Tables 4-5 (accuracy), 6-7 (nALRP), Figure 3 | `results/main_table.csv`, `results/summary.csv`, `figures/pareto_recall_nalrp.pdf` | `bash scripts/run_main.sh`, `python analysis/collect_results.py`, `python analysis/plot_pareto.py` |
| Table 8 (Calib@20), Table E.1 (paired tests) | `results/per_user_calib.csv`, `results/per_user_tests.csv` | `python analysis/per_user_tests.py` |
| Tables 9-11 (components, recency window N, sampler swaps) | `results/ablation_components.csv`, `results/ablation_recency.csv`, `results/ablation_samplers.csv` | `bash scripts/run_ablations_vg.sh`, `python analysis/ablation_tables.py` |
| Section 7.4, Figure 4, Table E.3 (block configurations) | `results/block_configurations.csv`, `figures/block_configurations.pdf` | `bash scripts/run_block_configs.sh`, `python analysis/block_config_table.py`, `python analysis/plot_block_configs.py` |
| Section 7.5, Tables 12-13 (embedding geometry) | `results/embedding_geometry.csv` | `python analysis/embedding_geometry.py` |
| Table E.2 (serving interventions) | `results/serving_interventions.csv` | `python analysis/serving_interventions.py` |
| Table E.4 (coverage, exposure Gini at @20) | `results/coverage_gini.csv`, `results/exposure_summary_k20.csv` | `python analysis/coverage_gini.py` |

Geometry: cos_pop is the mean cosine over 4,000 random pairs of items in the top 1% by
cumulative count, nrm_ratio the ratio of that group's median norm to the catalog's
median norm.  Item embeddings are the representation the serving score uses: MF the
learned item table; LightGCN the mean of layers 0..3 (layer 0 = the learned table
before propagation, layer l = l graph propagations over the training graph); SimGCL
the mean of layers 1..3, as in its reference encoder, without training-time noise.
This is the default of `analysis/embedding_geometry.py` (`--simgcl_include_layer0` is a
legacy option and not the manuscript's convention).  The interventions (norm removal,
direction removal) re-rank test users with the same serving representation of base
checkpoints, and the cos_pop / nrm_ratio columns of Table E.2 are computed on that
same representation.

Per-user accuracy in `analysis/per_user_tests.py` is computed from the saved top-K lists
(`--from_ranks` uses the rank dump instead; the two differ only when tied scores fall on
the cutoff boundary).

All analysis scripts read the run layout of `scripts/run_cell.py` (`--runs runs`) and
write to `results/` (`--out`); none of them trains or re-evaluates a model.
