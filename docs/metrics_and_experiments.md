# Metrics and reported experiments

## Popularity scale

With c_i the cumulative training count of item i and the catalog I,

    d_i = (log(1 + c_i) - min_j log(1 + c_j)) / (max_j log(1 + c_j) - min_j log(1 + c_j) + eps)

(Eq. 1, eps = 1e-12).  The fresh signal P_i^t applies the same transformation to the
block-t count within the items active in block t (zero for inactive items, Eq. 2).
For evaluated users U and top-K lists R_u^K,

    nALRP@K = 1 / (|U| K) sum_u sum_{i in R_u^K} d_i        (Eq. 3)

is the mean stale popularity of the recommendations (K = 20).  The training reference
(`analysis/train_vs_rec_popularity.py`) is mu_train = sum_i c_i d_i / sum_i c_i (every
training interaction weighted equally).

## Accuracy and mean-popularity gap

Recall@K and NDCG@K use binary relevance and full-catalog ranking with the masks and
targets of [data_and_evaluation.md](data_and_evaluation.md); NDCG is normalized by the ideal
DCG of min(K, number of targets); K = 20.  Test metrics are computed on the
validation-selected checkpoint.

    Calib@K(u) = | mean_{i in R_u^K} d_i - mean_{i in GT_u} d_i |        (Eq. 15)

with GT_u the user's test-window targets; Calib@20 is averaged over users within a seed and
then over the three seeds.

## Mean rank

Within each dataset-backbone setting the three seeds of every method are averaged first
(seed values are never rounded); Recall@20 and NDCG@20 are then ranked within the setting
on these unrounded means (rank 1 = highest).  The mean rank of a method on a backbone
averages its column ranks over the backbone's three datasets (six columns per backbone);
the overall mean rank averages the three backbones.  The nine methods (unmodified
backbone, IPS, DICE, DDC, PDA, TIDE, CausalEPP, TPAB, STFR) are ranked among themselves;
STFR (forecast) is ranked against the other eight methods with STFR removed, so it never
changes another method's rank.  The non-personalized fresh ranking and nALRP do not enter
the mean rank.  `analysis/collect_results.py` writes the displayed four-decimal values, the
column ranks and the mean ranks into `results/main_table.csv` and `results/mean_rank.csv`;
`reproduction/rebuild_ranks.py` recomputes them from the per-seed records.

In four columns two methods print the same four-decimal value (Amazon-VG / LightGCN
Recall@20: DICE and TPAB .0714; Amazon-VG / SimGCL NDCG@20: unmodified backbone and DICE
.0361; Amazon-Movies / LightGCN NDCG@20: unmodified backbone and TPAB .0234; Douban-Movie /
MF NDCG@20: unmodified backbone and DDC .0262).  The unrounded means order them, as in the
manuscript; `--rank_on displayed` instead gives equal displayed values the average rank,
which moves the mean rank of the unmodified backbone, DICE and DDC on the affected
backbones by 1/12 (half a column rank over six columns) and changes no mean rank of STFR,
STFR (forecast) or the injection-oriented methods.

## Coverage and exposure Gini

Over the training-active catalog I+ = {i : c_i > 0} (n = 14,189 / 54,774 / 23,425 for
Amazon-VG / Amazon-Movies / Douban-Movie), v_i counts the users whose top-K list contains i
(zero-exposure items included):

    Coverage@K = |{i in I+ : v_i > 0}| / n
    Gini@K     = sum_j (2j - n - 1) v_(j) / (n sum_j v_(j))     (ascending v_(j))

Both are computed per seed and averaged; standard deviations use the population
convention.  Lower nALRP does not imply broader coverage or lower Gini.

## Embedding geometry

Directional concentration (cos_pop) is the mean cosine similarity of 4,000 item pairs drawn
with replacement from the top 1% of items by stale popularity (cumulative training count),
excluding pairs with identical normalized embeddings; the norm ratio (nrm_ratio) is the
median norm of that group divided by the median norm of the catalog.  Item embeddings are
the representation the serving score uses: MF the learned item table; LightGCN the mean of
layers 0..3 (layer 0 = the learned table before propagation, layer l = l propagations over
the training graph); SimGCL the mean of layers 1..3, as in its reference encoder, without
training-time noise.  `analysis/embedding_geometry.py` rebuilds this representation from
the saved checkpoint and the training graph; `embedding_geometry.csv` holds the seed means
over the checkpoints.

## Experiments and commands

Run the experiments first (training writes the checkpoints, test metrics and top-20 list
dumps that every analysis reads); no analysis script trains a model.

| Experiment or analysis | Command | Output (`results/`, `figures/`) |
| --- | --- | --- |
| Main comparison: search, three-seed finals, STFR (forecast), test | `bash scripts/run_main.sh` | `runs/` |
| Recall@20, NDCG@20, nALRP@20, mean rank, comparison with the strongest compared method | `python analysis/collect_results.py` | `main_table.csv`, `mean_rank.csv`, `summary.csv`, `per_seed.csv`, `improvement_vs_strongest_baseline.csv` |
| Recall@20 vs nALRP@20 per setting (plot) | `python analysis/plot_pareto.py` (after `collect_results.py`) | `pareto_recall_nalrp.pdf` |
| Paired accuracy tests and Calib@20 | `python analysis/per_user_tests.py` | `per_user_tests.csv`, `calib_table.csv`, `calib_per_seed.csv` |
| Coverage and exposure Gini | `python analysis/coverage_gini.py` | `exposure_summary_k20.csv`, `coverage_gini.csv`, `coverage_gini_per_seed.csv` |
| Embedding analysis (directional concentration, norm ratio) | `python analysis/embedding_geometry.py` | `embedding_geometry.csv`, `embedding_geometry_per_seed.csv` |
| Popularity signals: non-personalized stale / fresh rankings, popularity persistence | `python analysis/popularity_signals.py` | `popularity_signals.csv` |
| Training vs recommendation stale popularity | `python analysis/train_vs_rec_popularity.py` (after `collect_results.py`) | `train_vs_rec_popularity.csv` |
| Dataset statistics, chronological split, window dates | `python analysis/dataset_stats.py`, `python analysis/plot_split.py` | `dataset_table.csv`, `split_blocks.csv`, `split_blocks.pdf` |
| Component ablation, training on recent history only, replacing SSNS with other negative samplers (Amazon-VG) | `bash scripts/run_ablations_vg.sh` (after `run_main.sh`), `python analysis/ablation_summary.py` | `ablation_means.csv`, `ablation_recency.csv`, `ablation_recency_all.csv` |
| Temporal block configurations on Amazon-VG (Section 7.5) | `bash scripts/run_block_configs.sh`, `python analysis/block_config_table.py`, `python analysis/plot_block_configs.py` | `block_configurations.csv`, `block_configurations.pdf` |
| Fixed settings and search ranges | – | `configs/grids.json`, [hyperparameters.md](hyperparameters.md) |

Per-user accuracy (paired tests), Calib@20, coverage and exposure Gini are computed from the
saved top-20 lists and targets (`test_recs_k20.txt`).  A missing list dump, or a stored list
shorter than the requested cutoff, stops the script; the rank dump (`test_ranks.npz`, 1 + the
number of items scored strictly higher than the target) is an optional raw output and no
analysis reads it, because targets with tied scores share a rank there while a list gives
them distinct positions.

Analyses of trained models read saved run outputs, recommendation lists or checkpoints
in the run layout of `scripts/run_cell.py` (`--runs runs`).  Dataset statistics
(`analysis/dataset_stats.py`) and the popularity-ranking diagnostics
(`analysis/popularity_signals.py`) read the preprocessed data instead (`--datasets`).  None
of these scripts trains a model, and all write to `results/` (`--out`).  See each script's
arguments for its required inputs.
