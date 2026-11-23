# Implementation and environment

## Code layout

| Path | Content |
| --- | --- |
| `stfr/config.py` | arguments, dataset / backbone defaults, run directories |
| `stfr/signals.py` | stale signal d_i, fresh signal P_i^t, PDA / DICE / CausalEPP tables, decayed popularity (C++) |
| `stfr/data.py` | BPR triplets with the SSNS negative sampler (and the DICE margin sampler) |
| `stfr/samplers.py` | DNS, AUC-NS and FairNeg samplers used as controls (replacing SSNS) |
| `stfr/models.py` | MF / LightGCN / SimGCL backbones; scores of the unmodified backbone (`base`), STFR, IPS, DICE, DDC, PDA, TIDE, CausalEPP, TPAB |
| `stfr/evaluate.py` | full-catalog evaluation, method-specific serving scores (including STFR (forecast)), dumps |
| `stfr/train.py`, `stfr/eval_ckpt.py` | entry points |
| `cppcode/` | time-decayed popularity of TIDE / CausalEPP (pybind11, built by cppimport) |
| `prep/` | dataset adapters and the chronological split |
| `scripts/` | protocol driver and experiment scripts |
| `analysis/` | measurements and summaries from saved run outputs (no training) |
| `reproduction/` | per-seed records of the manuscript's runs and the script that re-aggregates the main comparison |

## STFR

Training score `m_ui + softplus(w_i) P_i^t` (Eq. 4) with P_i^t the fresh signal of the
block containing the interaction; serving score `m_ui + softplus(w_i) P_i^{t*-1}` (Eq. 7)
from the same checkpoint.  The gains w_i are per-item parameters initialized so that
softplus(w_i) = 1 (`--shared_gain` replaces them by one scalar).  Both signals are
computed once from the training split (`stfr/signals.py`).

STFR (forecast) (Eq. 8) changes serving only: `--fresh_forecast eta` replaces P^{t*-1} by
clip(P^{t*-1} + eta (P^{t*-1} - P^{t*-2}), 0, 1) in `stfr/evaluate.py`, computed once for all
items from the last two training blocks for validation and test alike; eta = 0 is STFR.
The trained STFR checkpoint is used as it is (`python -m stfr.eval_ckpt --fresh_forecast`).

SSNS (`stfr/data.py`): a negative starts as a uniform draw from the catalog, redrawn
while it collides with the user's training positives (up to 16 rounds); a candidate
that still collides after the last round is replaced by an exact uniform draw from the
items outside the user's history, so no returned negative is a training positive (a
user whose history covers the whole catalog is rejected when the dataset is built).
With probability f the candidate is replaced by a draw from the catalog-level
distribution proportional to c_j^alpha (inverse-transform sampling on the cumulative
distribution); a replacement that collides with the user's training positives is
discarded and the uniform candidate is kept.  The 16-round limit was never the binding
case in the reported runs: the expected number of candidates still colliding after it
is below 1e-12 per epoch on every dataset, and the exact fallback consumes no random
number unless it is needed, so the negatives of a given seed are unchanged.

## Compared methods

| Method | Implementation |
| --- | --- |
| TIDE | the TIDE authors' code is the parent of this pipeline; `stfr/models.py` keeps its quality / conformity score and `cppcode/stfr_popularity.cpp` its time-decayed popularity (made dataset-independent); full serving (`q_i + b_i pop_i(t)`) as in its click-prediction mode |
| CausalEPP | the authors' code (ShiyinTan/CausalEPP @ df5be0c, built on TIDE's) with the settings of the original paper and official code: quality-ordering loss on item frequencies with weight 0.2 (`--cepp_lambda`; the released `main.py` adds it with weight 1); user sensitivity and item popularity tables over a six-month trailing window (three 60-day blocks, `--cepp_window 3`) with the top-20% high-popularity threshold recomputed for each window, as in the authors' preprocessing; smoothing 256 and alpha 0.5 as in the official configuration; served without the inference-time intervention.  Index error corrected: the released `main.py` (line 177) reads the frequency of the negative item from the positive item's index (`item_frequency_overall[item_i]` for both `frequency_i` and `frequency_j`), so the sign in the quality-ordering loss is always zero and the loss never moves the quality parameters; here `frequency_j` is read at `item_j` (`stfr/train.py`, `quality_loss`) |
| IPS | the MF-IPS baseline shipped with the TIDE / CausalEPP code (capped inverse propensity on the normalized log popularity), with the weighted BPR term averaged over the batch and the weights normalized to batch mean 1 (the lineage code sums over the batch, which makes the L2 and SimGCL terms relatively negligible) |
| DICE | the TIDE code's baseline aligned with the official DICE release (tsinghua-fib-lab/DICE @ 591b883): interest / conformity BPR terms per popularity case, each a mean over the full batch as in the official `mask_bpr_loss` (the lineage code summed them), mean-L1 discrepancy, margin-based popularity-conditioned negative sampling with per-epoch decay |
| PDA | rebuilt on the official popularity definition (zyang1580/PDA @ e8f1776): Laplace smoothing, per-block min-max normalization, linear extrapolation at serving from the last two training blocks; the forecast coefficient is searched over {0.05, 0.1, 0.2, 0.5} (0.5 is the official code's value) by re-serving the trained checkpoints |
| DDC | reimplementation of the official two-stage procedure (RecBole-based release, LingFeng-Liu-AI/DDC @ 37ae807) with its reproduction command: frozen base model; loss combination `b_a`, i.e. the positive item scored along the user's preference direction e_pre (beta_u ||e_u||) and the negative along the popularity direction e_pop (alpha_u ||e_u||); e_pre from the user's training items with the highest frozen base score (`y_uio`), e_pop as the head-tail centroid difference of the top / bottom 5% training-active items (paper Eq. 4); full serving with both offsets |
| TPAB | reimplementation of the official code (hsyoo32/tpab_ @ 7753efd; Yoo et al., KDD 2025) on this pipeline: popularity groups of the per-block absolute training counts (K groups, thresholds pmax^(g/K)), a per-user popularity embedding and per-group embeddings added to the matching score, popularity bootstrapping (BPR on [popularity, preference] with the popularity part permuted within the batch) weighted by lambda, and serving with the group of the extrapolated count c_{t*-1} + 0.2 (c_{t*-1} - c_{t*-2}); stages are the 60-day training blocks and pmax is taken over training blocks only |
| LightGCN | inherited from the TIDE code; the normalized graph is built sparsely (identical values) |
| SimGCL | encoder and InfoNCE ported from the SELFRec implementation (Coder-Yu/SELFRec @ 5b02294): per-layer noise of fixed norm, two perturbed passes, ego layer excluded from the layer mean, InfoNCE over the batch's unique users and positive items |
| DNS / AUC-NS / FairNeg | `stfr/samplers.py`; AUC-NS follows the authors' released code, FairNeg's genre groups are replaced by popularity deciles |

## Environment used for the manuscript

Ubuntu 24.04, one NVIDIA H100 NVL (95 GB); Python 3.13.11, PyTorch 2.10.0 (CUDA 12.8,
cuDNN 9.10), NumPy 2.5.1, pandas 3.0.0, SciPy 1.17.0, pybind11 3.0.4, cppimport 26.4.17,
matplotlib 3.10.8.  `requirements.txt` lists the minimum versions; the code also runs on
CPU (`--device cpu`).

The NumPy and PyTorch seeds are fixed and `torch.backends.cudnn.deterministic` is set.

What was verified (against the original experiment code and its saved runs):

* Data: `prep/prep_split.py` on the three raw interaction files gives splits whose content
  digests equal those of the manuscript's splits (`prep/verify_data.py`: "content same" for
  every file of the three datasets), and `analysis/dataset_stats.py` on them gives the
  manuscript's dataset statistics and window dates.
* Training: on the GPU, the first validation points of IPS and DICE with the batch-mean
  losses (Amazon-VG, seed 20: MF epochs 0, 3, 6 and SimGCL epochs 0, 3) equal the
  manuscript's search runs at the five recorded decimals.  Earlier checks of this
  repository whose code paths have not changed since: on CPU, two training epochs on Amazon-VG with the unmodified backbone, STFR, PDA and TIDE (MF), the
  unmodified backbone, STFR and TIDE (LightGCN), and the unmodified backbone and STFR
  (SimGCL) matched the original code to the printed digits; the CausalEPP Amazon-VG / MF
  seed-20 final matched the original run (training log and test metrics); and the first
  validation points (epochs 0 and 3) of the Amazon-VG / MF DDC and TPAB searches matched the
  original runs.
* Evaluation: final checkpoints of the manuscript's runs re-evaluate through
  `stfr/evaluate.py` to their logged Recall@20, NDCG@20 and nALRP@20 at the five recorded
  decimals: STFR (forecast) on validation and test in seven cases (Amazon-Movies MF,
  LightGCN, SimGCL; Douban-Movie MF; Amazon-VG MF; eta 0 to 0.5), and earlier DDC, TPAB and
  PDA finals (DDC: Amazon-VG / MF, Amazon-VG / LightGCN, Amazon-Movies / MF, Amazon-Movies /
  SimGCL, Douban-Movie / LightGCN; TPAB: Amazon-VG / MF, Amazon-Movies / LightGCN, Amazon-VG /
  SimGCL, Douban-Movie / SimGCL; PDA with forecast coefficient 0.5: Amazon-VG / LightGCN,
  Amazon-Movies / MF, Douban-Movie / SimGCL).
* Analysis: run on the saved outputs of the manuscript's final runs (test metrics, top-20
  list dumps, checkpoints) and on the splits of `prep/prep_split.py`, the analysis scripts
  reproduce the values of the original experiment records from which the manuscript reports
  the main comparison (Recall@20, NDCG@20, nALRP@20, mean ranks), the paired tests, Calib@20,
  coverage and exposure Gini, the embedding analysis, the popularity signals, the training
  vs recommendation stale popularity and the dataset statistics.  The per-seed records of
  the Amazon-VG control experiments and of the block configurations (`reproduction/`) were
  parsed again from the original logs and equal the manuscript's values.
  `scripts/run_cell.py`, including the STFR (forecast) selection, and the analysis scripts
  were run end to end on a short Amazon-VG / MF configuration (1-epoch caps) to check the
  pipeline, not to reproduce numbers.

Bit-identical repetition of a full GPU training run was not tested (sparse graph products
and scatter operations on CUDA are not guaranteed to be deterministic); results can also
differ across GPU models and library versions at the level of floating-point rounding.
