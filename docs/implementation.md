# Implementation and environment

## Code layout

| Path | Content |
| --- | --- |
| `stfr/config.py` | arguments, dataset / backbone defaults, run directories |
| `stfr/signals.py` | stale signal d_i, fresh signal P_i^t, PDA / DICE / CausalEPP tables, decayed popularity (C++) |
| `stfr/data.py` | BPR triplets with the SSNS negative sampler (and the DICE margin sampler) |
| `stfr/samplers.py` | DNS, AUC-NS and FairNeg samplers used as controls (Table 11); PNS is an auxiliary option not used in any reported table |
| `stfr/models.py` | MF / LightGCN / SimGCL backbones; base, STFR, IPS, DICE, DDC, PDA, TIDE, CausalEPP scores |
| `stfr/evaluate.py` | full-catalog evaluation, method-specific serving scores, dumps |
| `stfr/train.py`, `stfr/eval_ckpt.py` | entry points |
| `cppcode/` | time-decayed popularity of TIDE / CausalEPP (pybind11, built by cppimport) |
| `prep/` | dataset adapters and the chronological split |
| `scripts/` | protocol driver and experiment scripts |
| `analysis/` | tables and figures from saved outputs (no training) |
| `reproduction/` | records of the manuscript's runs and scripts that re-aggregate them |

## STFR

Training score `m_ui + softplus(w_i) P_i^t` (Eq. 4) with P_i^t the fresh signal of the
block containing the interaction; serving score `m_ui + softplus(w_i) P_i^{t*-1}` (Eq. 7)
from the same checkpoint.  The gains w_i are per-item parameters initialized so that
softplus(w_i) = 1 (`--shared_gain` replaces them by one scalar).  Both signals are
computed once from the training split (`stfr/signals.py`).

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
| CausalEPP | the authors' code (built on TIDE's), including the quality-ordering loss on item frequencies; user sensitivity and per-block item popularity tables from the split |
| IPS | the MF-IPS baseline shipped with the TIDE code (capped inverse propensity on the normalized log popularity) |
| DICE | the TIDE code's baseline aligned with the official DICE release: interest / conformity BPR terms per popularity case, mean-L1 discrepancy, margin-based popularity-conditioned negative sampling with per-epoch decay |
| PDA | rebuilt on the official popularity definition: Laplace smoothing, per-block min-max normalization, linear extrapolation at serving |
| DDC | port of the official two-stage procedure: frozen base embeddings, per-user offsets along the popularity direction (head-tail centroid difference) and the user's preference direction |
| LightGCN | inherited from the TIDE code; the normalized graph is built sparsely (identical values) |
| SimGCL | encoder and InfoNCE ported from the SELFRec implementation: per-layer noise of fixed norm, two perturbed passes, ego layer excluded from the layer mean, InfoNCE over the batch's unique users and positive items |
| PNS / DNS / AUC-NS / FairNeg | `stfr/samplers.py`; AUC-NS follows the authors' released code, FairNeg's genre groups are replaced by popularity deciles |

## Environment used for the manuscript

Ubuntu 24.04, one NVIDIA H100 NVL (95 GB); Python 3.13.11, PyTorch 2.10.0 (CUDA 12.8,
cuDNN 9.10), NumPy 2.5.1, pandas 3.0.0, SciPy 1.17.0, pybind11 3.0.4, cppimport 26.4.17,
matplotlib 3.10.8.  `requirements.txt` lists the minimum versions; the code also runs on
CPU (`--device cpu`).

The NumPy and PyTorch seeds are fixed and `torch.backends.cudnn.deterministic` is set.
What was verified: on CPU, two training epochs on Amazon-VG (every method with MF; base,
STFR and TIDE with LightGCN; base and STFR with SimGCL) match the original experiment
code to the printed digits, and saved checkpoints of the reported runs re-evaluate to
their logged test metrics.  Bit-identical repetition of a full GPU
training run was not tested (sparse graph products and scatter operations on CUDA are
not guaranteed to be deterministic); results can also differ across GPU models and
library versions at the level of floating-point rounding.
