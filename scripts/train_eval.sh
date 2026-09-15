#!/usr/bin/env bash
# Train one configuration and evaluate its selected checkpoint on the test split at K=20.
#
#   bash scripts/train_eval.sh <run_dir> <stfr.train arguments...>
#   bash scripts/train_eval.sh runs/demo/vg_mf_stfr --dataset Amazon-VG --backbone MF --method STFR \
#        --ssns_frac 0.7 --ssns_alpha 0.75 --seed 20
#
# The evaluation also writes the rank dump and the top-20 list dump used by analysis/.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=$1; shift
python -m stfr.train --run_dir "$RUN_DIR" "$@"
python -m stfr.eval_ckpt --run_dir "$RUN_DIR" --split test --topk 20 --rank_dump --dump_recs
grep -h "^\[test\]" "$RUN_DIR"/test_k20.log
