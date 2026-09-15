#!/usr/bin/env bash
# Section 7.4 (Figure 4, Table E.3): temporal block configurations on Amazon-VG, L in {15, 30, 120} days
# (L = 60 is the main comparison).  Every configuration keeps the same training
# interactions and the same evaluation-block start (--eval_block = 5520 / L, since the
# main split's evaluation block starts 5520 days after the first interaction); the
# block grid, the fresh signal and the validation/test windows follow the block length.
# base and STFR are trained per configuration with the main protocol (STFR: f/alpha
# search on validation; SimGCL: contrastive weight re-selected on base).
#
#   bash scripts/run_block_configs.sh
#   SELECTED=configs/selected_paper.json bash scripts/run_block_configs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-1}
OUT=${OUT:-runs}
BLOCKS=${BLOCKS:-"15 30 120"}
BACKBONES=${BACKBONES:-"MF LightGCN SimGCL"}
EXTRA=()
[ -n "${SELECTED:-}" ] && EXTRA+=(--selected "$SELECTED")
for L in $BLOCKS; do
  NAME=Amazon-VG_b$L
  if [ ! -f "data/$NAME/dataset_meta.json" ]; then
    python prep/prep_split.py --dataset Amazon-VG --block_days "$L" --eval_block $((5520 / L)) --out_name "$NAME"
  fi
  for BB in $BACKBONES; do
    echo "=== $NAME / $BB ==="
    python scripts/run_cell.py --dataset Amazon-VG --data_dir "$NAME" --backbone "$BB" \
        --out "$OUT" --jobs "$JOBS" --arms base,STFR "${EXTRA[@]}"
  done
done
python analysis/block_config_table.py --runs "$OUT"
