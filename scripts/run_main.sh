#!/usr/bin/env bash
# Main comparison: the eight dataset x backbone settings of the manuscript
# (SimGCL is not run on Douban).  Each cell runs the full protocol of scripts/run_cell.py:
# SimGCL weight selection, validation search of every method, three-seed finals, test evaluation.
#
#   bash scripts/run_main.sh                     # full search
#   SELECTED=configs/selected_paper.json bash scripts/run_main.sh   # skip the search, use the manuscript's configurations
#   JOBS=2 CELLS="Amazon-VG:MF Amazon-VG:LightGCN" bash scripts/run_main.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CELLS=${CELLS:-"Amazon-VG:MF Amazon-Movies:MF Amazon-VG:LightGCN Amazon-Movies:LightGCN Amazon-VG:SimGCL Amazon-Movies:SimGCL Douban-movie:MF Douban-movie:LightGCN"}
JOBS=${JOBS:-1}
OUT=${OUT:-runs}
EXTRA=()
[ -n "${SELECTED:-}" ] && EXTRA+=(--selected "$SELECTED")
for CELL in $CELLS; do
  DS=${CELL%:*}; BB=${CELL#*:}
  echo "=== $DS / $BB ==="
  python scripts/run_cell.py --dataset "$DS" --backbone "$BB" --out "$OUT" --jobs "$JOBS" "${EXTRA[@]}"
done
python analysis/collect_results.py --runs "$OUT"
