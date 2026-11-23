#!/usr/bin/env bash
# Main comparison: the nine dataset x backbone settings (three datasets x MF, LightGCN,
# SimGCL).  Each cell runs the full protocol of scripts/run_cell.py:
# SimGCL weight selection, validation search of every method, three-seed finals, the
# STFR (forecast) serving selection, and test evaluation.
#
#   bash scripts/run_main.sh
#   JOBS=2 CELLS="Amazon-VG:MF Amazon-VG:LightGCN" bash scripts/run_main.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CELLS=${CELLS:-"Amazon-VG:MF Amazon-Movies:MF Amazon-VG:LightGCN Amazon-Movies:LightGCN Amazon-VG:SimGCL Amazon-Movies:SimGCL Douban-movie:MF Douban-movie:LightGCN Douban-movie:SimGCL"}
JOBS=${JOBS:-1}
OUT=${OUT:-runs}
for CELL in $CELLS; do
  DS=${CELL%:*}; BB=${CELL#*:}
  echo "=== $DS / $BB ==="
  python scripts/run_cell.py --dataset "$DS" --backbone "$BB" --out "$OUT" --jobs "$JOBS"
done
python analysis/collect_results.py --runs "$OUT"
