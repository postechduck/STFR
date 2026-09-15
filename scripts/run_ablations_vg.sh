#!/usr/bin/env bash
# Section 7.3 controls on Amazon-VG (MF, LightGCN, SimGCL), starting from each cell's selected
# STFR configuration (runs/<cell>/selection.json from scripts/run_main.sh, or configs/selected_paper.json):
#   component ablation   SSNS_only, fresh_only, shared_gain
#   recency restriction  recent{1,2,3,6,12,24} and recent{N}_fresh
#   sampler replacement  swap_DNS, swap_AUCNS (validation search), swap_FairNeg
# Arms and their arguments are defined in configs/ablations_vg.json.
#
#   bash scripts/run_ablations_vg.sh
#   SELECTED=configs/selected_paper.json JOBS=2 bash scripts/run_ablations_vg.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-1}
OUT=${OUT:-runs}
BACKBONES=${BACKBONES:-"MF LightGCN SimGCL"}
ARMS=${ARMS:-"SSNS_only,fresh_only,shared_gain,recent1,recent2,recent3,recent6,recent12,recent24,recent1_fresh,recent2_fresh,recent3_fresh,recent6_fresh,recent12_fresh,recent24_fresh,swap_DNS,swap_AUCNS,swap_FairNeg"}
EXTRA=()
[ -n "${SELECTED:-}" ] && EXTRA+=(--selected "$SELECTED")
for BB in $BACKBONES; do
  echo "=== Amazon-VG / $BB : ablations ==="
  python scripts/run_cell.py --dataset Amazon-VG --backbone "$BB" --out "$OUT" --jobs "$JOBS" \
      --arms_file configs/ablations_vg.json --arms "$ARMS" "${EXTRA[@]}"
done
python analysis/ablation_tables.py --runs "$OUT"
