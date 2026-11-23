#!/usr/bin/env bash
# Section 7.4 controls on Amazon-VG (MF, LightGCN, SimGCL), starting from
# each setting's selected STFR configuration, which run_cell.py reads from the search results of
# scripts/run_main.sh (runs/Amazon-VG/<backbone>/selection.json):
#   component ablation   SSNS_only, fresh_only, shared_gain
#   recency restriction  recent{1,2,3,6,12,24} and recent{N}_fresh
#   sampler replacement  swap_DNS, swap_AUCNS (validation search), swap_FairNeg
# Arms and their arguments are defined in configs/ablations_vg.json.
#
#   bash scripts/run_main.sh            # first: the search that writes selection.json
#   JOBS=2 bash scripts/run_ablations_vg.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-1}
OUT=${OUT:-runs}
BACKBONES=${BACKBONES:-"MF LightGCN SimGCL"}
ARMS=${ARMS:-"SSNS_only,fresh_only,shared_gain,recent1,recent2,recent3,recent6,recent12,recent24,recent1_fresh,recent2_fresh,recent3_fresh,recent6_fresh,recent12_fresh,recent24_fresh,swap_DNS,swap_AUCNS,swap_FairNeg"}
for BB in $BACKBONES; do
  if [ ! -f "$OUT/Amazon-VG/$BB/selection.json" ]; then
    echo "missing $OUT/Amazon-VG/$BB/selection.json: run scripts/run_main.sh (Amazon-VG:$BB) first" >&2
    exit 1
  fi
  echo "=== Amazon-VG / $BB : ablations ==="
  python scripts/run_cell.py --dataset Amazon-VG --backbone "$BB" --out "$OUT" --jobs "$JOBS" \
      --arms_file configs/ablations_vg.json --arms "$ARMS"
done
python analysis/ablation_summary.py --runs "$OUT"
