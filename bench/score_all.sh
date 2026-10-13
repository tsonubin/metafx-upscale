#!/bin/bash
# Score every finished output that has no metrics yet (plus GT itself), N at a time.
# Usage: bench/score_all.sh [PARALLEL]
cd "$(dirname "$0")/.." && source bench/env.sh
CLIPS=output/sr-bench/clips
jobs=()
for c in "$CLIPS"/*/; do
  c=${c%/}
  [ -f "$c/gt.mkv" ] && [ ! -f "$c/metrics/gt.json" ] && jobs+=("$c gt")
  for j in "$c"/out/*.json; do
    [ -e "$j" ] || continue
    m=$(basename "$j" .json)
    case $m in *-nocut|flashvsr*) continue;; esac  # FlashVSR: matched ad-hoc comparisons only
    [ -f "$c/metrics/$m.json" ] && [ "$c/metrics/$m.json" -nt "$j" ] || jobs+=("$c $m")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "${1:-4}" -L 1 sh -c 'python bench/metrics.py "$0" "$1" 2>&1 | grep -v Warning | tail -1 | cut -c1-160'
