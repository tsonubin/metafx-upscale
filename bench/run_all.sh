#!/bin/bash
# Run every model on every clip (skipping finished ones), then cut-ignoring runs of the
# temporal models on clips with cuts. Usage: bench/run_all.sh MODEL [MODEL ...]
cd "$(dirname "$0")/.." && source bench/env.sh
CLIPS=output/sr-bench/clips
for m in "$@"; do
  extra=()
  # Heavy models: upscale only the scored centre of 4K clips (+ context margin), first 4 s.
  case $m in realesrgan|realbasicvsr|realviformer|flashvsr) extra=(--center-crop --frames 100);; esac
  for c in "$CLIPS"/*/; do
    c=${c%/}
    [ -f "$c/out/$m.json" ] || python bench/run_model.py "$m" "$c" "${extra[@]}" 2>&1 | grep -v Warning | tail -1 | cut -c1-200
  done
  case $m in apple|realbasicvsr|realviformer|flashvsr)
    for c in "$CLIPS"/*cut*/; do
      c=${c%/}
      [ -f "$c/out/$m-nocut.json" ] || python bench/run_model.py "$m" "$c" --ignore-cuts "${extra[@]}" 2>&1 | grep -v Warning | tail -1 | cut -c1-200
    done;;
  esac
done
