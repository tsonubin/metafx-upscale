#!/bin/bash
# Upscale a list of videos to 4K with the Apple AI engine (10-bit HEVC, 50 Mbps, quality-first), two at a time.
#
#   scripts/batch-upscale.sh JOBS_FILE      # one input path per line, processed in order
#
# Safe to re-run: finished outputs are verified and skipped. Each job writes to a hidden .partial file
# and is renamed only after its duration, frame count, size and audio check out. Progress is appended
# to output/BATCH-STATUS.md; per-job logs go to output/batch-logs/. Finished videos go to output/4K/.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/.build/release/metalfx-upscale"
OUT="$ROOT/output"
LOGS="$OUT/batch-logs"
TMP="$OUT/batch-tmp"
STATUS="$OUT/BATCH-STATUS.md"
PARALLEL="${PARALLEL:-2}"
BITRATE="${BITRATE:-50}"
DEST="${DEST:-$OUT/4K}"
MIN_FREE_GB=40

status() { printf -- '- %s %s\n' "$(date '+%Y-%m-%d %H:%M')" "$*" >> "$STATUS"; }

probe() { ffprobe -v error -select_streams "$2" -show_entries "$3" -of csv=p=0 "$1" 2>/dev/null | head -1; }

frame_count() { ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$1" 2>/dev/null; }

# verify SOURCE OUTPUT: same duration (±0.5 s), frame count = duration × fps (±max(2, 0.1%)),
# 4K height or width, audio kept. Source frames are derived from duration because packet counts
# include hidden pre-roll frames in edit-listed files.
verify() {
    local src="$1" out="$2"
    [ -s "$out" ] || return 1
    local sd od fps sf of w h
    sd=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$src")
    od=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out") || return 1
    awk -v a="$sd" -v b="$od" 'BEGIN { d = a - b; exit !(d < 0.5 && d > -0.5) }' || return 1
    fps=$(probe "$src" v:0 stream=avg_frame_rate)
    sf=$(awk -v d="$sd" -v r="$fps" 'BEGIN { split(r, f, "/"); printf "%d", d * f[1] / f[2] + 0.5 }')
    of=$(frame_count "$out")
    awk -v a="$sf" -v b="$of" 'BEGIN { d = a - b; if (d < 0) d = -d; t = a / 1000; if (t < 2) t = 2; exit !(d <= t) }' || return 1
    w=$(probe "$out" v:0 stream=width); h=$(probe "$out" v:0 stream=height)
    [ "$h" = 2160 ] || [ "$w" = 3840 ] || return 1
    if [ -n "$(probe "$src" a:0 stream=codec_type)" ]; then
        [ -n "$(probe "$out" a:0 stream=codec_type)" ] || return 1
    fi
}

run_one() {
    local input="$1" file stem final partial log src attempt t0
    file="$(basename "$input")"; stem="${file%.*}"
    final="$DEST/$stem - 4K.mp4"
    partial="$DEST/.partial-$stem.mp4"
    log="$LOGS/$stem.log"

    if verify "$input" "$final"; then status "SKIP (already done) $file"; return 0; fi

    local free_gb
    free_gb=$(df -g "$OUT" | awk 'NR==2 { print $4 }')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then status "FAILED $file: only ${free_gb} GB free"; return 1; fi

    # AVFoundation can't read WebM/Matroska/AV1/Opus: convert to near-lossless ProRes + AAC first.
    src="$input"
    case "${file##*.}" in
        mp4|MP4|mov|MOV|m4v|M4V) ;;
        *)
            src="$TMP/$stem.mov"
            status "PREP $file → ProRes intermediate"
            if ! ffmpeg -v error -y -i "$input" -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le \
                    -c:a aac -b:a 256k "$src" >> "$log" 2>&1; then
                status "FAILED $file: could not convert to an intermediate (see batch-logs/$stem.log)"
                return 1
            fi ;;
    esac

    for attempt in 1 2; do
        status "START $file (attempt $attempt)"
        t0=$(date +%s)
        if "$BIN" "$src" "$partial" --engine ml --bitrate "$BITRATE" --overwrite >> "$log" 2>&1 \
                && verify "$input" "$partial"; then
            mv -f "$partial" "$final"
            local summary
            summary=$(tr '\r' '\n' < "$log" | grep -E "frames in" | tail -1 | sed 's/^ *//')
            status "DONE $file in $(( ($(date +%s) - t0) / 60 )) min: $summary → output/4K/$(basename "$final")"
            rm -f "$TMP/$stem.mov"
            return 0
        fi
        status "ERROR $file attempt $attempt: $(tr '\r' '\n' < "$log" | grep -v '^ *$' | tail -1 | cut -c1-200)"
        rm -f "$partial"
    done
    status "FAILED $file after 2 attempts (see output/batch-logs/$stem.log)"
    return 1
}

if [ "${1:-}" = "--one" ]; then
    run_one "$2"
    exit $?
fi

JOBS="${1:?usage: $0 JOBS_FILE}"
mkdir -p "$LOGS" "$TMP" "$DEST"
status "BATCH START: $(grep -c . "$JOBS") jobs, $PARALLEL at a time, Apple AI engine, 10-bit HEVC ${BITRATE} Mbps"
grep -v '^ *$' "$JOBS" | tr '\n' '\0' | xargs -0 -P "$PARALLEL" -I{} "$0" --one {}
status "BATCH END: $(grep -cE ' (DONE|SKIP) ' "$STATUS") done/skipped, $(grep -c ' FAILED ' "$STATUS") failed (counts include earlier runs)"
rmdir "$TMP" 2>/dev/null
exit 0
