#!/usr/bin/env bash
# separate.sh — run StemForge stem separation on one audio file (Phase-1 headless harness).
#
#   scripts/separate.sh <input-audio> [output-dir] [model.onnx]
#
# Defaults: output -> "<input>_stems/", model -> Models/weights/htdemucs.onnx.
# Handles any codec ffmpeg can read (mp3/m4a/…) by transcoding to a 44.1 kHz stereo float
# WAV first; WAV/FLAC/AIFF/Ogg are fed to JUCE directly. HTDemucs' 4 packed stems are
# renamed to drums/bass/other/vocals for convenience.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build-onnx/StemForgeSeparate_artefacts/Debug/StemForgeSeparate"
MODEL_DEFAULT="$ROOT/Models/weights/htdemucs.onnx"

IN="${1:?usage: separate.sh <input-audio> [output-dir] [model.onnx]}"
OUTDIR="${2:-${IN%.*}_stems}"
MODEL="${3:-$MODEL_DEFAULT}"

[[ -x "$BIN"   ]] || { echo "binary missing — build it: cmake --build build-onnx --target StemForgeSeparate"; exit 1; }
[[ -f "$MODEL" ]] || { echo "model missing: $MODEL — run Models/spike/04_export_htdemucs.py"; exit 1; }
[[ -f "$IN"    ]] || { echo "input not found: $IN"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found (needed for non-WAV input)"; }

mkdir -p "$OUTDIR"

# JUCE reads WAV/FLAC/AIFF/Ogg natively; transcode everything else (e.g. MP3/M4A) first.
FEED="$IN"; TMPD=""
ext="$(printf '%s' "${IN##*.}" | tr '[:upper:]' '[:lower:]')"
case "$ext" in
  wav|flac|aif|aiff|ogg) ;;
  *) TMPD="$(mktemp -d)"; FEED="$TMPD/in.wav"
     echo "transcoding .$ext -> wav (44.1k stereo f32)…"
     ffmpeg -y -loglevel error -i "$IN" -ar 44100 -ac 2 -c:a pcm_f32le "$FEED" ;;
esac

echo "separating: $(basename "$IN")  ->  $OUTDIR"
"$BIN" "$MODEL" "$FEED" "$OUTDIR"

# HTDemucs output is one packed tensor with no per-stem names; map its fixed source order.
# (bash 3.2-safe — macOS default — so no mapfile.)
n_generic="$(ls "$OUTDIR"/stem_*.wav 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$n_generic" -eq 4 ]]; then
  names=(drums bass other vocals)
  for i in 0 1 2 3; do mv -f "$OUTDIR/stem_$i.wav" "$OUTDIR/${names[$i]}.wav"; done
  echo "named stems: ${names[*]}"
fi

[[ -n "$TMPD" ]] && rm -rf "$TMPD"
echo "done -> $OUTDIR"
