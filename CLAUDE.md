# CLAUDE.md

AI-assistant context for **StemForge** — an offline VST3/AU/Standalone plugin that
separates a finished song into stems (vocals, drums, bass, etc.) using a deep-learning
model run through ONNX Runtime in C++.

Full design lives in `docs/PLAN.md`. Learning notes in `docs/learning/`. Read those first
for *why*; this file is the *how* and the rules.

## Stack
- **JUCE 8** (C++17) — plugin shell + audio I/O, pulled via CMake `FetchContent`.
- **ONNX Runtime 1.18+** (C++) — inference engine for the separation model.
- **Separation model** (`.onnx`) — pipeline is model-agnostic. **HTDemucs (MIT)** is the
  shippable default; **BS-RoFormer-SW** is an opt-in higher-quality download (License:
  unknown, 699 MB — never bundled). Weights are *not* in git. See `docs/PLAN.md` R8.
- **CMake 3.22+** — build. **Python** — one-time model export only (not shipped).

## Build
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug   # configure (fetches JUCE 8.0.13)
cmake --build build                                 # VST3 + Standalone + test
ctest --test-dir build --output-on-failure          # runs roundtrip_lossless test
# Artifacts: build/StemForge_artefacts/Debug/{Standalone/StemForge.app, VST3/StemForge.vst3}
# Phase 1: add -DSTEMFORGE_WITH_ONNXRUNTIME=ON to fetch + link ONNX Runtime.
```
Test fixture: `python Tests/make_fixture.py` (needs `soundfile`) generates the WAV the
ctest uses. Phase 0 is complete; the skeleton builds clean with a bit-exact I/O roundtrip.

## Architecture (one screen)
```
PluginEditor (UI/message thread)  ──▶  PluginProcessor (parameter/state owner)
                                              │ dispatches
                                              ▼
                          SeparationWorker (juce::Thread, background)
                            load file → chunk → ONNX infer → overlap-add → write WAVs
                                              │ uses
                                              ▼
                          ONNXInferenceEngine (Ort::Session wrapper)
```
This plugin does **offline file processing**, not live DAW audio. `processBlock()` is
near-empty (pass-through). All heavy work is on the worker thread.

## Non-negotiable rules (audio-plugin correctness)
1. **Never block, allocate, lock, or do I/O on the audio thread** (`processBlock`,
   `prepareToPlay`-allocate-once pattern). Inference *never* runs there.
2. **No exceptions across the plugin boundary** — use `juce::Result` for fallible calls.
   A thrown exception can crash the host DAW.
3. **Annotate thread ownership** on every class with shared state
   (`// THREAD: Worker thread only` etc). Cross-thread reads use `std::atomic` or a lock.
4. **RAII everywhere** — `std::unique_ptr`, scoped `Ort::Value`. No raw `new`/`delete`,
   no raw `OrtValue*`.
5. **Model weights stay out of git** (`.gitignore` blocks `*.onnx`/`*.pt`). Track the
   export script, document how to regenerate.

## Conventions
- Classes `PascalCase`, methods/locals `camelCase`, members `m_prefixed`,
  constants `ALL_CAPS`. Matches JUCE so its docs and our code read alike.
- `#pragma once`. Header include order: system → JUCE → third-party → ours.
- Non-trivial sections get a CHUNK intent comment (see `docs/PLAN.md` §Coding Standards).
- Prefer `const`, references over copies for buffers, and explicit types in headers.

## Layout
- `Source/PluginProcessor.*`, `Source/PluginEditor.*` — JUCE entry points.
- `Source/Separation/` — worker, ONNX engine, chunker, stem writer.
- `Source/Drums/` — drum-stem decomposition (see PLAN risk note before building).
- `Source/UI/` — JUCE components.
- `Models/` — Python export script + regeneration docs (no weights).
- `Tests/` — `ctest` unit tests; chunking roundtrip must be lossless.

## When working here
- Default to the latest Claude models in any AI/tooling code.
- Match surrounding JUCE idiom; don't introduce a second style.
- Before adding a dependency, check JUCE already provides it (audio I/O, threading,
  JSON, files, DSP primitives — it usually does).
- If a task touches Stage-2 drum splitting, read the "Architectural Risks" section in
  `docs/PLAN.md` first — the original HPSS-only plan is flagged as insufficient.
