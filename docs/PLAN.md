# StemForge — VST3 Stem Separation Plugin
### Full Build Plan · Coding Standards · Learning Guide

---

## Table of Contents

1. [What We're Building](#what-were-building)
2. [The Science: Why AI, Not DSP](#the-science-why-ai-not-dsp)
3. [Model Architecture Decision](#model-architecture-decision)
4. [Stem Inventory: What We Can Extract](#stem-inventory-what-we-can-extract)
5. [Tech Stack](#tech-stack)
6. [System Architecture](#system-architecture)
7. [Phased Build Plan](#phased-build-plan)
8. [Coding Standards](#coding-standards)
9. [Learning Guide: Concepts You'll Encounter](#learning-guide-concepts-youll-encounter)
10. [File & Folder Structure](#file--folder-structure)
11. [Key Decisions Log](#key-decisions-log)

---

## What We're Building

**StemForge** is an offline-processing VST3/AU plugin for macOS, Windows, and Linux. You drop an audio file onto it (or render a track through it), hit **Analyze**, and it outputs separated stems as individual audio files or internal DAW tracks. No real-time constraint — we process the full file, write output to disk, and the DAW picks up the stems.

**Core user story:**
> "I have a finished song I want to remix. I drag it into StemForge, pick which stems I want (vocals, drums, melody/synth lead, bass, etc.), click Analyze, and 30 seconds later I have clean stems in my project folder ready to drag into my arrangement."

**Stem targets** (full list in section 4):
- Vocals (lead + backing split)
- Drums (as a group, then decomposed into kick, snare, hi-hat, clap, toms, cymbals)
- Bass
- Melody / lead synth
- Chords / pad / harmonic bed
- Guitar (acoustic + electric)
- Piano
- "Other" catch-all

---

## The Science: Why AI, Not DSP

### The old approach: Mid-Side (M/S) processing

Vocals are usually panned center. The "mid" channel of a stereo signal captures center-panned content, so you can subtract it to get a rough vocal-free mix. This is purely mathematical — no AI needed.

**Why this fails for us:**
- Also removes the kick drum, bass, and any lead synth that's panned center
- Creates hollow-sounding, phase-artifacted audio
- Completely breaks on modern music where vocals are stereo-spread, double-tracked, or layered with reverb
- Result: muddy, unusable for remix work

### The right approach: Source Separation via Deep Learning

The AI approach trains a neural network to *understand* the timbral and spectral fingerprint of each instrument type. It learns:
- A snare drum's transient envelope looks like this
- A human voice has formants in these frequency bands
- A bass guitar has energy concentrated here and this harmonic relationship

Given a mixed signal, the model predicts a **mask** for each stem — a per-frequency, per-time-frame weight that says "this part of the signal belongs to vocals, this part to drums."

The result: dramatically cleaner separation because the model learned from thousands of songs with known ground-truth stems.

---

## Model Architecture Decision

### The Landscape

There are three relevant architectures today:

| Model | Architecture | Best At | SDR (vocals) | License |
|---|---|---|---|---|
| **HTDemucs v4** | Hybrid U-Net (waveform + spectrogram) | Bass, drums, fast inference | ~8.5 dB | MIT |
| **BS-RoFormer** | Band-Split Rotary Transformer | Vocals, overall quality | ~10.9 dB | MIT (lucidrains impl.) |
| **MelBand-RoFormer** | Mel-scale band split + RoPE Transformer | Vocals, perceptual quality | ~10.8 dB | MIT |

**SDR (Signal-to-Distortion Ratio)** — higher is better. Every 1 dB improvement is audibly meaningful. The jump from Spleeter (~5.9 dB) to HTDemucs (~8.5 dB) to BS-RoFormer (~10.9 dB) is massive.

### Our Decision: Tiered Model Strategy

We use **two models in a pipeline**:

```
Stage 1: Separation model  (model-agnostic pipeline — see licensing note below)
  ├── DEFAULT  : HTDemucs (MIT, shippable)        → Vocals, Drums, Bass, Other (4-stem)
  └── OPT-IN   : BS-RoFormer-SW (license unknown)  → + Guitar, Piano (6-stem, higher SDR)

Stage 2: Drum decomposition  (see ⚠ architecture note below)
  └── Kick, Snare, Hi-Hat, Clap, Toms, Cymbals
      (runs on the Drums stem from Stage 1)
```

> ### ⚠ Licensing-driven model decision (revised after R8 investigation)
> The pipeline is **model-agnostic at Stage 1** — the engine loads whichever `.onnx` is
> present. We tier by *license*, not just quality:
>
> - **HTDemucs = the shippable default.** Meta releases the weights under **MIT**, so we can
>   redistribute/bundle them legally. Its ONNX export is the *easier, already-validated*
>   path (Mixxx GSoC precedent). Lower vocal SDR (~8.5 dB) but it ships today, clean.
> - **BS-RoFormer-SW = opt-in quality upgrade, NOT bundled.** Best-in-class SDR (~10.9 dB)
>   but its only available weights (`jarredou/BS-ROFO-SW-Fixed`) are **699 MB, License:
>   unknown** — trained by the community from undocumented data, re-hosted, no usage terms.
>   We must **not** redistribute it. Offer it as a user-initiated first-run download with an
>   in-UI provenance note; treat as best-effort/research. The lucidrains *code* is MIT, but
>   code license ≠ weights license.
> - **Security:** the checkpoint is a PyTorch **pickle** (arbitrary code on load). Export
>   tooling loads with `weights_only=True` and converts to **safetensors** before trusting.
>
> Net: build/validate against HTDemucs first (legally safe, easier export); keep
> BS-RoFormer as an optional upgrade. See risk **R8** and the Key Decisions Log.

> ### ⚠ Architecture correction — Stage 2 is NOT "DSP-only"
>
> The original plan claimed Stage 2 could split the drum group into 6 named sub-stems
> using **HPSS + frequency masking, no ML**. This is **not sound** and must not be built
> as written:
>
> - **HPSS separates *harmonic* from *percussive* content** — it cannot tell a kick from a
>   snare from a hi-hat, because all of those are percussive. It produces two outputs
>   (pitched vs transient), not six named instruments.
> - The well-known **`drumsep`** project (`github.com/inagoy/drumsep`) that produces
>   kick/snare/toms/cymbals is itself a **Demucs model fine-tuned on drums** — i.e. it *is*
>   ML, not DSP. The plan conflated "DrumSep the tool" with "a DSP method."
> - Frequency bands for kick/snare/hat/tom **overlap heavily**; band-pass masking alone
>   yields muddy, cross-bleeding stems.
>
> **Revised options (decide before Phase 3):**
> 1. **ML drum-sep model (recommended)** — export a Demucs drum fine-tune (drumsep /
>    LarsNet style) to ONNX and run it as a second inference stage. Highest quality,
>    consistent with our Stage-1 pipeline, but adds a second model to bundle/export.
> 2. **HPSS + classification hybrid** — use HPSS for harmonic/percussive cleanup, then a
>    transient detector + per-onset frequency-feature classifier to *label* hits as
>    kick/snare/hat. Real engineering effort, lower quality ceiling, deterministic and
>    weight-free.
> 3. **Ship Stage 1 only first** — deliver the 6 main stems (high value, lower risk) and
>    treat 6-way drum splitting as a fast-follow once Stage 1 is solid.
>
> See the `audio-dsp` skill for the HPSS scope limit and `docs/learning/02` for the data
> shapes involved.

### Why ONNX?

Models are trained in PyTorch. VST3 plugins are C++. ONNX (Open Neural Network Exchange) is the bridge:

```
PyTorch Model (.pt)
      ↓  export
ONNX Model (.onnx)
      ↓  load
ONNX Runtime (C++ library)
      ↓  run
Stems (WAV buffers)
```

The GSoC 2025 Mixxx project already validated a working ONNX export of HT-Demucs with < 0.1 dB quality difference. We'll do the same export step for BS-RoFormer.

---

## Stem Inventory: What We Can Extract

### Tier 1 — Stage 1 model (direct output, highest quality)
| Stem | Description |
|---|---|
| `vocals_lead` | Lead vocal, post lead/backing split |
| `vocals_backing` | Backing vocals, harmonies |
| `drums_group` | All drums as one stem |
| `bass` | Bass guitar, 808s, sub bass |
| `guitar` | Acoustic + electric combined |
| `piano` | Piano, keys |
| `other` | Everything else (synth pads, FX, etc.) |

### Tier 2 — Stage 2 DrumSep (from drums_group)
| Stem | Description |
|---|---|
| `kick` | Kick drum transients |
| `snare` | Snare drum + rimshots |
| `hihat` | Hi-hats (open + closed) |
| `clap` | Claps, snare layers |
| `toms` | Tom fills |
| `cymbals` | Crash, ride cymbals |

### Tier 3 — Future (fine-tuned models or ensemble)
| Stem | Path |
|---|---|
| `synth_lead` | Fine-tune on labeled electronic music |
| `synth_pad` | Same |
| `melody` | Post-process "other" with pitch salience |
| `fx` | Reverb/noise/atmos layer |

Total potential output: **13 stems** from one source file.

---

## Tech Stack

```
┌──────────────────────────────────────────────┐
│  Plugin Framework: JUCE 8 (C++17)            │
│  Plugin Format:    VST3 + AU + Standalone     │
│  Build System:     CMake 3.22+               │
│  Inference:        ONNX Runtime 1.18+ (C++)   │
│  Model Weights:    BS-RoFormer-SW (.onnx)     │
│  Drum Decomp:      Python subprocess / port  │
│  Audio I/O:        libsndfile (via JUCE)      │
│  UI:               JUCE Component system      │
│  Packaging:        CPack / installers         │
└──────────────────────────────────────────────┘
```

**Languages:**
- **C++17** — the VST3 plugin core, audio processing, ONNX inference calls
- **Python** — model export script (one-time, not shipped), DrumSep integration layer
- **CMake** — build system

**Why C++ and not Rust?**
JUCE is a C++ framework. While you've done Rust VST work (Auric's AudioWorklet model was a great concurrent systems story), JUCE is the industry standard for cross-platform VST3/AU/AAX and has the best ecosystem for this. We can isolate the processing core into a clean module boundary that could hypothetically be ported to Rust later. The plugin shell stays in C++.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│  JUCE Plugin Shell                                   │
│  ┌───────────────┐    ┌──────────────────────────┐  │
│  │ PluginEditor  │    │ PluginProcessor          │  │
│  │ (UI layer)    │───▶│ (audio thread interface)  │  │
│  │               │    │                           │  │
│  │ - File dropper│    │ - Parameter management    │  │
│  │ - Stem toggles│    │ - No real AI work here!   │  │
│  │ - Progress bar│    │ - Dispatches to Worker    │  │
│  │ - Export paths│    └──────────┬────────────────┘  │
│  └───────────────┘               │                   │
│                                  ▼                   │
│  ┌───────────────────────────────────────────────┐  │
│  │  SeparationWorker (background thread)         │  │
│  │                                               │  │
│  │  1. Load full audio file → float32 buffer     │  │
│  │  2. Chunk into overlapping segments           │  │
│  │  3. Run BS-RoFormer ONNX inference per chunk  │  │
│  │  4. Overlap-add reassembly                    │  │
│  │  5. Write 6 stem WAV files to output dir      │  │
│  │  6. Run DrumSep on drums stem → 6 sub-stems   │  │
│  │  7. Signal UI: done / progress updates        │  │
│  └──────────────────────┬────────────────────────┘  │
│                          │                           │
│  ┌───────────────────────▼────────────────────────┐ │
│  │  ONNXInferenceEngine                           │ │
│  │                                               │ │
│  │  - Session management (Ort::Session)          │ │
│  │  - Input tensor preparation                   │ │
│  │  - Output tensor extraction                   │ │
│  │  - GPU/CPU provider selection                 │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Critical Thread Safety Note

JUCE plugins have a strict threading model:
- **Audio thread** — real-time, no locks, no allocation, no I/O
- **Message (UI) thread** — UI events, parameter changes
- **Worker thread** — our AI inference lives here exclusively

Since we're offline processing, `processBlock()` on the audio thread does **nothing except maybe pass audio through**. All the heavy lifting is in the `SeparationWorker` which is a `juce::Thread` subclass that we kick off when the user hits Analyze.

---

## Architectural Review & Risks

A candid review of the design's load-bearing assumptions. Each risk has an owner phase and
a mitigation. Revisit this table at every phase boundary.

| # | Risk / Open question | Severity | Mitigation |
|---|---|---|---|
| R1 | **BS-RoFormer → ONNX export may not be clean.** RoFormer uses STFT/iSTFT + complex-number ops; ONNX support for these is opset-sensitive and historically fiddly. | ~~High~~ **Med** (spike done) | **SPIKE DONE (`Models/spike/`).** Native `torch.stft`/`istft` export is unreliable in our toolchain (torch 2.12 / ORT 1.26): TorchScript exporter rejects complex STFT; dynamo exporter emits a *silently wrong* istft. **Resolution:** substitute a conv-based STFT/ISTFT (`Conv1d`/`ConvTranspose1d`) — proven to export with basic ops only and match PyTorch to **4.8e-6 (122.8 dB)**, dynamic chunk lengths included. Remaining unknown is only the attention/band-split blocks (standard ops). HT-Demucs stays as a fallback. |
| R2 | **Stage-2 drum split was mis-specified** (see ⚠ note in Model Architecture). HPSS can't name drum sub-stems. | High | Pick one of the three revised options before Phase 3. Recommended: ship Stage 1 first (option 3), then add an ML drum model (option 1). |
| R3 | **Tensor layout / channel order mismatch** silently produces garbage stems. | Med | Confirm model IO names + dims with Netron; assert shapes at runtime; golden-file test in Phase 1. |
| R4 | **Memory pressure** — holding 6 full-length stem buffers ≈ 380 MB for a 3-min song, before counting model arenas. | Med | Stream each stem to disk as it finishes instead of holding all six; reuse segment buffers. Decide in Phase 1. |
| R5 | **Sample-rate handling.** Model expects a fixed rate (44.1 kHz); inputs vary. Plan only mentions *erroring* on unsupported rates. | Med | Resample in/out around inference (`juce::LagrangeInterpolator` / `juce::dsp`), don't reject. Document the canonical rate in one place. |
| R6 | **ONNX Runtime static linking + size.** ORT static libs are large and platform-specific; GPU providers add bulk and driver dependencies. | Med | Start CPU-only, statically linked. Add CoreML/CUDA/DirectML providers behind a runtime-detected fallback (CPU always works). |
| R7 | **"Offline plugin" is an unusual VST3 shape.** A VST3 normally processes the host's live audio stream; ours is a file-in/files-out utility hosted as a plugin and barely touches `processBlock`. Some hosts may behave oddly. | Low | Lead with the **Standalone** build (no host quirks). Treat VST3/AU as a convenience wrapper. Document the offline model for users. |
| R8 | **Model weight licensing & distribution.** | ~~Low~~ **High** (investigated) | **FINDINGS:** `jarredou/BS-ROFO-SW-Fixed` is **699 MB**, **License: unknown** (no terms, undocumented training data, pickle format). **Not redistributable** and too big to embed. **Resolution:** ship **HTDemucs (MIT)** as the default bundled model; offer BS-RoFormer only as an opt-in user-initiated download with a provenance note. Load checkpoints with `weights_only=True`, convert to safetensors. Keep all weights out of git. Revisit if a clearly-licensed BS-RoFormer checkpoint appears. |

### Design decisions that hold up well
- **Offline-only** correctly sidesteps real-time inference — the single biggest constraint
  in audio ML. Good call.
- **Worker-thread isolation** with a near-empty `processBlock` is exactly right for JUCE.
- **`juce::Result` over exceptions** is mandatory for host safety, not optional polish.
- **CMake + FetchContent** over Projucer is the modern, CI-friendly choice.

---

## Phased Build Plan

> Companion docs: read `docs/learning/01–03` for the C++/threading/data-structure mental
> models before Phase 0, and lean on the `juce-audio-plugin`, `onnx-runtime-cpp`, and
> `audio-dsp` skills while implementing.

### Phase 0 — Foundation (Week 1–2) — ✅ COMPLETE
**Goal:** A VST3 plugin that loads, opens a window, and accepts a file drag-drop. No AI yet.

Tasks:
- [x] Set up CMake project with JUCE 8 as FetchContent dependency (pinned **8.0.13**)
- [x] Get plugin building as VST3 + Standalone on macOS (arm64, AppleClang 17, Ninja)
- [x] Implement `PluginEditor` with file drop zone (`Source/UI/FileDropZone`)
- [x] Implement `PluginProcessor` boilerplate (pass-through `processBlock`, `ValueTree` state)
- [x] Load a WAV file into a `juce::AudioBuffer<float>` (`Source/IO/AudioFileIO`)
- [x] Write it back to disk — **roundtrip verified bit-exact (max diff 0.0)** via
      `StemForgeRoundtripTest` (ctest target, IEEE float32 WAV out)
- [x] Set up ONNX Runtime as a CMake dependency — **wired** in `cmake/onnxruntime.cmake`,
      gated behind `STEMFORGE_WITH_ONNXRUNTIME` (OFF until Phase 1)

**Result:** `build/StemForge_artefacts/Debug/{Standalone/StemForge.app, VST3/StemForge.vst3}`.
Clean build, zero warnings in our sources. Build: `cmake -B build -G Ninja && cmake --build
build && ctest --test-dir build`.

**Learning checkpoint:** Understand JUCE's two-class plugin model, CMake dependency management, and `juce::AudioBuffer`. → see `docs/learning/03-how-its-built.md`.

---

### Phase 1 — ONNX Inference Harness (Week 3–4)
**Goal:** Run BS-RoFormer ONNX model on a file from C++ and write stems to disk. Headless (no UI required yet).

Tasks:
- [ ] Export BS-RoFormer-SW from PyTorch to ONNX (Python script, documented)
- [ ] Write `ONNXInferenceEngine` class
- [ ] Implement audio chunking (segment + overlap logic)
- [ ] Implement overlap-add reconstruction
- [ ] Validate output quality vs Python reference implementation
- [ ] Write stems as 32-bit float WAV files via `juce::WavAudioFormat`

**Learning checkpoint:** Understand ONNX's session/tensor API, chunking strategies, overlap-add.

---

### Phase 2 — Worker Thread + Progress (Week 5–6)
**Goal:** Full async pipeline with progress reporting back to UI.

Tasks:
- [ ] Implement `SeparationWorker` as `juce::Thread` subclass
- [ ] Thread-safe progress reporting using `juce::AsyncUpdater`
- [ ] Cancel/abort mechanism
- [ ] UI: progress bar, stem toggle checkboxes, output directory picker
- [ ] Error handling: unsupported sample rates, mono files, etc.

**Learning checkpoint:** JUCE threading model, `AsyncUpdater`, `juce::ValueTree` for state.

---

### Phase 3 — Drum Decomposition (Week 7) — ⚠ approach must be chosen first
**Goal:** Take the drums stem and split into kick/snare/hihat/clap/toms/cymbals.

**Before any code:** resolve risk **R2** — HPSS alone cannot name drum sub-stems (see the
⚠ note under Model Architecture). Pick one:
- **(Recommended) ML drum model** — export a Demucs drum fine-tune (drumsep / LarsNet) to
  ONNX, run as a second inference stage reusing `ONNXInferenceEngine`. Best quality,
  consistent pipeline.
- **HPSS + transient classification hybrid** — DSP-only, deterministic, lower ceiling.
- **Defer** — ship Stage 1's 6 stems now, add drum splitting as a fast-follow.

Tasks (assuming ML-model path):
- [ ] Acquire/verify-license a drum-separation checkpoint; export to ONNX (same spike
      discipline as Phase 1, R1).
- [ ] Add a second `Ort::Session` stage to the worker, fed by the Stage-1 drums stem.
- [ ] Add drum sub-stems to the UI toggle list (nested under "Drums").

Tasks (if hybrid DSP path instead):
- [ ] STFT → HPSS (median filtering) to clean harmonic bleed.
- [ ] Spectral-flux transient detection → per-onset frequency-feature classification into
      kick/snare/hat/tom (see `audio-dsp` skill).

**Learning checkpoint:** STFT/ISTFT, median filtering in spectral domain, transient
detection — *and why classification, not just band-pass, is required to name drums.*

---

### Phase 4 — Polish + Distribution (Week 8–10)
**Goal:** Shippable, installable plugin.

Tasks:
- [ ] macOS: AU format support (add to CMake, test in Logic)
- [ ] Windows: VST3 build verification
- [ ] Model bundling strategy (embed .onnx in plugin binary or install alongside)
- [ ] First-run model download / bundled installer
- [ ] Installer: CPack + DMG (macOS), NSIS (Windows)
- [ ] Write README and usage docs

---

### Phase 5 — Future Features
- Lead/backing vocal split (second BS-RoFormer fine-tune or LALAL.AI-style two-stage)
- Electronic music fine-tune: synth lead vs pad separation
- Ensemble mode: combine HTDemucs + BS-RoFormer for even better quality
- Batch processing: folder of songs → all stems

---

## Coding Standards

These apply to all C++ code in this project. Written to be understood, not just followed.

### 1. Naming Conventions

```cpp
// Classes: PascalCase
class ONNXInferenceEngine {};
class SeparationWorker {};

// Methods: camelCase
void loadModel(const juce::File& modelPath);
bool runInference(const AudioBuffer& input);

// Member variables: prefix m_
float m_overlapFactor = 0.5f;
std::unique_ptr<Ort::Session> m_session;

// Constants: ALL_CAPS with underscores
constexpr int MAX_SEGMENT_SAMPLES = 441000; // 10s at 44.1kHz
constexpr float DEFAULT_OVERLAP   = 0.25f;

// Local variables: camelCase
auto inputTensor = prepareInputTensor(buffer);
int chunkStart   = 0;
```

**Why:** JUCE itself uses camelCase methods and PascalCase classes. We follow that so reading JUCE docs and reading our code feels consistent. The `m_` prefix is a deliberate signal: every time you see `m_`, you know this is state that lives across function calls and needs to be thought about in terms of thread safety.

### 2. File Organization

```
// Every .h file structure:
#pragma once
// ── 1. System includes
// ── 2. JUCE includes  
// ── 3. Third-party includes (ONNX Runtime)
// ── 4. Our own includes
// ── 5. Class declaration

// Every .cpp file structure:
// ── 1. Matching .h include first
// ── 2. Implementation in same order as declaration
// ── 3. Chunk boundaries annotated (see standard below)
```

### 3. Thread Safety Annotations

Every class that has shared state must declare which thread(s) own it:

```cpp
// THREAD: Audio thread only
// THREAD: Message thread only  
// THREAD: Worker thread only
// THREAD: Thread-safe (atomic / lock-protected)

class SeparationWorker : public juce::Thread
{
public:
    // THREAD: Message thread — call to kick off analysis
    void startSeparation(const juce::File& inputFile);
    
    // THREAD: Thread-safe — polled by UI for progress display
    float getProgress() const { return m_progress.load(); }

private:
    // THREAD: Worker thread — owns this exclusively
    std::unique_ptr<ONNXInferenceEngine> m_engine;
    
    // THREAD: Thread-safe (atomic)
    std::atomic<float> m_progress { 0.0f };
};
```

**Why:** Audio plugin bugs caused by wrong-thread access are the hardest to debug because they manifest as rare crashes under real-time load. Annotating thread ownership makes it reviewable at a glance.

### 4. Error Handling

We use a `Result` type (JUCE has `juce::Result`), not exceptions:

```cpp
// Good — explicit about success/failure
juce::Result ONNXInferenceEngine::loadModel(const juce::File& path)
{
    if (!path.existsAsFile())
        return juce::Result::fail("Model file not found: " + path.getFullPathName());
    
    // ... load ...
    return juce::Result::ok();
}

// Caller:
auto result = m_engine->loadModel(modelFile);
if (result.failed())
{
    // Show error in UI, log it, don't crash
    juce::Logger::writeToLog("Model load failed: " + result.getErrorMessage());
    showErrorToUser(result.getErrorMessage());
    return;
}
```

**Why:** Audio plugins cannot throw exceptions across the plugin boundary into the DAW — it will crash the host. Exceptions are off-limits in JUCE audio code. Result types force you to handle failure at the call site.

### 5. No Allocation on the Audio Thread

```cpp
// WRONG — vector allocation in processBlock
void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
{
    std::vector<float> temp(buffer.getNumSamples()); // ❌ heap allocation
}

// RIGHT — pre-allocate in prepareToPlay, reuse
void prepareToPlay(double sampleRate, int maxSamplesPerBlock) override
{
    m_scratchBuffer.resize(maxSamplesPerBlock); // ✅ done once
}

void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
{
    // m_scratchBuffer already allocated ✅
}
```

**Why:** Memory allocation can block for arbitrary time (the OS might trigger GC, page faults, etc.). The audio thread has a hard deadline — if it misses it, you get a dropout. `prepareToPlay` is called before the audio thread starts and is the right place to allocate.

### 6. Semantic Chunk Boundary Annotations

Following the dev-session standard — every non-trivial code section gets an intent comment that is independently understandable without reading surrounding code:

```cpp
// ┌─ CHUNK: AudioChunker::slice ─────────────────────────────────────────┐
// │ INTENT: Split a flat float buffer into overlapping segments for       │
// │         inference. Uses 25% overlap so reconstruction artifacts at   │
// │         segment boundaries are masked.                               │
// │ CONTRACT: output.size() == ceil(inputLen / hopSize)                  │
// │ DEPENDS ON: m_segmentSamples, m_overlapFactor (set in ctor)          │
// └──────────────────────────────────────────────────────────────────────┘
std::vector<Segment> AudioChunker::slice(const float* input, int inputLen) const
{
    // ...
}
```

### 7. ONNX-specific patterns

```cpp
// Always scope OrtValue to prevent leaks:
Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
    m_memoryInfo, inputData.data(), inputData.size(),
    inputShape.data(), inputShape.size()
);
// inputTensor auto-destructs at end of scope ✅

// Never store raw OrtValue* — always use the C++ wrapper:
// Bad:  OrtValue* raw = ...;
// Good: Ort::Value tensor = ...;
```

### 8. Ownership & RAII

Every heap object has exactly one owner expressed in the type system. No raw `new`/`delete`
in our code; no raw owning pointers.

```cpp
std::unique_ptr<Ort::Session> m_session;   // single owner, freed in destructor ✅
SomeType* m_thing = new SomeType();        // ❌ who frees this? when?
```

- Own with `std::unique_ptr` (or by value). Share (rarely) with `std::shared_ptr`.
- Borrow with `const T&` (read) or `T&` (write). Raw `T*` only at C-API boundaries (ONNX).
- Transfer ownership with `std::move`, never a copy.

**Why:** ownership-in-the-type-system means leaks and double-frees become compile-time or
scope-bound concerns, not runtime mysteries. See `docs/learning/01-cpp-mental-models.md`.

### 9. Const-correctness & pass-by-reference

Mark everything that doesn't mutate as `const`, and pass large objects by reference.

```cpp
juce::Result writeStems(const juce::AudioBuffer<float>& buffer) const;
//                       ^^^^^ no copy of the buffer            ^^^^^ method mutates nothing
```

**Why:** `const&` on a buffer parameter prevents an accidental multi-MB copy (a real perf
bug) and documents intent. `const` methods are safe to call on shared/read-only state.

### 10. Includes & headers

- `#pragma once` at the top of every header.
- Include what you use; don't rely on transitive includes.
- Header include order: **system → JUCE → third-party (ONNX) → ours.**
- Keep heavy includes (ONNX Runtime) out of headers where a forward-declaration works — it
  speeds compiles and shrinks the dependency surface.

---

## Learning Guide: Concepts You'll Encounter

> Deeper, example-driven versions of these concepts live in `docs/learning/`:
> `01-cpp-mental-models.md`, `02-data-structures.md`, `03-how-its-built.md`. This section
> is the inline primer; those are the full walkthroughs.

This is written assuming you know TypeScript/React deeply but are coming into C++ audio programming fresh-ish. Everything maps to things you already know.

---

### Concept 1: The JUCE Two-Class Model

Every JUCE plugin has exactly two required classes:

```
PluginProcessor  ──  like a Redux store + middleware
PluginEditor     ──  like a React component tree
```

**`PluginProcessor`** holds all state and audio logic. It's created once and lives as long as the plugin instance. The audio thread calls `processBlock()` on it continuously at ~44,100 samples/second.

**`PluginEditor`** is the UI window. It's created when the user opens the plugin window and destroyed when they close it. It holds a reference to the Processor so it can read/write state.

**Key rule:** The Editor is optional and may not exist. Never store important state in the Editor. Think of it exactly like "the UI is not the source of truth."

---

### Concept 2: The Audio Thread Is Sacred

Your DAW calls `processBlock()` ~every 5–20ms depending on buffer size. It expects to return within that window. If it doesn't: audio dropout, crackle, DAW unhappy.

Rules for `processBlock()`:
- No `malloc` / `new` / `std::vector` construction
- No file I/O
- No mutex locking (use lock-free atomics)
- No system calls
- No ONNX inference

Since we're offline-only, our `processBlock()` is nearly empty — it just passes audio through. All the real work is in the `SeparationWorker` thread.

This is the same constraint as AudioWorklet in the Web Audio API (which you've already navigated with Auric). Same philosophy, different language.

---

### Concept 3: ONNX Tensors

An **ONNX tensor** is just a flat array of floats with a shape descriptor. Exactly like a NumPy array or a JavaScript `Float32Array` with metadata.

```cpp
// Audio input: stereo, 10 seconds at 44.1kHz
// Shape: [batch=1, channels=2, samples=441000]
std::vector<int64_t> shape = {1, 2, 441000};
std::vector<float>   data(1 * 2 * 441000);

// Fill data from juce::AudioBuffer...

Ort::Value tensor = Ort::Value::CreateTensor<float>(
    memoryInfo,
    data.data(),     // pointer to flat array
    data.size(),     // total element count
    shape.data(),    // shape [1, 2, 441000]
    shape.size()     // number of dimensions = 3
);
```

The model outputs multiple tensors (one per stem). Each has the same shape as the input. You extract them by index:

```cpp
auto outputs = session.Run(runOptions, inputNames, &tensor, 1, outputNames, numStems);
// outputs[0] = vocals tensor
// outputs[1] = drums tensor
// outputs[2] = bass tensor  
// ...
```

---

### Concept 4: Chunking + Overlap-Add

BS-RoFormer can't process an entire 4-minute song at once — the tensor would be gigabytes. We split it into overlapping chunks and reconstruct.

```
Input: ─────────────────────────────────────────── (full song)
Chunk 1: [████████████████]                         (10 seconds)
Chunk 2:         [████████████████]                 (10 seconds, 25% overlap)
Chunk 3:                 [████████████████]
...

Output (overlap-add):
  Take chunk 1 output, apply fade-out at end
  Add chunk 2 output shifted by hop size, apply fade-in at start
  Artifacts at boundaries cancel out in the crossfade region
```

This is the same concept as STFT windowing (Hann window + 75% overlap in spectral analysis). If you've ever looked at how STFT frames are reconstructed into time-domain audio, this is the same idea one level up.

---

### Concept 5: HPSS (for drum decomposition)

**Harmonic-Percussive Source Separation** — a DSP trick that separates "smooth" spectral structures (harmonic = pitched instruments) from "spiky" structures (percussive = drums, transients).

How it works:
1. Take the spectrogram (2D: time × frequency)
2. Apply a **median filter horizontally** (across time) → keeps things that persist in frequency = harmonic
3. Apply a **median filter vertically** (across frequency) → keeps things that persist over time = percussive
4. Mask the original spectrogram by each → separate signals

This is why DrumSep doesn't need ML for its sub-decomposition — once you have the drums stem in isolation, HPSS + frequency band masking (kick is low, hi-hat is high) is reliable enough.

---

### Concept 6: JUCE's `ValueTree`

`ValueTree` is JUCE's state management primitive. It's like a Redux store that auto-serializes to XML for plugin preset save/load. For our plugin:

```cpp
juce::ValueTree state ("StemForge");
state.setProperty("outputDirectory", "/Users/rajat/stems", nullptr);
state.setProperty("stemVocals",      true,                 nullptr);
state.setProperty("stemDrums",       true,                 nullptr);
state.setProperty("stemBass",        true,                 nullptr);
// ...

// JUCE automatically uses this for getStateInformation/setStateInformation
// so DAW project save/load works for free
```

---

## File & Folder Structure

```
stemforge/
├── CMakeLists.txt              # Root build config (Phase 0 creates this)
├── CLAUDE.md                   # AI/contributor context — the "how" and the rules
├── .gitignore                  # Excludes build/, weights, deps, IDE cruft
├── README.md                   # User-facing intro
│
├── docs/
│   ├── PLAN.md                 # This document — the "why" and the roadmap
│   └── learning/               # C++ / DSP / architecture teaching notes
│       ├── 01-cpp-mental-models.md
│       ├── 02-data-structures.md
│       └── 03-how-its-built.md
│
├── .claude/skills/             # Project-local AI skills for the core libs/algos
│   ├── juce-audio-plugin/      # JUCE patterns, threading, Result, ValueTree
│   ├── onnx-runtime-cpp/       # Ort::Session, tensors, export, lifetime
│   └── audio-dsp/              # chunking, overlap-add, STFT, HPSS, transients
│
├── Source/
│   ├── PluginProcessor.h/.cpp  # JUCE audio processor
│   ├── PluginEditor.h/.cpp     # JUCE UI
│   │
│   ├── Separation/
│   │   ├── SeparationWorker.h/.cpp     # Background thread orchestrator
│   │   ├── ONNXInferenceEngine.h/.cpp  # ONNX Runtime wrapper
│   │   ├── AudioChunker.h/.cpp         # Segment + overlap-add
│   │   └── StemWriter.h/.cpp           # Write stems to WAV files
│   │
│   ├── Drums/
│   │   ├── DrumDecomposer.h/.cpp       # HPSS + frequency masking
│   │   └── TransientDetector.h/.cpp    # Onset detection for kick/snare
│   │
│   └── UI/
│       ├── StemTogglePanel.h/.cpp      # Per-stem enable/disable checkboxes
│       ├── ProgressPanel.h/.cpp        # Progress bar + status text
│       └── FileDropZone.h/.cpp         # Drag-and-drop audio file target
│
├── Models/
│   ├── export_bs_roformer.py   # One-time export: PyTorch → ONNX
│   └── README.md               # How to regenerate model weights
│
├── Tests/
│   ├── AudioChunkerTest.cpp    # Unit test: roundtrip chunking is lossless
│   ├── InferenceEngineTest.cpp # Unit test: known input → known output
│   └── DrumDecomposerTest.cpp  # Unit test: kick/snare detection
│
└── Packaging/
    ├── macOS/                  # DMG creation scripts
    └── Windows/                # NSIS installer config
```

---

## Key Decisions Log

This section records *why* we made major decisions. Append to it as the project evolves.

| Decision | Choice | Rationale |
|---|---|---|
| Plugin framework | JUCE 8 | Industry standard. VST3+AU+Standalone from one codebase. Best C++ audio ecosystem. |
| Inference runtime | ONNX Runtime C++ | Bridges Python-trained models to C++ plugin. MIT license. GPU support via CUDA/CoreML/DirectML execution providers. |
| Primary model | **HTDemucs (default) + BS-RoFormer (opt-in)** | Pipeline is model-agnostic. HTDemucs ships as default — **MIT-licensed**, redistributable, easier/validated ONNX export (~8.5 dB vocals). BS-RoFormer-SW offered as an opt-in download for higher SDR (~10.9 dB) but **License: unknown / 699 MB**, so never bundled. See R8 + Model Architecture ⚠ licensing note. |
| Drum decomposition | **OPEN — revised** | ⚠ Original "DrumSep = DSP, no ML" was incorrect: HPSS can't name kick/snare/hat, and the real `drumsep` is a Demucs ML fine-tune. Decision deferred to Phase 3; recommended path is ship Stage 1 first, then add an ONNX drum model. See R2 + Model Architecture ⚠ note. |
| Real-time processing | No (offline only) | Eliminates the hardest constraints (latency, real-time inference, lookahead buffer management). Full song quality >> real-time quality tradeoff. |
| Language | C++17 | Required by JUCE. No viable Rust path for JUCE-based plugins (JUCE's Rust bindings are immature). |
| Build system | CMake | Modern JUCE recommends CMake over Projucer. Better CI integration, VSCode compatible. |
| Error handling | `juce::Result` | No exceptions across plugin boundary. Explicit failure handling at call site. |

---

*Document version: 1.1 — architectural review pass, May 2026*
*Changes in 1.1: corrected Stage-2 drum-separation design (HPSS≠named drums); added
Architectural Review & Risks (R1–R8); added coding standards §8–10 (ownership, const-
correctness, includes); linked `docs/learning/` and `.claude/skills/`.*
*Next revision: after Phase 0 completion, and after the R1 (ONNX export) spike.*