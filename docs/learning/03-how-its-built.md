# Learning 03 — How StemForge Is Built (Assembly & Threads)

You know the pieces (`01`) and the data shapes (`02`). This doc shows how they assemble
into a running plugin and — critically — *which thread each piece runs on*. In audio
software, "what thread is this on?" is the question that prevents most bugs.

---

## The three threads

A JUCE plugin lives across three threads. Memorize this table; the whole architecture
follows from it.

| Thread | Runs | Deadline | Allowed to… |
|---|---|---|---|
| **Audio** | `processBlock()` | ~5–20 ms, hard | Touch pre-allocated buffers, read atomics. **Nothing else.** |
| **Message (UI)** | UI events, paint, button clicks | soft (~16 ms) | Update UI, read/write parameters, start the worker |
| **Worker** | our `SeparationWorker` | none (offline) | Allocate, load files, run ONNX, write WAVs |

The golden rule (CLAUDE.md rule 1): **the audio thread never waits for anything.** No
locks, no `new`, no file I/O, no inference. If `processBlock` ever blocks, the user hears
a click or dropout. Because StemForge is *offline*, our `processBlock` is essentially
empty — it passes audio through untouched. All real work is on the worker.

> Mental model: the audio thread is a real-time heartbeat you must never interrupt. The
> worker thread is where the "actual app" runs, like a web worker doing heavy compute off
> the main thread.

---

## The object graph

```
┌─ Message thread ───────────────────────────────────────────────┐
│  PluginEditor                    PluginProcessor                │
│  (the window, optional)  ──ref──▶ (always alive, owns state)    │
│   • FileDropZone                   • juce::ValueTree state       │
│   • StemTogglePanel                • owns the SeparationWorker   │
│   • ProgressPanel                  • processBlock() = passthrough│
└───────────────────────────────────────────┬────────────────────┘
                                  startSeparation(file)
                                             ▼
┌─ Worker thread (juce::Thread) ─────────────────────────────────┐
│  SeparationWorker::run()                                        │
│    1. read file        → AudioBuffer<float>   (02 §2)           │
│    2. chunk            → vector<Segment>       (02 §3)           │
│    3. for each segment → ONNXInferenceEngine.run()  (02 §4-5)   │
│    4. overlap-add      → per-stem AudioBuffer  (02 §6)          │
│    5. write stems      → WAV files             (02 §7)          │
│    6. publish progress → std::atomic<float>                     │
└────────────────────────────────────────────┬───────────────────┘
                                  owns / calls
                                             ▼
                          ONNXInferenceEngine  (Ort::Session wrapper)
```

### Why state lives in the Processor, never the Editor

The **Editor can be destroyed** any time the user closes the plugin window — but the
Processor keeps running (and the worker keeps separating). So anything that must survive a
window close (the chosen stems, output path, progress) lives in the **Processor**, in a
`juce::ValueTree`. The Editor is a *view* that reads from it. This is exactly "UI is not
the source of truth."

---

## How data crosses threads safely

Two crossings matter:

1. **UI → Worker (start):** the user clicks Analyze on the message thread; we call
   `worker.startSeparation(file)`. The worker copies what it needs and runs. One-way
   handoff, no shared mutable state. Clean.

2. **Worker → UI (progress):** the worker writes a `std::atomic<float> m_progress`
   (`01` §6). The UI thread *polls* it (or gets nudged via `juce::AsyncUpdater`) and
   repaints the progress bar. No lock, no race — a single atomic scalar is the only thing
   shared.

For the "done" signal and per-stem status, `juce::AsyncUpdater` lets the worker say
"something changed, repaint when you can" without touching UI objects from the worker
thread directly (touching JUCE Components off the message thread is illegal).

> Never call into a `juce::Component` from the worker thread. Publish data (atomics /
> AsyncUpdater) and let the message thread pull it. This is the whole concurrency story.

---

## The build pipeline (how source becomes a plugin)

```
CMakeLists.txt
   ├─ FetchContent: download JUCE 8 at configure time   (no submodule to manage)
   ├─ find / link ONNX Runtime (static)                 (the inference engine)
   ├─ juce_add_plugin(StemForge ...)  → defines VST3 + AU + Standalone targets
   └─ target_sources(... Source/**/*.cpp)
        │
   cmake --build  →  compiles each .cpp → .o → links → StemForge.vst3 / .app
```

- **CMake** is the build orchestrator (think `package.json` + bundler config). It resolves
  dependencies, picks compiler flags per platform, and emits the right plugin bundle
  format for each OS.
- **`FetchContent`** clones JUCE into `_deps/` at configure time so it's never committed
  (see `.gitignore`). One source of truth, reproducible builds.
- **One source tree → three artifacts:** JUCE compiles the same `Source/` into a VST3, an
  AU (macOS), and a Standalone `.app`. You write the plugin once.

---

## Lifecycle, end to end

1. DAW (or the Standalone shell) loads the plugin → constructs **one** `PluginProcessor`.
2. `prepareToPlay(sampleRate, blockSize)` is called → **allocate all reusable buffers
   here** (the one place allocation is fine before audio starts).
3. User opens the window → `PluginProcessor::createEditor()` builds a `PluginEditor`.
4. User drops a file, picks stems, clicks Analyze → message thread calls
   `worker.startSeparation()`.
5. Worker runs the 6-step pipeline above on its own thread, publishing progress.
6. Worker finishes → writes WAVs → flags done → UI shows "complete," lists output files.
7. User closes window → Editor destroyed; Processor + any in-flight worker keep living.

---

## Where the hard parts are (so you know where to slow down)

- **ONNX export of BS-RoFormer** — the model uses STFT/complex ops that don't always
  export cleanly to ONNX. Treat this as a research spike, not a checkbox. (PLAN risk.)
- **Tensor layout** — channel/sample order must match the model exactly (`02` §4).
- **Overlap-add window math** — wrong window → audible pulsing (`02` §6).
- **Stage-2 drum splitting** — plain HPSS can't tell a kick from a snare; the plan's
  "DSP-only" claim is flagged in `docs/PLAN.md`. Read that before building `Source/Drums/`.

That's the whole system. The rest is filling in each box, one phase at a time.
