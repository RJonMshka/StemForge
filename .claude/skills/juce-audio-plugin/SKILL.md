---
name: juce-audio-plugin
description: Use when writing or reviewing JUCE 8 plugin code in StemForge — PluginProcessor/PluginEditor, the audio/message/worker threading model, processBlock real-time safety, juce::Thread + AsyncUpdater background work, ValueTree state, AudioBuffer handling, or juce::Result error handling. Trigger on JUCE classes, processBlock, prepareToPlay, thread-safety questions, or plugin lifecycle.
---

# JUCE 8 Audio Plugin Patterns (StemForge)

Authoritative quick-reference for writing correct JUCE code here. Pair with
`docs/learning/03-how-its-built.md` for the threading model and CLAUDE.md for the rules.

## The two classes
- `PluginProcessor` — owns ALL state, lives for the whole plugin instance, audio thread
  calls `processBlock()`. State source of truth (use `juce::ValueTree`).
- `PluginEditor` — the window, optional, can be destroyed/recreated. Holds a reference to
  the Processor. **Never** store authoritative state here.

## Thread rules (non-negotiable)
1. **Audio thread** (`processBlock`): no alloc, no lock, no I/O, no exceptions, no
   inference. StemForge is offline → `processBlock` is pass-through. Pre-allocate in
   `prepareToPlay(sampleRate, maxBlockSize)`.
2. **Message thread**: UI + parameters. Only thread allowed to touch `juce::Component`.
3. **Worker thread** (`juce::Thread` subclass): all heavy work — file load, ONNX, WAV write.
4. Annotate every class holding shared state: `// THREAD: Worker thread only` etc.

## Cross-thread communication
- Scalar worker→UI: `std::atomic<T>` (e.g. progress). UI polls or `triggerAsyncUpdate()`.
- "Repaint when convenient": `juce::AsyncUpdater` — worker calls `triggerAsyncUpdate()`,
  `handleAsyncUpdate()` runs on the message thread where touching Components is legal.
- Never call a `Component` method from the worker thread.

## Background work skeleton
```cpp
// THREAD: declared per-member below
class SeparationWorker : public juce::Thread {
public:
    SeparationWorker() : juce::Thread("StemForge Separation") {}
    // THREAD: Message thread
    void startSeparation(juce::File f) { m_input = std::move(f); startThread(); }
    // THREAD: Thread-safe
    float getProgress() const { return m_progress.load(); }
    void run() override {                      // THREAD: Worker thread
        while (!threadShouldExit()) { /* pipeline; check threadShouldExit() in loops */ }
    }
private:
    juce::File m_input;
    std::atomic<float> m_progress { 0.0f };
};
```
Always check `threadShouldExit()` inside long loops for cancel support.

## Error handling — `juce::Result`, never exceptions
```cpp
juce::Result loadModel(const juce::File& p) {
    if (!p.existsAsFile()) return juce::Result::fail("Model not found: " + p.getFullPathName());
    return juce::Result::ok();
}
auto r = loadModel(f);
if (r.failed()) { juce::Logger::writeToLog(r.getErrorMessage()); return; }
```
A thrown exception can crash the host DAW. Keep all fallible calls behind `Result`.

## AudioBuffer essentials
```cpp
juce::AudioBuffer<float> buf(numChannels, numSamples);   // planar, owns memory (RAII)
const float* in  = buf.getReadPointer(ch);
float*       out = buf.getWritePointer(ch);
buf.clear();                                              // zero-fill (for overlap-add dest)
```
Pass buffers as `const juce::AudioBuffer<float>&` to avoid copying. Allocate once, reuse.

## State that survives DAW save/load
```cpp
juce::ValueTree state { "StemForge" };
state.setProperty("outputDir", path, nullptr);
// wire into getStateInformation/setStateInformation → presets + project recall for free
```

## File I/O & WAV output
```cpp
juce::WavAudioFormat fmt;
std::unique_ptr<juce::AudioFormatWriter> w {
    fmt.createWriterFor(new juce::FileOutputStream(outFile),
                        sampleRate, (unsigned)numChannels, 32, {}, 0) };
w->writeFromAudioSampleBuffer(stemBuffer, 0, stemBuffer.getNumSamples());
```
Write stems as 32-bit float WAV (lossless). Reading uses `AudioFormatManager` +
`AudioFormatReader`.

## Build (CMake + FetchContent)
- JUCE pulled via `FetchContent_Declare(juce ...)`, never committed.
- `juce_add_plugin(StemForge FORMATS VST3 AU Standalone ...)` → one source tree, 3 targets.
- Link ONNX Runtime statically (`target_link_libraries`). Keep model weights out of git.

## Review checklist
- [ ] No allocation/lock/IO in `processBlock`.
- [ ] Every shared member has a `// THREAD:` annotation.
- [ ] Worker→UI only via atomics / AsyncUpdater; no Component access off message thread.
- [ ] Fallible paths return `juce::Result`, callers handle `.failed()`.
- [ ] Buffers passed by `const&`; allocations hoisted to `prepareToPlay`/ctor.
- [ ] Long worker loops poll `threadShouldExit()`.
