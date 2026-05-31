#include "Separation/SeparationWorker.h"

#include "Separation/AudioChunker.h"
#include "Separation/StemWriter.h"
#include "IO/AudioFileIO.h"

#include <cmath>
#include <memory>
#include <vector>

namespace stemforge
{
namespace
{
// Resample every channel with a Lagrange interpolator (PLAN R5: resample around inference
// rather than rejecting off-rate inputs). Returns a copy at `to` Hz. Mirrors SeparateCli so
// the worker and the headless driver share one resampling contract.
juce::AudioBuffer<float> resample (const juce::AudioBuffer<float>& in, double from, double to)
{
    if (juce::approximatelyEqual (from, to) || from <= 0.0 || to <= 0.0)
        return in;

    const double ratio  = from / to;                                 // input samples / output sample
    const int    outLen = (int) std::ceil (in.getNumSamples() / ratio);
    juce::AudioBuffer<float> out (in.getNumChannels(), outLen);
    out.clear();

    for (int ch = 0; ch < in.getNumChannels(); ++ch)
    {
        juce::LagrangeInterpolator interp;
        interp.process (ratio, in.getReadPointer (ch), out.getWritePointer (ch), outLen);
    }
    return out;
}

// HTDemucs expects stereo input. Upmix a mono buffer by duplicating its single channel;
// pass anything already multi-channel through untouched. (PLAN Phase 2: handle mono inputs
// instead of erroring on them.)
juce::AudioBuffer<float> ensureStereo (const juce::AudioBuffer<float>& in)
{
    if (in.getNumChannels() != 1)
        return in;

    juce::AudioBuffer<float> out (2, in.getNumSamples());
    out.copyFrom (0, 0, in, 0, 0, in.getNumSamples());
    out.copyFrom (1, 0, in, 0, 0, in.getNumSamples());
    return out;
}
} // namespace

SeparationWorker::SeparationWorker() : juce::Thread ("StemForge Separation") {}

SeparationWorker::~SeparationWorker()
{
    // Stop the worker before we tear down its state; give in-flight inference time to
    // unwind. Then drop any pending async callback so it can't fire into a dead object.
    stopThread (5000);
    cancelPendingUpdate();
}

void SeparationWorker::startSeparation (const juce::File& modelFile,
                                        const juce::File& inputFile,
                                        const juce::File& outputDir,
                                        const juce::StringArray& displayNames,
                                        const juce::StringArray& enabledStems)
{
    if (isBusy())
        return;

    m_modelFile    = modelFile;
    m_inputFile    = inputFile;
    m_outputDir    = outputDir;
    m_displayNames = displayNames;
    m_enabledStems = enabledStems;

    {
        const juce::ScopedLock sl (m_resultLock);
        m_message.clear();
        m_writtenStems.clear();
    }
    m_progress.store (0.0f);
    m_status.store (Status::Running);   // publish "running" before the thread launches
    startThread();                       // also clears the threadShouldExit flag
}

void SeparationWorker::cancel()
{
    signalThreadShouldExit();
}

juce::String SeparationWorker::getStatusMessage() const
{
    const juce::ScopedLock sl (m_resultLock);
    return m_message;
}

juce::StringArray SeparationWorker::getWrittenStems() const
{
    const juce::ScopedLock sl (m_resultLock);
    return m_writtenStems;
}

void SeparationWorker::finish (Status status, const juce::String& message)
{
    {
        const juce::ScopedLock sl (m_resultLock);
        m_message = message;
    }
    m_status.store (status);   // terminal status — release of m_message under the lock above
    triggerAsyncUpdate();       // → handleAsyncUpdate() on the message thread
}

void SeparationWorker::handleAsyncUpdate()
{
    if (onFinished)
        onFinished();
}

// ┌─ CHUNK: SeparationWorker::run ───────────────────────────────────────────────┐
// │ INTENT: The whole offline pipeline on the worker thread. Each stage is fallible│
// │         (juce::Result) and routes to finish(Failed, …); cancel is honoured at  │
// │         every coarse boundary via threadShouldExit(). Progress is mapped so the │
// │         per-segment inference loop — the dominant cost — fills 0.05‥0.90.       │
// │ CONTRACT: exactly one finish() call per run (success or failure).               │
// └────────────────────────────────────────────────────────────────────────────────┘
void SeparationWorker::run()
{
    // ── 1. Load audio. ─────────────────────────────────────────────────────────
    juce::AudioBuffer<float> mix;
    double inputRate = 0.0;
    if (auto r = AudioFileIO::readToBuffer (m_inputFile, mix, inputRate); r.failed())
        return finish (Status::Failed, "Read failed: " + r.getErrorMessage());

    if (threadShouldExit()) return finish (Status::Cancelled, "Cancelled.");
    m_progress.store (0.03f);

    // ── 2. Resample to the model rate and ensure stereo. ───────────────────────
    juce::AudioBuffer<float> work = ensureStereo (resample (mix, inputRate, MODEL_SAMPLE_RATE));

    // ── 3. Load the model. ─────────────────────────────────────────────────────
    if (auto r = m_engine.loadModel (m_modelFile); r.failed())
        return finish (Status::Failed, "Model load failed: " + r.getErrorMessage());
    if (threadShouldExit()) return finish (Status::Cancelled, "Cancelled.");
    m_progress.store (0.05f);

    // ── 4. Chunk → infer → overlap-add. ────────────────────────────────────────
    // Size segments to the model: HTDemucs requires a fixed 343980-sample chunk; models with
    // a dynamic sample axis report 0, so we fall back to the 10 s default. (PLAN R3.)
    const int  modelSamples = m_engine.getRequiredInputSamples();
    AudioChunker chunker = modelSamples > 0 ? AudioChunker (modelSamples) : AudioChunker();
    const int  total = work.getNumSamples();
    const int  numCh = work.getNumChannels();
    const auto starts = chunker.planSegments (total);

    std::vector<std::unique_ptr<OverlapAdder>> adders;   // one per stem, created after run 0
    juce::AudioBuffer<float> segment;
    std::vector<juce::AudioBuffer<float>> stemSegs;

    for (size_t i = 0; i < starts.size(); ++i)
    {
        if (threadShouldExit()) return finish (Status::Cancelled, "Cancelled.");

        chunker.readSegment (work, starts[i], segment);
        if (auto r = m_engine.run (segment, stemSegs); r.failed())
            return finish (Status::Failed, "Inference failed: " + r.getErrorMessage());

        if (adders.empty())
            for (size_t s = 0; s < stemSegs.size(); ++s)
                adders.push_back (std::make_unique<OverlapAdder> (numCh, total, chunker.segmentSamples()));

        for (size_t s = 0; s < adders.size(); ++s)
            adders[s]->addSegment (starts[i], stemSegs[s]);

        m_progress.store (0.05f + 0.85f * (float) (i + 1) / (float) starts.size());
    }

    if (adders.empty())
        return finish (Status::Failed, "Model produced no stems.");

    // ── 5. Finalize, resample back to the source rate, filter to enabled stems. ──
    // Effective name per stem: caller's display name if given, else the model's own name.
    // Selection matches on that effective name (case-insensitive); empty selection ⇒ all.
    const juce::StringArray modelNames = m_engine.getStemNames();
    std::vector<juce::AudioBuffer<float>> stems;
    juce::StringArray names;

    for (size_t s = 0; s < adders.size(); ++s)
    {
        juce::String name = (s < (size_t) m_displayNames.size() && m_displayNames[(int) s].isNotEmpty())
                                ? m_displayNames[(int) s]
                                : (s < (size_t) modelNames.size() ? modelNames[(int) s]
                                                                   : "stem_" + juce::String ((int) s));

        if (! m_enabledStems.isEmpty() && ! m_enabledStems.contains (name, /*ignoreCase*/ true))
            continue;

        juce::AudioBuffer<float> full;
        adders[s]->finalize (full);
        stems.push_back (resample (full, MODEL_SAMPLE_RATE, inputRate));   // back to source rate
        names.add (name);
    }
    m_progress.store (0.95f);

    if (stems.empty())
        return finish (Status::Failed, "No stems selected to write.");
    if (threadShouldExit()) return finish (Status::Cancelled, "Cancelled.");

    // ── 6. Write stems to disk. ────────────────────────────────────────────────
    if (auto r = StemWriter::writeStems (m_outputDir, names, stems, inputRate); r.failed())
        return finish (Status::Failed, "Write failed: " + r.getErrorMessage());

    {
        const juce::ScopedLock sl (m_resultLock);
        m_writtenStems = names;
    }
    m_progress.store (1.0f);
    finish (Status::Succeeded,
            "Wrote " + juce::String (names.size()) + " stem"
                + (names.size() == 1 ? "" : "s") + " to " + m_outputDir.getFileName());
}
} // namespace stemforge
