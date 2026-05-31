#pragma once
// ── System ───────────────────────────────────────────────────────────────────
#include <atomic>
#include <functional>
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>      // juce::AsyncUpdater
// ── Ours ─────────────────────────────────────────────────────────────────────
#include "Separation/ONNXInferenceEngine.h"   // ORT-free header (PIMPL) — safe here

namespace stemforge
{
// THREAD: see per-member annotations. A juce::Thread that runs the full offline separation
// pipeline (load → resample → stereo → chunk → infer → overlap-add → resample back → write)
// off the message thread. Progress is published through an atomic the UI polls; completion
// is signalled on the message thread via juce::AsyncUpdater. Never throws across its
// boundary — every fallible step returns a juce::Result that becomes a Failed status.
//
// This is the Phase-2 lift of the Phase-1 headless driver (Tests/SeparateCli.cpp) into a
// background thread with progress + cancel.
class SeparationWorker : public juce::Thread,
                         private juce::AsyncUpdater
{
public:
    enum class Status { Idle, Running, Succeeded, Failed, Cancelled };

    SeparationWorker();
    ~SeparationWorker() override;

    // THREAD: Message thread — configure inputs then launch. No-op while already running.
    //   displayNames : friendly name per stem index (HTDemucs: drums/bass/other/vocals).
    //                  Empty ⇒ use the model's own stem names. Drives output filenames.
    //   enabledStems : subset of effective names to actually write. Empty ⇒ write all.
    void startSeparation (const juce::File& modelFile,
                          const juce::File& inputFile,
                          const juce::File& outputDir,
                          const juce::StringArray& displayNames = {},
                          const juce::StringArray& enabledStems = {});

    // THREAD: Message thread — request cancel; run() stops at the next loop check.
    void cancel();

    // THREAD: Thread-safe (atomic).
    float  getProgress() const noexcept { return m_progress.load(); }
    Status getStatus()   const noexcept { return m_status.load(); }
    bool   isBusy()      const noexcept { return m_status.load() == Status::Running; }

    // THREAD: Thread-safe (lock-protected) — meaningful once a run has ended.
    juce::String      getStatusMessage() const;
    juce::StringArray getWrittenStems()  const;

    // THREAD: invoked on the MESSAGE thread when a run reaches a terminal status.
    std::function<void()> onFinished;

    // THREAD: Worker thread.
    void run() override;

private:
    // THREAD: Message thread.
    void handleAsyncUpdate() override;

    // THREAD: Worker thread — stash a terminal message, publish the status, notify the UI.
    void finish (Status status, const juce::String& message);

    // ── Config: written on the message thread before startThread(), then only read on the
    //    worker. Not mutated while running, so no lock is needed. ──
    juce::File        m_modelFile;
    juce::File        m_inputFile;
    juce::File        m_outputDir;
    juce::StringArray m_displayNames;
    juce::StringArray m_enabledStems;

    // THREAD: Worker thread only — the inference engine and its loaded model live here.
    ONNXInferenceEngine m_engine;

    // THREAD: Thread-safe (atomic).
    std::atomic<float>  m_progress { 0.0f };
    std::atomic<Status> m_status   { Status::Idle };

    // THREAD: Thread-safe (lock-protected) — written by the worker at finish, read by the UI.
    juce::CriticalSection m_resultLock;
    juce::String          m_message;
    juce::StringArray     m_writtenStems;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SeparationWorker)
};
} // namespace stemforge
