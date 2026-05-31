#pragma once
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_audio_processors/juce_audio_processors.h>
// ── Ours ─────────────────────────────────────────────────────────────────────
#include "PluginProcessor.h"
#include "UI/FileDropZone.h"

namespace stemforge
{
// THREAD: Message thread only. Optional/destroyable — holds no authoritative state, just a
// reference to the processor (the source of truth).
class StemForgeEditor : public juce::AudioProcessorEditor
{
public:
    explicit StemForgeEditor (StemForgeProcessor& processor);
    ~StemForgeEditor() override = default;

    void paint (juce::Graphics& g) override;
    void resized() override;

private:
    // Runs the Phase 0 roundtrip check and reports the result in the status label.
    void handleFile (const juce::File& file);

    StemForgeProcessor& m_processor;

    juce::Label    m_title;
    FileDropZone   m_dropZone;
    juce::Label    m_status;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StemForgeEditor)
};
} // namespace stemforge
