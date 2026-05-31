#pragma once
// ── System ───────────────────────────────────────────────────────────────────
#include <memory>
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_audio_processors/juce_audio_processors.h>
// ── Ours ─────────────────────────────────────────────────────────────────────
#include "PluginProcessor.h"
#include "UI/FileDropZone.h"

namespace stemforge
{
// THREAD: Message thread only. Optional/destroyable — holds no authoritative state, just a
// reference to the processor (the source of truth) and the transient UI selection.
class StemForgeEditor : public juce::AudioProcessorEditor
#if defined(STEMFORGE_HAS_ONNX)
                      , private juce::Timer
#endif
{
public:
    explicit StemForgeEditor (StemForgeProcessor& processor);
    ~StemForgeEditor() override;

    void paint (juce::Graphics& g) override;
    void resized() override;

private:
#if defined(STEMFORGE_HAS_ONNX)
    // ── Phase 2 separation flow ──
    void fileChosen (const juce::File& file);   // drop/selection → ready to analyze
    void chooseOutputDir();
    void startAnalysis();
    void timerCallback() override;              // polls the worker's progress
    void onWorkerFinished();                    // message-thread completion callback
    void refreshControls();                     // enable/disable per worker state
    void setStatus (const juce::String& text, juce::uint32 colour);
    juce::StringArray enabledStemNames() const;

    juce::File m_inputFile;
    juce::File m_outputDir;

    juce::TextButton                     m_outputButton { "Output folder…" };
    juce::TextButton                     m_analyzeButton { "Analyze" };
    juce::TextButton                     m_cancelButton  { "Cancel" };
    juce::OwnedArray<juce::ToggleButton> m_stemToggles;
    double                               m_progressValue { 0.0 };
    juce::ProgressBar                    m_progressBar { m_progressValue };
    std::unique_ptr<juce::FileChooser>   m_chooser;
#else
    // ── Phase 0 fallback (no ONNX Runtime): WAV I/O roundtrip demo ──
    void handleFile (const juce::File& file);
#endif

    StemForgeProcessor& m_processor;

    juce::Label  m_title;
    FileDropZone m_dropZone;
    juce::Label  m_status;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StemForgeEditor)
};
} // namespace stemforge
