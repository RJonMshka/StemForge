#pragma once
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
// ── Ours ─────────────────────────────────────────────────────────────────────
#if defined(STEMFORGE_HAS_ONNX)
 #include "Separation/SeparationWorker.h"
#endif

namespace stemforge
{
// THREAD: Created once and lives for the whole plugin instance.
//   - processBlock() runs on the AUDIO thread — offline plugin, so it's pass-through.
//   - everything else here is called on the MESSAGE thread.
// State source of truth lives in m_state (juce::ValueTree) so DAW save/load works.
class StemForgeProcessor : public juce::AudioProcessor
{
public:
    StemForgeProcessor();
    ~StemForgeProcessor() override = default;

    // ── AudioProcessor: audio lifecycle ──
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    // THREAD: Audio thread — offline plugin, this is intentionally a pass-through.
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // ── AudioProcessor: editor ──
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    // ── AudioProcessor: identity / programs ──
    const juce::String getName() const override { return "StemForge"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    // ── AudioProcessor: state persistence ──
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // ── StemForge API (message thread) ──
    // Resolves the model the worker should run. Phase-2 dev resolution (proper bundling is
    // Phase 4): STEMFORGE_MODEL_PATH env override, else the repo weights path baked in by
    // CMake. Returns an empty File if neither is available.
    juce::File getDefaultModelFile() const;

#if defined(STEMFORGE_HAS_ONNX)
    // Records the paths in state (DAW save/load) and kicks off the background separation.
    void startSeparation (const juce::File& input,
                          const juce::File& outputDir,
                          const juce::StringArray& displayNames,
                          const juce::StringArray& enabledStems);
    void cancelSeparation();

    // THREAD: Message thread — the editor reads progress/status and wires onFinished.
    SeparationWorker& worker() noexcept { return m_worker; }
#else
    // Phase 0 fallback (builds without ONNX Runtime): records the input file in state and
    // verifies a lossless WAV roundtrip, writing a copy to `writtenCopy`. Returns the
    // largest per-sample diff.
    juce::Result loadAndVerifyRoundtrip (const juce::File& input,
                                         const juce::File& writtenCopy,
                                         float& maxAbsDiffOut);
#endif

private:
    juce::ValueTree m_state { "StemForge" };

#if defined(STEMFORGE_HAS_ONNX)
    // THREAD: Message thread owns it; the worker runs its own background thread internally.
    SeparationWorker m_worker;
#endif

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StemForgeProcessor)
};
} // namespace stemforge
