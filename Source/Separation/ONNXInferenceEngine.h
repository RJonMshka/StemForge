#pragma once
// ── System ───────────────────────────────────────────────────────────────────
#include <memory>
#include <vector>
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
//
// ONNX Runtime headers are intentionally kept OUT of this header (CLAUDE.md §10): all
// Ort:: types live behind the PIMPL in the .cpp. Callers don't pay the ORT include cost.

namespace stemforge
{
// THREAD: Worker thread only — owns an Ort::Session; not safe to share across threads.
//
// Model-agnostic wrapper around one separation model. Stem count and IO names come from
// the loaded model, not hard-coded — the same engine runs HTDemucs (4 stems) or
// BS-RoFormer (6 stems). Never throws across its boundary: Ort::Exception is caught and
// converted to juce::Result (a thrown exception would crash the host DAW).
//
// Output layouts handled (discovered at load):
//   • one rank-4 output [batch, stems, channels, samples]  — our export's "stems" tensor
//   • N rank-3 outputs  [batch, channels, samples]          — one tensor per stem
class ONNXInferenceEngine
{
public:
    ONNXInferenceEngine();
    ~ONNXInferenceEngine();

    // Loads an .onnx separation model. Safe: catches Ort::Exception → Result::fail.
    juce::Result loadModel (const juce::File& modelPath);
    bool isLoaded() const noexcept;

    int getNumStems() const noexcept;        // 0 until loaded (or until first run if dynamic)
    juce::StringArray getStemNames() const;  // model-provided names, else "stem_0..N"

    // The exact input length the model requires, or 0 if its sample axis is dynamic. Some
    // models (HTDemucs: fixed 7.8 s / 343980 samples) reject any other length; callers size
    // the chunker to this. 0 means "any length works" (e.g. BS-RoFormer's dynamic axis).
    int getRequiredInputSamples() const noexcept;

    // Runs one segment. input is [channels, samples]; on success stemsOut holds one buffer
    // per stem, each [channels, samples]. stemsOut is resized/overwritten here.
    juce::Result run (const juce::AudioBuffer<float>& input,
                      std::vector<juce::AudioBuffer<float>>& stemsOut);

private:
    struct Impl;                  // ONNX Runtime state hidden in the .cpp
    std::unique_ptr<Impl> m_impl;

    JUCE_DECLARE_NON_COPYABLE (ONNXInferenceEngine)
};
} // namespace stemforge
