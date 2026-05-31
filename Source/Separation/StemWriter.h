#pragma once
// ── System ───────────────────────────────────────────────────────────────────
#include <vector>
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

namespace stemforge
{
// THREAD: Worker thread only — does disk I/O, never call from the audio thread.
//
// Writes the separated stems to disk, one 32-bit float WAV per stem, reusing the Phase 0
// AudioFileIO writer so the output format matches the verified roundtrip path.
class StemWriter
{
public:
    // Writes "<names[i]>.wav" into outputDir for each stem buffer. Output is clamped to
    // [-1, 1] (audio-dsp numerical hygiene) before writing. names.size() must equal
    // stems.size(); creates outputDir if needed; fails fast on the first write error.
    static juce::Result writeStems (const juce::File& outputDir,
                                    const juce::StringArray& names,
                                    const std::vector<juce::AudioBuffer<float>>& stems,
                                    double sampleRate);

private:
    StemWriter() = delete;
};
} // namespace stemforge
