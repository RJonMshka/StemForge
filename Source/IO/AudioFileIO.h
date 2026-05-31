#pragma once
// ── System ───────────────────────────────────────────────────────────────────
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_audio_formats/juce_audio_formats.h>

namespace stemforge
{
// THREAD: Thread-safe — pure static utilities, no shared state. Today they run on the
// message thread (Phase 0); in Phase 2 the worker thread will call the same functions.
//
// Phase 0 helper: read audio into a buffer, write a 32-bit WAV, and prove the read→write
// →read roundtrip is lossless. This is the foundation the SeparationWorker will reuse.
class AudioFileIO
{
public:
    // Reads any format JUCE knows (WAV/AIFF/FLAC/…) into a planar float buffer.
    static juce::Result readToBuffer (const juce::File& file,
                                      juce::AudioBuffer<float>& out,
                                      double& sampleRateOut);

    // Writes a planar float buffer as a 32-bit WAV. Overwrites if the file exists.
    static juce::Result writeWav (const juce::File& file,
                                  const juce::AudioBuffer<float>& buffer,
                                  double sampleRate);

    // Reads `input`, writes it to `output`, reads `output` back, and reports the largest
    // per-sample difference. For a lossless path this is ~0 (below float32 epsilon).
    static juce::Result verifyRoundtripLossless (const juce::File& input,
                                                 const juce::File& output,
                                                 float& maxAbsDiffOut);

private:
    AudioFileIO() = delete;
};
} // namespace stemforge
