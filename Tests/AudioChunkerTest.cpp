// Phase 1 verification: prove chunk → (identity) → overlap-add reconstruction is lossless.
// This is the contract CLAUDE.md calls out: "chunking roundtrip must be lossless." The
// OverlapAdder divides by the accumulated window-sum, so an identity "model" must return
// the original signal to within float epsilon — including the very first/last samples.
#include "Separation/AudioChunker.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>
#include <cstdio>

using namespace stemforge;

namespace
{
// Fill a buffer with a distinct, non-trivial signal per channel so any seam, drop, or
// amplitude pulse shows up as a difference.
void fillTestSignal (juce::AudioBuffer<float>& buf)
{
    for (int ch = 0; ch < buf.getNumChannels(); ++ch)
    {
        float* d = buf.getWritePointer (ch);
        for (int n = 0; n < buf.getNumSamples(); ++n)
        {
            const float t = (float) n;
            d[n] = 0.6f * std::sin (0.013f * t + (float) ch)
                 + 0.3f * std::sin (0.071f * t)
                 + 0.1f * std::sin (0.250f * t + 0.5f * (float) ch);
        }
    }
}

float roundtripMaxDiff (int total, int seg, float overlap, int channels)
{
    juce::AudioBuffer<float> in (channels, total);
    fillTestSignal (in);

    AudioChunker chunker (seg, overlap);
    OverlapAdder adder (channels, total, chunker.segmentSamples());

    juce::AudioBuffer<float> segment;
    for (int start : chunker.planSegments (total))
    {
        chunker.readSegment (in, start, segment);   // identity model: feed it straight back
        adder.addSegment (start, segment);
    }

    juce::AudioBuffer<float> out;
    adder.finalize (out);

    float maxDiff = 0.0f;
    for (int ch = 0; ch < channels; ++ch)
    {
        const float* a = in.getReadPointer (ch);
        const float* b = out.getReadPointer (ch);
        for (int n = 0; n < total; ++n)
            maxDiff = juce::jmax (maxDiff, std::abs (a[n] - b[n]));
    }
    return maxDiff;
}
} // namespace

int main()
{
    struct Case { int total; int seg; float overlap; int channels; const char* name; };
    const Case cases[] = {
        { 44100, 1024, 0.25f, 2, "stereo 1s / 1024 @ 25%" },
        { 10000, 1000, 0.50f, 2, "stereo / 1000 @ 50% (COLA)" },
        {  5000,  777, 0.30f, 2, "stereo / odd seg @ 30%" },
        {  2048, 2048, 0.25f, 2, "single full-length segment" },
        {   100, 1024, 0.25f, 1, "mono shorter-than-segment (zero-pad)" },
        { 30000,  512, 0.75f, 1, "mono / heavy 75% overlap" },
        // Regression: a default-sized 441k window on a short signal — the window weight
        // near index 0 is ~1e-11, which an absolute-epsilon guard would wrongly zero.
        { 88200, DEFAULT_SEGMENT_SAMPLES, DEFAULT_OVERLAP, 2, "short signal / 441k window edge" },
    };

    constexpr float kTolerance = 1.0e-5f;
    bool allPass = true;

    for (const auto& c : cases)
    {
        const float diff = roundtripMaxDiff (c.total, c.seg, c.overlap, c.channels);
        const bool pass = diff < kTolerance;
        allPass = allPass && pass;
        std::printf ("[%s] %-38s max abs diff = %.3e\n",
                     pass ? "PASS" : "FAIL", c.name, (double) diff);
    }

    std::printf ("%s\n", allPass ? "ALL LOSSLESS" : "RECONSTRUCTION NOT LOSSLESS");
    return allPass ? 0 : 1;
}
