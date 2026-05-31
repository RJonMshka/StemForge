#include "Separation/StemWriter.h"
// ── Ours ─────────────────────────────────────────────────────────────────────
#include "IO/AudioFileIO.h"

namespace stemforge
{
juce::Result StemWriter::writeStems (const juce::File& outputDir,
                                     const juce::StringArray& names,
                                     const std::vector<juce::AudioBuffer<float>>& stems,
                                     double sampleRate)
{
    if (names.size() != (int) stems.size())
        return juce::Result::fail ("Stem name/count mismatch ("
                                   + juce::String (names.size()) + " names, "
                                   + juce::String ((int) stems.size()) + " buffers)");

    if (auto r = outputDir.createDirectory(); r.failed())
        return r;

    for (int s = 0; s < (int) stems.size(); ++s)
    {
        // Copy so we can clamp without mutating the caller's buffers; clamping avoids
        // wrap-around clipping artifacts if the model overshoots ±1.
        juce::AudioBuffer<float> clamped (stems[(size_t) s]);
        for (int ch = 0; ch < clamped.getNumChannels(); ++ch)
            juce::FloatVectorOperations::clip (clamped.getWritePointer (ch),
                                               clamped.getReadPointer (ch),
                                               -1.0f, 1.0f, clamped.getNumSamples());

        const juce::File out = outputDir.getChildFile (names[s] + ".wav");
        if (auto r = AudioFileIO::writeWav (out, clamped, sampleRate); r.failed())
            return r;
    }

    return juce::Result::ok();
}
} // namespace stemforge
