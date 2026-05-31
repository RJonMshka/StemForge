// Phase 0 verification: prove the read -> 32-bit WAV write -> read path is lossless.
// Headless counterpart to the GUI drop-zone check, so CI/ctest can assert it.
#include "IO/AudioFileIO.h"

#include <juce_core/juce_core.h>
#include <cstdio>

int main (int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf ("usage: StemForgeRoundtripTest <input-audio-file>\n");
        return 2;
    }

    const juce::String path = juce::String::fromUTF8 (argv[1]);
    const juce::File input = juce::File::isAbsolutePath (path)
                                 ? juce::File (path)
                                 : juce::File::getCurrentWorkingDirectory().getChildFile (path);
    const juce::File output = input.getParentDirectory()
                                  .getChildFile (input.getFileNameWithoutExtension() + "_roundtrip.wav");

    float maxDiff = 0.0f;
    const auto result = stemforge::AudioFileIO::verifyRoundtripLossless (input, output, maxDiff);
    if (result.failed())
    {
        std::printf ("FAIL: %s\n", result.getErrorMessage().toRawUTF8());
        return 1;
    }

    const bool lossless = maxDiff < 1.0e-6f;
    std::printf ("roundtrip max abs diff = %.9g  ->  %s\n",
                 (double) maxDiff, lossless ? "LOSSLESS PASS" : "CHECK (not lossless)");
    return lossless ? 0 : 1;
}
