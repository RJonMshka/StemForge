// Phase 1 headless harness: run a separation model on an audio file from C++ and write the
// stems to disk — no UI, no worker thread yet (that's Phase 2). This is the end-to-end
// integration of AudioChunker → ONNXInferenceEngine → OverlapAdder → StemWriter.
//
//   StemForgeSeparate <model.onnx> <input-audio> <output-dir>
//
#include "Separation/AudioChunker.h"
#include "Separation/ONNXInferenceEngine.h"
#include "Separation/StemWriter.h"
#include "IO/AudioFileIO.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace stemforge;

namespace
{
juce::File resolve (const char* arg)
{
    const juce::String p = juce::String::fromUTF8 (arg);
    return juce::File::isAbsolutePath (p)
               ? juce::File (p)
               : juce::File::getCurrentWorkingDirectory().getChildFile (p);
}

// Resample every channel with a Lagrange interpolator (PLAN R5: resample around inference
// rather than rejecting off-rate inputs). Returns a copy at `to` Hz.
juce::AudioBuffer<float> resample (const juce::AudioBuffer<float>& in, double from, double to)
{
    if (juce::approximatelyEqual (from, to) || from <= 0.0 || to <= 0.0)
        return in;

    const double ratio = from / to;                                  // input samples / output sample
    const int outLen = (int) std::ceil (in.getNumSamples() / ratio);
    juce::AudioBuffer<float> out (in.getNumChannels(), outLen);
    out.clear();

    for (int ch = 0; ch < in.getNumChannels(); ++ch)
    {
        juce::LagrangeInterpolator interp;
        interp.process (ratio, in.getReadPointer (ch), out.getWritePointer (ch), outLen);
    }
    return out;
}
} // namespace

int main (int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf ("usage: StemForgeSeparate <model.onnx> <input-audio> <output-dir>\n");
        return 2;
    }

    const juce::File modelFile = resolve (argv[1]);
    const juce::File inputFile = resolve (argv[2]);
    const juce::File outputDir = resolve (argv[3]);

    // ── 1. Load audio. ────────────────────────────────────────────────────────
    juce::AudioBuffer<float> mix;
    double inputRate = 0.0;
    if (auto r = AudioFileIO::readToBuffer (inputFile, mix, inputRate); r.failed())
    {
        std::printf ("FAIL (read): %s\n", r.getErrorMessage().toRawUTF8());
        return 1;
    }
    std::printf ("loaded %s: %d ch, %d samples, %.0f Hz\n",
                 inputFile.getFileName().toRawUTF8(),
                 mix.getNumChannels(), mix.getNumSamples(), inputRate);

    // ── 2. Resample to the model's canonical rate if needed. ───────────────────
    juce::AudioBuffer<float> work = resample (mix, inputRate, MODEL_SAMPLE_RATE);
    if (! juce::approximatelyEqual (inputRate, MODEL_SAMPLE_RATE))
        std::printf ("resampled %.0f → %.0f Hz (%d samples)\n",
                     inputRate, MODEL_SAMPLE_RATE, work.getNumSamples());

    // ── 3. Load the model. ─────────────────────────────────────────────────────
    ONNXInferenceEngine engine;
    if (auto r = engine.loadModel (modelFile); r.failed())
    {
        std::printf ("FAIL (load): %s\n", r.getErrorMessage().toRawUTF8());
        return 1;
    }

    // ── 4. Chunk → infer → overlap-add. ────────────────────────────────────────
    // Size segments to the model: HTDemucs requires a fixed 343980-sample chunk; models with
    // a dynamic sample axis report 0, so we fall back to the 10 s default. (PLAN R3/Phase 1.)
    const int modelSamples = engine.getRequiredInputSamples();
    AudioChunker chunker = modelSamples > 0 ? AudioChunker (modelSamples) : AudioChunker();
    if (modelSamples > 0)
        std::printf ("model requires a fixed %d-sample chunk\n", modelSamples);
    const int total = work.getNumSamples();
    const int numCh = work.getNumChannels();
    const auto starts = chunker.planSegments (total);

    std::vector<std::unique_ptr<OverlapAdder>> adders;   // one per stem, created after run 0
    juce::AudioBuffer<float> segment;
    std::vector<juce::AudioBuffer<float>> stemSegs;

    for (size_t i = 0; i < starts.size(); ++i)
    {
        chunker.readSegment (work, starts[i], segment);
        if (auto r = engine.run (segment, stemSegs); r.failed())
        {
            std::printf ("FAIL (infer): %s\n", r.getErrorMessage().toRawUTF8());
            return 1;
        }

        if (adders.empty())
            for (size_t s = 0; s < stemSegs.size(); ++s)
                adders.push_back (std::make_unique<OverlapAdder> (numCh, total, chunker.segmentSamples()));

        for (size_t s = 0; s < adders.size(); ++s)
            adders[s]->addSegment (starts[i], stemSegs[s]);

        std::printf ("\rsegment %d / %d", (int) i + 1, (int) starts.size());
        std::fflush (stdout);
    }
    std::printf ("\n");

    if (adders.empty())
    {
        std::printf ("FAIL: model produced no stems\n");
        return 1;
    }

    // ── 5. Finalize, resample back, write. ─────────────────────────────────────
    std::vector<juce::AudioBuffer<float>> stems (adders.size());
    for (size_t s = 0; s < adders.size(); ++s)
    {
        juce::AudioBuffer<float> full;
        adders[s]->finalize (full);
        stems[s] = resample (full, MODEL_SAMPLE_RATE, inputRate);   // back to source rate
    }

    if (auto r = StemWriter::writeStems (outputDir, engine.getStemNames(), stems, inputRate);
        r.failed())
    {
        std::printf ("FAIL (write): %s\n", r.getErrorMessage().toRawUTF8());
        return 1;
    }

    std::printf ("wrote %d stems to %s\n",
                 (int) stems.size(), outputDir.getFullPathName().toRawUTF8());
    return 0;
}
