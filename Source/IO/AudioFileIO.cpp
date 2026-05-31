#include "IO/AudioFileIO.h"

namespace stemforge
{
juce::Result AudioFileIO::readToBuffer (const juce::File& file,
                                        juce::AudioBuffer<float>& out,
                                        double& sampleRateOut)
{
    if (! file.existsAsFile())
        return juce::Result::fail ("File not found: " + file.getFullPathName());

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();   // WAV, AIFF, FLAC, Ogg, MP3 (platform-dependent)

    std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor (file));
    if (reader == nullptr)
        return juce::Result::fail ("Unsupported or unreadable audio file: " + file.getFileName());

    const auto numChannels = (int) reader->numChannels;
    const auto numSamples  = (int) reader->lengthInSamples;
    if (numChannels <= 0 || numSamples <= 0)
        return juce::Result::fail ("Audio file has no samples: " + file.getFileName());

    out.setSize (numChannels, numSamples);
    reader->read (&out, 0, numSamples, 0, true, true);
    sampleRateOut = reader->sampleRate;
    return juce::Result::ok();
}

juce::Result AudioFileIO::writeWav (const juce::File& file,
                                    const juce::AudioBuffer<float>& buffer,
                                    double sampleRate)
{
    if (buffer.getNumChannels() <= 0 || buffer.getNumSamples() <= 0)
        return juce::Result::fail ("Refusing to write an empty buffer");

    file.deleteFile();   // overwrite cleanly (FileOutputStream would otherwise append)

    // ┌─ CHUNK: writer ownership handoff (JUCE 8 AudioFormatWriterOptions API) ──┐
    // │ createWriterFor() takes the stream FROM the unique_ptr on success (leaves │
    // │ it null); on failure ownership stays with us and the unique_ptr frees it. │
    // │ We write IEEE float32 — lossless for our float buffers and the format the │
    // │ stem pipeline uses downstream.                                           │
    // └──────────────────────────────────────────────────────────────────────────┘
    auto fileStream = std::make_unique<juce::FileOutputStream> (file);
    if (! fileStream->openedOk())
        return juce::Result::fail ("Cannot open for writing: " + file.getFullPathName());

    std::unique_ptr<juce::OutputStream> stream = std::move (fileStream);

    const auto options = juce::AudioFormatWriterOptions{}
                             .withSampleRate (sampleRate)
                             .withNumChannels (buffer.getNumChannels())
                             .withBitsPerSample (32)
                             .withSampleFormat (juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer = wav.createWriterFor (stream, options);
    if (writer == nullptr)
        return juce::Result::fail ("Could not create WAV writer for: " + file.getFileName());

    if (! writer->writeFromAudioSampleBuffer (buffer, 0, buffer.getNumSamples()))
        return juce::Result::fail ("Write failed for: " + file.getFileName());

    return juce::Result::ok();   // writer flushes + closes on destruction here
}

juce::Result AudioFileIO::verifyRoundtripLossless (const juce::File& input,
                                                   const juce::File& output,
                                                   float& maxAbsDiffOut)
{
    maxAbsDiffOut = 0.0f;

    juce::AudioBuffer<float> a;
    double srA = 0.0;
    if (auto r = readToBuffer (input, a, srA); r.failed())
        return r;

    if (auto r = writeWav (output, a, srA); r.failed())
        return r;

    juce::AudioBuffer<float> b;
    double srB = 0.0;
    if (auto r = readToBuffer (output, b, srB); r.failed())
        return r;

    if (a.getNumChannels() != b.getNumChannels() || a.getNumSamples() != b.getNumSamples())
        return juce::Result::fail ("Roundtrip dimension mismatch (read ≠ written)");

    // Largest absolute per-sample deviation across the whole buffer.
    for (int ch = 0; ch < a.getNumChannels(); ++ch)
    {
        const float* pa = a.getReadPointer (ch);
        const float* pb = b.getReadPointer (ch);
        for (int i = 0; i < a.getNumSamples(); ++i)
            maxAbsDiffOut = juce::jmax (maxAbsDiffOut, std::abs (pa[i] - pb[i]));
    }
    return juce::Result::ok();
}
} // namespace stemforge
