#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "IO/AudioFileIO.h"

namespace stemforge
{
StemForgeProcessor::StemForgeProcessor()
    : juce::AudioProcessor (BusesProperties()
        .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
{
}

void StemForgeProcessor::prepareToPlay (double, int)
{
    // Offline plugin: no real-time DSP, so nothing to pre-allocate yet. When inference
    // arrives it stays on the worker thread, never here. (See CLAUDE.md rule 1.)
}

bool StemForgeProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& main = layouts.getMainOutputChannelSet();
    if (main != juce::AudioChannelSet::mono() && main != juce::AudioChannelSet::stereo())
        return false;
    // Input and output layouts must match (we pass audio through unchanged).
    return layouts.getMainInputChannelSet() == main;
}

void StemForgeProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    // Pass-through: clear any output channels that have no matching input, then leave the
    // signal untouched. All separation work happens off the audio thread (Phase 2+).
    for (int ch = getTotalNumInputChannels(); ch < getTotalNumOutputChannels(); ++ch)
        buffer.clear (ch, 0, buffer.getNumSamples());
}

juce::AudioProcessorEditor* StemForgeProcessor::createEditor()
{
    return new StemForgeEditor (*this);
}

void StemForgeProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = m_state.createXml())
        copyXmlToBinary (*xml, destData);
}

void StemForgeProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
    {
        auto restored = juce::ValueTree::fromXml (*xml);
        if (restored.isValid())
            m_state = restored;
    }
}

juce::Result StemForgeProcessor::loadAndVerifyRoundtrip (const juce::File& input,
                                                         const juce::File& writtenCopy,
                                                         float& maxAbsDiffOut)
{
    m_state.setProperty ("inputFile", input.getFullPathName(), nullptr);
    return AudioFileIO::verifyRoundtripLossless (input, writtenCopy, maxAbsDiffOut);
}
} // namespace stemforge

// ── JUCE plugin entry point ───────────────────────────────────────────────────
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new stemforge::StemForgeProcessor();
}
