#include "PluginEditor.h"

namespace stemforge
{
StemForgeEditor::StemForgeEditor (StemForgeProcessor& proc)
    : juce::AudioProcessorEditor (&proc), m_processor (proc)
{
    m_title.setText ("StemForge", juce::dontSendNotification);
    m_title.setFont (juce::Font (juce::FontOptions (26.0f).withStyle ("Bold")));
    m_title.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (m_title);

    addAndMakeVisible (m_dropZone);
    m_dropZone.onFileDropped = [this] (const juce::File& f) { handleFile (f); };

    m_status.setText ("Drop a WAV/AIFF/FLAC file to verify the I/O roundtrip.",
                      juce::dontSendNotification);
    m_status.setJustificationType (juce::Justification::centredLeft);
    m_status.setColour (juce::Label::textColourId, juce::Colour (0xffaab4c0));
    addAndMakeVisible (m_status);

    setSize (560, 360);
}

void StemForgeEditor::handleFile (const juce::File& file)
{
    const auto out = file.getParentDirectory()
                         .getChildFile (file.getFileNameWithoutExtension() + "_roundtrip.wav");

    float maxDiff = 0.0f;
    const auto result = m_processor.loadAndVerifyRoundtrip (file, out, maxDiff);

    if (result.failed())
    {
        m_status.setColour (juce::Label::textColourId, juce::Colour (0xffe06c6c));
        m_status.setText ("Error: " + result.getErrorMessage(), juce::dontSendNotification);
        return;
    }

    const bool lossless = maxDiff < 1.0e-6f;
    m_status.setColour (juce::Label::textColourId,
                        lossless ? juce::Colour (0xff62d0c4) : juce::Colour (0xffe0b86c));
    m_status.setText (file.getFileName()
                          + "  →  roundtrip max diff " + juce::String (maxDiff, 9)
                          + (lossless ? "   (lossless ✓)" : "   (check!)"),
                      juce::dontSendNotification);
}

void StemForgeEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff151a21));
}

void StemForgeEditor::resized()
{
    auto area = getLocalBounds().reduced (20);
    m_title.setBounds (area.removeFromTop (40));
    area.removeFromTop (8);
    m_status.setBounds (area.removeFromBottom (44));
    area.removeFromBottom (8);
    m_dropZone.setBounds (area);
}
} // namespace stemforge
