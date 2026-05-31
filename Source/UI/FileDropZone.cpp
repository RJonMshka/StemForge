#include "UI/FileDropZone.h"

namespace stemforge
{
FileDropZone::FileDropZone()
{
    setInterceptsMouseClicks (false, false);   // purely a drop target in Phase 0
}

bool FileDropZone::isAudioFile (const juce::String& path)
{
    static const juce::StringArray exts { ".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg" };
    const auto lower = path.toLowerCase();
    for (const auto& e : exts)
        if (lower.endsWith (e))
            return true;
    return false;
}

bool FileDropZone::isInterestedInFileDrag (const juce::StringArray& files)
{
    for (const auto& f : files)
        if (isAudioFile (f))
            return true;
    return false;
}

void FileDropZone::fileDragEnter (const juce::StringArray&, int, int)
{
    m_dragging = true;
    repaint();
}

void FileDropZone::fileDragExit (const juce::StringArray&)
{
    m_dragging = false;
    repaint();
}

void FileDropZone::filesDropped (const juce::StringArray& files, int, int)
{
    m_dragging = false;
    repaint();

    for (const auto& f : files)
    {
        if (isAudioFile (f))
        {
            if (onFileDropped != nullptr)
                onFileDropped (juce::File (f));
            return;   // Phase 0: take the first audio file only
        }
    }
}

void FileDropZone::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced (8.0f);
    const float corner = 12.0f;

    g.setColour (m_dragging ? juce::Colour (0xff2c5d63) : juce::Colour (0xff202833));
    g.fillRoundedRectangle (bounds, corner);

    g.setColour (m_dragging ? juce::Colour (0xff62d0c4) : juce::Colour (0xff52606e));
    g.drawRoundedRectangle (bounds, corner, m_dragging ? 2.4f : 1.6f);

    g.setColour (m_dragging ? juce::Colours::white : juce::Colour (0xffaab4c0));
    g.setFont (juce::Font (juce::FontOptions (17.0f)));
    g.drawText (m_dragging ? "Release to load" : "Drop an audio file here",
                getLocalBounds(), juce::Justification::centred);
}
} // namespace stemforge
