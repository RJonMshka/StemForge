#pragma once
// ── System ───────────────────────────────────────────────────────────────────
#include <functional>
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_gui_basics/juce_gui_basics.h>

namespace stemforge
{
// THREAD: Message thread only (it's a juce::Component).
//
// A drag-and-drop target for audio files. Owns no state beyond a transient "is something
// being dragged over me" flag; reports a dropped file via the onFileDropped callback so the
// editor stays in control of what happens next.
class FileDropZone : public juce::Component,
                     public juce::FileDragAndDropTarget
{
public:
    FileDropZone();
    ~FileDropZone() override = default;

    // Called on the message thread with the first audio file dropped.
    std::function<void (const juce::File&)> onFileDropped;

    // ── FileDragAndDropTarget ──
    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void fileDragEnter (const juce::StringArray& files, int x, int y) override;
    void fileDragExit  (const juce::StringArray& files) override;
    void filesDropped  (const juce::StringArray& files, int x, int y) override;

    // ── Component ──
    void paint (juce::Graphics& g) override;

private:
    static bool isAudioFile (const juce::String& path);

    bool m_dragging = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FileDropZone)
};
} // namespace stemforge
