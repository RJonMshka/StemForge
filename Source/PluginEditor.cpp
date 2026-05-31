#include "PluginEditor.h"

#if ! defined(STEMFORGE_HAS_ONNX)
 #include "IO/AudioFileIO.h"
#endif

namespace stemforge
{
namespace
{
constexpr juce::uint32 kColInfo  = 0xffaab4c0;   // neutral grey-blue
constexpr juce::uint32 kColOk    = 0xff62d0c4;   // teal
constexpr juce::uint32 kColWarn  = 0xffe0b86c;   // amber
constexpr juce::uint32 kColError = 0xffe06c6c;   // red

#if defined(STEMFORGE_HAS_ONNX)
// HTDemucs packs its four stems in this fixed order (see Models/spike/04_export_htdemucs.py).
// These are the display/output names; the model itself reports only generic stem_0..3.
const juce::StringArray kStemNames { "drums", "bass", "other", "vocals" };
#endif
} // namespace

StemForgeEditor::StemForgeEditor (StemForgeProcessor& proc)
    : juce::AudioProcessorEditor (&proc), m_processor (proc)
{
    m_title.setText ("StemForge", juce::dontSendNotification);
    m_title.setFont (juce::Font (juce::FontOptions (26.0f).withStyle ("Bold")));
    m_title.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (m_title);

    addAndMakeVisible (m_dropZone);

    m_status.setJustificationType (juce::Justification::centredLeft);
    m_status.setColour (juce::Label::textColourId, juce::Colour (kColInfo));
    addAndMakeVisible (m_status);

#if defined(STEMFORGE_HAS_ONNX)
    m_dropZone.onFileDropped = [this] (const juce::File& f) { fileChosen (f); };

    for (const auto& name : kStemNames)
    {
        auto* toggle = new juce::ToggleButton (name);
        toggle->setToggleState (true, juce::dontSendNotification);
        toggle->setColour (juce::ToggleButton::textColourId, juce::Colour (kColInfo));
        addAndMakeVisible (toggle);
        m_stemToggles.add (toggle);
    }

    m_outputButton.onClick  = [this] { chooseOutputDir(); };
    m_analyzeButton.onClick = [this] { startAnalysis(); };
    m_cancelButton.onClick  = [this] { m_processor.cancelSeparation(); };
    addAndMakeVisible (m_outputButton);
    addAndMakeVisible (m_analyzeButton);
    addAndMakeVisible (m_cancelButton);

    m_progressBar.setPercentageDisplay (true);
    addChildComponent (m_progressBar);   // only shown while a run is in flight

    // The worker outlives the editor (the processor owns it), so the editor clears this in
    // its destructor to avoid the async callback firing into a destroyed object.
    m_processor.worker().onFinished = [this] { onWorkerFinished(); };

    m_status.setText ("Drop an audio file to begin.", juce::dontSendNotification);
    refreshControls();
    setSize (560, 440);
#else
    m_dropZone.onFileDropped = [this] (const juce::File& f) { handleFile (f); };
    m_status.setText ("Drop a WAV/AIFF/FLAC file to verify the I/O roundtrip.",
                      juce::dontSendNotification);
    setSize (560, 360);
#endif
}

StemForgeEditor::~StemForgeEditor()
{
#if defined(STEMFORGE_HAS_ONNX)
    stopTimer();
    m_processor.worker().onFinished = nullptr;
#endif
}

#if defined(STEMFORGE_HAS_ONNX)
void StemForgeEditor::setStatus (const juce::String& text, juce::uint32 colour)
{
    m_status.setColour (juce::Label::textColourId, juce::Colour (colour));
    m_status.setText (text, juce::dontSendNotification);
}

void StemForgeEditor::fileChosen (const juce::File& file)
{
    if (m_processor.worker().isBusy())
        return;

    m_inputFile = file;
    // Default the output to a "<name>_stems" folder beside the source until the user picks one.
    if (m_outputDir == juce::File())
        m_outputDir = file.getParentDirectory()
                          .getChildFile (file.getFileNameWithoutExtension() + "_stems");

    setStatus ("Ready: " + file.getFileName() + "  →  " + m_outputDir.getFileName() + "/",
               kColInfo);
    refreshControls();
}

void StemForgeEditor::chooseOutputDir()
{
    const auto start = m_outputDir != juce::File()
                           ? m_outputDir
                           : juce::File::getSpecialLocation (juce::File::userMusicDirectory);

    m_chooser = std::make_unique<juce::FileChooser> ("Choose an output folder", start);
    m_chooser->launchAsync (juce::FileBrowserComponent::openMode
                                | juce::FileBrowserComponent::canSelectDirectories,
        [this] (const juce::FileChooser& fc)
        {
            const auto dir = fc.getResult();
            if (dir == juce::File())
                return;

            m_outputDir = dir;
            if (m_inputFile != juce::File())
                setStatus ("Ready: " + m_inputFile.getFileName() + "  →  "
                               + m_outputDir.getFileName() + "/", kColInfo);
            refreshControls();
        });
}

void StemForgeEditor::startAnalysis()
{
    if (m_inputFile == juce::File() || m_processor.worker().isBusy())
        return;

    const auto model = m_processor.getDefaultModelFile();
    if (! model.existsAsFile())
    {
        setStatus ("Model not found: " + model.getFullPathName()
                       + "  (set STEMFORGE_MODEL_PATH)", kColError);
        return;
    }

    if (enabledStemNames().isEmpty())
    {
        setStatus ("Select at least one stem to extract.", kColWarn);
        return;
    }

    m_outputDir.createDirectory();
    m_progressValue = 0.0;
    m_progressBar.setVisible (true);
    setStatus ("Separating " + m_inputFile.getFileName() + "…", kColInfo);

    m_processor.startSeparation (m_inputFile, m_outputDir, kStemNames, enabledStemNames());
    startTimerHz (20);
    refreshControls();
}

void StemForgeEditor::timerCallback()
{
    m_progressValue = (double) m_processor.worker().getProgress();
    if (! m_processor.worker().isBusy())   // completion handled by onWorkerFinished()
        stopTimer();
}

void StemForgeEditor::onWorkerFinished()
{
    stopTimer();
    m_progressValue = (double) m_processor.worker().getProgress();
    m_progressBar.setVisible (false);

    const auto status = m_processor.worker().getStatus();
    const auto msg    = m_processor.worker().getStatusMessage();
    using S = SeparationWorker::Status;

    setStatus (msg, status == S::Succeeded ? kColOk
                  : status == S::Cancelled ? kColWarn
                                           : kColError);
    refreshControls();
}

void StemForgeEditor::refreshControls()
{
    const bool busy     = m_processor.worker().isBusy();
    const bool hasInput = m_inputFile != juce::File();

    m_analyzeButton.setEnabled (hasInput && ! busy);
    m_cancelButton.setEnabled (busy);
    m_outputButton.setEnabled (! busy);
    m_dropZone.setEnabled (! busy);
    for (auto* toggle : m_stemToggles)
        toggle->setEnabled (! busy);
}

juce::StringArray StemForgeEditor::enabledStemNames() const
{
    juce::StringArray names;
    for (auto* toggle : m_stemToggles)
        if (toggle->getToggleState())
            names.add (toggle->getButtonText());
    return names;
}

#else  // ── Phase 0 fallback ──────────────────────────────────────────────────

void StemForgeEditor::handleFile (const juce::File& file)
{
    const auto out = file.getParentDirectory()
                         .getChildFile (file.getFileNameWithoutExtension() + "_roundtrip.wav");

    float maxDiff = 0.0f;
    const auto result = m_processor.loadAndVerifyRoundtrip (file, out, maxDiff);

    if (result.failed())
    {
        m_status.setColour (juce::Label::textColourId, juce::Colour (kColError));
        m_status.setText ("Error: " + result.getErrorMessage(), juce::dontSendNotification);
        return;
    }

    const bool lossless = maxDiff < 1.0e-6f;
    m_status.setColour (juce::Label::textColourId,
                        juce::Colour (lossless ? kColOk : kColWarn));
    m_status.setText (file.getFileName()
                          + "  →  roundtrip max diff " + juce::String (maxDiff, 9)
                          + (lossless ? "   (lossless ✓)" : "   (check!)"),
                      juce::dontSendNotification);
}
#endif

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

#if defined(STEMFORGE_HAS_ONNX)
    auto controls = area.removeFromBottom (32);
    m_cancelButton.setBounds  (controls.removeFromRight (90));
    controls.removeFromRight (8);
    m_analyzeButton.setBounds (controls.removeFromRight (110));
    m_outputButton.setBounds  (controls.removeFromLeft (140));
    area.removeFromBottom (8);

    m_progressBar.setBounds (area.removeFromBottom (22));
    area.removeFromBottom (8);

    auto toggles = area.removeFromBottom (28);
    const int tw = toggles.getWidth() / juce::jmax (1, m_stemToggles.size());
    for (auto* toggle : m_stemToggles)
        toggle->setBounds (toggles.removeFromLeft (tw));
    area.removeFromBottom (8);
#endif

    m_dropZone.setBounds (area);
}
} // namespace stemforge
