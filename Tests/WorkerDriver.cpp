// Phase-2 verification harness: drives the real SeparationWorker through its public
// interface under a live JUCE MessageManager — exactly the path the editor's "Analyze"
// button triggers (startSeparation → background thread → atomic progress → AsyncUpdater
// → onFinished on the message thread). NOT a unit test of internals: it pumps the message
// loop and samples progress just as the editor's Timer does, confirms onFinished fires on
// the message thread, and checks the stems land on disk.
//
//   StemForgeWorkerDriver <model.onnx> <input-audio> <output-dir> [cancel]
//
#include "Separation/SeparationWorker.h"

#include <juce_events/juce_events.h>
#include <cstdio>

using namespace stemforge;

namespace
{
juce::File resolve (const char* arg)
{
    const juce::String p = juce::String::fromUTF8 (arg);
    return juce::File::isAbsolutePath (p) ? juce::File (p)
                                          : juce::File::getCurrentWorkingDirectory().getChildFile (p);
}
} // namespace

int main (int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf ("usage: StemForgeWorkerDriver <model.onnx> <input-audio> <output-dir> [cancel]\n");
        return 2;
    }

    const juce::File model    = resolve (argv[1]);
    const juce::File input    = resolve (argv[2]);
    const juce::File outDir   = resolve (argv[3]);
    const bool       doCancel = (argc >= 5 && juce::String (argv[4]) == "cancel");

    juce::ScopedJuceInitialiser_GUI juceInit;   // gives us a MessageManager to pump

    SeparationWorker worker;
    bool onFinishedFired = false;
    worker.onFinished = [&worker, &onFinishedFired]
    {
        std::printf ("\n[onFinished @ message thread] status=%d  \"%s\"\n",
                     (int) worker.getStatus(), worker.getStatusMessage().toRawUTF8());
        onFinishedFired = true;
    };

    // Mirror the editor: HTDemucs display order, and a SUBSET selection to prove filtering.
    const juce::StringArray displayNames { "drums", "bass", "other", "vocals" };
    const juce::StringArray enabled      { "vocals", "drums" };

    std::printf ("starting: model=%s\n  input=%s\n  out=%s\n  enabled=%s  cancel=%s\n",
                 model.getFileName().toRawUTF8(), input.getFileName().toRawUTF8(),
                 outDir.getFullPathName().toRawUTF8(), enabled.joinIntoString (",").toRawUTF8(),
                 doCancel ? "yes" : "no");

    worker.startSeparation (model, input, outDir, displayNames, enabled);

    // Pump messages (so AsyncUpdater → onFinished can fire) while sampling the atomic
    // progress the editor's Timer would poll. Loop until the async completion lands.
    float lastShown = -1.0f;
    bool  cancelRequested = false;
    const auto startMs = juce::Time::getMillisecondCounter();
    while (! onFinishedFired)
    {
        juce::MessageManager::getInstance()->runDispatchLoopUntil (50);
        const float p = worker.getProgress();
        if (p - lastShown >= 0.05f || (p >= 1.0f && lastShown < 1.0f))
        {
            std::printf ("\rprogress %.0f%%   ", p * 100.0f);
            std::fflush (stdout);
            lastShown = p;
        }
        if (doCancel && ! cancelRequested && p > 0.30f)
        {
            std::printf ("\n[requesting cancel @ %.0f%%]\n", p * 100.0f);
            worker.cancel();
            cancelRequested = true;
        }
        if (juce::Time::getMillisecondCounter() - startMs > 600000)   // 10-min safety
        {
            std::printf ("\nTIMEOUT (worker still running)\n");
            return 1;
        }
    }

    std::printf ("final status=%d  message=\"%s\"\n",
                 (int) worker.getStatus(), worker.getStatusMessage().toRawUTF8());

    const auto written = worker.getWrittenStems();
    std::printf ("written stems: [%s]\n", written.joinIntoString (", ").toRawUTF8());
    for (const auto& name : written)
    {
        const auto f = outDir.getChildFile (name + ".wav");
        std::printf ("  %-10s %s  (%lld bytes)\n", (name + ".wav").toRawUTF8(),
                     f.existsAsFile() ? "EXISTS" : "MISSING", (long long) f.getSize());
    }

    const auto status = worker.getStatus();
    const bool pass = doCancel ? status == SeparationWorker::Status::Cancelled
                               : status == SeparationWorker::Status::Succeeded;
    return pass ? 0 : 1;
}
