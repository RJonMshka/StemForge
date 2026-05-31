#include "Separation/ONNXInferenceEngine.h"
// ── Third-party (ONNX Runtime) ───────────────────────────────────────────────
#include <onnxruntime_cxx_api.h>
// ── System ───────────────────────────────────────────────────────────────────
#include <algorithm>
#include <array>
#include <string>

namespace stemforge
{
// ┌─ CHUNK: ONNXInferenceEngine::Impl ───────────────────────────────────────────┐
// │ INTENT: Hold all Ort:: state. Env + Session + SessionOptions are EXPENSIVE —    │
// │         created once in loadModel(), reused for every segment's run(). IO names  │
// │         and the output layout are discovered at load so run() does no querying.  │
// └──────────────────────────────────────────────────────────────────────────────┘
struct ONNXInferenceEngine::Impl
{
    Ort::Env            env { ORT_LOGGING_LEVEL_WARNING, "StemForge" };
    Ort::SessionOptions sessionOptions;
    Ort::MemoryInfo     memInfo = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::Session> session;

    std::string              inputName;
    std::vector<std::string> outputNames;

    // Discovered at load. numStems == 0 means "determine on the first run" (dynamic axis).
    int  numStems          = 0;
    bool singlePackedOutput = false;   // true: one rank-4 [B,stems,C,N] output
    int  requiredSamples    = 0;       // fixed model input length, 0 if the axis is dynamic

    // Reused across segments (allocated on first run; same shape every segment).
    std::vector<float> inputData;
};

ONNXInferenceEngine::ONNXInferenceEngine() : m_impl (std::make_unique<Impl>())
{
    // CPU-only for now (PLAN R6: start CPU-only, statically simple; add CoreML/CUDA/
    // DirectML providers behind a runtime-detected fallback later). 0 = ORT picks threads.
    m_impl->sessionOptions.SetIntraOpNumThreads (0);
    m_impl->sessionOptions.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
}

ONNXInferenceEngine::~ONNXInferenceEngine() = default;

bool ONNXInferenceEngine::isLoaded() const noexcept { return m_impl->session != nullptr; }
int  ONNXInferenceEngine::getNumStems() const noexcept { return m_impl->numStems; }
int  ONNXInferenceEngine::getRequiredInputSamples() const noexcept { return m_impl->requiredSamples; }

juce::StringArray ONNXInferenceEngine::getStemNames() const
{
    juce::StringArray names;

    // When the model packs all stems into one tensor we have no per-stem names — emit
    // generic, unambiguous labels rather than guessing an order (the #1 integration bug).
    if (m_impl->singlePackedOutput || (int) m_impl->outputNames.size() != m_impl->numStems)
    {
        for (int s = 0; s < m_impl->numStems; ++s)
            names.add ("stem_" + juce::String (s));
        return names;
    }

    for (const auto& n : m_impl->outputNames)
        names.add (juce::String (n));
    return names;
}

juce::Result ONNXInferenceEngine::loadModel (const juce::File& modelPath)
{
    if (! modelPath.existsAsFile())
        return juce::Result::fail ("Model file not found: " + modelPath.getFullPathName());

    try
    {
        // Hold the path string alive while ORT copies it (toRawUTF8 points into `full`).
        const juce::String full = modelPath.getFullPathName();
       #if defined (_WIN32)
        m_impl->session = std::make_unique<Ort::Session> (
            m_impl->env, full.toWideCharPointer(), m_impl->sessionOptions);
       #else
        m_impl->session = std::make_unique<Ort::Session> (
            m_impl->env, full.toRawUTF8(), m_impl->sessionOptions);
       #endif

        Ort::AllocatorWithDefaultOptions allocator;
        auto& session = *m_impl->session;

        if (session.GetInputCount() < 1)
            return juce::Result::fail ("Model has no inputs");

        m_impl->inputName = session.GetInputNameAllocated (0, allocator).get();

        // Fixed-length models (HTDemucs) carry a concrete sample dim; dynamic ones (-1) don't.
        const auto inShape = session.GetInputTypeInfo (0)
                                    .GetTensorTypeAndShapeInfo().GetShape();   // [B, C, N]
        m_impl->requiredSamples = (inShape.size() == 3 && inShape[2] > 0) ? (int) inShape[2] : 0;

        const size_t outCount = session.GetOutputCount();
        if (outCount < 1)
            return juce::Result::fail ("Model has no outputs");

        m_impl->outputNames.clear();
        for (size_t i = 0; i < outCount; ++i)
            m_impl->outputNames.emplace_back (session.GetOutputNameAllocated (i, allocator).get());

        // ── Determine the output layout (see header). ──────────────────────────
        const auto outShape = session.GetOutputTypeInfo (0)
                                     .GetTensorTypeAndShapeInfo().GetShape();

        if (outCount == 1 && outShape.size() == 4)
        {
            m_impl->singlePackedOutput = true;                       // [B, stems, C, N]
            m_impl->numStems = outShape[1] > 0 ? (int) outShape[1] : 0;  // 0 => dynamic
        }
        else
        {
            m_impl->singlePackedOutput = false;                      // one [B, C, N] per stem
            m_impl->numStems = (int) outCount;
        }

        return juce::Result::ok();
    }
    catch (const Ort::Exception& e)
    {
        m_impl->session.reset();
        return juce::Result::fail (juce::String ("ONNX load failed: ") + e.what());
    }
}

juce::Result ONNXInferenceEngine::run (const juce::AudioBuffer<float>& input,
                                       std::vector<juce::AudioBuffer<float>>& stemsOut)
{
    if (! isLoaded())
        return juce::Result::fail ("run() called before a model was loaded");

    const int channels = input.getNumChannels();
    const int samples   = input.getNumSamples();
    if (channels <= 0 || samples <= 0)
        return juce::Result::fail ("run() given an empty input segment");

    try
    {
        // ── Build the input tensor: planar [batch=1, channels, samples]. ──────────
        // `inputData` MUST outlive `inputTensor` (the tensor is a non-owning view).
        auto& data = m_impl->inputData;
        const auto samplesSz = (size_t) samples;
        data.resize ((size_t) channels * samplesSz);
        for (int c = 0; c < channels; ++c)
            std::copy_n (input.getReadPointer (c), samples, data.data() + (size_t) c * samplesSz);

        const std::array<int64_t, 3> inShape { 1, channels, samples };
        Ort::Value inputTensor = Ort::Value::CreateTensor<float> (
            m_impl->memInfo, data.data(), data.size(), inShape.data(), inShape.size());

        // ── Run. IO names come from the model, never guessed. ────────────────────
        const char* inNames[] = { m_impl->inputName.c_str() };
        std::vector<const char*> outNames;
        outNames.reserve (m_impl->outputNames.size());
        for (const auto& n : m_impl->outputNames)
            outNames.push_back (n.c_str());

        auto outputs = m_impl->session->Run (
            Ort::RunOptions { nullptr },
            inNames,         &inputTensor, 1,
            outNames.data(), outNames.size());

        // ── Extract per-stem buffers. ─────────────────────────────────────────────
        if (m_impl->singlePackedOutput)
        {
            const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1,S,C,N]
            if (shape.size() != 4)
                return juce::Result::fail ("Expected a rank-4 packed output tensor");

            const int numStems = (int) shape[1];
            const int outCh     = (int) shape[2];
            const int outN      = (int) shape[3];
            m_impl->numStems = numStems;   // resolve a dynamic stem axis

            const float* src = outputs[0].GetTensorData<float>();
            stemsOut.assign ((size_t) numStems, {});
            for (int s = 0; s < numStems; ++s)
            {
                auto& dst = stemsOut[(size_t) s];
                dst.setSize (outCh, outN);
                const auto outNSz = (size_t) outN;
                for (int c = 0; c < outCh; ++c)
                {
                    const float* plane = src + (((size_t) s * (size_t) outCh + (size_t) c) * outNSz);
                    std::copy_n (plane, outN, dst.getWritePointer (c));
                }
            }
        }
        else
        {
            stemsOut.assign (outputs.size(), {});
            for (size_t s = 0; s < outputs.size(); ++s)
            {
                const auto shape = outputs[s].GetTensorTypeAndShapeInfo().GetShape();  // [1,C,N]
                if (shape.size() != 3)
                    return juce::Result::fail ("Expected rank-3 per-stem output tensors");

                const int outCh = (int) shape[1];
                const int outN  = (int) shape[2];
                const float* src = outputs[s].GetTensorData<float>();

                auto& dst = stemsOut[s];
                dst.setSize (outCh, outN);
                const auto outNSz = (size_t) outN;
                for (int c = 0; c < outCh; ++c)
                    std::copy_n (src + (size_t) c * outNSz, outN, dst.getWritePointer (c));
            }
        }

        return juce::Result::ok();
    }
    catch (const Ort::Exception& e)
    {
        return juce::Result::fail (juce::String ("ONNX inference failed: ") + e.what());
    }
}
} // namespace stemforge
