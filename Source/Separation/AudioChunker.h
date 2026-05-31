#pragma once
// ── System ───────────────────────────────────────────────────────────────────
#include <vector>
// ── JUCE ─────────────────────────────────────────────────────────────────────
#include <juce_audio_basics/juce_audio_basics.h>

namespace stemforge
{
// Canonical inference rate. Inputs at any other rate are resampled around inference
// (PLAN R5). Documented once, here, so the whole pipeline agrees.
constexpr double MODEL_SAMPLE_RATE = 44100.0;

// 10 s segments at 44.1 kHz, 25% overlap — see PLAN Concept 4 + the audio-dsp skill.
// The model accepts variable-length chunks (dynamic sample axis), so these are tunable.
constexpr int   DEFAULT_SEGMENT_SAMPLES = 441000;
constexpr float DEFAULT_OVERLAP         = 0.25f;

// THREAD: Worker thread only. Stateless plan math (only the ctor-derived sizes are kept);
// safe to reuse across segments on the owning thread.
//
// Splits a full-length buffer into fixed-length overlapping segments for inference. The
// reconstruction half lives in OverlapAdder below.
class AudioChunker
{
public:
    AudioChunker (int segmentSamples = DEFAULT_SEGMENT_SAMPLES,
                  float overlap      = DEFAULT_OVERLAP);

    int segmentSamples() const noexcept { return m_segmentSamples; }
    int hopSize()        const noexcept { return m_hopSize; }

    // ┌─ CHUNK: AudioChunker::planSegments ──────────────────────────────────────┐
    // │ INTENT: Segment start offsets covering [0, totalSamples). The last segment │
    // │         runs past the end and is zero-padded on read.                      │
    // │ CONTRACT: starts.front()==0; starts.back() < totalSamples;                 │
    // │           starts.back()+segmentSamples >= totalSamples (full coverage).    │
    // └────────────────────────────────────────────────────────────────────────────┘
    std::vector<int> planSegments (int totalSamples) const;

    // Copies [start, start+segmentSamples) from src into dst (resized to
    // [src.channels, segmentSamples]), zero-padding any samples past the end of src.
    void readSegment (const juce::AudioBuffer<float>& src,
                      int start,
                      juce::AudioBuffer<float>& dst) const;

private:
    int m_segmentSamples;
    int m_hopSize;
};

// THREAD: Worker thread only. Accumulates windowed, overlapping processed segments back
// into a full-length buffer, then normalizes by the summed window weights.
//
// ┌─ CHUNK: OverlapAdder ────────────────────────────────────────────────────────┐
// │ INTENT: Reassemble per-segment model output into one continuous stem with no   │
// │         audible seams. Uses a PERIODIC Hann window (nonzero at its endpoints)   │
// │         and divides by the accumulated window-sum, so reconstruction is exact   │
// │         (lossless for an identity "model") even at the file's first/last sample.│
// │ CONTRACT: with identity segments, finalize() == the original input (≤1e-5).     │
// │ DEPENDS ON: window length == segmentSamples used by the feeding AudioChunker.   │
// └──────────────────────────────────────────────────────────────────────────────┘
class OverlapAdder
{
public:
    OverlapAdder (int numChannels, int totalSamples, int segmentSamples);

    // Windows one processed segment and adds it into the accumulator at `start`.
    // segment must be [numChannels, >=segmentSamples]; samples past totalSamples drop.
    void addSegment (int start, const juce::AudioBuffer<float>& segment);

    // Normalizes by the accumulated window weights into `out` (resized to full length).
    void finalize (juce::AudioBuffer<float>& out) const;

private:
    int m_numChannels;
    int m_totalSamples;
    int m_segmentSamples;
    juce::AudioBuffer<float> m_accum;   // [numChannels, totalSamples]
    std::vector<float> m_window;        // [segmentSamples]
    std::vector<float> m_winSum;        // [totalSamples] — window weight per output sample
};
} // namespace stemforge
