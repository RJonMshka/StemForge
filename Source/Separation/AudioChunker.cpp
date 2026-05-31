#include "Separation/AudioChunker.h"
// ── System ───────────────────────────────────────────────────────────────────
#include <cmath>

namespace stemforge
{
AudioChunker::AudioChunker (int segmentSamples, float overlap)
    : m_segmentSamples (juce::jmax (1, segmentSamples))
{
    // Keep at least one sample of hop so planSegments always makes progress.
    overlap   = juce::jlimit (0.0f, 0.95f, overlap);
    m_hopSize = juce::jmax (1, (int) std::lround (m_segmentSamples * (1.0f - overlap)));
}

std::vector<int> AudioChunker::planSegments (int totalSamples) const
{
    std::vector<int> starts;
    if (totalSamples <= 0)
        return starts;

    for (int start = 0; start < totalSamples; start += m_hopSize)
        starts.push_back (start);

    return starts;   // last start < totalSamples, and start+segment covers the tail
}

void AudioChunker::readSegment (const juce::AudioBuffer<float>& src,
                                int start,
                                juce::AudioBuffer<float>& dst) const
{
    const int numCh = src.getNumChannels();
    dst.setSize (numCh, m_segmentSamples, false, false, true);
    dst.clear();   // zero-pads the tail of the final (over-running) segment

    const int copyLen = juce::jlimit (0, m_segmentSamples, src.getNumSamples() - start);
    if (copyLen <= 0)
        return;

    for (int ch = 0; ch < numCh; ++ch)
        dst.copyFrom (ch, 0, src, ch, start, copyLen);
}

// ──────────────────────────────────────────────────────────────────────────────

OverlapAdder::OverlapAdder (int numChannels, int totalSamples, int segmentSamples)
    : m_numChannels    (juce::jmax (1, numChannels)),
      m_totalSamples   (juce::jmax (0, totalSamples)),
      m_segmentSamples (juce::jmax (1, segmentSamples)),
      m_window  ((size_t) m_segmentSamples),
      m_winSum  ((size_t) m_totalSamples, 0.0f)
{
    m_accum.setSize (m_numChannels, m_totalSamples);
    m_accum.clear();

    // Periodic Hann in its sin² form: sin²(π(i+0.5)/N) ≡ 0.5(1−cos(2π(i+0.5)/N)). The
    // sin² form is numerically stable at the edges — the equivalent 0.5−0.5·cos form
    // suffers catastrophic cancellation there (cos(tiny) rounds to 1.0f, collapsing the
    // ~1e-11 edge weight to exactly 0 and zeroing the file's first/last samples). The
    // (i+0.5) half-sample shift keeps the window strictly positive at i==0 and i==N-1.
    for (int i = 0; i < m_segmentSamples; ++i)
    {
        const float s = std::sin (juce::MathConstants<float>::pi
                                  * ((float) i + 0.5f) / (float) m_segmentSamples);
        m_window[(size_t) i] = s * s;
    }
}

void OverlapAdder::addSegment (int start, const juce::AudioBuffer<float>& segment)
{
    const int len = juce::jlimit (0, m_segmentSamples, m_totalSamples - start);
    if (len <= 0)
        return;

    // Window weights are channel-independent — accumulate the sum once.
    for (int i = 0; i < len; ++i)
        m_winSum[(size_t) (start + i)] += m_window[(size_t) i];

    for (int ch = 0; ch < m_numChannels; ++ch)
    {
        float* dest      = m_accum.getWritePointer (ch);
        const float* seg = segment.getReadPointer (ch);
        for (int i = 0; i < len; ++i)
            dest[start + i] += m_window[(size_t) i] * seg[i];
    }
}

void OverlapAdder::finalize (juce::AudioBuffer<float>& out) const
{
    out.setSize (m_numChannels, m_totalSamples);

    for (int ch = 0; ch < m_numChannels; ++ch)
    {
        const float* acc = m_accum.getReadPointer (ch);
        float* dst       = out.getWritePointer (ch);
        for (int n = 0; n < m_totalSamples; ++n)
        {
            // winSum is EXACTLY 0 only where no window covered the sample; the periodic
            // Hann is strictly positive everywhere it reaches. So `> 0` is the right test:
            // a small-but-positive sum (e.g. ~1e-11 near a large window's edge) still
            // reconstructs exactly (w·x / w == x) — an absolute epsilon would wrongly
            // zero those edge samples. Denormals are not flushed here (no ScopedNoDenormals).
            const float ws = m_winSum[(size_t) n];
            dst[n] = ws > 0.0f ? acc[n] / ws : 0.0f;
        }
    }
}
} // namespace stemforge
