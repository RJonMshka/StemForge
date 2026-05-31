---
name: audio-dsp
description: Use when writing or reviewing the DSP in StemForge — audio chunking, overlap-add reconstruction, windowing (Hann), STFT/ISTFT, HPSS (harmonic-percussive separation via median filtering), transient/onset detection, frequency-band masking, or sample-rate conversion. Trigger on overlap-add, window, spectrogram, FFT/STFT, HPSS, median filter, transient, or drum-decomposition work.
---

# Audio DSP Patterns (StemForge)

The signal-processing math the plugin relies on, with the gotchas. Pair with
`docs/learning/02-data-structures.md`.

## Audio = float samples in [-1, 1], planar per channel
Sample rate (e.g. 44.1 kHz) = samples/sec/channel. Everything below operates on
`float` arrays. Resample to the model's expected rate before inference (`juce::dsp` /
`LagrangeInterpolator`) and document the rate in one place.

## Chunking + overlap-add (the core reconstruction)
Models process fixed-length **segments** that overlap (~25%). Reassemble with a window so
seams are inaudible.

```
hopSize    = segmentSamples * (1 - overlap)      // distance between segment starts
numSegments = ceil(totalSamples / hopSize)
```
Reconstruction:
```cpp
dest.clear();                                    // full-length, zero-filled
std::vector<float> winSum(totalSamples, 0.0f);   // accumulate window weights
for (seg : segments) {
    for (i = 0; i < seg.length; ++i) {
        float w = window[i];                     // Hann
        dest[ch][seg.start + i] += w * segOut[ch][i];
        winSum[seg.start + i]   += w;            // (window only depends on i, but track sum)
    }
}
// normalize so overlapping windows sum to unity → no amplitude pulsing
for (n) if (winSum[n] > 1e-8f) dest[ch][n] /= winSum[n];
```

### Window choice & the COLA condition
Use a **Hann** window. For artifact-free overlap-add the window must satisfy **COLA**
(Constant Overlap-Add): the shifted windows sum to a constant. Hann at 50% or 75% overlap
satisfies COLA; arbitrary overlaps need the explicit `winSum` normalization above. Symptom
of getting it wrong: a periodic amplitude pulse at the hop rate.

## STFT / ISTFT
Short-Time Fourier Transform = slide a window over the signal, FFT each frame → a
**spectrogram** (2D: time × frequency, complex values). ISTFT inverts it via overlap-add
(same machinery as above, one level down). Use `juce::dsp::FFT`. Keep window + hop
consistent between STFT and ISTFT or reconstruction won't be unity-gain.

## HPSS — Harmonic/Percussive Source Separation
Separates "smooth-in-time" (harmonic/pitched) from "smooth-in-frequency"
(percussive/transient) content of a spectrogram:
1. Magnitude spectrogram `S` (time × freq).
2. **Median filter across time** (horizontal) → `H` (harmonic estimate — sustained tones).
3. **Median filter across frequency** (vertical) → `P` (percussive estimate — transients).
4. Soft masks: `maskH = H / (H + P)`, `maskP = P / (H + P)`; multiply onto complex `S`.
5. ISTFT each masked spectrogram.

> ⚠ **Scope limit — read before building `Source/Drums/`.** HPSS splits *harmonic vs
> percussive*. It does **not** distinguish kick from snare from hi-hat — those are all
> percussive. Splitting the drum group into 6 named sub-stems (kick/snare/hihat/clap/toms/
> cymbals) needs more than HPSS: either a trained drum-separation model (e.g. a Demucs
> drum fine-tune / LarsNet) or HPSS **plus** per-instrument frequency-band masking +
> transient classification. The original PLAN's "DSP-only, no ML" claim for 6-way drum
> splitting is flagged as insufficient in `docs/PLAN.md`. Confirm the approach first.

## Frequency-band masking (coarse drum cues)
Rough energy bands help *after* isolating drums, but overlap heavily — not a clean split:
- Kick: ~40–120 Hz (+ a click transient ~2–5 kHz)
- Snare: ~150–250 Hz body + ~2–8 kHz noise
- Hi-hat / cymbals: mostly > 6 kHz, broadband, long decay (cymbals) vs short (closed hat)
- Toms: ~80–350 Hz, pitched, with decay
Use as features feeding a classifier, not as the separator itself.

## Transient / onset detection
Find drum hits via **spectral flux**: per frame, sum the positive change in magnitude
across bins; peaks above an adaptive threshold = onsets. Useful for slicing kick/snare hits
and as a feature for classification. Watch for double-triggers (use a minimum
inter-onset interval / peak-picking with a refractory window).

## Numerical hygiene
- Add `1e-8` epsilon before any division (masks, normalization) to avoid NaN/Inf.
- Keep everything `float`; only go `double` for accumulators if you see drift.
- Clamp final output to [-1, 1] before writing WAV to avoid clipping artifacts.

## Review checklist
- [ ] Overlap-add normalized by window-sum (no periodic pulsing).
- [ ] Window satisfies COLA for the chosen overlap, or explicit normalization is applied.
- [ ] STFT and ISTFT share window + hop.
- [ ] Any division guarded with epsilon.
- [ ] Resampling to/from model sample rate is explicit and documented.
- [ ] No claim that HPSS alone yields named drum sub-stems.
