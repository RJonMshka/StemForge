# Learning 02 — The Data Structures Audio Flows Through

A song enters StemForge as a file and leaves as several stem files. In between it changes
shape several times. This doc walks each shape, *why* it exists, and the cost of each.

```
file.wav ─▶ AudioBuffer<float> ─▶ Segments (chunks) ─▶ ONNX Tensor ─▶
            ─▶ per-stem Tensors ─▶ overlap-add ─▶ AudioBuffer<float> ─▶ stem.wav
```

---

## 1. PCM audio = a big array of floats

All digital audio is just numbers sampled over time. One number ("sample") per channel,
44,100 times per second (the **sample rate**). Each sample is a `float` in `[-1.0, 1.0]`
(amplitude). That's it. A 3-minute stereo song ≈ `3*60 * 44100 * 2 ≈ 16 million floats`.

**Interleaved vs planar** — the same data, two layouts. This trips everyone up:

```
Interleaved (how WAV files store it):   L R L R L R L R ...
Planar / non-interleaved (what we use): L L L L ... | R R R R ...
```

JUCE and ML models want **planar**: each channel is its own contiguous array. WAV on disk
is **interleaved**. JUCE's reader de-interleaves for you, but when you hand data to ONNX
you must know which layout the model expects (BS-RoFormer = planar `[channel][sample]`).

---

## 2. `juce::AudioBuffer<float>` — the in-memory song

JUCE's core audio container. Think: **an array of channels, each channel an array of
samples.** Conceptually `float[numChannels][numSamples]`, planar.

```cpp
juce::AudioBuffer<float> buffer(2, 441000);   // 2 channels, 10s @ 44.1k
buffer.getNumChannels();                       // 2
buffer.getNumSamples();                         // 441000
const float* left = buffer.getReadPointer(0);   // raw pointer to channel 0's samples
float*       leftW = buffer.getWritePointer(0); // writable version
```

`getReadPointer` / `getWritePointer` hand you the raw `float*` so you can loop fast or
copy into a tensor. The buffer **owns** that memory and frees it (RAII). Don't free it
yourself.

> Cost: allocating a buffer is a heap allocation. That's why we never create one inside
> `processBlock` — we allocate once up front and reuse. (See `01` §RAII, CLAUDE.md rule 1.)

---

## 3. `Segment` — one overlapping chunk

The model can't eat a whole song (the tensors would be gigabytes). We slice the buffer
into fixed-length **segments** (e.g. 10s each) that *overlap* their neighbours by ~25%.

```cpp
struct Segment {
    int   startSample;      // where in the original it begins
    int   length;           // how many samples (usually segmentSamples)
    // the actual float data is copied into a tensor at inference time
};
```

Why overlap? The model is slightly worse at the very edges of a chunk. Overlapping +
crossfading neighbours hides those edge artifacts. This is the same trick as STFT
windowing, one level up — see `audio-dsp` skill.

**Data-structure shape:** a `std::vector<Segment>` — a growable, contiguous array. The
number of segments is `ceil(totalSamples / hopSize)` where `hopSize = segmentSamples *
(1 - overlap)`.

---

## 4. The ONNX tensor — a flat array + a shape

An ML tensor is **one flat `std::vector<float>` plus a list of dimensions** telling you
how to interpret it. Exactly a NumPy array or a typed `Float32Array` with `.shape`.

```cpp
// stereo, 10s @ 44.1k  →  shape [batch, channels, samples]
std::vector<int64_t> shape = { 1, 2, 441000 };
std::vector<float>   data(1 * 2 * 441000);   // 882,000 floats, flat

// index of (channel c, sample s) in the flat array:
//   data[c * 441000 + s]      (row-major / C order)
```

The model reads this flat block; the **shape** is the only thing that says "treat the
first 441000 as the left channel." Get the shape order wrong and you feed garbage — this
is the #1 ONNX integration bug. Always confirm the model's expected `[dims]` and channel
layout.

`Ort::Value` wraps that data+shape into something ONNX Runtime accepts. It's a *view* — it
does not copy your `data` vector, so the vector must outlive the tensor (lifetime!).

---

## 5. Output: one tensor per stem, same shape as input

The model returns N tensors (one per stem), each the same `[1, 2, samples]` shape as the
input. You pull them out by index:

```
outputs[0] → vocals     outputs[1] → drums     outputs[2] → bass    ...
```

So for each segment you now hold ~6 same-sized float blocks. Multiply by ~25 segments for
a full song and you see why this is memory-heavy and lives on a background thread, never
the audio thread.

---

## 6. Overlap-add accumulator — stitching chunks back

To turn per-segment stem outputs back into a full-length stem, you **overlap-add**: write
each segment's output into a full-length destination buffer at its `startSample`, applying
a fade so overlapping regions crossfade instead of clicking.

```cpp
juce::AudioBuffer<float> stemOut(2, totalSamples);   // destination, zero-filled
// for each segment, for each sample:
//   stemOut[ch][startSample + i] += window[i] * segmentOut[ch][i];
```

The `+=` with a windowing function is the "add" in overlap-add. A matching **window-sum
normalization** divides out the total window weight so the crossfade is unity-gain. Get
the window math right and the seams are inaudible; get it wrong and you hear a pulse every
hop.

Data structure: a destination `AudioBuffer<float>` per stem + a `std::vector<float>`
window of length `segmentSamples`.

---

## 7. Out to disk — `juce::WavAudioFormat`

Final stem buffers are written as **32-bit float WAV** (lossless, no quantization noise)
via `juce::WavAudioFormat` → an `AudioFormatWriter`. The writer re-interleaves planar back
to interleaved L R L R for the file. Roundtrip (read → write, no processing) must be
bit-exact — that's a Phase-0 test.

---

## Memory budget intuition

| Shape | Size (3-min stereo song) |
|---|---|
| Full mix `AudioBuffer` | ~63 MB (16M floats) |
| One 10s segment tensor | ~3.5 MB |
| 6 stem outputs, one segment | ~21 MB |
| 6 full-length stem buffers | ~380 MB |

That ~380 MB of simultaneous stem buffers is the reason this is a worker-thread,
allocate-carefully job — and why streaming stems to disk as they finish (instead of
holding all six) is worth considering. (Open question logged in `docs/PLAN.md`.)

Next: `03-how-its-built.md` — how these pieces assemble and the threads they run on.
