---
name: onnx-runtime-cpp
description: Use when writing or reviewing ONNX Runtime C++ inference code in StemForge — Ort::Session/Env setup, building input tensors from audio, running the model, extracting per-stem output tensors, execution providers (CPU/CoreML/CUDA/DirectML), memory/lifetime of Ort::Value, or the PyTorch→ONNX export of BS-RoFormer. Trigger on Ort::, OrtValue, .onnx, session.Run, tensor shape, or model export questions.
---

# ONNX Runtime C++ Inference (StemForge)

How to run the separation model from C++ correctly. Pair with
`docs/learning/02-data-structures.md` (tensor = flat float array + shape).

## Mental model
An ONNX tensor is a flat `std::vector<float>` + a `shape` (`std::vector<int64_t>`). The
shape is the *only* thing that says how to interpret the flat block. Wrong shape order =
silent garbage output. This is the #1 integration bug — verify the model's expected dims
and channel layout first.

## One-time setup (per engine instance, NOT per inference)
```cpp
Ort::Env       m_env { ORT_LOGGING_LEVEL_WARNING, "StemForge" };
Ort::SessionOptions m_opts;
m_opts.SetIntraOpNumThreads(0);                 // 0 = let ORT pick
// Optional GPU/accelerator (try/catch — fall back to CPU if append fails):
// CoreML (macOS), CUDA (NVIDIA), DirectML (Windows). Always keep CPU fallback.
std::unique_ptr<Ort::Session> m_session;        // built in loadModel(), lives in the engine
Ort::MemoryInfo m_memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
```
`Ort::Env` and `Ort::Session` are expensive — create once, reuse for every chunk. Never
construct a Session inside the per-segment loop.

## Build an input tensor from audio
```cpp
// BS-RoFormer expects planar [batch=1, channels=2, samples]
std::vector<int64_t> shape { 1, channels, samples };
std::vector<float>   data(1 * channels * samples);
for (int c = 0; c < channels; ++c)
    std::copy_n(buffer.getReadPointer(c), samples, data.data() + c * samples);

Ort::Value input = Ort::Value::CreateTensor<float>(
    m_memInfo, data.data(), data.size(), shape.data(), shape.size());
// ⚠ `input` is a VIEW over `data` — `data` must outlive `input`. Don't let it go out of scope.
```

## Run
```cpp
const char* inputNames[]  = { "mix" };           // must match the exported model's IO names
const char* outputNames[] = { "vocals","drums","bass","guitar","piano","other" };
auto outputs = m_session->Run(
    Ort::RunOptions{nullptr},
    inputNames,  &input, 1,
    outputNames, std::size(outputNames));
// outputs[i] is an Ort::Value tensor, same shape as input, one per stem.
```
Get IO names from the exported model (Netron, or `session.GetInputNameAllocated`). Don't
hard-code wrong names.

## Extract output
```cpp
float* out = outputs[0].GetTensorMutableData<float>();
auto info  = outputs[0].GetTensorTypeAndShapeInfo();
auto outShape = info.GetShape();                  // confirm [1, channels, samples]
// copy planar `out` back into a juce::AudioBuffer<float> per channel
```

## Lifetime & RAII rules
- Use the C++ wrapper `Ort::Value`, never raw `OrtValue*`. It frees on scope exit.
- The input `data` vector MUST outlive the input `Ort::Value` (tensor is a non-owning view).
- Session/Env are members owned by `unique_ptr` / by value; freed in the engine destructor.
- Wrap fallible setup (`loadModel`) in `juce::Result`; catch ORT's `Ort::Exception` at the
  boundary and convert to a `Result::fail` — never let it propagate to the host.

## Chunked inference loop shape
```
for each Segment:
    fill data[] from the segment's samples   (resample to model SR if needed)
    Run() → 6 output tensors
    overlap-add each stem output into its full-length destination buffer
publish progress (atomic) after each segment; check threadShouldExit()
```
Reuse the `data` vector and shape across segments (allocate once) when lengths are equal.

## PyTorch → ONNX export (BS-RoFormer) — the hard part
- RoFormer uses STFT/iSTFT + complex ops. These do **not** always export cleanly:
  `torch.onnx.export` needs a recent opset (≥17 for `STFT`), and complex tensors may need
  manual real/imag splitting. **Treat export as a research spike, not a checkbox.**
- Reference: Mixxx GSoC 2025 validated an HT-Demucs ONNX export at <0.1 dB difference —
  but BS-RoFormer is harder than HT-Demucs. Budget time for op-support surprises.
- Validate: run the same audio through PyTorch and ONNX, assert per-stem SDR delta is
  negligible (<0.1 dB) before trusting the C++ path.
- Pin opset + ORT version together; record both in `Models/README.md`. Keep `dynamic_axes`
  for the sample dimension so variable-length chunks work.

### ✅ Spike finding (R1, `Models/spike/`) — DO NOT export native `torch.stft`/`istft`
Validated on torch 2.12 / onnxruntime 1.26 / opset 17–18:
- TorchScript exporter → `STFT does not currently support complex types`.
- Dynamo exporter → exports ONNX `STFT`/`DFT` ops but they're **broken at runtime**
  (int32 `ScatterND`, then a `Mul` broadcast bug in the istft path). Silently wrong.
- **Fix that works:** replace the model's STFT front/back-end with a **conv-based
  STFT/ISTFT** — `Conv1d` against a fixed DFT+Hann basis for analysis,
  `ConvTranspose1d` overlap-add (window-sum normalized) for synthesis. Emits only basic
  ops (`Conv`/`ConvTranspose`/`Pad`/`Div`), matches PyTorch to **~5e-6**, supports dynamic
  chunk lengths. Reference impl: `Models/spike/01b_conv_stft_probe.py`. Use the TorchScript
  exporter for the conv path. Wire this substitution into the model **before**
  `torch.onnx.export` (see `Models/spike/02_export_bs_roformer.py`).

## Review checklist
- [ ] Session/Env created once, reused across all segments.
- [ ] Input `data` vector outlives its `Ort::Value`.
- [ ] Tensor shape + channel layout verified against the actual model (Netron).
- [ ] IO names read from the model, not guessed.
- [ ] `Ort::Exception` caught at the boundary → `juce::Result`, never thrown to host.
- [ ] CPU execution-provider fallback if GPU provider append fails.
