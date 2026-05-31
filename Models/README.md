# Models — export & regeneration

Model **weights are never committed** (`.gitignore` blocks `*.onnx`/`*.pt`/`*.ckpt`/
`*.safetensors`). This directory tracks the *scripts* and *how to regenerate*, not the
binaries. See `docs/PLAN.md` R8 (licensing) and R1 (export feasibility).

## Version pins (keep these in lockstep)

| Thing | Pinned value | Where |
|---|---|---|
| ONNX **opset** | **17** | export scripts (`--opset 17`) |
| ONNX Runtime (C++) | **1.20.1** | `cmake/onnxruntime.cmake` (`ORT_VERSION`) |
| Exporter | **TorchScript** (`dynamo=False`) | R1 spike — dynamo emits a broken iSTFT |

When you bump ORT, bump the opset compatibility note here too. Opset 17 + ORT 1.20.1 is the
validated pair.

## The IO contract (what the C++ engine expects)

`Source/Separation/ONNXInferenceEngine.cpp` is model-agnostic but assumes one of:

- **Packed output (our default):** input `mix` `[batch, channels, samples]` →
  output `stems` `[batch, num_stems, channels, samples]`. One rank-4 tensor; the engine
  slices the stems dimension.
- **Per-stem outputs:** N rank-3 tensors, each `[batch, channels, samples]`.

The sample axis is dynamic (`dynamic_axes`) **for BS-RoFormer**, so variable-length chunks
work. **HTDemucs is the exception:** it is trained on a fixed 7.8 s segment, so its export
has a **FIXED sample axis = 343980** (7.8 s @ 44.1 kHz) and only the batch axis is dynamic.
The C++ chunker must feed exactly 343980-sample chunks for HTDemucs (zero-pad the tail).
Channel order is **stereo, fixed** — pin it at export (see `StereoWrapper`/`ExportWrapper`
in the spike scripts) to avoid the #1 integration bug (tensor-layout mismatch, PLAN R3).

> Stem **names**: a packed single-tensor export carries no per-stem names, so the engine
> labels them `stem_0..N` rather than guessing an order. Map those to real names
> (vocals/drums/…) per the specific model when you wire up the UI (Phase 2).

## HTDemucs (default, MIT) — ✅ exported & validated

The shippable default. Weights auto-download (MIT, ~80 MB `.th`) via `demucs.pretrained`, so
acquisition is fully scripted — no manual checkpoint hunt. One command acquires + exports:

```bash
cd Models/spike
source .venv/bin/activate                      # py3.14 venv: torch 2.12, onnx, onnxruntime
# one-time deps (kept off torch's resolver so 2.12 isn't downgraded):
pip install --no-deps demucs openunmix tqdm "torchaudio==2.11.0"
pip install einops julius dora-search omegaconf pyyaml

python 04_export_htdemucs.py --out ../weights/htdemucs.onnx   # acquire + export + self-check
python 05_validate_htdemucs.py --onnx ../weights/htdemucs.onnx # per-stem SDR parity vs native
```

Key facts (see `04_export_htdemucs.py` header for the full rationale):
- **Sources (packed order):** `drums, bass, other, vocals` → output `[batch, 4, 2, 343980]`.
- **Fixed 343980-sample chunk** (7.8 s) — *not* dynamic (see IO contract caveat above).
- **Conv-STFT substitution (R1a):** HTDemucs' spectral branch uses `torch.stft`, which does
  not export. The script monkeypatches `_spec/_magnitude/_mask/_ispec` to carry a real-stacked
  `[B,C,2,Fr,T]` layout via a `Conv1d`/`ConvTranspose1d` STFT/ISTFT — no complex ops in the
  graph (verified: emitted ops contain no `STFT`/`DFT`). `cac=True` makes this clean.
- **MHA fast path disabled** at export (`torch.backends.mha.set_fastpath_enabled(False)`) —
  the cross-transformer's `MultiheadAttention(need_weights=False)` otherwise dispatches to the
  fused `aten::_native_multi_head_attention`, which the TorchScript exporter can't lower.
- **Validation:** conv-substitution self-check **117 dB SNR** (native vs patched, in-process);
  end-to-end ONNX-vs-native parity **worst-stem 54.7 dB** ≫ the 40 dB gate / 0.1 dB audible bar.
- **Size:** ~289 MB fp32 — larger than the 42 M params alone because the DFT conv basis
  (`Wa`/`Ws`, `[4098,1,4096]`) ships as initializers. fp16 / computed-basis is a later optimization.

## BS-RoFormer (opt-in upgrade) — gated on a weights download (NOT automated)

- **BS-RoFormer-SW** — opt-in quality upgrade. Weights are **699 MB, License: unknown** —
  fine to use locally for testing, **do not redistribute or bundle**. Load checkpoints with
  `weights_only=True` and convert to safetensors before trusting (pickle = arbitrary code).

Export + parity-check workflow (from the R1 spike — needs a checkpoint + YAML config you
supply):

```bash
cd Models/spike
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt bs-roformer pyyaml soundfile

# 1) Export (substitutes conv-STFT for torch.stft before export — see the script).
python 02_export_bs_roformer.py --config <cfg.yaml> --checkpoint <ckpt> \
       --exporter torchscript --opset 17 --out ../weights/<model>.onnx

# 2) Confirm PyTorch↔ONNX parity (acceptance: worst-stem SDR delta ≫ the 0.1 dB audible bar)
python 03_validate_parity.py --onnx ../weights/<model>.onnx \
       --config <cfg.yaml> --checkpoint <ckpt> --audio <clip.wav>
```

Then run the C++ harness end-to-end:

```bash
cmake -B build-onnx -G Ninja -DSTEMFORGE_WITH_ONNXRUNTIME=ON && cmake --build build-onnx
./build-onnx/StemForgeSeparate_artefacts/Debug/StemForgeSeparate \
       Models/weights/<model>.onnx <song.wav> <out-dir>
```

## Dummy model (for testing the C++ harness without real weights)

`Tests/make_dummy_model.py` emits `Tests/fixtures/dummy_half_split.onnx` — a tiny model with
the exact IO contract above that splits the mix into two stems, each `0.5 * mix`. It lets
`StemForgeSeparate` be exercised end-to-end (chunk → infer → overlap-add → write) and proves
correctness: `stem_0 + stem_1 == mix` (verified to ~1e-8). It is **not** a separator.

```bash
python Tests/make_dummy_model.py        # needs torch (use the spike venv)
```
