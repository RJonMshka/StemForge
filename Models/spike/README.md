# R1 Spike — PyTorch → ONNX export feasibility (BS-RoFormer)

**Risk under test (PLAN R1):** BS-RoFormer's STFT/iSTFT + complex ops may not export to
ONNX cleanly, which would sink the whole "train in PyTorch, run in C++" architecture.

**Status: R1a RESOLVED (the STFT primitive). R1b PENDING (full model, needs weights).**

---

## How to reproduce

```bash
cd Models/spike
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, onnx, onnxruntime, numpy (+onnxscript)
pip install onnxscript                    # required by the dynamo exporter

python 01_stft_onnx_probe.py             # tests native stft export 3 ways (records failures)
python 01b_conv_stft_probe.py            # proves the portable conv-STFT path  ← the answer
```

Environment used: macOS arm64 (Apple Silicon, 8-core), CPython **3.14**,
torch **2.12.0**, onnx **1.21.0**, onnxruntime **1.26.0**, onnxscript 0.7.0.

> Note: PyTorch 2.12 **does** ship a cp314 wheel, so no separate 3.12 interpreter was
> needed. If a future torch/python combo lacks a wheel, `brew install python@3.12` and make
> the venv with `python3.12`.

---

## Findings

### The native `torch.stft` / `torch.istft` export path is unreliable (here)

`01_stft_onnx_probe.py` tried three strategies on a minimal stft→touch→istft round-trip:

| Strategy | Result | Why |
|---|---|---|
| A — native stft + **TorchScript** exporter (opset 17) | ❌ export fails | `SymbolicValueError: STFT does not currently support complex types` |
| B — native stft + **Dynamo** exporter (opset 18) | ❌ runtime fails | Exports real ONNX `STFT`/`DFT` ops, but ORT rejects it: first an `int32` `ScatterND` (castable to int64), then a genuine `Mul` **broadcast bug** (`1025 by 2048`) in the istft decomposition. |
| C — manual stft via `tensor.unfold` + TorchScript | ❌ export fails | `Unsupported: ONNX export of operator Unfold` |

**Takeaway:** do **not** rely on the framework exporting `torch.stft`/`torch.istft`. In this
toolchain the dynamo exporter even emits a *silently wrong* istft. This matches what real
BS-RoFormer ONNX exports (UVR / python-audio-separator) do — they replace the STFT.

### The conv-based STFT/ISTFT path works — exactly and portably ✅

`01b_conv_stft_probe.py` implements STFT as a `Conv1d` against a fixed DFT+Hann basis and
ISTFT as a `ConvTranspose1d` overlap-add with window-sum normalization:

| Check | Result |
|---|---|
| Implementation round-trip (eager vs input) | **123.1 dB SNR** (effectively lossless) |
| ONNX ops emitted | only basic ops: `Conv`, `ConvTranspose`, `Pad`, `Div`, `Mul`, `Concat`, `Slice`… — **no** `STFT`/`DFT` op |
| ONNX Runtime vs PyTorch eager | **max abs diff 4.8e-6 / 122.8 dB SNR — PASS** |
| Dynamic chunk length (22 050 samples) | max abs diff 6.0e-6 — **PASS** |

**Conclusion:** the scary part of R1 — the STFT/iSTFT — has a proven, exact, ONNX-portable
implementation that runs in stock onnxruntime with no custom ops and supports variable-length
chunks. R1a is closed.

---

## Decision (feeds back into `docs/PLAN.md`)

1. **Stage-1 export will substitute the model's `torch.stft`/`torch.istft` with the
   conv-STFT/ISTFT** from `01b` before `torch.onnx.export`. This is the single contained
   code change the whole export hinges on.
2. **Use the TorchScript exporter** (opset 17) for the conv path — it produces clean basic
   ops and avoids the dynamo istft bug. (Re-evaluate dynamo on future torch versions.)
3. R1's severity drops from **High → Medium** (remaining unknown is only the
   attention/band-split/RoPE blocks, which are standard matmul/softmax ops that export
   routinely — far lower risk than the STFT was).

## What's left — R1b (full model)

> **Model decision update (R8):** HTDemucs (MIT) is now the **shippable default**;
> BS-RoFormer-SW is an **opt-in** upgrade (License: unknown, 699 MB — never bundled). The
> conv-STFT finding above applies to **both** (HTDemucs' hybrid branch also uses STFT).
> Validate R1b against **HTDemucs first** (clean license, easier export); the BS-RoFormer
> scripts below stay for the opt-in path and for local dev testing.

`02_export_bs_roformer.py` and `03_validate_parity.py` are ready but need a real
**checkpoint + YAML config** (large, **not in git**, see `.gitignore`). For BS-RoFormer-SW,
note its 699 MB pickle is **License: unknown** — fine to use locally for testing, do not
redistribute. Once acquired:

```bash
pip install bs-roformer pyyaml soundfile
python 02_export_bs_roformer.py --config <cfg.yaml> --checkpoint <ckpt> --exporter torchscript \
       --out ../weights/bs_roformer_sw.onnx
python 03_validate_parity.py --onnx ../weights/bs_roformer_sw.onnx \
       --config <cfg.yaml> --checkpoint <ckpt> --audio <clip.wav>
```
Acceptance: `03` reports worst-stem SDR delta > 40 dB (≪ the 0.1 dB audible bar).
Before that run, wire the conv-STFT substitution into `02` (marked `# STFT substitution note`).

## Files
- `01_stft_onnx_probe.py` — native-export attempts (records the failures above).
- `01b_conv_stft_probe.py` — **the portable conv-STFT proof** (the actionable result).
- `02_export_bs_roformer.py` — full-model export (gated on checkpoint).
- `03_validate_parity.py` — PyTorch-vs-ONNX SDR parity check (gated on checkpoint).
- `results_stft_probe.json` — machine-readable results from `01`.
- `requirements.txt` — spike deps. `.venv/`, `*.onnx` are gitignored.
