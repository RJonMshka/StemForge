#!/usr/bin/env python3
"""
R1 spike — Step 1: Can STFT / ISTFT + complex ops survive a PyTorch -> ONNX export?

This is the CHEAP, HIGH-INFORMATION test. BS-RoFormer's hard-to-export parts are its
front-end `torch.stft` (produces a complex spectrogram) and back-end `torch.istft`. We
test that primitive in isolation — no model weights, no multi-GB download — before
committing to the full export.

We try three strategies and print a result matrix:

  A) native torch.stft/istft  + TorchScript ONNX exporter (opset 17)
  B) native torch.stft/istft  + Dynamo ONNX exporter (decomposes to primitive ops)
  C) manual DFT-as-matmul STFT/ISTFT + TorchScript exporter   <- guaranteed-portable fallback

Each strategy: export -> run in onnxruntime -> compare against PyTorch eager
(max abs diff + SNR in dB). A strategy "passes" if it exports, runs, and matches eager
to < 1e-3 max abs diff (round-trip identity should be near-exact).

Run:
    python 01_stft_onnx_probe.py
Writes: results_stft_probe.json  (machine-readable findings for the README table)
"""
from __future__ import annotations
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Config: typical BS-RoFormer-ish STFT params ───────────────────────────────
N_FFT = 2048
HOP = 512
SAMPLES = 44100          # 1 second mono; enough to exercise many frames
OPSET = 17               # ONNX 'STFT' op exists at >=17; ISTFT does NOT exist at all
HERE = Path(__file__).parent
TOL = 1e-3               # round-trip identity should be ~exact


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-noise of `test` vs `ref`, in dB. Higher = closer. inf = identical."""
    noise = np.sum((ref - test) ** 2)
    if noise == 0:
        return math.inf
    return 10.0 * math.log10(np.sum(ref ** 2) / noise)


# ── Strategy A/B model: native torch.stft / torch.istft ───────────────────────
class NativeStftRoundTrip(nn.Module):
    """
    Mirrors the shape of what BS-RoFormer does at its edges: real waveform -> complex
    spectrogram -> touch real & imaginary parts with a (trivial) real-valued op ->
    inverse back to waveform. If THIS exports, the model's STFT plumbing can too.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT))
        # A trivial real op over the real/imag channels, to force complex<->real handling.
        self.scale = nn.Parameter(torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: [B, samples]
        spec = torch.stft(x, N_FFT, HOP, window=self.window,
                          center=True, return_complex=True)    # [B, F, T] complex
        ri = torch.view_as_real(spec)                          # [B, F, T, 2] real
        ri = ri * self.scale                                   # touch real & imag
        spec2 = torch.view_as_complex(ri.contiguous())         # back to complex
        y = torch.istft(spec2, N_FFT, HOP, window=self.window,
                        center=True, length=x.shape[-1])       # [B, samples]
        return y


# ── Strategy C model: manual STFT/ISTFT as real matmuls (no complex dtype) ────
class ManualStftRoundTrip(nn.Module):
    """
    STFT implemented as conv/matmul with precomputed DFT basis, kept entirely in REAL
    arithmetic. No torch.stft, no complex tensors -> exports to ONNX everywhere. This is
    the fallback we fall back to if A and B fail. Uses framing + real/imag DFT matrices
    and an overlap-add inverse with window normalization.
    """
    def __init__(self):
        super().__init__()
        n = N_FFT
        k = np.arange(n)
        # DFT basis for the n_fft//2+1 non-redundant bins.
        freqs = np.arange(n // 2 + 1)[:, None]
        ang = 2.0 * np.pi * freqs * k[None, :] / n             # [F, n]
        self.register_buffer("dft_cos", torch.tensor(np.cos(ang), dtype=torch.float32))
        self.register_buffer("dft_sin", torch.tensor(-np.sin(ang), dtype=torch.float32))
        win = np.hanning(n + 1)[:-1].astype(np.float32)        # periodic Hann
        self.register_buffer("window", torch.tensor(win))

    def _frame(self, x: torch.Tensor) -> torch.Tensor:
        # center pad like torch.stft(center=True): reflect pad n_fft//2 each side
        pad = N_FFT // 2
        xp = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
        return xp.unfold(-1, N_FFT, HOP)                       # [B, T, n_fft]

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: [B, samples]
        frames = self._frame(x) * self.window                 # [B, T, n]
        re = torch.matmul(frames, self.dft_cos.t())           # [B, T, F]
        im = torch.matmul(frames, self.dft_sin.t())           # [B, T, F]
        # (identity touch, mirrors strategy A's trivial op)
        # inverse: real iDFT of the (re, im) bins, overlap-add with window^2 norm
        cos_t = self.dft_cos                                   # [F, n]
        sin_t = self.dft_sin
        # Reconstruct frames: sum over bins. Account for one-sided spectrum (x2 for 1..F-2).
        scale = torch.ones(cos_t.shape[0], 1)
        scale[1:-1] = 2.0
        rec = (torch.matmul(re, cos_t * scale) - torch.matmul(im, sin_t * scale)) / N_FFT
        rec = rec * self.window                               # synthesis window
        # overlap-add
        B, T, n = rec.shape
        out_len = (T - 1) * HOP + n
        y = torch.zeros(B, out_len)
        wsum = torch.zeros(out_len)
        idx = torch.arange(n)
        for t in range(T):
            s = t * HOP
            y[:, s:s + n] += rec[:, t, :]
            wsum[s:s + n] += self.window ** 2
        wsum = torch.clamp(wsum, min=1e-8)
        y = y / wsum
        pad = N_FFT // 2
        return y[:, pad:pad + x.shape[-1]]


# ── Export helpers ────────────────────────────────────────────────────────────
def export_torchscript(model, example, path: Path) -> None:
    torch.onnx.export(
        model, (example,), str(path),
        input_names=["x"], output_names=["y"],
        opset_version=OPSET,
        dynamic_axes={"x": {0: "batch", 1: "samples"}, "y": {0: "batch", 1: "samples"}},
        dynamo=False,
    )


def export_dynamo(model, example, path: Path) -> None:
    # torch>=2.5 unified API: dynamo=True. Falls back to dynamo_export on older torch.
    try:
        torch.onnx.export(model, (example,), str(path), opset_version=OPSET, dynamo=True)
    except TypeError:
        onnx_prog = torch.onnx.dynamo_export(model, example)   # legacy API
        onnx_prog.save(str(path))


def run_ort(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: x})[0]
    return np.asarray(out)


def try_strategy(name: str, model: nn.Module, exporter, x: torch.Tensor) -> dict:
    rec = {"strategy": name, "exported": False, "ran_in_ort": False,
           "max_abs_diff": None, "snr_db": None, "passed": False, "error": None}
    path = HERE / f"_probe_{name}.onnx"
    model.eval()
    with torch.no_grad():
        eager = model(x).cpu().numpy()
    try:
        exporter(model, x, path)
        rec["exported"] = True
    except Exception as e:                                      # noqa: BLE001
        rec["error"] = f"export: {type(e).__name__}: {e}"
        return rec
    try:
        got = run_ort(path, x.cpu().numpy())
        rec["ran_in_ort"] = True
        diff = float(np.max(np.abs(eager - got)))
        rec["max_abs_diff"] = diff
        rec["snr_db"] = snr_db(eager, got)
        rec["passed"] = diff < TOL
    except Exception as e:                                      # noqa: BLE001
        rec["error"] = f"ort: {type(e).__name__}: {e}"
    return rec


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(1, SAMPLES)
    print(f"torch {torch.__version__} | opset {OPSET} | n_fft {N_FFT} hop {HOP} "
          f"| input {tuple(x.shape)}\n")

    strategies = [
        ("A_native_torchscript", NativeStftRoundTrip(), export_torchscript),
        ("B_native_dynamo",      NativeStftRoundTrip(), export_dynamo),
        ("C_manual_torchscript", ManualStftRoundTrip(), export_torchscript),
    ]
    results = []
    for name, model, exporter in strategies:
        print(f"── {name} ".ljust(60, "─"))
        rec = try_strategy(name, model, exporter, x)
        results.append(rec)
        status = "PASS ✓" if rec["passed"] else ("ran, mismatch" if rec["ran_in_ort"]
                  else ("exported, ort failed" if rec["exported"] else "export FAILED"))
        print(f"   exported={rec['exported']}  ran={rec['ran_in_ort']}  "
              f"maxdiff={rec['max_abs_diff']}  snr={rec['snr_db']}  -> {status}")
        if rec["error"]:
            print(f"   error: {rec['error'].splitlines()[0]}")
        print()

    out = HERE / "results_stft_probe.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out.name}")

    any_native = any(r["passed"] for r in results if r["strategy"].startswith(("A_", "B_")))
    fallback_ok = next((r["passed"] for r in results if r["strategy"].startswith("C_")), False)
    print("\n── VERDICT ─────────────────────────────────────────────")
    if any_native:
        print("Native torch.stft/istft exports cleanly. R1a is LOW risk — proceed to full")
        print("model export (02) using the passing exporter above.")
    elif fallback_ok:
        print("Native STFT export FAILS, but the manual DFT-matmul fallback (C) WORKS.")
        print("R1a is MEDIUM risk: full export must swap the model's stft/istft for the")
        print("conv/matmul implementation (a known, contained code change). Proceed to 02")
        print("with that substitution.")
    else:
        print("All strategies failed. R1a is HIGH risk — escalate: reconsider HT-Demucs")
        print("(easier export) as the Stage-1 model, per PLAN risk R1 mitigation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
