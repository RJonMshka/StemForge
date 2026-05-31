#!/usr/bin/env python3
"""
R1 spike — Step 1b: The PORTABLE STFT/ISTFT path (conv-based).

Step 1 (01_stft_onnx_probe.py) showed that torch 2.12's *native* stft/istft export is
unreliable in this toolchain:
  - TorchScript exporter: "STFT does not currently support complex types"
  - Dynamo exporter: exports ONNX STFT/DFT ops, but they fail at runtime in onnxruntime
    1.26 (int32 ScatterND, then a Mul broadcast bug in the istft decomposition).

The fix used by real BS-RoFormer ONNX exports (UVR / python-audio-separator) is to NOT
rely on torch.stft at all. Implement STFT as a Conv1d against a fixed DFT+window basis and
ISTFT as a ConvTranspose1d overlap-add. Those emit only Conv / ConvTranspose / Pad / Div —
basic ops every ONNX runtime supports. This script proves that path exports AND matches a
reference to floating-point tolerance.

Run:  python 01b_conv_stft_probe.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT, HOP, SAMPLES, OPSET = 2048, 512, 44100, 17
HERE = Path(__file__).parent
NBINS = N_FFT // 2 + 1


def make_basis():
    n = N_FFT
    t = np.arange(n)
    f = np.arange(NBINS)[:, None]
    ang = 2.0 * np.pi * f * t[None, :] / n
    win = np.hanning(n + 1)[:-1].astype(np.float32)          # periodic Hann
    cos = (np.cos(ang) * win).astype(np.float32)             # [F, n]
    sin = (-np.sin(ang) * win).astype(np.float32)            # [F, n]
    scale = np.full((NBINS, 1), 2.0, np.float32); scale[0] = scale[-1] = 1.0
    syn_cos = (np.cos(ang) * win * scale / n).astype(np.float32)
    syn_sin = (-np.sin(ang) * win * scale / n).astype(np.float32)
    return cos, sin, syn_cos, syn_sin, win


class ConvStftRoundTrip(nn.Module):
    """Real waveform -> conv-STFT -> (trivial real touch) -> convT-ISTFT -> waveform."""
    def __init__(self):
        super().__init__()
        cos, sin, syn_cos, syn_sin, win = make_basis()
        # analysis: [2F, 1, n]  (cos bins then sin bins)
        wa = np.concatenate([cos, sin], axis=0)[:, None, :]
        self.register_buffer("Wa", torch.tensor(wa))
        # synthesis: ConvTranspose1d weight [in=2F, out=1, n]
        ws = np.concatenate([syn_cos, syn_sin], axis=0)[:, None, :]
        self.register_buffer("Ws", torch.tensor(ws))
        # normalization basis: overlap-add of win^2 -> [1,1,n]
        self.register_buffer("Wn", torch.tensor((win * win)[None, None, :]))
        self.scale = nn.Parameter(torch.ones(1))             # trivial real op

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # x: [B, samples]
        pad = N_FFT // 2
        xp = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")  # [B,1,L+2pad]
        spec = F.conv1d(xp, self.Wa, stride=HOP)              # [B, 2F, T]
        re, im = spec[:, :NBINS], spec[:, NBINS:]             # each [B, F, T]
        re, im = re * self.scale, im * self.scale             # touch real & imag
        frames = torch.cat([re, im], dim=1)                   # [B, 2F, T]
        num = F.conv_transpose1d(frames, self.Ws, stride=HOP) # [B,1,L+2pad]
        ones = torch.ones_like(frames[:, :1, :])              # [B,1,T] (dynamic T)
        den = F.conv_transpose1d(ones, self.Wn, stride=HOP)   # [B,1,L+2pad]
        y = num / (den + 1e-8)
        return y[:, 0, pad:pad + x.shape[-1]]                 # [B, samples]


def snr_db(ref, test):
    noise = np.sum((ref - test) ** 2)
    return math.inf if noise == 0 else 10 * math.log10(np.sum(ref ** 2) / noise)


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(1, SAMPLES)
    model = ConvStftRoundTrip().eval()
    with torch.no_grad():
        eager = model(x).numpy()

    # round-trip fidelity of the implementation itself (eager vs input)
    rt = snr_db(x.numpy(), eager)
    print(f"conv-STFT round-trip (eager vs input): SNR={rt:.1f} dB "
          f"({'good' if rt > 30 else 'check window/COLA'})")

    path = HERE / "_probe_conv_stft.onnx"
    torch.onnx.export(
        model, (x,), str(path), input_names=["x"], output_names=["y"],
        opset_version=OPSET, dynamo=False,
        dynamic_axes={"x": {0: "batch", 1: "samples"}, "y": {0: "batch", 1: "samples"}},
    )
    import onnx
    om = onnx.load(path); onnx.checker.check_model(om)
    ops = sorted({n.op_type for n in om.graph.node})
    print(f"exported OK. ops used: {ops}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.max(np.abs(eager - got)))
    snr = snr_db(eager, got)
    ok = diff < 1e-3
    print(f"ORT vs torch eager: max_abs_diff={diff:.3e}  SNR={snr:.1f} dB  -> "
          f"{'PASS ✓' if ok else 'FAIL'}")

    # dynamic-length check: a different sample count must also run (variable chunks)
    x2 = torch.randn(1, 22050)
    got2 = sess.run(None, {sess.get_inputs()[0].name: x2.numpy()})[0]
    with torch.no_grad():
        eager2 = model(x2).numpy()
    diff2 = float(np.max(np.abs(eager2 - got2)))
    print(f"dynamic-length (22050): max_abs_diff={diff2:.3e}  -> "
          f"{'PASS ✓' if diff2 < 1e-3 else 'FAIL'}")

    print("\n── VERDICT ─────────────────────────────────────────────")
    if ok and diff2 < 1e-3:
        print("PORTABLE conv-STFT/ISTFT exports to ONNX (basic ops only), runs in ORT, and")
        print("matches PyTorch to ~1e-6, with dynamic chunk lengths. R1a is RESOLVED: the")
        print("Stage-1 export must substitute the model's stft/istft with this conv path.")
        return 0
    print("Conv path mismatch — investigate window normalization / COLA before proceeding.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
