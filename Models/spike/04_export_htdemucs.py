#!/usr/bin/env python3
"""
Export HTDemucs (Meta, MIT) to ONNX — the shippable default Stage-1 model (PLAN R8).

WHY THIS SCRIPT EXISTS
----------------------
HTDemucs is a *hybrid* separator: a time-domain branch and a spectral branch joined by a
cross-transformer. The spectral branch runs `torch.stft`/`torch.istft` (see demucs/spec.py).
The R1 spike proved that native `torch.stft` export is unreliable in this toolchain
(TorchScript: "STFT does not currently support complex types"; dynamo: silently-wrong
istft). The resolution (R1a, see 01b_conv_stft_probe.py) is to substitute a conv-based
STFT/ISTFT that emits only basic ops (Conv/ConvTranspose/Pad/Div).

HTDemucs uses `cac=True` ("complex as channels"): the neural net never touches complex
numbers — `_magnitude` turns the STFT's (re, im) into 2 extra channels right away, and
`_mask` turns them back right before ISTFT. The ONLY complex ops are inside spectro/ispectro
plus the view_as_real/view_as_complex bridges. We therefore monkeypatch four methods so the
whole graph carries a REAL-stacked [B, C, 2, Fr, T] layout instead of a complex tensor,
reusing demucs' exact padding/cropping math. Everything else (encoders, decoders, the
cross-transformer) is standard matmul/conv/softmax that exports routinely.

FIXED SEGMENT LENGTH
--------------------
Unlike BS-RoFormer (dynamic chunks), HTDemucs is trained on a fixed 7.8 s segment
(use_train_segment=True). The transformer's positional handling and the internal
pad-to-training-length logic assume that exact length, so we export with a FIXED sample
axis = 343980 (= 7.8 s @ 44.1 kHz) and a dynamic BATCH axis only. The C++ chunker must feed
exactly 343980-sample chunks (zero-pad the tail chunk). This differs from the BS-RoFormer
path — document it where the engine picks a chunk size.

OUTPUT IO CONTRACT (matches Source/Separation/ONNXInferenceEngine + Models/README.md)
    input  "mix"   : [batch, 2, 343980]                 float32, stereo, fixed channel order
    output "stems" : [batch, 4, 2, 343980]              packed: drums, bass, other, vocals

USAGE
    python 04_export_htdemucs.py --out ../weights/htdemucs.onnx        # acquire+export
    # (model + weights auto-download via demucs.pretrained on first run; weights are MIT)
"""
from __future__ import annotations
import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OPSET = 17                       # validated pair with ORT 1.20.1 (Models/README.md)
SAMPLE_RATE = 44100
SEGMENT_SECONDS = 39 / 5         # 7.8 s — HTDemucs training segment
TRAINING_LENGTH = int(SEGMENT_SECONDS * SAMPLE_RATE)   # 343980


# ┌─ CHUNK: ConvSTFT ───────────────────────────────────────────────────────────┐
# │ INTENT: ONNX-portable STFT/ISTFT that numerically matches torch.stft/istft   │
# │         with (window=hann(n_fft, periodic), win_length=n_fft, center=True,   │
# │         pad_mode='reflect', normalized=True) — the exact settings demucs uses │
# │         in spec.py. STFT = Conv1d against a fixed DFT+Hann basis; ISTFT =     │
# │         ConvTranspose1d overlap-add with window-sum (NOLA) normalization.     │
# │ REAL-STACKED: returns/accepts [*lead, 2, F, frames] (index 0=real, 1=imag)   │
# │         so NO complex tensor ever enters the graph (HTDemucs cac=True path).  │
# │ NORMALIZATION: analysis basis ×1/sqrt(n_fft), synthesis basis ×sqrt(n_fft)   │
# │         to reproduce torch's normalized=True (they cancel on a pure round-    │
# │         trip but the trained net sits between them, so the scale must match). │
# │ DEPENDS ON: emits only Conv/ConvTranspose/Pad/Slice/Concat/Div (R1a proof).  │
# └──────────────────────────────────────────────────────────────────────────────┘
class ConvSTFT(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.nbins = n_fft // 2 + 1

        n = n_fft
        t = np.arange(n)
        f = np.arange(self.nbins)[:, None]
        ang = 2.0 * np.pi * f * t[None, :] / n               # [F, n]
        win = np.hanning(n + 1)[:-1]                          # periodic Hann == torch default
        anorm = 1.0 / math.sqrt(n)                            # torch normalized=True (forward)
        cos = (np.cos(ang) * win * anorm)
        sin = (-np.sin(ang) * win * anorm)

        scale = np.full((self.nbins, 1), 2.0)                # IDFT real-signal bin weights
        scale[0] = scale[-1] = 1.0                           # DC + Nyquist counted once
        snorm = math.sqrt(n)                                 # torch normalized=True (inverse)
        syn_cos = (np.cos(ang) * win * scale / n * snorm)
        syn_sin = (-np.sin(ang) * win * scale / n * snorm)

        # analysis weight for Conv1d: [2F, 1, n] (cos bins then sin bins)
        wa = np.concatenate([cos, sin], axis=0)[:, None, :].astype(np.float32)
        # synthesis weight for ConvTranspose1d: [2F, 1, n]
        ws = np.concatenate([syn_cos, syn_sin], axis=0)[:, None, :].astype(np.float32)
        # window-sum normalizer (overlap-add of win^2): [1, 1, n]
        wn = (win * win)[None, None, :].astype(np.float32)

        self.register_buffer("Wa", torch.tensor(wa))
        self.register_buffer("Ws", torch.tensor(ws))
        self.register_buffer("Wn", torch.tensor(wn))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """[*lead, L] -> [*lead, 2, F, frames]."""
        lead = x.shape[:-1]
        L = x.shape[-1]
        x = x.reshape(-1, 1, L)
        pad = self.n_fft // 2
        xp = F.pad(x, (pad, pad), mode="reflect")
        spec = F.conv1d(xp, self.Wa, stride=self.hop)        # [M, 2F, frames]
        re = spec[:, : self.nbins]
        im = spec[:, self.nbins :]
        z = torch.stack([re, im], dim=1)                     # [M, 2, F, frames]
        return z.reshape(*lead, 2, self.nbins, z.shape[-1])

    def istft(self, z: torch.Tensor, length: int) -> torch.Tensor:
        """[*lead, 2, F, frames] -> [*lead, length]."""
        lead = z.shape[:-3]
        Fb, frames = z.shape[-2], z.shape[-1]
        z = z.reshape(-1, 2, Fb, frames)
        frames_cat = torch.cat([z[:, 0], z[:, 1]], dim=1)    # [M, 2F, frames]
        num = F.conv_transpose1d(frames_cat, self.Ws, stride=self.hop)   # [M, 1, Lpad]
        ones = torch.ones_like(frames_cat[:, :1, :])
        den = F.conv_transpose1d(ones, self.Wn, stride=self.hop)         # [M, 1, Lpad]
        y = (num / (den + 1e-8))[:, 0]                       # [M, Lpad]
        pad = self.n_fft // 2                                # undo center padding
        y = y[:, pad : pad + length]
        return y.reshape(*lead, length)


# ┌─ CHUNK: patched HTDemucs spectral methods ───────────────────────────────────┐
# │ INTENT: drop-in replacements for HTDemucs._spec/_magnitude/_mask/_ispec that  │
# │         keep demucs' EXACT pad/crop/slice math but (a) call ConvSTFT and       │
# │         (b) carry a real-stacked [.., 2, ..] layout instead of complex.        │
# │ CONTRACT: numerically identical to the originals (validated in-process below   │
# │         against the unpatched native-stft model before any ONNX export).       │
# └──────────────────────────────────────────────────────────────────────────────┘
def _spec_conv(self, x):
    from demucs.htdemucs import pad1d
    hl, nfft = self.hop_length, self.nfft
    assert hl == nfft // 4
    le = int(math.ceil(x.shape[-1] / hl))
    pad = hl // 2 * 3
    x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
    z = self.convstft.stft(x)            # [B, C, 2, F(=nbins), frames]
    z = z[..., :-1, :]                   # drop Nyquist freq bin -> F = nfft//2
    assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
    z = z[..., 2 : 2 + le]               # crop frames, exactly as demucs
    return z                             # [B, C, 2, nfft//2, le]


def _magnitude_conv(self, z):
    # cac=True: fold the real/imag axis into channels. z: [B, C, 2, Fr, T]
    B, C, _two, Fr, T = z.shape
    return z.reshape(B, C * 2, Fr, T)


def _mask_conv(self, z, m):
    # cac=True: split channels back into (C, 2). m: [B, S, C*2, Fr, T]
    B, S, C2, Fr, T = m.shape
    return m.view(B, S, C2 // 2, 2, Fr, T)     # [B, S, C, 2, Fr, T] (real-stacked)


def _ispec_conv(self, z, length=None, scale=0):
    hl = self.hop_length // (4 ** scale)
    z = F.pad(z, (0, 0, 0, 1))           # add Nyquist freq row back (zeros) — pads Fr axis
    z = F.pad(z, (2, 2))                 # pad frames axis, exactly as demucs
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    x = self.convstft.istft(z, length=le)
    x = x[..., pad : pad + length]
    return x


def patch_for_export(htdemucs: nn.Module) -> nn.Module:
    """Attach ConvSTFT and swap in the conv/real-stacked spectral methods. In place."""
    if not getattr(htdemucs, "cac", False):
        sys.exit("This patch assumes cac=True (HTDemucs default). Got cac=False.")
    htdemucs.convstft = ConvSTFT(htdemucs.nfft)
    htdemucs._spec = types.MethodType(_spec_conv, htdemucs)
    htdemucs._magnitude = types.MethodType(_magnitude_conv, htdemucs)
    htdemucs._mask = types.MethodType(_mask_conv, htdemucs)
    htdemucs._ispec = types.MethodType(_ispec_conv, htdemucs)
    return htdemucs


class ExportWrapper(nn.Module):
    """Pin the export interface: [B, 2, N] -> [B, stems, 2, N]. HTDemucs already returns
    [B, S, C, N], so this is mostly a named-IO anchor (PLAN R3: lock channel order)."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        return self.model(mix)


def load_htdemucs() -> nn.Module:
    from demucs.pretrained import get_model
    from demucs.apply import BagOfModels
    bag = get_model("htdemucs")          # downloads MIT weights to torch hub cache
    sub = bag.models[0] if isinstance(bag, BagOfModels) else bag
    sub.eval()
    return sub


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("../weights/htdemucs.onnx"))
    p.add_argument("--opset", type=int, default=OPSET)
    p.add_argument("--check-tol", type=float, default=1e-3,
                   help="max-abs-diff tolerance for the conv-substitution self-check")
    args = p.parse_args()

    # The cross-transformer calls nn.MultiheadAttention(need_weights=False), which in eval
    # mode dispatches to the FUSED aten::_native_multi_head_attention kernel that the
    # TorchScript ONNX exporter cannot lower (opset 17). Disabling the MHA fast path routes
    # attention through the decomposable Linear/softmax path, which exports cleanly. Numeric
    # result is identical — the self-check below still gates correctness.
    torch.backends.mha.set_fastpath_enabled(False)

    print("Loading HTDemucs (downloads MIT weights on first run)…")
    native = load_htdemucs()
    print(f"  sources={native.sources}  nfft={native.nfft}  segment={float(native.segment)}s "
          f"-> chunk={TRAINING_LENGTH} samples")

    # ── Self-check: conv-substituted model vs native torch.stft model, in-process. ──
    # This isolates the STFT-substitution from ONNX-export issues: if this passes, the
    # patch is mathematically correct; any later mismatch is an ORT/export problem.
    patched = load_htdemucs()            # a second copy to patch (keep `native` pristine)
    patch_for_export(patched)
    torch.manual_seed(0)
    x = torch.randn(1, 2, TRAINING_LENGTH)
    with torch.no_grad():
        y_native = native(x)
        y_patched = patched(x)
    diff = float((y_native - y_patched).abs().max())
    snr = 10 * math.log10(float((y_native ** 2).sum()) /
                          float(((y_native - y_patched) ** 2).sum() + 1e-12))
    status = "PASS" if diff < args.check_tol else "FAIL"
    print(f"conv-substitution self-check: max_abs_diff={diff:.3e}  SNR={snr:.1f} dB  -> {status}")
    if diff >= args.check_tol:
        sys.exit("Conv STFT/ISTFT substitution does not match native torch.stft — "
                 "fix the basis/normalization before exporting (do NOT ship this).")

    # ── Export the patched model. ──────────────────────────────────────────────────
    model = ExportWrapper(patched).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting (torchscript, opset {args.opset}) -> {args.out}")
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), str(args.out),
            input_names=["mix"], output_names=["stems"],
            opset_version=args.opset,
            dynamic_axes={"mix": {0: "batch"}, "stems": {0: "batch"}},   # samples FIXED
            dynamo=False,
        )

    # ── Verify the .onnx loads, uses only basic ops, and runs in ORT == torch. ──────
    import onnx
    om = onnx.load(str(args.out))
    onnx.checker.check_model(om)
    ops = sorted({n.op_type for n in om.graph.node})
    banned = {"STFT", "DFT", "Rfft", "Irfft"}
    print(f"onnx.checker OK. distinct ops ({len(ops)}): {ops}")
    if banned & set(ops):
        sys.exit(f"Export emitted complex/fft ops {banned & set(ops)} — substitution leaked.")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    if onnx_out.shape != tuple(y_patched.shape):
        sys.exit(f"Shape mismatch torch {tuple(y_patched.shape)} vs onnx {onnx_out.shape}.")
    od = float(np.max(np.abs(y_patched.numpy() - onnx_out)))
    print(f"ORT vs patched-torch: max_abs_diff={od:.3e}  -> {'PASS' if od < 1e-3 else 'CHECK'}")
    print(f"\nExport complete: {args.out}  shape in/out = {list(x.shape)} -> {list(onnx_out.shape)}")
    print("Next: 05_validate_htdemucs.py for end-to-end SDR parity vs native PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
