#!/usr/bin/env python3
"""
HTDemucs parity check — does the exported ONNX match the NATIVE PyTorch model?

Runs the same 7.8 s stereo chunk through (a) the unpatched HTDemucs (real torch.stft) and
(b) the exported ONNX (conv-STFT substitution), then reports per-stem SDR. This is the real
acceptance gate for the conv-STFT substitution end to end: the export script's self-check
already compares native-vs-patched in-process (~117 dB), this additionally pins the ONNX
RUNTIME path against native PyTorch.

Acceptance (from PLAN): worst-stem SDR delta > 40 dB  (== far below the 0.1 dB audible bar).

    python 05_validate_htdemucs.py --onnx ../weights/htdemucs.onnx              # synthetic clip
    python 05_validate_htdemucs.py --onnx ../weights/htdemucs.onnx --audio x.wav # real clip (44.1k)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 44100
CHUNK = int(39 / 5 * SAMPLE_RATE)        # 343980, HTDemucs training segment


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    num = float(np.sum(ref ** 2))
    den = float(np.sum((ref - est) ** 2)) + 1e-12
    return 10.0 * np.log10((num + 1e-12) / den)


def make_chunk(audio_path: Path | None) -> np.ndarray:
    """Return one [1, 2, CHUNK] float32 chunk (synthetic if no audio given)."""
    if audio_path is None:
        # Reproducible pseudo-musical signal: a few partials + light noise, per channel.
        rng = np.random.default_rng(0)
        t = np.arange(CHUNK) / SAMPLE_RATE
        sig = np.zeros((2, CHUNK), dtype=np.float32)
        for ch, base in enumerate((110.0, 147.0)):
            for k in (1, 2, 3, 5):
                sig[ch] += np.sin(2 * np.pi * base * k * t) / k
            sig[ch] += 0.05 * rng.standard_normal(CHUNK)
        sig *= 0.2 / np.max(np.abs(sig))
        return sig[None].astype(np.float32)

    import soundfile as sf
    audio, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)  # [N, ch]
    if file_sr != SAMPLE_RATE:
        sys.exit(f"Audio is {file_sr} Hz; resample to {SAMPLE_RATE} Hz first (model SR).")
    audio = audio.T                                          # [ch, N]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio = audio[:2]
    if audio.shape[1] < CHUNK:                               # zero-pad short clips
        audio = np.pad(audio, ((0, 0), (0, CHUNK - audio.shape[1])))
    return audio[None, :, :CHUNK].astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", type=Path, default=Path("../weights/htdemucs.onnx"))
    p.add_argument("--audio", type=Path, default=None)
    args = p.parse_args()
    if not args.onnx.exists():
        sys.exit(f"ONNX not found: {args.onnx}. Run 04_export_htdemucs.py first.")

    from demucs.pretrained import get_model
    from demucs.apply import BagOfModels
    bag = get_model("htdemucs")                              # NATIVE: real torch.stft
    native = bag.models[0] if isinstance(bag, BagOfModels) else bag
    native.eval()
    sources = native.sources

    chunk = make_chunk(args.audio)
    with torch.no_grad():
        torch_out = native(torch.from_numpy(chunk)).cpu().numpy()    # [1, S, 2, N]

    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: chunk})[0]

    if torch_out.shape != onnx_out.shape:
        sys.exit(f"Shape mismatch torch {torch_out.shape} vs onnx {onnx_out.shape} (PLAN R3).")

    print(f"{'stem':>8}  {'SDR(onnx vs native torch) dB':>30}")
    worst = 1e9
    for s, name in enumerate(sources):
        d = sdr(torch_out[0, s], onnx_out[0, s])
        worst = min(worst, d)
        print(f"{name:>8}  {d:>30.2f}")
    verdict = "PASS ✓ (parity)" if worst > 40 else "INVESTIGATE"
    print(f"\nworst-stem SDR delta: {worst:.2f} dB  -> {verdict}")
    return 0 if worst > 40 else 1


if __name__ == "__main__":
    sys.exit(main())
