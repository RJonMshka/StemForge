#!/usr/bin/env python3
"""
R1 spike — Step 3: End-to-end parity — does the exported ONNX model match PyTorch?

Run AFTER 02 produces an .onnx. Feeds the same real audio chunk through both the PyTorch
model and the ONNX model and reports per-stem SDR delta. The acceptance bar (from the plan)
is < 0.1 dB difference — below that the C++ path can be trusted.

    python 03_validate_parity.py \
        --onnx ../weights/bs_roformer_sw.onnx \
        --config ./model_bs_roformer.yaml \
        --checkpoint ./bs_roformer_sw.ckpt \
        --audio ./test_clip.wav

Needs the checkpoint + config (same as 02) and a short stereo test clip.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Signal-to-distortion ratio in dB (global). Higher = closer."""
    num = np.sum(ref ** 2)
    den = np.sum((ref - est) ** 2) + 1e-12
    return 10.0 * np.log10((num + 1e-12) / den)


def load_audio(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # [N, ch]
    audio = audio.T                                                        # [ch, N]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)                                # mono -> stereo
    audio = audio[:2]
    if file_sr != sr:
        sys.exit(f"Audio is {file_sr} Hz; resample to {sr} Hz first (model SR).")
    return audio


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from importlib import import_module
    exporter = import_module("02_export_bs_roformer".replace("02_", "_02_")) \
        if False else None  # noqa: F841  (kept explicit; build_model reused below)

    # Rebuild the torch model (reuse 02's helpers)
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp", Path(__file__).parent / "02_export_bs_roformer.py")
    exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
    cfg = exp.load_config(args.config)
    model = exp.StereoWrapper(exp.build_model(cfg))
    state = torch.load(args.checkpoint, map_location="cpu")
    model.model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    audio = load_audio(args.audio, args.sample_rate)
    n = int(args.chunk_seconds * args.sample_rate)
    chunk = audio[:, :n][None, ...]                          # [1, 2, n]
    x = torch.tensor(chunk, dtype=torch.float32)

    with torch.no_grad():
        torch_out = model(x).cpu().numpy()                  # [1, stems, 2, n]

    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: chunk})[0]

    if torch_out.shape != onnx_out.shape:
        sys.exit(f"Shape mismatch torch {torch_out.shape} vs onnx {onnx_out.shape} — "
                 "check tensor layout (onnx-runtime-cpp skill, #1 bug).")

    n_stems = torch_out.shape[1]
    print(f"{'stem':>6}  {'SDR(onnx vs torch) dB':>22}")
    worst = 1e9
    for s in range(n_stems):
        d = sdr(torch_out[0, s], onnx_out[0, s])
        worst = min(worst, d)
        print(f"{s:>6}  {d:>22.2f}")
    print(f"\nworst-stem SDR delta: {worst:.2f} dB  -> "
          f"{'PASS ✓ (parity)' if worst > 40 else 'INVESTIGATE'}")
    # >40 dB SDR between the two implementations == far below 0.1 dB audible difference.
    return 0 if worst > 40 else 1


if __name__ == "__main__":
    sys.exit(main())
