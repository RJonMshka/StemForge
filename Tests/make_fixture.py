#!/usr/bin/env python3
"""Generate a small stereo 32-bit-float WAV fixture for the Phase 0 roundtrip test.

    python Tests/make_fixture.py            # -> Tests/fixtures/sine_stereo_f32.wav

Stereo, 44.1 kHz, 2 s: L = 220 Hz, R = 330 Hz sines at -6 dBFS. Float32 content exercises
full-precision fidelity through the read -> 32-bit WAV write -> read path.
"""
from pathlib import Path
import numpy as np
import soundfile as sf

SR, SECS = 44100, 2.0
t = np.arange(int(SR * SECS)) / SR
left = 0.5 * np.sin(2 * np.pi * 220.0 * t)
right = 0.5 * np.sin(2 * np.pi * 330.0 * t)
stereo = np.stack([left, right], axis=1).astype(np.float32)

out = Path(__file__).parent / "fixtures" / "sine_stereo_f32.wav"
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, stereo, SR, subtype="FLOAT")
print(f"wrote {out}  ({stereo.shape[0]} frames, {stereo.shape[1]} ch, float32)")
