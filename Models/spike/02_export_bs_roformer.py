#!/usr/bin/env python3
"""
R1 spike — Step 2: Export the FULL BS-RoFormer model to ONNX.

GATED ON STEP 1: run 01_stft_onnx_probe.py first. The exporter you pass here (--exporter)
should be the strategy that PASSED in step 1:
  - native stft exported fine  -> --exporter torchscript   (or dynamo)
  - only the manual fallback worked -> see "STFT substitution" note below.

This script needs an actual checkpoint + its config — those are large and licensed, so they
are NOT in the repo (see .gitignore). Acquire jarredou's BS-RoFormer-SW checkpoint (verify
license before redistributing) and its YAML config, then:

    python 02_export_bs_roformer.py \
        --config   ./model_bs_roformer.yaml \
        --checkpoint ./bs_roformer_sw.ckpt \
        --exporter torchscript \
        --out ../weights/bs_roformer_sw.onnx

Architecture comes from the `bs_roformer` package (lucidrains):  pip install bs-roformer
The checkpoint must match the config's dims/depth exactly or state_dict loading fails.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def load_config(path: Path) -> dict:
    import yaml  # pip install pyyaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate BSRoformer from a ZFTurbo/lucidrains-style config. The exact key layout
    depends on the checkpoint; adjust the mapping to your config. This is intentionally
    explicit so a mismatch fails loudly rather than silently mis-shaping.
    """
    try:
        from bs_roformer import BSRoformer
    except ImportError:
        sys.exit("Missing dependency: pip install bs-roformer")
    model_kwargs = cfg.get("model", cfg)            # configs vary; support both shapes
    return BSRoformer(**model_kwargs)


class StereoWrapper(nn.Module):
    """
    Normalizes the export interface to: input [B, 2, samples] -> output [B, stems, 2, samples].
    The plugin always feeds stereo, fixed channel order; pinning it here avoids the #1
    integration bug (tensor layout mismatch — see onnx-runtime-cpp skill).
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)        # bs_roformer returns [B, stems, channels, samples]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--exporter", choices=["torchscript", "dynamo"], default="torchscript")
    p.add_argument("--out", type=Path, default=Path("../weights/bs_roformer_sw.onnx"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--sample-rate", type=int, default=44100)
    args = p.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\n"
                 "Acquire the BS-RoFormer-SW checkpoint (verify license) — it is not in git.")

    cfg = load_config(args.config)
    model = StereoWrapper(build_model(cfg))
    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict mismatch — missing={len(missing)} unexpected={len(unexpected)}")
        print("       If many, the config does not match the checkpoint. Fix before trusting output.")
    model.eval()

    # ── STFT substitution note ───────────────────────────────────────────────
    # If step 1 showed native torch.stft does NOT export, monkeypatch the model's STFT
    # front/back-end with the manual DFT-matmul implementation from 01 here, BEFORE export.
    # Keep it in one place so the swap is auditable.

    samples = int(args.chunk_seconds * args.sample_rate)
    example = torch.randn(1, 2, samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ({args.exporter}, opset {args.opset}) -> {args.out}")
    with torch.no_grad():
        if args.exporter == "dynamo":
            try:
                torch.onnx.export(model, (example,), str(args.out),
                                  opset_version=args.opset, dynamo=True)
            except TypeError:
                torch.onnx.dynamo_export(model, example).save(str(args.out))
        else:
            torch.onnx.export(
                model, (example,), str(args.out),
                input_names=["mix"], output_names=["stems"],
                opset_version=args.opset,
                dynamic_axes={"mix": {0: "batch", 2: "samples"},
                              "stems": {0: "batch", 3: "samples"}},
                dynamo=False,
            )
    print("Export OK. Next: run 03_validate_parity.py to confirm <0.1 dB SDR delta vs PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
