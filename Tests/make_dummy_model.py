#!/usr/bin/env python3
"""
Generate a tiny dummy separation model for testing the C++ inference harness — NOT a real
separator. It emits the exact IO contract the engine expects (input [B,C,N] "mix",
output [B,stems,C,N] "stems"), so StemForgeSeparate can be exercised end-to-end without the
large, license-gated real weights.

The "model" splits the mix into two stems, each = 0.5 * mix. So a correct harness produces
stem_0 and stem_1 that each equal half the input, and stem_0 + stem_1 == the input. That
makes the C++ pipeline (chunk → infer → overlap-add → write) verifiable.

    python Tests/make_dummy_model.py            # writes Tests/fixtures/dummy_half_split.onnx
"""
from pathlib import Path
import torch
import torch.nn as nn


class DummyHalfSplit(nn.Module):
    def forward(self, mix: torch.Tensor) -> torch.Tensor:   # mix: [B, C, N]
        half = mix * 0.5
        return torch.stack([half, half], dim=1)             # [B, stems=2, C, N]


def main() -> None:
    out = Path(__file__).resolve().parent / "fixtures" / "dummy_half_split.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    model = DummyHalfSplit().eval()
    example = torch.randn(1, 2, 1000)
    with torch.no_grad():
        torch.onnx.export(
            model, (example,), str(out),
            input_names=["mix"], output_names=["stems"],
            opset_version=17,
            dynamic_axes={"mix": {0: "batch", 2: "samples"},
                          "stems": {0: "batch", 3: "samples"}},
            dynamo=False,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
