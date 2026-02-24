#!/usr/bin/env python3
"""Export ONNX encoder and decoder_joint from multitalker .nemo checkpoint.

Removes multitalker speaker kernel hooks before export (they use forward
pre-hooks that can't be traced by torch.jit). For single-speaker inference,
spk_targets=1.0 applies a fixed transform that we handle separately.

Patches torch.onnx.export to use legacy exporter (dynamo=False) for
PyTorch 2.10+ compatibility.
"""
# ruff: noqa: E402

import os
import sys

for key in list(sys.modules.keys()):
    if "numba" in key:
        del sys.modules[key]

import torch

_original_onnx_export = torch.onnx.export


def _patched_onnx_export(*args, **kwargs):
    kwargs["dynamo"] = False
    return _original_onnx_export(*args, **kwargs)


torch.onnx.export = _patched_onnx_export

print("Loading NeMo model...")
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

model = EncDecHybridRNNTCTCBPEModel.restore_from(
    "data/nemo/multitalker-parakeet-streaming-0.6b-v1.nemo", map_location="cpu"
)
model.eval()
model.set_export_config({"cache_support": "True"})
print(f"Loaded: {type(model).__name__}")

if hasattr(model, "encoder_hooks"):
    for hook in model.encoder_hooks:
        hook.remove()
    model.encoder_hooks.clear()
    print("Removed speaker kernel hooks from encoder")

for param in model.parameters():
    param.requires_grad = False

encoder_path = "data/parakeet/encoder_new.onnx"
decoder_path = "data/parakeet/decoder_joint_new.onnx"

print(f"\n--- Exporting encoder to {encoder_path} ---")
encoder = model.get_export_subnet("encoder")
encoder.eval()
encoder._export(
    output=encoder_path,
    check_trace=False,
    do_constant_folding=True,
    onnx_opset_version=17,
)
print("Encoder exported.")

print(f"\n--- Exporting decoder_joint to {decoder_path} ---")
dec_joint = model.get_export_subnet("decoder_joint")
dec_joint.eval()
dec_joint._export(
    output=decoder_path,
    check_trace=False,
    do_constant_folding=True,
    onnx_opset_version=17,
)
print("Decoder exported.")

torch.onnx.export = _original_onnx_export

print("\n--- Exporting speaker kernel ---")
spk_kernel = model.spk_kernels["0"]
spk_kernel.eval()
dummy_input = torch.randn(1, 1, 1024)
torch.onnx.export(
    spk_kernel,
    dummy_input,
    "data/parakeet/spk_kernel.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 1: "time"}, "output": {0: "batch", 1: "time"}},
    opset_version=17,
    dynamo=False,
)
print("Speaker kernel exported.")

print("\n--- Exported files ---")
for p in [encoder_path, decoder_path, "data/parakeet/spk_kernel.onnx"]:
    if os.path.exists(p):
        sz = os.path.getsize(p) / (1024 * 1024)
        print(f"  {p}: {sz:.1f} MB")
    data_file = p + ".data"
    if os.path.exists(data_file):
        sz = os.path.getsize(data_file) / (1024 * 1024)
        print(f"  {data_file}: {sz:.1f} MB")
