#!/usr/bin/env python3
"""Re-export encoder.onnx with correct external data reference."""
# ruff: noqa: E402

import os
import sys

for key in list(sys.modules.keys()):
    if "numba" in key:
        del sys.modules[key]

import torch  # noqa: F401

_original_onnx_export = torch.onnx.export


def _patched_onnx_export(*args, **kwargs):
    kwargs["dynamo"] = False
    return _original_onnx_export(*args, **kwargs)


torch.onnx.export = _patched_onnx_export

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

model = EncDecHybridRNNTCTCBPEModel.restore_from(
    "data/nemo/multitalker-parakeet-streaming-0.6b-v1.nemo", map_location="cpu"
)
model.eval()
model.set_export_config({"cache_support": "True"})

if hasattr(model, "encoder_hooks"):
    for hook in model.encoder_hooks:
        hook.remove()
    model.encoder_hooks.clear()

for param in model.parameters():
    param.requires_grad = False

for f in [
    "data/parakeet/encoder.onnx",
    "data/parakeet/encoder.onnx.data",
    "data/parakeet/decoder_joint.onnx",
]:
    if os.path.exists(f):
        os.unlink(f)

encoder = model.get_export_subnet("encoder")
encoder.eval()
encoder._export(
    output="data/parakeet/encoder.onnx",
    check_trace=False,
    do_constant_folding=True,
    onnx_opset_version=17,
)
print("Encoder exported")

import onnx
from onnx.external_data_helper import convert_model_to_external_data

enc_model = onnx.load("data/parakeet/encoder.onnx")
convert_model_to_external_data(
    enc_model,
    all_tensors_to_one_file=True,
    location="encoder.onnx.data",
    size_threshold=1024,
)
for init in enc_model.graph.initializer:
    if init.data_location == 1:
        for entry in init.external_data:
            if entry.key == "location":
                path = os.path.join("data/parakeet", entry.value)
                if os.path.exists(path) and entry.value != "encoder.onnx.data":
                    os.unlink(path)
                break

onnx.save(enc_model, "data/parakeet/encoder.onnx")
print("Encoder consolidated")

dec_joint = model.get_export_subnet("decoder_joint")
dec_joint.eval()
dec_joint._export(
    output="data/parakeet/decoder_joint.onnx",
    check_trace=False,
    do_constant_folding=True,
    onnx_opset_version=17,
)
print("Decoder exported")

torch.onnx.export = _original_onnx_export

for p in [
    "data/parakeet/encoder.onnx",
    "data/parakeet/encoder.onnx.data",
    "data/parakeet/decoder_joint.onnx",
]:
    if os.path.exists(p):
        sz = os.path.getsize(p) / (1024 * 1024)
        print(f"  {p}: {sz:.1f} MB")

import onnxruntime as ort

sess = ort.InferenceSession(
    "data/parakeet/encoder.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("\nEncoder inputs:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: {inp.shape}")
print("Encoder outputs:")
for out in sess.get_outputs():
    print(f"  {out.name}: {out.shape}")

sess2 = ort.InferenceSession(
    "data/parakeet/decoder_joint.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("\nDecoder inputs:")
for inp in sess2.get_inputs():
    print(f"  {inp.name}: {inp.shape}")
print("Decoder outputs:")
for out in sess2.get_outputs():
    print(f"  {out.name}: {out.shape}")
