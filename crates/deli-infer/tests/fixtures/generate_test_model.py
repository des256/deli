#!/usr/bin/env python3
"""
Generate a minimal ONNX model for testing.
This creates an "add" model that takes two float tensors (X and Y) and returns their sum.

Requirements: pip install onnx numpy
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3])

Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [2, 3])

add_node = helper.make_node(
    'Add',
    inputs=['X', 'Y'],
    outputs=['Z']
)

graph = helper.make_graph(
    nodes=[add_node],
    name='TestAddModel',
    inputs=[X, Y],
    outputs=[Z]
)

model = helper.make_model(graph, producer_name='deli-infer-test')
model.opset_import[0].version = 11
model.ir_version = 8

onnx.checker.check_model(model)
onnx.save(model, 'test_add.onnx')
print("Generated test_add.onnx successfully")
