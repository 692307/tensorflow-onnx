# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for shape inference."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from onnx import TensorProto
from tf2onnx import utils
from tf2onnx.graph import Graph
from backend_test_base import Tf2OnnxBackendTestBase
from common import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-docstring

INPUT1 = "input1"
INPUT2 = "input2"
INPUT3 = "input3"


class ONNXShapeInferenceTests(Tf2OnnxBackendTestBase):
    """
    Test shape inference, it's just a subset of all cases that can be inferred shape.
    For more information, please refer to onnx shape inference test:
    https://github.com/onnx/onnx/blob/master/onnx/test/shape_inference_test.py
    """

    def _run_test_case(self, graph, feed_dict):
        """Run model with onnxruntime and compare results' shape with internal shape inference."""
        outputs = graph.outputs
        results = self.run_backend(graph, outputs, feed_dict)

        for actual, inferred in zip(results, outputs):
            actual_shape = actual.shape
            inferred_shape = tuple(graph.get_shape(inferred))
            self.assertTrue(utils.are_shapes_compatible(actual_shape, inferred_shape))

            actual_dtype = actual.dtype
            inferred_dtype = utils.ONNX_TO_NUMPY_DTYPE[graph.get_dtype(inferred)]
            self.assertEqual(actual_dtype, inferred_dtype)

    def _create_empty_graph(self, inputs, shapes, dtypes):
        graph = Graph([], target=self.config.target, opset=self.config.opset)
        for inp, shape, dtype in zip(inputs, shapes, dtypes):
            graph.add_graph_input(inp, dtype, shape)
        return graph

    def _generate_random_inputs(self, inputs, shapes, dtypes):
        """Generate random input of shape and ONNX dtypes"""
        res = {}
        for inp, shape, dtype in zip(inputs, shapes, dtypes):
            new_shape = [1 if s == -1 else s for s in shape]
            res[inp] = np.random.random(new_shape).astype(utils.ONNX_TO_NUMPY_DTYPE[dtype])
        return res

    # @check_opset_min_version(2, "Split")
    # def test_split(self):
    #     inputs = [INPUT1]
    #     shapes = [[5, 6, 6]]
    #     dtypes = [TensorProto.FLOAT]
    #     graph = self._create_empty_graph(inputs, shapes, dtypes)
    #     node = graph.make_node("Split", [INPUT1], output_count=2, attr={"axis": 2})
    #     graph.add_graph_output(node.output[0])
    #     graph.add_graph_output(node.output[1])
    #     self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_gather(self):
        inputs = [INPUT1]
        shapes = [[4, 3]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        const = graph.make_const("index", np.array([0], dtype=np.int64))
        node = graph.make_node("Gather", [INPUT1, const.output[0]])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

if __name__ == "__main__":
    unittest_main()
