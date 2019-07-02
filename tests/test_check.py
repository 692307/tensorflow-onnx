# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for optimizers such as TransposeOptimizer."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from onnx import helper, TensorProto, OperatorSetIdProto
from tf2onnx import utils
from tf2onnx.graph import GraphUtil
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, group_nodes_by_type, check_opset_min_version


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class OptimizerTests(Tf2OnnxBackendTestBase):
    """Run original model proto and modified model proto with onnxruntime, compare the results."""

    def run_and_compare(self, output_names_with_port, onnx_feed_dict, origin_proto, op_type,
                        remaining_op_num, debug=False, rtol=1e-07):
        utils.make_sure(op_type is not None, "op_type should be specified")
        utils.make_sure(remaining_op_num is not None, "remaining_op_num should be specified")

        origin_model_path = self.save_onnx_model(origin_proto, onnx_feed_dict, postfix="_origin")

        new_proto = GraphUtil.optimize_model_proto(origin_proto)

        self.assertTrue(new_proto, msg="model proto after optimizer should not be None")

        new_model_path = self.save_onnx_model(new_proto, onnx_feed_dict, postfix="_opt")
        current = GraphUtil.get_node_count_from_onnx_graph(new_proto.graph)

        self.assertTrue(current[op_type] == remaining_op_num,
                        msg="Expect " + str(remaining_op_num) + " " + op_type + " ops left, but actually " + str(
                            current[op_type]) + " left")

        if self.config.is_onnxruntime_backend:
            expected = self.run_onnxruntime(origin_model_path, onnx_feed_dict, output_names_with_port)
            actual = self.run_onnxruntime(new_model_path, onnx_feed_dict, output_names_with_port)
        else:
            raise ValueError("only onnxruntime is supported to test transpose optimizer")

        for expected_val, actual_val in zip(expected, actual):
            self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=1e-5)
            self.assertEqual(expected_val.dtype, actual_val.dtype)
            self.assertEqual(expected_val.shape, actual_val.shape)

        return new_proto

    def run_transpose_compare(self, output_names_with_port, onnx_feed_dict, origin_proto,
                              remaining_transpose_num=None, debug=False, rtol=1e-07):
        return self.run_and_compare(output_names_with_port, onnx_feed_dict, origin_proto, op_type="Transpose",
                                    remaining_op_num=remaining_transpose_num, debug=debug, rtol=rtol)

    def test_transpose_add_with_conv_2(self):
        # case where bias's dim is not 1D and can't be merged into Conv
        # add handler just remove the transpose around Add node
        const_b_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
        const_b = helper.make_tensor("const_b", TensorProto.FLOAT, (1, 3, 3, 1), const_b_val.flatten())
        const_b_node = helper.make_node("Constant", [], ["const_b"], value=const_b, name="const_b")

        node0 = helper.make_node("Conv", ["x", "W"], ["X"], name="conv", pads=[0, 0, 0, 0])
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node2 = helper.make_node("Add", ["Y", "const_b"], ["Z"], name="add")
        node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [const_b_node, node0, node1, node2, node3],
            "transpose-add-test-with-conv-2",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 1, 5, 5)),
             helper.make_tensor_value_info("W", TensorProto.FLOAT, (1, 1, 3, 3))],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, (1, 1, 3, 3))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"x": np.random.randn(1, 1, 5, 5).astype(np.float32),
                                             "W": np.random.randn(1, 1, 3, 3).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_transpose_max_input_non_const(self):
        const_1_val = [2.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        const_1_node = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")

        const_2_val = np.random.randn(2, 4, 5, 3).astype(np.float32)
        const_2 = helper.make_tensor("const_2", TensorProto.FLOAT, (2, 4, 5, 3), const_2_val.flatten())
        const_2_node = helper.make_node("Constant", [], ["const_2"], value=const_2, name="const_2")

        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node2 = helper.make_node("Max", ["Y", "non_const", "const_2", "const_1"], ["Z"], name="max")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [const_1_node, const_2_node, node1, node2, node3],
            "Max-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5)),
             helper.make_tensor_value_info("non_const", TensorProto.FLOAT, (2, 4, 5, 3))],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32),
                                            "non_const": np.random.randn(2, 4, 5, 3).astype(np.float32)},
                                   model_proto, remaining_transpose_num=1)

if __name__ == "__main__":
    unittest_main()
