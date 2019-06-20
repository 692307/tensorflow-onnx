# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit tests using onnx backends."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from itertools import product

import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
# pylint reports unused-wildcard-import which is false positive, __all__ is defined in common
from common import *  # pylint: disable=wildcard-import,unused-wildcard-import
from tf2onnx import constants

# pylint: disable=missing-docstring,invalid-name,unused-argument


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]

_STRIDE1x1 = [1, 1, 1, 1]
_KERNEL3x3 = [3, 3, 1, 1]

# names for input and outputs for tests
_TFINPUT = "input"
_INPUT = "input:0"
_TFINPUT1 = "input1"
_INPUT1 = "input1:0"
_TFINPUT2 = "input2"
_INPUT2 = "input2:0"
_TFOUTPUT = "output"
_OUTPUT = "output:0"
_TFOUTPUT1 = "output1"
_OUTPUT1 = "output1:0"
_TFOUTPUT2 = "output2"
_OUTPUT2 = "output2:0"

def make_xval(shape):
    x_val = (2*np.arange(np.prod(shape))).astype("float32").reshape(shape)
    return x_val

def get_conv_getdata(kind=1):
    if kind == 0:
        # generate all combinations (costly)
        dims = [
            ("padding", ["SAME", "VALID"]),
            ("input_sizes", [[32, 35, 35, 3], [32, 17, 17, 3], [1, 28, 28, 3], [32, 8, 8, 3]]),
            ("filter_sizes", [[1, 3, 3, 1], [1, 2, 2, 1], [1, 5, 5, 1], [1, 1, 1, 1], [1, 5, 2, 1], [1, 2, 5, 1]]),
            ("strides", [[1, 2, 2, 1], [1, 1, 1, 1]]),
        ]
        values = [key_values[1] for key_values in dims]
        for idx, v in enumerate(product(*values)):
            if True or idx == 30:
                yield (idx,) + v
    elif kind == 1:
        # some combination to that give decent padding coverage
        data = [
            ('SAME', [32, 35, 35, 3], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 3], [1, 2, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 3], [1, 1, 1, 1], [1, 1, 1, 1]),
            ('SAME', [32, 35, 35, 3], [1, 5, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 3], [1, 2, 5, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 3], [1, 2, 5, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 2, 2, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 2, 2, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 5, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 5, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 8, 8, 3], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [32, 8, 8, 3], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('VALID', [32, 35, 35, 3], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('VALID', [32, 35, 35, 3], [1, 2, 2, 1], [1, 2, 2, 1]),
        ]
        for idx, v in enumerate(data):
            yield (idx,) + v
    else:
        raise ValueError("kind not known")

def get_maxpoolwithargmax_getdata():
    data = [
        ('SAME', [1, 3, 3, 1], [1, 3, 3, 1], [1, 2, 2, 1]),
        ('SAME', [1, 5, 5, 1], [1, 4, 4, 1], [1, 2, 2, 1]),
        ('SAME', [1, 10, 5, 1], [1, 2, 2, 1], [1, 2, 2, 1]),
        ('SAME', [1, 10, 5, 1], [1, 4, 4, 1], [1, 1, 1, 1]),
        ('VALID', [1, 3, 3, 1], [1, 3, 3, 1], [1, 2, 2, 1]),
        ('VALID', [1, 5, 5, 1], [1, 4, 4, 1], [1, 2, 2, 1]),
    ]
    for idx, v in enumerate(data):
        yield (idx,) + v


class BackendTests(Tf2OnnxBackendTestBase):
    def _run_test_case(self, output_names_with_port, feed_dict, **kwargs):
        # print("in _run_test_case")
        # print(output_names_with_port)
        # print(feed_dict)
        kwargs["convert_var_to_const"] = False
        kwargs["constant_fold"] = False
        return self.run_test_case(feed_dict, [], output_names_with_port, **kwargs)

    # @skip_caffe2_backend("fails on caffe2 with dim issue")
    # @check_onnxruntime_incompatibility("Mul")
    # @check_tf_min_version("1.6")
    # def test_leaky_relu_int(self):
    #     # starting from tf 1.6, leaky_relu supports `feature` x of int type
    #     x_types = [np.int32, np.int64]
    #     for x_type in x_types:
    #         x_val = 1000 * np.random.random_sample([1000, 100]).astype(x_type)
    #         for alpha in [0.1, -0.1, 1.0, -1.0]:
    #             x = tf.placeholder(x_val.dtype, [None] * x_val.ndim, name=_TFINPUT)
    #             x_ = tf.nn.leaky_relu(x, alpha)
    #             _ = tf.identity(x_, name=_TFOUTPUT)
    #             self._run_test_case([_OUTPUT], {_INPUT: x_val})
    #             tf.reset_default_graph()

    # test for gemm pattern0: alpha*A*B + beta*C
    # def test_gemm_pattern0(self):
    #     x_val1 = np.array([[1., 2.], [4., 5.]], dtype=np.float32)   #np.random
    #     x_val2 = np.array([[7., 8.], [10., 11.]], dtype=np.float32)
    #     x_val3 = np.array([[13., 14.], [16., 17.]], dtype=np.float32)
    #     a = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
    #     b = tf.placeholder(tf.float32, x_val2.shape, name=_TFINPUT1)
    #     c = tf.placeholder(tf.float32, x_val3.shape, name=_TFINPUT2)
    #     alpha = tf.constant(1.0)
    #     beta = tf.constant(2.0)
    #     # x_ = tf.matmul(a, b) + c
    #     x_ = tf.multiply(alpha, tf.matmul(a, b)) + tf.multiply(beta, c)
    #     # x_ = tf.multiply(alpha, tf.matmul(a, b))
    #     _ = tf.identity(x_, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, _INPUT2: x_val3},
    #                         graph_validator=lambda g: check_op_count(g, "Gemm", 1))

    # def test_batch_to_spacend(self):
    #     block_size = [2, 2]
    #     crop = [[0, 1], [2, 1]]
    #     input_val = np.random.random_sample([40, 3, 5, 100]).astype(np.float32)
    #     input_x = tf.placeholder(dtype=tf.float32, shape=input_val.shape, name=_TFINPUT)  # NHWC
    #     _ = tf.batch_to_space_nd(input_x, block_size, crop, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: input_val})

    # def test_silly_add(self):
    #     a_values = np.array([[1, 2], [4, 5]], dtype=np.int32)
    #     b_values = np.array([[1, 2], [4, 5]], dtype=np.int32)
    #     A = tf.placeholder(dtype=tf.int32, shape=[2, 2], name=_TFINPUT)
    #     B = tf.placeholder(dtype=tf.int32, shape=[2, 2], name=_TFINPUT1)
    #     _ = tf.add(A, B, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: a_values, _INPUT1: b_values})

    def test_batch_to_spacend_New(self):
        block_size = [2, 2]
        crop_value = [[0, 1], [2, 1]]

        crop = tf.placeholder(dtype=tf.int32, shape=[2, 2])
        input_val = np.random.random_sample([40, 3, 5, 100]).astype(np.float32)
        input_x = tf.placeholder(dtype=tf.float32, shape=input_val.shape, name=_TFINPUT)  # NHWC
        _ = tf.batch_to_space_nd(input_x, block_size, crop, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], feed_dict={_INPUT: input_val, crop: crop_value})

    # @check_onnxruntime_incompatibility("Add")
    # def test_add_bcast1(self):
    #     # example taken from onnx doc
    #     x1_val = np.random.randn(3, 4, 5).astype(np.float32)
    #     x2_val = np.random.randn(5).astype(np.float32)
    #     x1 = tf.placeholder(tf.float32, x1_val.shape, name="input")
    #     x2 = tf.placeholder(tf.float32, x2_val.shape, name=_TFINPUT1)
    #     x_ = tf.add(x1, x2)
    #     _ = tf.identity(x_, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: x1_val, _INPUT1: x2_val})

if __name__ == '__main__':
    unittest_main()

