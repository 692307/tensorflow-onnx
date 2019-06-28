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

    # @check_opset_min_version(10, "Slice in opset 10 can accept dymaic 'start' and 'ends'")
    # def test_slice_with_dynamic_starts_and_size(self):
    #     x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    #     t1_value = np.array([0, 1], dtype=np.int32)
    #     t2_value = np.array([2, 2], dtype=np.int32)
    #
    #     t1 = tf.placeholder(dtype=tf.int32, shape=t1_value.shape, name=_TFINPUT1)
    #     t2 = tf.placeholder(dtype=tf.int32, shape=t2_value.shape, name=_TFINPUT2)
    #     x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
    #
    #     x_ = tf.slice(x0, t1, t2)
    #     _ = tf.identity(x_, name=_TFOUTPUT)
    #
    #     self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: t1_value, _INPUT2: t2_value})

    # def test_equal(self):
    #     x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
    #     x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
    #     x1 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT)
    #     x2 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT1)
    #     mi = tf.equal(x1, x2)
    #     _ = tf.identity(mi, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})
    #
    #     tf.reset_default_graph()
    #     x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
    #     x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
    #     x1 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
    #     x2 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT1)
    #     mi = tf.equal(x1, x2)
    #     _ = tf.identity(mi, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    # @check_opset_min_version(10, "Slice in opset 10 can accept dymaic 'start' and 'ends'")
    # def test_slice_with_non_const(self):
    #     x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    #     t1 = np.array([0, 1], dtype=np.int32)
    #     t2 = np.array([2, 2], dtype=np.int32)
    #
    #     t1_ = tf.placeholder(tf.int32, t1.shape, name=_TFINPUT1)
    #     t2_ = tf.placeholder(tf.int32, t2.shape, name=_TFINPUT2)
    #     x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
    #
    #     x_ = tf.slice(x0, t1_, t2_)
    #     _ = tf.identity(x_, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: t1, _INPUT2: t2})

    # def test_batch_to_spacend(self):
    #     block_size = [2, 2]
    #     crop = [[0, 1], [2, 1]]
    #     input_val = np.random.random_sample([40, 3, 5, 100]).astype(np.float32)
    #     input_x = tf.placeholder(dtype=tf.float32, shape=input_val.shape, name=_TFINPUT)  # NHWC
    #     _ = tf.batch_to_space_nd(input_x, block_size, crop, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: input_val})

    def test_batch_to_spacend_with_dynamic_crop(self):
        block_size = [2, 2]
        crop_value = np.array([[0, 1], [2, 1]], dtype=np.int32)
        input_val = np.random.random_sample([40, 3, 5, 100]).astype(np.float32)

        input_x = tf.placeholder(dtype=tf.float32, shape=input_val.shape, name=_TFINPUT)
        crop = tf.placeholder(dtype=tf.int32, shape=[2, 2], name=_TFINPUT1)
        _ = tf.batch_to_space_nd(input_x, block_size, crop, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], feed_dict={_INPUT: input_val, _INPUT1: crop_value})

    # def test_gather(self):
    #     x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    #     idx = np.array([1, 0, 2], dtype=np.int32)
    #     idx_flattened = np.array([i * x_val.shape[1] + idx for i in range(0, x_val.shape[0])])
    #     x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
    #     x_ = tf.gather(tf.reshape(x, [-1]), tf.constant(idx_flattened))
    #     _ = tf.identity(x_, name=_TFOUTPUT)
    #     self._run_test_case([_OUTPUT], {_INPUT: x_val})
    #
    # def test_split_Gao(self):
    #     crop_value = [[0, 1], [2, 1]]
    #     crop = tf.placeholder(dtype=tf.int32, shape=[2, 2])

if __name__ == '__main__':
    unittest_main()

