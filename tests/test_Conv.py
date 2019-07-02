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
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

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
    x_val = np.arange(np.prod(shape)).astype("float32").reshape(shape)
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
        kwargs["convert_var_to_const"] = False
        kwargs["constant_fold"] = False
        return self.run_test_case(feed_dict, [], output_names_with_port, **kwargs)

    def _conv_test(self, x_val, w, strides=None, padding="VALID", dilations=None, rtol=1e-07):
        if strides is None:
            strides = _STRIDE1x1
        if dilations is None:
            dilations = [1, 1, 1, 1]
        tf.reset_default_graph()
        kernel = tf.constant(w, dtype=tf.float32, name='k')
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        conv = tf.nn.conv2d(x, kernel, strides=strides, padding=padding, dilations=dilations)
        _ = tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=rtol)


if __name__ == '__main__':
    unittest_main()
