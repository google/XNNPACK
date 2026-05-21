"""Definition of sigmoid kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("sigmoid")
def sigmoid_fp32(a, x):
  return store(1 / (1 + exp(-load(a))), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("sigmoid")
def sigmoid_fp64(a, x):
  return store(1 / (1 + exp(-load(a))), x)
