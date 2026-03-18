"""Definition of unary kernels."""

import math

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("round")
def round_fp32(a, x):
  return store(round(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("ceil")
def ceil_fp32(a, x):
  return store(ceil(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("floor")
def floor_fp32(a, x):
  return store(floor(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("square_root")
def square_root_fp32(a, x):
  return store(sqrt(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("reciprocal_square_root")
def reciprocal_square_root_fp32(a, x):
  return store(1.0 / sqrt(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("abs")
def abs_fp32(a, x):
  return store(abs(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("negate")
def negate_fp32(a, x):
  return store(-load(a), x)
