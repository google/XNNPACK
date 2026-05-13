"""Definition of unary kernels."""

import math

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("round")
def round_fp32(a, x):
  return store(round(load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("round")
def round_fp64(a, x):
  return store(round(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("ceil")
def ceil_fp32(a, x):
  return store(ceil(load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("ceil")
def ceil_fp64(a, x):
  return store(ceil(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("floor")
def floor_fp32(a, x):
  return store(floor(load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("floor")
def floor_fp64(a, x):
  return store(floor(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("square")
def square_fp32(a, x):
  vx = load(a)
  return store(vx * vx, x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("square")
def square_fp64(a, x):
  vx = load(a)
  return store(vx * vx, x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("square_root")
def square_root_fp32(a, x):
  return store(sqrt(load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("square_root")
def square_root_fp64(a, x):
  return store(sqrt(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("reciprocal_square_root")
def reciprocal_square_root_fp32(a, x):
  return store(1.0 / sqrt(load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("reciprocal_square_root")
def reciprocal_square_root_fp64(a, x):
  return store(1.0 / sqrt(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("abs")
def abs_fp32(a, x):
  return store(abs(load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("abs")
def abs_fp64(a, x):
  return store(abs(load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("negate")
def negate_fp32(a, x):
  return store(-load(a), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("negate")
def negate_fp64(a, x):
  return store(-load(a), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("c0", Float(32)),
    Scalar("c1", Float(32)),
    Scalar("c2", Float(32)),
    Scalar("c3", Float(32)),
)
@operator_name("poly3")
def poly3_fp32(a, x, c0, c1, c2, c3):
  vx = load(a)
  vp = multiply_add(vx, c3, c2)
  vp = multiply_add(vx, vp, c1)
  vp = multiply_add(vx, vp, c0)
  return store(vp, x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("c0", Float(64)),
    Scalar("c1", Float(64)),
    Scalar("c2", Float(64)),
    Scalar("c3", Float(64)),
)
@operator_name("poly3")
def poly3_fp64(a, x, c0, c1, c2, c3):
  vx = load(a)
  vp = multiply_add(vx, c3, c2)
  vp = multiply_add(vx, vp, c1)
  vp = multiply_add(vx, vp, c0)
  return store(vp, x)
