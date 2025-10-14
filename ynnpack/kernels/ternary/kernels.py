"""Definition of ternary kernels."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_fp32_fp32_fp32(a, b, c, x):
  return store(load(a) * load(b) * load(c), x)


@const_buffer("a", Int(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_int32_fp32_fp32(a, b, c, x):
  return store(cast(Float(32), load(a)) * load(b) * load(c), x)


@const_buffer("a", Int(32))
@const_buffer("b", Int(32))
@const_buffer("c", Int(32))
@buffer("x", Int(32))
@operator_name("subtract_multiply")
def subtract_multiply_int32_int32_int32(a, b, c, x):
  return store(load(a) - load(b) * load(c), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply_add")
def multiply_add_fp32_fp32_fp32(a, b, c, x):
  return store(load(a) * load(b) + load(c), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("clamp")
def clamp_fp32_fp32_fp32(a, b, c, x):
  return store(min(max(load(a), load(b)), load(c)), x)
