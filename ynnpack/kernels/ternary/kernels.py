"""Definition of ternary kernels."""

# pylint: disable=undefined-variable
# pylint: disable=g-unsafe-pickle-load
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
@const_buffer("c", Int(32))
@buffer("x", Float(32))
@operator_name("subtract_multiply")
def subtract_multiply_fp32_fp32_int32(a, b, c, x):
  return store(load(a) - load(b) * load(c), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply_add")
def multiply_add_fp32_fp32_fp32(a, b, c, x):
  return store(load(a) * load(b) + load(c), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@const_buffer("c", Float(64))
@buffer("x", Float(64))
@operator_name("multiply_add")
def multiply_add_fp64_fp64_fp64(a, b, c, x):
  return store(load(a) * load(b) + load(c), x)


@const_buffer("a", Int(8))
@const_buffer("b", Int(8))
@const_buffer("c", Int(32))
@buffer("x", Int(32))
@operator_name("multiply_add")
def multiply_add_int8_int8_int32(a, b, c, x):
  a_i16 = cast(Int(16), load(a))
  b_i16 = cast(Int(16), load(b))
  return store(cast(Int(32), a_i16 * b_i16) + load(c), x)


@const_buffer("a", BFloat(16))
@const_buffer("b", BFloat(16))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply_add")
def multiply_add_bf16_bf16_fp32(a, b, c, x):
  return store(cast(Float(32), load(a)) * cast(Float(32), load(b)) + load(c), x)


@const_buffer("a", Float(16))
@const_buffer("b", Float(16))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply_add")
def multiply_add_fp16_fp16_fp32(a, b, c, x):
  return store(cast(Float(32), load(a)) * cast(Float(32), load(b)) + load(c), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("clamp")
def clamp_fp32_fp32_fp32(a, b, c, x):
  return store(min(max(load(a), load(b)), load(c)), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@const_buffer("c", Float(64))
@buffer("x", Float(64))
@operator_name("clamp")
def clamp_fp64_fp64_fp64(a, b, c, x):
  return store(min(max(load(a), load(b)), load(c)), x)
