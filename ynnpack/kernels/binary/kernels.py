"""Definition of binary kernels."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("add")
def add_fp32(a, b, x):
  return store(load(a) + load(b), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("subtract")
def subtract_fp32(a, b, x):
  return store(load(a) - load(b), x)


@const_buffer("a", Float(32))
@const_buffer("b", BFloat(16))
@buffer("x", BFloat(16))
@operator_name("subtract")
def subtract_fp32_bf16(a, b, x):
  return store(cast(BFloat(16), load(a) - cast(Float(32), load(b))), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_fp32(a, b, x):
  return store(load(a) * load(b), x)


@const_buffer("a", Int(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_int32_fp32(a, b, x):
  return store(cast(Float(32), load(a)) * load(b), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("divide")
def divide_fp32(a, b, x):
  return store(load(a) / load(b), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("max")
def max_fp32(a, b, x):
  return store(max(load(a), load(b)), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("min")
def min_fp32(a, b, x):
  return store(min(load(a), load(b)), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("copysign")
def copysign_fp32(a, b, x):
  va = load(a)
  vb = load(b)
  # TODO (vksnk): we shouldn't need this cast if we add patterns for bit binary
  # ops which do reinterpret_cast themselves.
  mask = reinterpret_cast(Float(32), 0x7FFFFFFF)
  return store(select_bits(mask, va, vb), x)
