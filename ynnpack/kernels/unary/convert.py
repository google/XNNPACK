"""Definition of unary convert kernels."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(16))
@buffer("x", Float(32))
@operator_name("convert")
def convert_fp16_to_fp32(a, x):
  return store(cast(Float(32), load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(16))
@operator_name("convert")
def convert_fp32_to_fp16(a, x):
  return store(cast(Float(16), load(a)), x)
