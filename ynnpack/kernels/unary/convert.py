"""Definition of unary convert kernels."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(16))
@buffer("x", Float(32))
@operator_name("convert")
def convert(a, x):
  return store(cast(Float(32), load(a)), x)
