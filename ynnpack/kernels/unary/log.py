"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("input_multiplier", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("log")
def log_fp32(a, x, input_multiplier, output_multiplier):
  return store(output_multiplier * log(input_multiplier * load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("input_multiplier", Float(64)),
    Scalar("output_multiplier", Float(64)),
)
@operator_name("log")
def log_fp64(a, x, input_multiplier, output_multiplier):
  return store(output_multiplier * log(input_multiplier * load(a)), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("input_multiplier", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("log1p")
def log1p_fp32(a, x, input_multiplier, output_multiplier):
  return store(output_multiplier * log1p(input_multiplier * load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("input_multiplier", Float(64)),
    Scalar("output_multiplier", Float(64)),
)
@operator_name("log1p")
def log1p_fp64(a, x, input_multiplier, output_multiplier):
  return store(output_multiplier * log1p(input_multiplier * load(a)), x)
