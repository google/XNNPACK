"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("erf")
def erf_fp32(a, x, output_offset, output_multiplier, input_multiplier):
  va = load(a) * input_multiplier
  return store(multiply_add(erf(va), output_multiplier, output_offset), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("output_offset", Float(64)),
    Scalar("output_multiplier", Float(64)),
    Scalar("input_multiplier", Float(64)),
)
@operator_name("erf")
def erf_fp64(a, x, output_offset, output_multiplier, input_multiplier):
  va = load(a) * input_multiplier
  return store(multiply_add(erf(va), output_multiplier, output_offset), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("approx_erf")
def approx_erf_fp32(a, x, output_offset, output_multiplier, input_multiplier):
  va = load(a) * input_multiplier
  return store(
      multiply_add(approx_erf(va), output_multiplier, output_offset), x
  )
