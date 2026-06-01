"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("exp")
def exp_fp32(a_buf, x_buf, output_multiplier, input_multiplier):
  return store(exp(load(a_buf) * input_multiplier) * output_multiplier, x_buf)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    # These are actually fp32, but we convert them to fp64 when loading them.
    Scalar("output_multiplier", Float(64)),
    Scalar("input_multiplier", Float(64)),
)
@operator_name("exp")
def exp_fp64(a_buf, x_buf, output_multiplier, input_multiplier):
  return store(exp(load(a_buf) * input_multiplier) * output_multiplier, x_buf)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("expm1")
def expm1_fp32(a_buf, x_buf, output_multiplier, input_multiplier):
  return store(expm1(load(a_buf) * input_multiplier) * output_multiplier, x_buf)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    # These are actually fp32, but we convert them to fp64 when loading them.
    Scalar("output_multiplier", Float(64)),
    Scalar("input_multiplier", Float(64)),
)
@operator_name("expm1")
def expm1_fp64(a_buf, x_buf, output_multiplier, input_multiplier):
  return store(expm1(load(a_buf) * input_multiplier) * output_multiplier, x_buf)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("approx_exp")
def approx_exp_fp32(a_buf, x_buf, output_multiplier, input_multiplier):
  return store(
      approx_exp(load(a_buf) * input_multiplier) * output_multiplier, x_buf
  )


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("approx_expm1")
def approx_expm1_fp32(a_buf, x_buf, output_multiplier, input_multiplier):
  return store(
      approx_expm1(load(a_buf) * input_multiplier) * output_multiplier, x_buf
  )
