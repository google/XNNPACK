"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("erf")
def erf_fp32(a, x, output_offset, output_multiplier, input_multiplier):
  # Cap the inputs to this value as `erf(x)` will always be `+/-1.0f`
  # beyond this point. This value is chosen roughly as the first floating point
  # number as of which the interpolation returns +/-1.0f.
  vmax_abs_x = 3.8

  # The monomial coefficients of the numerator polynomial (`valpha_0` = 0.0).
  # We distribute the output_multiplier into the coefficients, these multiplies
  # are evaluated outside the loop over the output.
  valpha_1 = 1.1283791149e00 * output_multiplier
  valpha_3 = 1.8942591284e-01 * output_multiplier
  valpha_5 = 5.2645591597e-02 * output_multiplier
  valpha_7 = 3.7304820991e-03 * output_multiplier
  valpha_9 = 2.8532683811e-04 * output_multiplier
  valpha_11 = 2.0742698573e-06 * output_multiplier

  # The monomial coefficients of the denominator polynomial (`vbeta_0 = 1.0).
  vbeta_2 = 5.0120705366e-01
  vbeta_4 = 1.1372791231e-01
  vbeta_6 = 1.4898274094e-02
  vbeta_8 = 1.1562824948e-03
  vbeta_10 = 3.8364178182e-05

  # Clamp the inputs to the interpolation range.
  vx = load(a) * input_multiplier
  vx = min(vmax_abs_x, vx)
  vx = max(-vmax_abs_x, vx)

  # Since the polynomials are odd/even, we need x^2.
  vx2 = vx * vx

  # Evaluate the numerator polynomial p.
  vp = multiply_add(vx2, valpha_11, valpha_9)
  vp = multiply_add(vx2, vp, valpha_7)
  vp = multiply_add(vx2, vp, valpha_5)
  vp = multiply_add(vx2, vp, valpha_3)
  vp = multiply_add(vx2, vp, valpha_1)
  vp = vx * vp

  # Evaluate the denominator polynomial q.
  vq = multiply_add(vx2, vbeta_10, vbeta_8)
  vq = multiply_add(vx2, vq, vbeta_6)
  vq = multiply_add(vx2, vq, vbeta_4)
  vq = multiply_add(vx2, vq, vbeta_2)
  vq = multiply_add(vx2, vq, 1.0)

  # Divide the numerator by the denominator.
  return store((vp / vq) + output_offset, x)
