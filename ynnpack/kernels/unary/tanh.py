# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Definition of tanh kernels."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("tanh")
def tanh_fp32(a, x, output_offset, output_multiplier):
  # Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  # this point. This value is chosen as the first floating point number as of
  # which the interpolation returns 1.0f.
  vmax_x = 7.9807181358e00
  vmin_x = -7.9807181358e00

  # Polynomial coefficients. p(x) is an odd polynomial of x^2; q(x) is an even
  # polynomial of x^2.
  p = [
      2.0879957319e-08 * output_multiplier,
      2.5928236937e-05 * output_multiplier,
      3.8165270817e-03 * output_multiplier,
      1.3651061058e-01 * output_multiplier,
      1.0000001192e00 * output_multiplier,
  ]
  q = [
      1.0818473584e-06,
      3.8116899668e-04,
      2.7097105980e-02,
      4.6984484792e-01,
      1.0000000000e00,
  ]

  va = load(a)

  # Clamp the inputs to the interpolation range.
  vx = min(vmax_x, va)
  vx = max(vmin_x, vx)

  vx2 = vx * vx
  vp = eval_polynomial(vx2, p) * vx
  vq = eval_polynomial(vx2, q)

  return store(vp / vq + output_offset, x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("output_offset", Float(64)),
    Scalar("output_multiplier", Float(64)),
)
@operator_name("tanh")
def tanh_fp64(a, x, output_offset, output_multiplier):
  # tanh(x) ~= 1 for large x, and we can avoid Inf/Inf = NaN by limiting x such
  # that exp(x*2) does not overflow a double
  e2a = expm1(min(load(a) * 2, 300))
  result = e2a / (e2a + 2)
  return store(multiply_add(result, output_multiplier, output_offset), x)
