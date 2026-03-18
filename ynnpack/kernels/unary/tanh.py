# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This algorithm is a scalar implementation of the rational polynomial
# approximation of `tanh(x)` from
# `src/f32-vtanh/rational-9-8.c.in`.

"""Definition of tanh kernels."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("tanh")
def tanh_fp32(a, x):
  # Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  # this point. This value is chosen as the first floating point number as of
  # which the interpolation returns 1.0f.
  vmax_x = 7.9807181358e00
  vmin_x = -7.9807181358e00

  # The monomial coefficients of the numerator polynomial (odd).
  valpha_3 = 1.3412411511e-01
  valpha_5 = 3.5330520477e-03
  valpha_7 = 2.1235626264e-05
  valpha_9 = 1.4248920266e-08

  # The monomial coefficients of the denominator polynomial (even).
  vbeta_2 = 4.6745735407e-01
  vbeta_4 = 2.6018999517e-02
  vbeta_6 = 3.3472978976e-04
  vbeta_8 = 8.1365948290e-07

  va = load(a)

  # Clamp the inputs to the interpolation range.
  vx = min(vmax_x, va)
  vx = max(vmin_x, vx)

  # Since the polynomials are odd/even, we need x^2.
  vx2 = vx * vx

  # Evaluate the numerator polynomial p.
  # p = x * (1 + x^2 * (alpha_3 + x^2 * (alpha_5 + x^2 * (alpha_7 + x^2 * alpha_9))))
  vp = multiply_add(vx2, valpha_9, valpha_7)
  vp = multiply_add(vx2, vp, valpha_5)
  vp = multiply_add(vx2, vp, valpha_3)
  vp = multiply_add(vx2, vp, 1.0)
  vp = vx * vp

  # Evaluate the denominator polynomial q.
  # q = 1 + x^2 * (beta_2 + x^2 * (beta_4 + x^2 * (beta_6 + x^2 * beta_8)))
  vq = multiply_add(vx2, vbeta_8, vbeta_6)
  vq = multiply_add(vx2, vq, vbeta_4)
  vq = multiply_add(vx2, vq, vbeta_2)
  vq = multiply_add(vx2, vq, 1.0)

  # Divide the numerator by the denominator.
  vy = vp / vq

  return store(vy, x)
