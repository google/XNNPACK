# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This algorithm is a scalar implementation of the rational polynomial
# approximation of `sin(x)/cos(x)` from
# `src/f32-vsin/rational-5-4.c.in`.

"""Definition of sine/cosine kernels."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def sine_cosine_impl(a, x, is_cosine):
  # Some mathematical constants. We don't use pre-defined macros to ensure
  # that they are rounded exactly as we expect them to be.
  vpi = 3.1415927  # M_PI
  vpi_half = 1.5707964  # M_PI / 2
  v2pi_inv = 0.15915494  # 0.5 / M_PI

  # The following two values sum to 2*Pi with ~33 bits of accuracy. We use
  # them to accurately subtract integer multiples of 2*Pi from large inputs.
  v2pi_hi = 6.28125  # 2.0 * M_PI (first 11 bits of mantissa)
  v2pi_lo = 1.9353072e-3  # 2.0 * M_PI (remaining bits)

  # The monomial coefficients of the numerator polynomial (odd,
  # `valpha_1` = `vone`).
  valpha_3 = -1.3314664364e-01
  valpha_5 = 3.2340581529e-03

  # The monomial coefficients of the denominator polynomial (even,
  # `vbeta_0` = `vone`).
  vbeta_2 = 3.3519912511e-02
  vbeta_4 = 4.8770775902e-04

  va = load(a)

  # Map the inputs to the interpolation range [-pi, pi].
  vx_div_2pi = va * v2pi_inv
  vx_div_2pi = round(vx_div_2pi)
  vx = va - vx_div_2pi * v2pi_hi
  vx = vx - vx_div_2pi * v2pi_lo

  if is_cosine:
    # Use sine approximation for cosine.
    vx = vpi_half - vx

  # Fold the range to [-pi/2, pi/2]
  vx = min(vx, vpi - vx)
  vx = max(vx, -vpi - vx)
  vx = min(vx, vpi - vx)

  # Since the polynomials are odd/even, we need x^2.
  vx2 = vx * vx

  # Evaluate the numerator polynomial p.
  # p = x * (1 + x^2 * (alpha_3 + x^2 * alpha_5))
  vp = multiply_add(vx2, valpha_5, valpha_3)
  vp = multiply_add(vx2, vp, 1.0)
  vp = vx * vp

  # Evaluate the denominator polynomial q.
  # q = 1 + x^2 * (beta_2 + x^2 * beta_4)
  vq = multiply_add(vx2, vbeta_4, vbeta_2)
  vq = multiply_add(vx2, vq, 1.0)

  # Divide the numerator by the denominator.
  vy = vp / vq

  return store(vy, x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("sine")
def sine_fp32(a, x):
  return sine_cosine_impl(a, x, is_cosine=False)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("cosine")
def cosine_fp32(a, x):
  return sine_cosine_impl(a, x, is_cosine=True)
