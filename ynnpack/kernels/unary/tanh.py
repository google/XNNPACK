# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Definition of tanh kernels."""

import math

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
  # Polynomial coefficients for 2^r approx. These are the same coefficients as
  # exp_fp64
  p = [
      f64(3.430671987749682348e-06),
      f64(1.754214714900316551e-04),
      f64(3.930681642138933278e-03),
      f64(4.871246780757146344e-02),
      f64(3.331084219217221309e-01),
      f64(9.999999999999998890e-01),
  ]
  q = [
      f64(-7.126417699553038831e-06),
      f64(2.821496207245147419e-04),
      f64(-5.316864104706260294e-03),
      f64(5.804581129084830649e-02),
      f64(-3.600387586382234328e-01),
      f64(1.000000000000000000e00),
  ]

  log2_e = f64(math.log2(math.e))

  va = load(a)
  # tanh(x) is ~1 for large inputs, we can avoid numerical issues by just
  # clamping before that happens.
  va = max(min(va, f64(100)), f64(-100))

  v_prime = va * log2_e
  vz = round_small_fp64(v_prime)
  vr = v_prime - vz
  vr2 = vr * vr

  # exp(a) = 2^vz * vp_pos / vq_pos
  # exp(-a) = 2^-vz * vp_neg / vq_neg
  #
  # vx = (exp(a) - exp(-a)) / (exp(a) + exp(-a))
  # vx = (2^{2*vz} * vp_pos * vq_neg - vp_neg * vq_pos) /
  #      (2^{2*vz} * vp_pos * vq_neg + vp_neg * vq_pos)
  vp_even = eval_polynomial(vr2, [p[1], p[3], p[5]])
  vp_odd = eval_polynomial(vr2, [p[0], p[2], p[4]]) * vr
  vp_pos = vp_even + vp_odd
  vp_neg = vp_even - vp_odd

  vq_even = eval_polynomial(vr2, [q[1], q[3], q[5]])
  vq_odd = eval_polynomial(vr2, [q[0], q[2], q[4]]) * vr
  vq_pos = vq_even + vq_odd
  vq_neg = vq_even - vq_odd

  v2z2 = exp2_round(vz + vz)
  v_num_pos = v2z2 * vp_pos * vq_neg
  v_num_neg = vp_neg * vq_pos

  vx = (v_num_pos - v_num_neg) / (v_num_pos + v_num_neg)

  return store(vx * output_multiplier + output_offset, x)
