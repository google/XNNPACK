"""Definition of exp kernel."""

import math

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def setexp_f32(x):
  # If `x` is an floating point value in the range [-127, 128], then
  # `(x + magic) << 23` will generate the floating point value corresponding
  # to `2^round(x)` (2^-127 and 2^128 will flush to zero and infinity,
  # respectively).
  vmagic = 8388735.0
  return reinterpret_cast(
      Float(32),
      logical_shift_left(reinterpret_cast(Int(32), x + vmagic), i32(23)),
  )


# Quick-and-dirty round to nearest, only works for floats in the range
# `[2^-22, 2^22)`.
def qd_round_f32(a):
  # If `x` is an floating point value in the range `[2^-22, 2^22)`, then
  # `(x + magic) - magic`` will generate the floating point value corresponding
  # to `round(x)`.
  vmagic = 12582912.0
  return (vmagic + a) - vmagic


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("exp")
def exp_fp32(a, x):
  # The monomial coefficients of the numerator polynomial (`valpha_0` = 1.0).
  valpha_1 = 4.1594290733e-01
  valpha_2 = 7.2068706155e-02
  valpha_3 = 5.5380910635e-03

  # The monomial coefficients of the denominator polynomial (`vbeta_0 = 1.0).
  vbeta_1 = -2.7720427513e-01
  vbeta_2 = 2.3986088112e-02

  va = load(a)
  # Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
  vz_prime = min(max(va * f32(math.log2(math.e)), -127.0), 128.0)

  # Decompose x * log2e into `z` (integer part) and `r` (remainder).
  vz = qd_round_f32(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = setexp_f32(vz)

  # Evaluate the numerator polynomial p(f).
  vp = multiply_add(vr, valpha_3, valpha_2)
  vp = multiply_add(vr, vp, valpha_1)
  vp = multiply_add(vr, vp, 1.0)

  # Evaluate the denominator polynomial q(r).
  vq = multiply_add(vr, vbeta_2, vbeta_1)
  vq = multiply_add(vr, vq, 1.0)

  # Divide the numerator by the denominator, obtaining 2^r.
  v2r = vp / vq

  # Compute 2^z * 2^r.
  vx = v2z * v2r

  return store(vx, x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("erf")
def erf_fp32(a, x):
  # Cap the inputs to this value as `erf(x)` will always be `+/-1.0f`
  # beyond this point. This value is chosen as the first floating point
  # number as of which the interpolation returns +/-1.0f.
  vmax_abs_x = 3.65137805943

  # The monomial coefficients of the numerator polynomial (`valpha_0` = 0.0).
  valpha_1 = 1.1283791149e00
  valpha_3 = 1.8942591284e-01
  valpha_5 = 5.2645591597e-02
  valpha_7 = 3.7304820991e-03
  valpha_9 = 2.8532683811e-04
  valpha_11 = 2.0742698573e-06

  # The monomial coefficients of the denominator polynomial (`vbeta_0 = 1.0).
  vbeta_2 = 5.0120705366e-01
  vbeta_4 = 1.1372791231e-01
  vbeta_6 = 1.4898274094e-02
  vbeta_8 = 1.1562824948e-03
  vbeta_10 = 3.8364178182e-05

  # Clamp the inputs to the interpolation range.
  vx = load(a)
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
  return store(vp / vq, x)
