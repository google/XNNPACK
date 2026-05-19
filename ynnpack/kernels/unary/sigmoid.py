"""Definition of sigmoid kernel."""

import math

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


@const_buffer('a', Float(32))
@buffer('x', Float(32))
@operator_name('sigmoid')
def sigmoid_fp32(a, x):
  # We want to compute 1/(1 + e^-x)
  # Polynomial coefficients for 2^r. These coefficients have been optimized for
  # the purposes of computing sigmoid.
  p = [
      1.3182145543e-02,
      1.3825711608e-01,
      6.0665678978e-01,
      1.0000000000e+00
  ]
  q = [
      7.5802938081e-03,
      -4.2016692460e-02,
      -8.6490385234e-02,
      1.0000000000e+00
  ]
  log2_e = math.log2(math.e)

  va = load(a) * -log2_e
  vz_prime = min(max(va, -127.0), 128.0)

  # Decompose x * log2e into `z` (integer part) and `r` (remainder).
  vz = round_small_fp32(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = exp2_round(vz)
  v2z = copynan(v2z, va)

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)

  # This is 1 / (1 + v2z * vp / vq), rearranged to avoid the extra division.
  vx = vq / multiply_add(v2z, vp, vq)

  return store(vx, x)


@const_buffer('a', Float(64))
@buffer('x', Float(64))
@operator_name('sigmoid')
def sigmoid_fp64(a, x):
  # We want to compute 1/(1 + e^-x)
  # Polynomial coefficients for 2^r
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

  va = load(a) * -log2_e
  vz_prime = min(max(va, f64(-1023.0)), f64(1024.0))

  # Decompose x * log2e into `z` (integer part) and `r` (remainder).
  vz = round_small_fp64(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = exp2_round(vz)
  v2z = copynan(v2z, va)

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)

  # This is 1 / (1 + v2z * vp / vq), rearranged to avoid the extra division.
  vx = vq / multiply_add(v2z, vp, vq)

  return store(vx, x)
