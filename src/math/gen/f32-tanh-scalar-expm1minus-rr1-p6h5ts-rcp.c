// Auto-generated file. Do not edit!
//   Template: src/math/f32-tanh-scalar-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_rcp(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // The smallest z for which tanhf(-z) is saturated at -1.0f.
  const float vsat_cutoff = 0x1.205968p+3f;
  const float vminus_log2e = -0x1.715476p+0f;
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 63.5 mod 2**21.
  const float vmagic_bias = 0x1.8000FEp+22f;
  const float vln2 = 0x1.62E430p-1f;
  // Coefficients of polynomial approximation
  //   exp(-2t) - 1 ~ t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
  // on [-log(2)/4, log(2)/4]
  const float vc6 = 0x1.6B7338p-4f;
  const float vc5 = -0x1.12278Ep-2f;
  const float vc4 = 0x1.555716p-1f;
  const float vc3 = -0x1.5554B0p+0f;
  const float vc2 = 0x1.FFFFFEp+0f;
  const float vminus_two = -2.0f;
  const float vone = 1.0f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // General structure of the algorithm:
    //
    //           / -expm1(-2x) / (2 + expm1(-2x)) if x >= 0
    //   f(x) :=
    //           \ -f(-x) if x <= 0
    //
    // First we compute y := expm1(-2z) / (2 + expm1(-2z)) where z = abs(x),
    // then set its sign according to the sign of x: f(x) := sign(x) * abs(y).
    float vz = fabsf(vx);

    // The function saturates at -1 for large positive inputs: tanhf(-z) == -1.0f for z >= sat_cutoff ~= 9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vz = math_pmin_f32(vz, vsat_cutoff);

    // Compute reduced argument n := round(-z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 1 fractional bit,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**21, i.e. |z| <= 0x1.62E43p+20 = 1453635.0), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x).
    // Additionally, we fuse addition of the floating-point exponent bias (127) into the magic bias.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float vn = vz * vminus_log2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't cause underflow, i.e.
    // 0 <= z <= 9.010913, and -13 <= n <= 0 accordingly.
    const float vs = uint32_as_float(float_as_uint32(vn) << 23);

    // Subtract the large number back to get final n := round(-z / log(2), 1) as a floating-point number.
    vn -= vmagic_bias;

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    const float vt = vn * vln2 + vz;

    // Compute degree-6 polynomial approximation for exp(-2t) - 1 on [-log(2)/4, log(2)/4].
    //   P(t) = t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //        = t * p
    float vp = vc6 * vt + vc5;
    vp = vp * vt + vc4;
    vp = vp * vt + vc3;
    vp = vp * vt + vc2;
    vp = vp * vt + vminus_two;

    // Reconstruct the exp(-2z) - 1 value:
    //   exp(-2z) - 1 = s * (t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))) + 1) - 1
    //                = s * t * p + (s - 1)
    //                = (s - 1) + (t * s) * p
    const float vts = vt * vs;
    const float vsmo = vs - vone;
    const float vemo = vp * vts + vsmo;

    // Denominator of the tanh fraction: exp(-2z) + 1 = expm1(-2z) + 2
    const float vepo = vemo - vminus_two;

    // Compute reciprocal of denominator.
    const float vrepo = vone / vepo;

    // Reconstruct y = expm1(-2z) / (expm1(-2z) + 2)
    float vy = vemo * vrepo;

    // Reconstruct tanh(x) = copysign(y, x)
    vy = copysignf(vy, vx);

    *output++ = vy;
  }
}
