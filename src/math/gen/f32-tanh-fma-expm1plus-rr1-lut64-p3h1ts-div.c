// Auto-generated file. Do not edit!
//   Template: src/math/f32-tanh-scalar-expm1plus.c.in
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

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/math-stubs.h"


// Table of exp2(k / 64) values decremented (as integer) by (k << 17), k = 0..63
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_64[64];

void xnn_math_f32_tanh__fma_expm1plus_rr1_lut64_p3h1ts_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // The smallest z for which tanhf(z) is saturated at 1.0f.
  const float vsat_cutoff = 0x1.205968p+3f;
  const float vlog2e = 0x1.715476p+0f;
  // Large number such that ulp(magic bias) == exp2(-7)
  const float vmagic_bias = 0x1.800000p+16f;
  // Mask for the lowest 6 bits
  const uint32_t vindex_mask = UINT32_C(0x3F);
  const float vminus_ln2 = -0x1.62E430p-1f;
  // Coefficients of polynomial approximation
  //   exp(2t) - 1 ~ 2 * (t + t * (t * (c2 + t * c3)))
  // on [-log(2)/256, log(2)/256]
  const float vc3 = 0x1.55555Ep-1f;
  const float vc2 = 0x1.00001Ep+0f;
  const float vone = 1.0f;
  const float vtwo = 2.0f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x >= 0
    //   f(x) :=
    //           \ -f(-x) if x <= 0
    //
    // First we compute y := expm1(2z) / (2 + expm1(2z)) where z = abs(x),
    // then set its sign according to the sign of x: f(x) := sign(x) * abs(y).
    float vz = fabsf(vx);

    // The function saturates at -1 for large positive inputs: tanhf(-z) == -1.0f for z >= sat_cutoff ~= 9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vz = math_pmin_f32(vz, vsat_cutoff);

    // Compute reduced argument n := round(z / log(2), 7).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 7 fractional bits,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|z / log(2)| <= 2**15, i.e. |z| <= 0x1.62E43p+14 = 22713.046875), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x).
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float vn = fmaf(vz, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**(2n) for valid inputs, i.e. 0 <= z <= 9.010913. As
    // n has 7 fractional bits, we split s == 2**(2n) = 2**int(2n) * 2**frac(2n). We create s in two steps:
    // 1. Fetch 2**frac(2n) from the table using the 6 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their unbiased floating-point exponent is 0.
    // 2. Adjust fetched value by addition of int(2n) to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 9.010913 we have 0 <= int(n) <= 13, and thus the adjusted exponent is not
    //    greater than 13.
    //
    // Shift bits 6:14 into 23:31 (position of floating-point exponent).
    const uint32_t vb = float_as_uint32(vn);
    const uint32_t ve = vb << 17;

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    const uint32_t vidx = vb & vindex_mask;
    const uint32_t vl = xnn_table_exp2minus_k_over_64[vidx];

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float vs = uint32_as_float(vl + ve);

    // Subtract the large number back to get final n := round(z / log(2), 7) as a floating-point number.
    vn -= vmagic_bias;

    // Compute reduced argument t := z - n * log(2).
    const float vt = fmaf(vn, vminus_ln2, vz);

    // Compute degree-3 polynomial approximation for exp(2t) - 1 on [-log(2)/256, log(2)/256].
    //   P(t) = 2 * (t + t * (t * (c2 + t * c3)))
    //        = 2 * (t + t * p)
    float vp = fmaf(vc3, vt, vc2);
    vp *= vt;

    // Reconstruct the exp(2z) - 1 value:
    //   exp(2z) - 1 = s * (2 * (t + t * (t * (c2 + t * c3))) + 1) - 1
    //               = s * (2 * (t + t * p) + 1) - 1
    //               = (s - 1) + 2 * ((t * s) + (t * s) * p)
    const float vts = vt * vs;
    const float vsmo = vs - vone;
    vp = fmaf(vp, vts, vts);
    const float vemo = fmaf(vp, vtwo, vsmo);

    // Denominator of the tanh fraction: exp(2z) + 1 = expm1(2z) + 2
    const float vepo = vemo + vtwo;

    // Reconstruct y = expm1(2z) / (expm1(2z) + 2)
    float vy = vemo / vepo;

    // Reconstruct tanh(x) = copysign(y, x)
    vy = copysignf(vy, vx);

    *output++ = vy;
  }
}
