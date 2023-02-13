// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_math_f32_expm1minus__scalar_rr2_lut8_p4(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Large number such that ulp(magic bias) == exp2(-3)
  const float vmagic_bias = 0x1.800000p20f;
  const float vlog2e = 0x1.715476p+0f;
  // Mask for the lowest 3 bits
  const uint32_t vindex_mask = UINT32_C(0x7);
  // The largest x for which expm1f(x) is saturated at -1.0f.
  const float vsat_cutoff = -0x1.154246p+4f;
  // Last 8 bits are zeroes
  const float vminus_ln2_hi = -0x1.62E400p-1f;
  const float vminus_ln2_lo = -0x1.7F7D1Cp-20f;
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * (c3 + t * c4)))
  // on [-log(2)/16, log(2)/16]
  const float vc4 = 0x1.5558ECp-5f;
  const float vc3 = 0x1.555C20p-3f;
  const float vc2 = 0x1.000000p-1f;
  const float vone = 1.0f;

  for (; n != 0; n -= sizeof(float)) {
    float vx = *input++;

    // Compute reduced argument n := round(x / log(2), 3).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 3 fractional bits, then
    // subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**19, i.e. |x| <= 0x1.62E43p+18 = 363408.75), but that is acceptable, because inputs x are
    // restricted to [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float vn = vx * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s := 2**n for valid inputs, i.e. -17.328680 <= x <= 0.0. As n
    // has 3 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s in two steps:
    // 1. Fetch 2**frac(n) from the table using the 3 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -17.328680 <= x <= 0.0 we have -25 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -25.
    //
    // Shift bits 3:11 into 23:31 (position of floating-point exponent).
    const uint32_t ven = float_as_uint32(vn) << 20;

    // Use bits 0:3 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
    // Adjust exponent of the value l fetched from the table to get the final s value.
    float vs = uint32_as_float(xnn_table_exp2minus_k_over_8[vidx] + ven);

    // Subtract the large number back to get final n := round(x / log(2), 3).
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt = vn * vminus_ln2_hi + vx;
    vt = vn * vminus_ln2_lo + vt;

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we zero out s (scale) and t (reduced argument) for x <= sat_cutoff.
    if XNN_UNPREDICTABLE(vx <= vsat_cutoff) {
      vs = 0.0f;
      vt = 0.0f;
    }

    // Compute degree-4 polynomial approximation for exp(t) - 1 on [-log(2)/16, log(2)/16].
    //   P(t) = t * (1 + t * (c2 + t * (c3 + t * c4))) = t + t * (t * (c2 + t * (c3 + t * c4))) = t + t * p
    float vp = vc4 * vt + vc3;
    vp = vp * vt + vc2;
    vp *= vt;

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * (c3 + t * c4)))) - 1
    //              = (s - 1) + s * (t + t * p)
    //              = ((t * s) + (t * s) * p) + (s - 1)
    vt *= vs;
    const float vsm1 = vs - vone;
    vp = vp * vt + vt;
    const float vf = vp + vsm1;

    *output++ = vf;
  }
}
