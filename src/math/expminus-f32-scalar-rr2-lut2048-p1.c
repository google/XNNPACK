// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 2048) values decremented (as integer) by (k << 12), k = 0..2048
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_2048[2048];

void xnn_math_f32_expminus__scalar_rr2_lut2048_p1(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // Large number such that ulp(magic bias) == exp2(-11)
  const float vmagic_bias = 0x1.800000p12f;
  const float vlog2e  = 0x1.715476p0f;
  // Mask for the lowest 11 bits
  const uint32_t vindex_mask = UINT32_C(0x7FF);
  // Last 18 bits are zeroes
  const float vminus_ln2_hi = -0x1.600000p-1f;
  const float vminus_ln2_lo = -0x1.7217F8p-8f;
  // Coefficient of polynomial approximation
  //   exp(t) ~ 1 + t * c1
  // on [-log(2)/2048, log(2)/2048]
  const float vc1 = 0x1.FFFFFEp-1f;
  // The smallest x for which expf(x) is normalized.
  const float vdenorm_cutoff = -0x1.5D589Ep6f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // Compute reduced argument n := round(x / log(2), 11).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 11 fractional bits, then
    // subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**11, i.e. |x| <= 0x1.62E43p+10 = 1419.5654296875), but that is acceptable, because inputs x
    // outside of [-87.336544, 0] underflow expf(x). We fixup the result for such inputs at the very end of the
    // algorithm.
    float vn = vx * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s := 2**n for such inputs that expf(x) is normalized, i.e.
    // -87.336544 <= x <= 0. As n has 11 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s in
    // two steps:
    // 1. Fetch 2**frac(n) from the table using the 11 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -87.33642 <= x <= 0 (inputs for which expf(x) is normalized) we have -126 <= int(n) <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Shift bits 11:19 into 23:31 (position of floating-point exponent).
    const uint32_t ve = float_as_uint32(vn) << 12;

    // Use bits 0:11 of n, as integer, as an index for table lookup of l := 2**frac(n).
    const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float vs = uint32_as_float(xnn_table_exp2minus_k_over_2048[vidx] + ve);

    // Subtract the large number back to get final n := round(x / log(2), 11) as a floating-point number.
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2)
    // Use Cody-Waite range reduction method (note the two constants representing log(2)) to improve accuracy.
    float vt = vn * vminus_ln2_hi + vx;
    vt = vn * vminus_ln2_lo + vt;

    // Compute degree-1 polynomial approximation for exp(t) on [-log(2)/2048, log(2)/2048].
    //   P(t) = 1 + t * c1 = 1 + t * c1 = 1 + p
    const float vp = vt * vc1;

    // Reconstruct the exp(x) value:
    //   exp(x) = s * (1 + t * c1)
    //          = s * (1 + p)
    //          = s + s * p
    float vf = vp * vs + vs;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vx < vdenorm_cutoff) {
      vf = 0.0f;
    }

    *output++ = vf;
  }
}
