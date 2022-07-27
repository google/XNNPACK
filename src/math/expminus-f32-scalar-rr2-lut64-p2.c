// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 64) values decremented (as integer) by (k << 17), k = 0..63
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_64[64];

void xnn_math_f32_expminus__scalar_rr2_lut64_p2(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // Large number such that ulp(magic bias) == exp2(-6)
  const float vmagic_bias = 0x1.800000p17f;
  const float vlog2e  = 0x1.715476p0f;
  // Mask for the lowest 6 bits
  const uint32_t vindex_mask = UINT32_C(0x3F);
  // Last 13 bits are zeroes
  const float vminus_ln2_hi = -0x1.630000p-1f;
  const float vminus_ln2_lo =  0x1.BD0106p-13f;
  // Coefficient of polynomial approximation
  //   exp(t) ~ 1 + t * (1 + t * c2)
  // on [-log(2)/128, log(2)/128]
  const float vc2 = 0x1.FFFF0Ap-2f;
  // The smallest x for which expf(x) is normalized.
  const float vdenorm_cutoff = -0x1.5D589Ep6f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // Compute reduced argument n := round(x / log(2), 6).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 6 fractional bits, then
    // subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**16, i.e. |x| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs x
    // outside of [-87.336544, 0] underflow expf(x). We fixup the result for such inputs at the very end of the
    // algorithm.
    float vn = vx * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s := 2**n for such inputs that expf(x) is normalized, i.e.
    // -87.336544 <= x <= 0. As n has 6 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s in
    // two steps:
    // 1. Fetch 2**frac(n) from the table using the 6 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -87.33642 <= x <= 0 (inputs for which expf(x) is normalized) we have -126 <= int(n) <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Shift bits 6:14 into 23:31 (position of floating-point exponent).
    const uint32_t ve = float_as_uint32(vn) << 17;

    // Use bits 0:6 of n, as integer, as an index for table lookup of l := 2**frac(n).
    const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float vs = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx] + ve);

    // Subtract the large number back to get the final n := round(x / log(2), 6) as a floating-point number.
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2)
    // Use Cody-Waite range reduction method (note the two constants representing log(2)) to improve accuracy.
    float vt = vn * vminus_ln2_hi + vx;
    vt = vn * vminus_ln2_lo + vt;

    // Compute degree-2 polynomial approximation for exp(t) on [-log(2)/128, log(2)/128].
    //   P(t) = 1 + t * (1 + t * c2) = 1 + (t + t * (t * c2)) = 1 + p
    float vp = vt * vc2;
    vp = vp * vt + vt;

    // Reconstruct the exp(x) value:
    //   exp(x) = s * (1 + t * (1 + t * c2))
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
