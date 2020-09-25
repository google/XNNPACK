// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>

#include <fp16/bitcasts.h>


// Table of exp2(k / 2048) values decremented (as integer) by (k << 12), k = 0..2048
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_2048[2048];

void xnn_math_f32_expminus__scalar_lut2048_p1(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  const float vmagic_bias = 0x1.800000p23f;
  // The smallest x for which expf(x) is normalized.
  const float vdenorm_cutoff = -0x1.5D589Ep6f;
  const float vlog2e_x2048  = 0x1.715476p11f;
  // Last 18 bits are zeroes
  const float vminus_ln2_o2048_hi = -0x1.600000p-12f;
  const float vminus_ln2_o2048_lo = -0x1.7217F8p-19f;

  const float vc1 = 0x1.FFFFFEp-1f;

  const uint32_t vindex_mask = UINT32_C(0x7FF);

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // Compute reduced argument n := round(x * 2048 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x * 2048 / log(2)| <= 2**22, i.e.
    // |x| <= 0x1.62E43p+10 = 1419.5654296875), but that is acceptable, because inputs outside of [-87.336540, 0.0]
    // result in denormalized or underflown expf(x). We fixup the result for such inputs at the very end of the
    // algorithm.
    float vn = vx * vlog2e_x2048 + vmagic_bias;

    // Create a floating-point number s (scale) such that s := 2**(n / 2048) for such inputs that expf(x) is normalized,
    // i.e. -87.33642 <= x <= 0.0. As n has 11 fractional bits, we split s == 2**(n / 2048) = 2**e * 2**(n / 2048 - e),
    // where e := int(n / 2048). We create s in two steps:
    // 1. Fetch 2**(n / 2048 - e) = 2**(n % 2048) from the table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for -87.33642 <= x <= 0.0 (inputs for which expf(x) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Shift bits 11:19 into 23:31 (position of floating-point exponent).
    const uint32_t ve = fp32_to_bits(vn) << 12;

    // Use bits 0:11 bits of n, as integer, as an index for table lookup of l := 2**(n % 2048).
    const uint32_t vidx = fp32_to_bits(vn) & vindex_mask;
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float vs = fp32_from_bits(xnn_table_exp2minus_k_over_2048[vidx] + ve);

    // Subtract the large number back to get final n := round(x * 2048 / log(2)) as a floating-point number.
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2) / 2048.
    // Use Cody-Waite range reduction method (note the two constants representing log(2) / 2048) to improve accuracy.
    float vt = vn * vminus_ln2_o2048_hi + vx;
    vt = vn * vminus_ln2_o2048_lo + vt;

    // Compute degree-1 polynomial approxiatmion for exp(t) on [-log(2)/2048, log(2)/2048].
    const float vp = vt * vc1;

    // Reconstruct the final f value:
    //   f = s * (1 + t * c1)
    //     = s + s * (t * c1))
    //     = s + s * p
    float vf = vp * vs + vs;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vx < vdenorm_cutoff) {
      vf = 0.0f;
    }

    *output++ = vf;
  }
}
