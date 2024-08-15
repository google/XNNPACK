// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/scalar-rr2-lut64-p2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/raddstoreexpminusmax.h"


// Note redefine as uint32[] to avoid redundant bitcasts.
extern XNN_INTERNAL const uint32_t xnn_table_exp2_k_over_64[64];

void xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u1(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float vlog2e  = 0x1.715476p0f;
  const float vmagic_bias = 0x1.800000p17f;
  const uint32_t vindex_mask = UINT32_C(0x3F);
  const float vminus_ln2_hi = -0x1.630000p-1f;
  const float vminus_ln2_lo = 0x1.BD0106p-13f;
  const float vc2 = 0x1.FFFF0Ap-2f;
  const float vdenorm_cutoff = -0x1.5D589Ep6f;

  const float vi_max = *max;

  float vacc = 0.0f;
  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    // Load 1 input at a time.
    const float vi = *input++;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float vx = vi - vi_max;

    // Compute reduced argument n := round(x * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x * 64 / log(2)| <= 2**22, i.e.
    // |x| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs outside of [-87.336540, 0.0]
    // result in denormalized or underflown expf(x). We fixup the result for such inputs at the very end of the
    // algorithm.
    float vn = vx * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s := 2**(n / 64) for such inputs that expf(x) is normalized,
    // i.e. -87.33642 <= x <= 0.0. As n has 6 fractional bits, we split s == 2**(n / 64) = 2**e * 2**(n / 64 - e), where
    // e := int(n / 64). We create s in two steps:
    // 1. Fetch 2**(n / 64 - e) = 2**(n % 64) from the table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for -87.33642 <= x <= 0.0 (inputs for which expf(x) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Extract e from bits 6:14 of n and shift it into bits 23:31 (position of floating-point exponent).
    const uint32_t ve = (float_as_uint32(vn) & UINT32_C(0xFFFFFFC0)) << 17;

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**(n % 64).
    const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float vs = uint32_as_float(xnn_table_exp2_k_over_64[vidx] + ve);

    // Subtract the large number back to get final n := round(x * 64 / log(2)) as a floating-point number.
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note the two constants representing log(2) / 64) to improve accuracy.
    float vt = vn * vminus_ln2_hi + vx;
    vt = vn * vminus_ln2_lo + vt;

    // Compute degree-2 polynomial approximation for exp(t) on [-log(2)/128, log(2)/128].
    float vp = vt * vc2;
    vp = vp * vt + vt;

    // Reconstruct the final f value:
    //   f = s * (1 + t * (1 + t * c2))
    //     = s * (1 + t + t * (t * c2))
    //     = s + s * (t + t * (t * c2))
    //     = s + s * p
    float vf = vp * vs + vs;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vx < vdenorm_cutoff) {
      vf = 0.0f;
    }

    // Store 1 output at a time.
    *output++ = vf;

    // Accumulate computed exponents.
    vacc += vf;
  }
  *sum = vacc;
}
