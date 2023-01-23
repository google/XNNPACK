// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 16) values decremented (as integer) by (k << 19), k = 0..15
extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16];

void xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // The largest x for which expm1f(x) is saturated at -1.0f.
  const float32x4_t vsat_cutoff = vmovq_n_f32(-0x1.154246p+4f);
  // Large number such that ulp(magic bias) == exp2(-4)
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p19f);
  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  // Mask for the lowest 4 bits
  const int32x4_t vindex_mask = vmovq_n_s32(0xF);
  const float32x4_t vminus_ln2 = vmovq_n_f32(-0x1.62E430p-1f);
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * c3))
  // on [-log(2)/32, log(2)/32]
  const float32x4_t vc3 = vmovq_n_f32(0x1.55561Cp-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.0001ECp-1f);
  const float32x4_t vone = vmovq_n_f32(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(input); input += 4;

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we clip input at sat_cutoff, and leverage the fact that for our implementation
    // expm1f(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vx = vmaxq_f32(vx, vsat_cutoff);

    // Compute reduced argument n := round(x / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 4 fractional bits, then
    // subtracing the large number back. The addition is combined with multiplication by log2e into a single FMA
    // instruction. The trick with adding large number is valid only within certain bounds (|x / log(2)| <= 2**18, i.e.
    // |x| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x are restricted to [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    // Create a floating-point number s (scale) such that s := 2**n for valid inputs, i.e. -17.328680 <= x <= 0.0. As n
    // has 4 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s in two steps:
    // 1. Fetch 2**frac(n) from the table using the 4 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -17.328680 <= x <= 0.0 we have -25 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -25.
    //
    // Shift bits 4:12 into 23:31 (position of floating-point exponent).
    const int32x4_t ven = vshlq_n_s32(vreinterpretq_s32_f32(vn), 19);

    // Use bits 0:4 of n, as integer, as an index for table lookup of l := 2**frac(n).
    const uint64x2_t vidx = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask), 2));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    float32x2_t vl_hi = vld1_dup_f32((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    vl_lo = vld1_lane_f32((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)), vl_lo, 1);
    vl_hi = vld1_lane_f32((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)), vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ven));

    // Subtract the large number back to get final n := round(x / log(2), 4).
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2);

    // Compute degree-3 polynomial approximation for exp(t) - 1 on [-log(2)/32, log(2)/32].
    //   P(t) = t * (1 + t * (c2 + t * c3)) = t + t * (t * (c2 + t * c3)) = t + t * p
    float32x4_t vp = vfmaq_f32(vc2, vc3, vt);
    vp = vmulq_f32(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * c3))) - 1
    //              = (s - 1) + s * (t + t * p)
    //              = ((t * s) + (t * s) * p) + (s - 1)
    vt = vmulq_f32(vt, vs);
    const float32x4_t vsm1 = vsubq_f32(vs, vone);
    vp = vfmaq_f32(vt, vp, vt);
    const float32x4_t vf = vaddq_f32(vp, vsm1);

    vst1q_f32(output, vf); output += 4;
  }
}
