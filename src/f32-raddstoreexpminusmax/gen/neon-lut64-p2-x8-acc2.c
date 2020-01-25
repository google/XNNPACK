// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/raddstoreexpminusmax.h>


extern XNN_INTERNAL const float xnn_table_exp2_k_over_64[64];

void xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8_acc2(
    size_t elements,
    const float* input,
    float* output,
    float* sum,
    float max)
{
  assert(elements % sizeof(float) == 0);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p23f);
  // The smallest x for which expf(x) is normalized.
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(-0x1.5D589Ep6f);
  const float32x4_t vlog2e_x64  = vmovq_n_f32(0x1.715476p6f);
  // Last 13 bits are zeroes
  const float32x4_t vminus_ln2_o64_hi = vmovq_n_f32(-0x1.630000p-7f);
  const float32x4_t vminus_ln2_o64_lo = vmovq_n_f32(0x1.BD0106p-19f);

  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFF0Ap-2f);

  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0x3F));

  const float32x4_t vi_max = vdupq_n_f32(max);

  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  for (; elements >= 8 * sizeof(float); elements -= 8 * sizeof(float)) {
    // Load 8 (2x4) inputs at a time.
    const float32x4_t vi0123 = vld1q_f32(input); input += 4;
    const float32x4_t vi4567 = vld1q_f32(input); input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float32x4_t vx0123 = vsubq_f32(vi0123, vi_max);
    const float32x4_t vx4567 = vsubq_f32(vi4567, vi_max);

    // Compute reduced argument n := round(x * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x * 64 / log(2)| <= 2**22, i.e.
    // |x| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs outside of [-87.336540, 0.0]
    // result in denormalized or underflown expf(x). We fixup the result for such inputs at the very end of the
    // algorithm.
    float32x4_t vn0123 = vmlaq_f32(vmagic_bias, vx0123, vlog2e_x64);
    float32x4_t vn4567 = vmlaq_f32(vmagic_bias, vx4567, vlog2e_x64);

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
    const int32x4_t ve0123 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn0123), vmovq_n_s32(INT32_C(0x3F))), 17);
    const int32x4_t ve4567 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn4567), vmovq_n_s32(INT32_C(0x3F))), 17);

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**(n % 64).
    const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask));
    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask));
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);

    float32x2_t vl01 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx01]);
    float32x2_t vl23 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx23]);
    float32x2_t vl45 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx45]);
    float32x2_t vl67 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx67]);

    vl01 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx01 >> 32)], vl01, 1);
    vl23 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx23 >> 32)], vl23, 1);
    const float32x4_t vl0123 = vcombine_f32(vl01, vl23);
    vl45 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx45 >> 32)], vl45, 1);
    vl67 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx67 >> 32)], vl67, 1);
    const float32x4_t vl4567 = vcombine_f32(vl45, vl67);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl0123), ve0123));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl4567), ve4567));

    // Subtract the large number back to get final n := round(x * 64 / log(2)) as a floating-point number.
    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);

    // Compute reduced argument t := x - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note the two constants representing log(2) / 64) to improve accuracy.
    float32x4_t vt0123 = vmlaq_f32(vx0123, vn0123, vminus_ln2_o64_hi);
    float32x4_t vt4567 = vmlaq_f32(vx4567, vn4567, vminus_ln2_o64_hi);

    vt0123 = vmlaq_f32(vt0123, vn0123, vminus_ln2_o64_lo);
    vt4567 = vmlaq_f32(vt4567, vn4567, vminus_ln2_o64_lo);

    // Compute degree-2 polynomial approxiatmion for exp(t) on [-log(2)/128, log(2)/128].
    float32x4_t vp0123 = vmulq_f32(vt0123, vc2);
    float32x4_t vp4567 = vmulq_f32(vt4567, vc2);

    vp0123 = vmlaq_f32(vt0123, vt0123, vp0123);
    vp4567 = vmlaq_f32(vt4567, vt4567, vp4567);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (1 + t * c2))
    //     = s * (1 + t + t * (t * c2))
    //     = s + s * (t + t * (t * c2))
    //     = s + s * p
    float32x4_t vf0123 = vmlaq_f32(vs0123, vs0123, vp0123);
    float32x4_t vf4567 = vmlaq_f32(vs4567, vs4567, vp4567);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcltq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcltq_f32(vx4567, vdenorm_cutoff)));

    // Store 8 (2x4) outputs at a time.
    vst1q_f32(output, vf0123); output += 4;
    vst1q_f32(output, vf4567); output += 4;

    // Accumulate computed exponents.
    vacc0 = vaddq_f32(vacc0, vf0123);
    vacc0 = vaddq_f32(vacc0, vf4567);
  }
  // Add up all accumulators to vacc0
  vacc0 = vaddq_f32(vacc0, vacc1);

  float32x4_t vacc = vacc0;
  for (; elements >= 4 * sizeof(float); elements -= 4 * sizeof(float)) {
    // Load 4 inputs at a time.
    const float32x4_t vi = vld1q_f32(input); input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float32x4_t vx = vsubq_f32(vi, vi_max);

    // Compute reduced argument n := round(x * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x * 64 / log(2)| <= 2**22, i.e.
    // |x| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs outside of [-87.336540, 0.0]
    // result in denormalized or underflown expf(x). We fixup the result for such inputs at the very end of the
    // algorithm.
    float32x4_t vn = vmlaq_f32(vmagic_bias, vx, vlog2e_x64);

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
    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17);

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**(n % 64).
    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    // Subtract the large number back to get final n := round(x * 64 / log(2)) as a floating-point number.
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note the two constants representing log(2) / 64) to improve accuracy.
    float32x4_t vt = vmlaq_f32(vx, vn, vminus_ln2_o64_hi);
    vt = vmlaq_f32(vt, vn, vminus_ln2_o64_lo);

    // Compute degree-2 polynomial approxiatmion for exp(t) on [-log(2)/128, log(2)/128].
    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vmlaq_f32(vt, vt, vp);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (1 + t * c2))
    //     = s * (1 + t + t * (t * c2))
    //     = s + s * (t + t * (t * c2))
    //     = s + s * p
    float32x4_t vf = vmlaq_f32(vs, vs, vp);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));

    // Store 4 outputs at a time.
    vst1q_f32(output, vf); output += 4;

    // Accumulate computed exponents.
    vacc = vaddq_f32(vacc, vf);
  }
#if XNN_ARCH_ARM64
  float vacc_lo = vaddvq_f32(vacc);
#else
  float32x2_t vacc_lo = vadd_f32(vget_high_f32(vacc), vget_low_f32(vacc));
#endif
  if (elements != 0) {
    assert(elements >= 1 * sizeof(float));
    assert(elements <= 3 * sizeof(float));
    // Load 4 inputs at a time.
    const float32x4_t vi = vld1q_f32(input); input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float32x4_t vx = vsubq_f32(vi, vi_max);

    // Compute reduced argument n := round(x * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x * 64 / log(2)| <= 2**22, i.e.
    // |x| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs outside of [-87.336540, 0.0]
    // result in denormalized or underflown expf(x). We fixup the result for such inputs at the very end of the
    // algorithm.
    float32x4_t vn = vmlaq_f32(vmagic_bias, vx, vlog2e_x64);

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
    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17);

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**(n % 64).
    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    // Subtract the large number back to get final n := round(x * 64 / log(2)) as a floating-point number.
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note the two constants representing log(2) / 64) to improve accuracy.
    float32x4_t vt = vmlaq_f32(vx, vn, vminus_ln2_o64_hi);
    vt = vmlaq_f32(vt, vn, vminus_ln2_o64_lo);

    // Compute degree-2 polynomial approxiatmion for exp(t) on [-log(2)/128, log(2)/128].
    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vmlaq_f32(vt, vt, vp);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (1 + t * c2))
    //     = s * (1 + t + t * (t * c2))
    //     = s + s * (t + t * (t * c2))
    //     = s + s * p
    float32x4_t vf = vmlaq_f32(vs, vs, vp);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));

    float32x2_t vf_lo = vget_low_f32(vf);
    if (elements & (2 * sizeof(float))) {
      // Store 2 outputs at a time.
      vst1_f32(output, vf_lo); output += 2;

      // Accumulate 2 computed exponents.
      #if XNN_ARCH_ARM64
        vacc_lo += vaddv_f32(vf_lo);
      #else
        vacc_lo = vadd_f32(vacc_lo, vf_lo);
      #endif

      vf_lo = vget_high_f32(vf);
    }
    if (elements & (1 * sizeof(float))) {
      // Store 1 output at a time.
      vst1_lane_f32(output, vf_lo, 0);

      // Accumulate 1 computed exponent.
      #if XNN_ARCH_ARM64
        vacc_lo += vget_lane_f32(vf_lo, 0);
      #else
        vacc_lo = vadd_f32(vacc_lo, vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vf_lo), 32)));
      #endif
    }
  }
  // Reduce 4 elements in the SIMD register
#if XNN_ARCH_ARM64
  *sum = vacc_lo;
#else
  vst1_lane_f32(sum, vpadd_f32(vacc_lo, vacc_lo), 0);
#endif
}
