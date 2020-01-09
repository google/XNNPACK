// Auto-generated file. Do not edit!
//   Template: src/f32-sigmoid/neon-lut2048-p1.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


extern XNN_INTERNAL const float xnn_table_exp2_k_over_2048[2048];

void xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8(
    size_t n,
    const float* x,
    float* y,
    const void* params)
{
  assert(n % sizeof(float) == 0);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p23f);
  // The largest z for which sigmoidf(-z) is normalized.
  // This number is also the largest z for which expf(-z) is normalized.
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(0x1.5D589Ep+6f);
  const float32x4_t vminus_log2e_x2048  = vmovq_n_f32(-0x1.715476p11f);
  const float32x4_t vln2_o2048 = vmovq_n_f32(0x1.62E43p-12f);
  const float32x4_t vone = vmovq_n_f32(1.0f);

  const float32x4_t vc1 = vmovq_n_f32(-0x1.FFFFFEp-1f);

  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0x7FF));

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(x); x += 4;
    const float32x4_t vx4567 = vld1q_f32(x); x += 4;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const float32x4_t vz0123 = vabsq_f32(vx0123);
    const float32x4_t vz4567 = vabsq_f32(vx4567);

    // Compute reduced argument n := round(-z * 2048 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|z * 2048 / log(2)| <= 2**22, i.e.
    // |z| <= 0x1.62E43p+10 = 1419.5654296875), but that is acceptable, because inputs x outside of
    // [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup the result
    // for such inputs at the very end of the algorithm.
    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vminus_log2e_x2048);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vminus_log2e_x2048);

    // Create a floating-point number s (scale) such that s := 2**(n / 2048) for such inputs that sigmoidf(-z) is
    // normalized, i.e. 0 <= z <= 87.33642. As n has 11 fractional bits, we split s == 2**(n / 2048) =
    // = 2**e * 2**(n / 2048 - e), where e := int(n / 2048). We create s in two steps:
    // 1. Fetch 2**(n / 2048 - e) = 2**(n % 2048) from the table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 87.33642 (inputs for which sigmoidf(-z) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Extract e from bits 11:19 of n and shift it into bits 23:31 (position of floating-point exponent).
    const int32x4_t ve0123 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn0123), vmovq_n_s32(INT32_C(0x7FF))), 12);
    const int32x4_t ve4567 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn4567), vmovq_n_s32(INT32_C(0x7FF))), 12);

    // Use bits 0:11 bits of n, as integer, as an index for table lookup of l := 2**(n % 2048).
    const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask));
    const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask));

    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    float32x2_t vl01 = vld1_dup_f32(&xnn_table_exp2_k_over_2048[(uint32_t) vidx01]);
    float32x2_t vl23 = vld1_dup_f32(&xnn_table_exp2_k_over_2048[(uint32_t) vidx23]);
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);
    float32x2_t vl45 = vld1_dup_f32(&xnn_table_exp2_k_over_2048[(uint32_t) vidx45]);
    float32x2_t vl67 = vld1_dup_f32(&xnn_table_exp2_k_over_2048[(uint32_t) vidx67]);

    vl01 = vld1_lane_f32(&xnn_table_exp2_k_over_2048[(uint32_t) (vidx01 >> 32)], vl01, 1);
    vl23 = vld1_lane_f32(&xnn_table_exp2_k_over_2048[(uint32_t) (vidx23 >> 32)], vl23, 1);
    const float32x4_t vl0123 = vcombine_f32(vl01, vl23);
    vl45 = vld1_lane_f32(&xnn_table_exp2_k_over_2048[(uint32_t) (vidx45 >> 32)], vl45, 1);
    vl67 = vld1_lane_f32(&xnn_table_exp2_k_over_2048[(uint32_t) (vidx67 >> 32)], vl67, 1);
    const float32x4_t vl4567 = vcombine_f32(vl45, vl67);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl0123), ve0123));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl4567), ve4567));

    // Subtract the large number back to get the final n := round(-z * 2048 / log(2)) as a floating-point number.
    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);

    // Compute reduced argument t := (z + n * log(2) / 2048). Note that -t = -z - n * log(2) / 2048.
    float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vln2_o2048);
    float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vln2_o2048);

    // Compute degree-1 polynomial approximation for exp(-t) on [-log(2)/2048, log(2)/2048]:
    //   P1(t) = 1 + t * c1
    const float32x4_t vp0123 = vmulq_f32(vt0123, vc1);
    const float32x4_t vp4567 = vmulq_f32(vt4567, vc1);

    // Reconstruct the exp(-z) value:
    //   y = s * (1 + t * c1)
    //     = s + s * (t * c1))
    //     = s + s * p
    const float32x4_t vy0123 = vfmaq_f32(vs0123, vs0123, vp0123);
    const float32x4_t vy4567 = vfmaq_f32(vs4567, vs4567, vp4567);

    // Denominator of the sigmoid fraction: 1.0 + exp(-z)
    const float32x4_t vd0123 = vaddq_f32(vy0123, vone);
    const float32x4_t vd4567 = vaddq_f32(vy4567, vone);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    float32x4_t vr0123 = vrecpeq_f32(vd0123);
    float32x4_t vr4567 = vrecpeq_f32(vd4567);

    vr0123 = vmulq_f32(vr0123, vrecpsq_f32(vr0123, vd0123));
    vr4567 = vmulq_f32(vr4567, vrecpsq_f32(vr4567, vd4567));

    vr0123 = vmulq_f32(vr0123, vrecpsq_f32(vr0123, vd0123));
    vr4567 = vmulq_f32(vr4567, vrecpsq_f32(vr4567, vd4567));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    float32x4_t vf0123 = vmulq_f32(vy0123, vr0123);
    float32x4_t vf4567 = vmulq_f32(vy4567, vr4567);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcagtq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcagtq_f32(vx4567, vdenorm_cutoff)));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    const uint32x4_t vm0123 = vcltq_f32(vx0123, vmovq_n_f32(0.0f));
    const uint32x4_t vm4567 = vcltq_f32(vx4567, vmovq_n_f32(0.0f));

    vf0123 = vbslq_f32(vm0123, vf0123, vsubq_f32(vone, vf0123));
    vf4567 = vbslq_f32(vm4567, vf4567, vsubq_f32(vone, vf4567));

    vst1q_f32(y, vf0123); y += 4;
    vst1q_f32(y, vf4567); y += 4;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(x); x += 4;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const float32x4_t vz = vabsq_f32(vx);

    // Compute reduced argument n := round(-z * 2048 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|z * 2048 / log(2)| <= 2**22, i.e.
    // |z| <= 0x1.62E43p+10 = 1419.5654296875), but that is acceptable, because inputs x outside of
    // [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup the result
    // for such inputs at the very end of the algorithm.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e_x2048);

    // Create a floating-point number s (scale) such that s := 2**(n / 2048) for such inputs that sigmoidf(-z) is
    // normalized, i.e. 0 <= z <= 87.33642. As n has 11 fractional bits, we split s == 2**(n / 2048) =
    // = 2**e * 2**(n / 2048 - e), where e := int(n / 2048). We create s in two steps:
    // 1. Fetch 2**(n / 2048 - e) = 2**(n % 2048) from exp2_k_over_2048_table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 87.33642 (inputs for which sigmoidf(-z) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Extract e from bits 11:19 of n and shift it into bits 23:31 (position of floating-point exponent).
    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x7FF))), 12);

    // Use bits 0:11 bits of n, as integer, as an index for table lookup of l := 2**(n % 2048).
    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2_k_over_2048[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2_k_over_2048[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2_k_over_2048[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2_k_over_2048[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);
    // Adjust exponent of the value l fetched from the exp2_k_over_2048_table to get the final s value.
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    // Subtract the large number back to get the final n := round(-z * 2048 / log(2)) as a floating-point number.
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := (z + n * log(2) / 2048). Note that -t = -z - n * log(2) / 2048.
    float32x4_t vt = vfmaq_f32(vz, vn, vln2_o2048);

    // Compute degree-1 polynomial approximation for exp(-t) on [-log(2)/2048, log(2)/2048]:
    //   P1(t) = 1 + t * c1
    const float32x4_t vp = vmulq_f32(vt, vc1);

    // Reconstruct the exp(-z) value:
    //   y = s * (1 + t * c1)
    //     = s + s * (t * c1))
    //     = s + s * p
    const float32x4_t vy = vfmaq_f32(vs, vs, vp);

    // Denominator of the sigmoid fraction: 1.0 + exp(-z)
    const float32x4_t vd = vaddq_f32(vy, vone);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    float32x4_t vr = vrecpeq_f32(vd);

    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    float32x4_t vf = vmulq_f32(vy, vr);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    vst1q_f32(y, vf); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const float32x4_t vx = vld1q_f32(x);

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const float32x4_t vz = vabsq_f32(vx);

    // Compute reduced argument n := round(-z * 2048 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|z * 2048 / log(2)| <= 2**22, i.e.
    // |z| <= 0x1.62E43p+10 = 1419.5654296875), but that is acceptable, because inputs x outside of
    // [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup the result
    // for such inputs at the very end of the algorithm.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e_x2048);

    // Create a floating-point number s (scale) such that s := 2**(n / 2048) for such inputs that sigmoidf(-z) is
    // normalized, i.e. 0 <= z <= 87.33642. As n has 11 fractional bits, we split s == 2**(n / 2048) =
    // = 2**e * 2**(n / 2048 - e), where e := int(n / 2048). We create s in two steps:
    // 1. Fetch 2**(n / 2048 - e) = 2**(n % 2048) from exp2_k_over_2048_table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 87.33642 (inputs for which sigmoidf(-z) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Extract e from bits 11:19 of n and shift it into bits 23:31 (position of floating-point exponent).
    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x7FF))), 12);

    // Use bits 0:11 bits of n, as integer, as an index for table lookup of l := 2**(n % 2048).
    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2_k_over_2048[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2_k_over_2048[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2_k_over_2048[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2_k_over_2048[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);
    // Adjust exponent of the value l fetched from the exp2_k_over_2048_table to get the final s value.
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    // Subtract the large number back to get the final n := round(-z * 2048 / log(2)) as a floating-point number.
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := (z + n * log(2) / 2048). Note that -t = -z - n * log(2) / 2048.
    float32x4_t vt = vfmaq_f32(vz, vn, vln2_o2048);

    // Compute degree-1 polynomial approximation for exp(-t) on [-log(2)/2048, log(2)/2048]:
    //   P1(t) = 1 + t * c1
    const float32x4_t vp = vmulq_f32(vt, vc1);

    // Reconstruct the exp(-z) value:
    //   y = s * (1 + t * c1)
    //     = s + s * (t * c1))
    //     = s + s * p
    const float32x4_t vy = vfmaq_f32(vs, vs, vp);

    // Denominator of the sigmoid fraction: 1.0 + exp(-z)
    const float32x4_t vd = vaddq_f32(vy, vone);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    float32x4_t vr = vrecpeq_f32(vd);

    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    float32x4_t vf = vmulq_f32(vy, vr);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    float32x2_t vf_lo = vget_low_f32(vf);
    if (n & (2 * sizeof(float))) {
      vst1_f32(y, vf_lo); y += 2;
      vf_lo = vget_high_f32(vf);
    }
    if (n & (1 * sizeof(float))) {
      vst1_lane_f32(y, vf_lo, 0);
    }
  }
}
