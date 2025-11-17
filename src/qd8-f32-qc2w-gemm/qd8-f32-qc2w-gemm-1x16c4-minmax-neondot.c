// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"

void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x16c4__neondot(
    size_t mr, size_t nc, size_t kc, const int8_t* restrict a, size_t a_stride,
    const void* restrict w, float* restrict c, size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* params,
    const float* row_sum,
    const struct xnn_qd8_quantization_params* quantization_params) XNN_OOB_READS {
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    //
    // Variable names:
    //  * vlh_zero_point_M: per-i zero points, int32_t, bcast to all lanes.
    //  * rh_col_sum_NNNN: per-j column sums, int32_t, one column per lane.
    //  * vacc_MxNNNN: per-ij pair accumularors, int32_t, one column per lane.
    //
    const int32x4_t vlh_zero_point_0 =
        vld1q_dup_s32(&quantization_params[0].zero_point);
    const int32x4_t rh_col_sum_0123 = vld1q_s32(w);
    w = (const int32_t*)w + 4;
    int32x4_t vacc_0x0123 = vmulq_s32(rh_col_sum_0123, vlh_zero_point_0);
    const int32x4_t rh_col_sum_4567 = vld1q_s32(w);
    w = (const int32_t*)w + 4;
    int32x4_t vacc_0x4567 = vmulq_s32(rh_col_sum_4567, vlh_zero_point_0);
    const int32x4_t rh_col_sum_89AB = vld1q_s32(w);
    w = (const int32_t*)w + 4;
    int32x4_t vacc_0x89AB = vmulq_s32(rh_col_sum_89AB, vlh_zero_point_0);
    const int32x4_t rh_col_sum_CDEF = vld1q_s32(w);
    w = (const int32_t*)w + 4;
    int32x4_t vacc_0xCDEF = vmulq_s32(rh_col_sum_CDEF, vlh_zero_point_0);

    // Initialize the bias with the scaled left-hand weight sums.
    //
    // Variable names:
    //  * lh_row_sum_M: per-i row sums, float, bcast to all lanes.
    //  * rh_zero_points_NNNN: per-j zero points, float, one column per lane.
    //  * scaled_lh_row_sum_MxNNNN: per-ij biases, float, one column per lane.
    //
    const float32x4_t lh_row_sum_0 =
        vld1q_dup_f32(&row_sum[0]);
    const float32x4_t rh_zero_points_0123 = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_zero_points_4567 = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_zero_points_89AB = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_zero_points_CDEF = vld1q_f32(w);
    w = (const float*)w + 4;

    const float32x4_t scaled_lh_row_sum_0x0123 =
        vmulq_f32(rh_zero_points_0123, lh_row_sum_0);
    const float32x4_t scaled_lh_row_sum_0x4567 =
        vmulq_f32(rh_zero_points_4567, lh_row_sum_0);
    const float32x4_t scaled_lh_row_sum_0x89AB =
        vmulq_f32(rh_zero_points_89AB, lh_row_sum_0);
    const float32x4_t scaled_lh_row_sum_0xCDEF =
        vmulq_f32(rh_zero_points_CDEF, lh_row_sum_0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;

    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 16 * sizeof(int8_t)) {
      // Load a single 1x16 block of activations.
      //
      // Variable names:
      //  * va_MxKK: partial left-hand row, int8_t, 8 entries for one row.
      //  * vb_NNNNxK: chunk of packed right-hand columns, 2bit packed, 4
      //               columns, 16 entries per column.
      //  * vb_NNNNxKKKK: chunk of unpacked right hand columns, int8_t, four
      //                  columns times four entries.
      const int8x16_t va_0x16 = vld1q_s8(a0);
      a0 += 16;

      // Load a 8x16 block of weights.
      const int8x16_t vb_0123x16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;
      const int8x16_t vb_4567x16 = vld1q_s8(w);
      w = (const int8_t *)w + 16;
      const int8x16_t vb_89ABx16 = vld1q_s8(w);
      w = (const int8_t *)w + 16;
      const int8x16_t vb_CDEFx16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;

      const int8x16_t v2 = vdupq_n_s8(2);
      const int8x16_t v3 = vdupq_n_s8(3);

      // First crumb.
      const int8x16_t vb_0123x0123 = vsubq_s8(veorq_s8(vandq_s8(vb_0123x16, v3), v2), v2);
      const int8x16_t vb_4567x0123 = vsubq_s8(veorq_s8(vandq_s8(vb_4567x16, v3), v2), v2);
      const int8x16_t vb_89ABx0123 = vsubq_s8(veorq_s8(vandq_s8(vb_89ABx16, v3), v2), v2);
      const int8x16_t vb_CDEFx0123 = vsubq_s8(veorq_s8(vandq_s8(vb_CDEFx16, v3), v2), v2);

      // Second crumb.
      const int8x16_t vb_0123x4567 = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_0123x16, 2), v3), v2), v2);
      const int8x16_t vb_4567x4567 = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_4567x16, 2), v3), v2), v2);
      const int8x16_t vb_89ABx4567 = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_89ABx16, 2), v3), v2), v2);
      const int8x16_t vb_CDEFx4567 = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_CDEFx16, 2), v3), v2), v2);

      // Third crumb.
      const int8x16_t vb_0123x89AB = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_0123x16, 4), v3), v2), v2);
      const int8x16_t vb_4567x89AB = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_4567x16, 4), v3), v2), v2);
      const int8x16_t vb_89ABx89AB = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_89ABx16, 4), v3), v2), v2);
      const int8x16_t vb_CDEFx89AB = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_CDEFx16, 4), v3), v2), v2);

      // Fourth crumb.
      const int8x16_t vb_0123xCDEF = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_0123x16, 6), v3), v2), v2);
      const int8x16_t vb_4567xCDEF = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_4567x16, 6), v3), v2), v2);
      const int8x16_t vb_89ABxCDEF = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_89ABx16, 6), v3), v2), v2);
      const int8x16_t vb_CDEFxCDEF = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_CDEFx16, 6), v3), v2), v2);

      // Multiply-accumulate: 1x16 * 16x16 --> 1x16.
      vacc_0x0123 =
        vdotq_lane_s32(vacc_0x0123, vb_0123x0123, vget_low_s8(va_0x16), 0);
      vacc_0x4567 =
          vdotq_lane_s32(vacc_0x4567, vb_4567x0123, vget_low_s8(va_0x16), 0);
      vacc_0x89AB =
          vdotq_lane_s32(vacc_0x89AB, vb_89ABx0123, vget_low_s8(va_0x16), 0);
      vacc_0xCDEF =
          vdotq_lane_s32(vacc_0xCDEF, vb_CDEFx0123, vget_low_s8(va_0x16), 0);

      vacc_0x0123 =
          vdotq_lane_s32(vacc_0x0123, vb_0123x4567, vget_low_s8(va_0x16), 1);
      vacc_0x4567 =
          vdotq_lane_s32(vacc_0x4567, vb_4567x4567, vget_low_s8(va_0x16), 1);
      vacc_0x89AB =
          vdotq_lane_s32(vacc_0x89AB, vb_89ABx4567, vget_low_s8(va_0x16), 1);
      vacc_0xCDEF =
          vdotq_lane_s32(vacc_0xCDEF, vb_CDEFx4567, vget_low_s8(va_0x16), 1);

      vacc_0x0123 =
          vdotq_lane_s32(vacc_0x0123, vb_0123x89AB, vget_high_s8(va_0x16), 0);
      vacc_0x4567 =
          vdotq_lane_s32(vacc_0x4567, vb_4567x89AB, vget_high_s8(va_0x16), 0);
      vacc_0x89AB =
          vdotq_lane_s32(vacc_0x89AB, vb_89ABx89AB, vget_high_s8(va_0x16), 0);
      vacc_0xCDEF =
          vdotq_lane_s32(vacc_0xCDEF, vb_CDEFx89AB, vget_high_s8(va_0x16), 0);

      vacc_0x0123 =
          vdotq_lane_s32(vacc_0x0123, vb_0123xCDEF, vget_high_s8(va_0x16), 1);
      vacc_0x4567 =
          vdotq_lane_s32(vacc_0x4567, vb_4567xCDEF, vget_high_s8(va_0x16), 1);
      vacc_0x89AB =
          vdotq_lane_s32(vacc_0x89AB, vb_89ABxCDEF, vget_high_s8(va_0x16), 1);
      vacc_0xCDEF =
          vdotq_lane_s32(vacc_0xCDEF, vb_CDEFxCDEF, vget_high_s8(va_0x16), 1);

      k -= 16 * sizeof(int8_t);
    }

    // Handle 4, 8, or 12 final positions of `k`.
    if XNN_UNLIKELY (k > 0) {
      // Load a 4x16 block of weights.
      int8x16_t vb_0123x16 = vld1q_s8(w);
      w = (const int8_t *)w + 16;
      int8x16_t vb_4567x16 = vld1q_s8(w);
      w = (const int8_t *)w + 16;
      int8x16_t vb_89ABx16 = vld1q_s8(w);
      w = (const int8_t *)w + 16;
      int8x16_t vb_CDEFx16 = vld1q_s8(w);
      w = (const int8_t *)w + 16;

      if XNN_UNLIKELY (k >= 8 * sizeof(int8_t)) {
        // Load a single 1x8 block of activations.
        const int8x8_t va_0x01234567 = vld1_s8(a0);
        a0 += 8;

        const int8x16_t v2 = vdupq_n_s8(2);
        const int8x16_t v3 = vdupq_n_s8(3);

        // First crumb.
        const int8x16_t vb_0123x0123 = vsubq_s8(veorq_s8(vandq_s8(vb_0123x16, v3), v2), v2);
        const int8x16_t vb_4567x0123 = vsubq_s8(veorq_s8(vandq_s8(vb_4567x16, v3), v2), v2);
        const int8x16_t vb_89ABx0123 = vsubq_s8(veorq_s8(vandq_s8(vb_89ABx16, v3), v2), v2);
        const int8x16_t vb_CDEFx0123 = vsubq_s8(veorq_s8(vandq_s8(vb_CDEFx16, v3), v2), v2);

        // Second crumb.
        const int8x16_t vb_0123x4567 = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_0123x16, 2), v3), v2), v2);
        const int8x16_t vb_4567x4567 = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_4567x16, 2), v3), v2), v2);
        const int8x16_t vb_89ABx4567 = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_89ABx16, 2), v3), v2), v2);
        const int8x16_t vb_CDEFx4567 = vsubq_s8(veorq_s8(vandq_s8(vshrq_n_s8(vb_CDEFx16, 2), v3), v2), v2);

        // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
        vacc_0x0123 =
            vdotq_lane_s32(vacc_0x0123, vb_0123x0123, va_0x01234567, 0);
        vacc_0x4567 =
            vdotq_lane_s32(vacc_0x4567, vb_4567x0123, va_0x01234567, 0);
        vacc_0x89AB =
            vdotq_lane_s32(vacc_0x89AB, vb_89ABx0123, va_0x01234567, 0);
        vacc_0xCDEF =
            vdotq_lane_s32(vacc_0xCDEF, vb_CDEFx0123, va_0x01234567, 0);

        vacc_0x0123 =
            vdotq_lane_s32(vacc_0x0123, vb_0123x4567, va_0x01234567, 1);
        vacc_0x4567 =
            vdotq_lane_s32(vacc_0x4567, vb_4567x4567, va_0x01234567, 1);
        vacc_0x89AB =
            vdotq_lane_s32(vacc_0x89AB, vb_89ABx4567, va_0x01234567, 1);
        vacc_0xCDEF =
            vdotq_lane_s32(vacc_0xCDEF, vb_CDEFx4567, va_0x01234567, 1);

        k -= 8 * sizeof(int8_t);

        if XNN_UNLIKELY (k > 0) {
          vb_0123x16 = vshrq_n_s8(vb_0123x16, 4);
          vb_4567x16 = vshrq_n_s8(vb_4567x16, 4);
          vb_89ABx16 = vshrq_n_s8(vb_89ABx16, 4);
          vb_CDEFx16 = vshrq_n_s8(vb_CDEFx16, 4);
        }
      }

      // Handle up to 4 final positions of `k`.
      if XNN_UNLIKELY (k >= 4 * sizeof(int8_t)) {
        // Load a single 1x4 block of activations.
        const int8x8_t va_0x0123 = vreinterpret_s8_u32(
            vld1_lane_u32((const uint32_t *)a0, vmov_n_u32(0), 0));
        a0 += 4;

        const int8x16_t v2 = vdupq_n_s8(2);
        const int8x16_t v3 = vdupq_n_s8(3);

        // First crumb.
        const int8x16_t vb_0123x0123 = vsubq_s8(veorq_s8(vandq_s8(vb_0123x16, v3), v2), v2);
        const int8x16_t vb_4567x0123 = vsubq_s8(veorq_s8(vandq_s8(vb_4567x16, v3), v2), v2);
        const int8x16_t vb_89ABx0123 = vsubq_s8(veorq_s8(vandq_s8(vb_89ABx16, v3), v2), v2);
        const int8x16_t vb_CDEFx0123 = vsubq_s8(veorq_s8(vandq_s8(vb_CDEFx16, v3), v2), v2);

        // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
        vacc_0x0123 = vdotq_lane_s32(vacc_0x0123, vb_0123x0123, va_0x0123, 0);
        vacc_0x4567 = vdotq_lane_s32(vacc_0x4567, vb_4567x0123, va_0x0123, 0);
        vacc_0x89AB = vdotq_lane_s32(vacc_0x89AB, vb_89ABx0123, va_0x0123, 0);
        vacc_0xCDEF = vdotq_lane_s32(vacc_0xCDEF, vb_CDEFx0123, va_0x0123, 0);

        k -= 4 * sizeof(int8_t);
      }
    }

    // Make sure there were no leftovers.
    assert(k == 0);

    // Convert the accumulated values to `float32`.
    float32x4_t vout_0x0123 = vcvtq_f32_s32(vacc_0x0123);
    float32x4_t vout_0x4567 = vcvtq_f32_s32(vacc_0x4567);
    float32x4_t vout_0x89AB = vcvtq_f32_s32(vacc_0x89AB);
    float32x4_t vout_0xCDEF = vcvtq_f32_s32(vacc_0xCDEF);

    // Subtract out the scaled left-hand row sums.
    vout_0x0123 = vsubq_f32(vout_0x0123, scaled_lh_row_sum_0x0123);
    vout_0x4567 = vsubq_f32(vout_0x4567, scaled_lh_row_sum_0x4567);
    vout_0x89AB = vsubq_f32(vout_0x89AB, scaled_lh_row_sum_0x89AB);
    vout_0xCDEF = vsubq_f32(vout_0xCDEF, scaled_lh_row_sum_0xCDEF);

    // Add the product of left/right-hand zero points and `kc`.
    const float32x4_t vscaled_lh_zero_point_0 =
        vdupq_n_f32((float)kc * quantization_params[0].zero_point);
    vout_0x0123 =
        vmlaq_f32(vout_0x0123, rh_zero_points_0123, vscaled_lh_zero_point_0);
    vout_0x4567 =
        vmlaq_f32(vout_0x4567, rh_zero_points_4567, vscaled_lh_zero_point_0);
    vout_0x89AB =
        vmlaq_f32(vout_0x89AB, rh_zero_points_89AB, vscaled_lh_zero_point_0);
    vout_0xCDEF =
        vmlaq_f32(vout_0xCDEF, rh_zero_points_CDEF, vscaled_lh_zero_point_0);

    // Load the left-hand scaling factor.
    const float32x4_t lhs_scale_0 =
        vld1q_dup_f32(&quantization_params[0].inv_scale);

    // Apply the left-hand scaling factor to the accumulated values.
    vout_0x0123 = vmulq_f32(vout_0x0123, lhs_scale_0);
    vout_0x4567 = vmulq_f32(vout_0x4567, lhs_scale_0);
    vout_0x89AB = vmulq_f32(vout_0x89AB, lhs_scale_0);
    vout_0xCDEF = vmulq_f32(vout_0xCDEF, lhs_scale_0);

    // Load the right-hand scaling factors.
    const float32x4_t rh_scale_0123 = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_scale_4567 = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_scale_89AB = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_scale_CDEF = vld1q_f32(w);
    w = (const float*)w + 4;

    // Load and apply the biases with the right-hand scaling factor.
    const float32x4_t vbias0123 = vld1q_f32(w);
    w = (const float*)w + 4;
    vout_0x0123 = vmlaq_f32(vbias0123, vout_0x0123, rh_scale_0123);
    const float32x4_t vbias4567 = vld1q_f32(w);
    w = (const float*)w + 4;
    vout_0x4567 = vmlaq_f32(vbias4567, vout_0x4567, rh_scale_4567);
    const float32x4_t vbias89AB = vld1q_f32(w);
    w = (const float*)w + 4;
    vout_0x89AB = vmlaq_f32(vbias89AB, vout_0x89AB, rh_scale_89AB);
    const float32x4_t vbiasCDEF = vld1q_f32(w);
    w = (const float*)w + 4;
    vout_0xCDEF = vmlaq_f32(vbiasCDEF, vout_0xCDEF, rh_scale_CDEF);

    // Apply the min/max scaling.
    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout_0x0123 = vmaxq_f32(vout_0x0123, voutput_min);
    vout_0x4567 = vmaxq_f32(vout_0x4567, voutput_min);
    vout_0x89AB = vmaxq_f32(vout_0x89AB, voutput_min);
    vout_0xCDEF = vmaxq_f32(vout_0xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout_0x0123 = vminq_f32(vout_0x0123, voutput_max);
    vout_0x4567 = vminq_f32(vout_0x4567, voutput_max);
    vout_0x89AB = vminq_f32(vout_0x89AB, voutput_max);
    vout_0xCDEF = vminq_f32(vout_0xCDEF, voutput_max);

    if XNN_LIKELY (nc >= 16) {
      vst1q_f32(c0, vout_0x0123);
      vst1q_f32(c0 + 4, vout_0x4567);
      vst1q_f32(c0 + 8, vout_0x89AB);
      vst1q_f32(c0 + 12, vout_0xCDEF);

      a0 = (const int8_t*)((uintptr_t)a0 - kc);

      c0 = (float*)((uintptr_t)c0 + cn_stride);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f32(c0, vout_0x0123);
        c0 += 4;
        vout_0x0123 = vout_0x89AB;
        vst1q_f32(c0, vout_0x4567);
        c0 += 4;
        vout_0x4567 = vout_0xCDEF;
      }
      if (nc & 4) {
        vst1q_f32(c0, vout_0x0123);
        c0 += 4;
        vout_0x0123 = vout_0x4567;
      }
      float32x2_t vout_0x01 = vget_low_f32(vout_0x0123);
      if (nc & 2) {
        vst1_f32(c0, vout_0x01);
        c0 += 2;
        vout_0x01 = vget_high_f32(vout_0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vout_0x01, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
