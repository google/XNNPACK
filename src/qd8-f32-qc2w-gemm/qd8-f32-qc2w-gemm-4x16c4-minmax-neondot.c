// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"

void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_4x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* params,
    const float* row_sum,
    const struct xnn_qd8_quantization_params* quantization_params) XNN_OOB_READS {
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8x16_t vmask = vmovq_n_s8(0x03);

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
    const int32x4_t vlh_zero_point_1 =
        vld1q_dup_s32(&quantization_params[1].zero_point);
    const int32x4_t vlh_zero_point_2 =
        vld1q_dup_s32(&quantization_params[2].zero_point);
    const int32x4_t vlh_zero_point_3 =
        vld1q_dup_s32(&quantization_params[3].zero_point);
    const int32x4_t rh_col_sum_0123 = vld1q_s32(w);
    w = (const int32_t*)w + 4;
    const int32x4_t rh_col_sum_4567 = vld1q_s32(w);
    w = (const int32_t*)w + 4;
    const int32x4_t rh_col_sum_89AB = vld1q_s32(w);
    w = (const int32_t*)w + 4;
    const int32x4_t rh_col_sum_CDEF = vld1q_s32(w);
    w = (const int32_t*)w + 4;

    int32x4_t vacc_0x0123 = vmulq_s32(rh_col_sum_0123, vlh_zero_point_0);
    int32x4_t vacc_0x4567 = vmulq_s32(rh_col_sum_4567, vlh_zero_point_0);
    int32x4_t vacc_0x89AB = vmulq_s32(rh_col_sum_89AB, vlh_zero_point_0);
    int32x4_t vacc_0xCDEF = vmulq_s32(rh_col_sum_CDEF, vlh_zero_point_0);

    int32x4_t vacc_1x0123 = vmulq_s32(rh_col_sum_0123, vlh_zero_point_1);
    int32x4_t vacc_1x4567 = vmulq_s32(rh_col_sum_4567, vlh_zero_point_1);
    int32x4_t vacc_1x89AB = vmulq_s32(rh_col_sum_89AB, vlh_zero_point_1);
    int32x4_t vacc_1xCDEF = vmulq_s32(rh_col_sum_CDEF, vlh_zero_point_1);

    int32x4_t vacc_2x0123 = vmulq_s32(rh_col_sum_0123, vlh_zero_point_2);
    int32x4_t vacc_2x4567 = vmulq_s32(rh_col_sum_4567, vlh_zero_point_2);
    int32x4_t vacc_2x89AB = vmulq_s32(rh_col_sum_89AB, vlh_zero_point_2);
    int32x4_t vacc_2xCDEF = vmulq_s32(rh_col_sum_CDEF, vlh_zero_point_2);

    int32x4_t vacc_3x0123 = vmulq_s32(rh_col_sum_0123, vlh_zero_point_3);
    int32x4_t vacc_3x4567 = vmulq_s32(rh_col_sum_4567, vlh_zero_point_3);
    int32x4_t vacc_3x89AB = vmulq_s32(rh_col_sum_89AB, vlh_zero_point_3);
    int32x4_t vacc_3xCDEF = vmulq_s32(rh_col_sum_CDEF, vlh_zero_point_3);

    // Initialize the bias with the scaled left-hand weight sums.
    //
    // Variable names:
    //  * lh_row_sum_M: per-i row sums, float, bcast to all lanes.
    //  * rh_zero_points_NNNN: per-j zero points, float, one column per lane.
    //  * scaled_lh_row_sum_MxNNNN: per-ij biases, float, one column per lane.
    //
    const float32x4_t lh_row_sum_0 = vld1q_dup_f32(&row_sum[0]);
    const float32x4_t lh_row_sum_1 = vld1q_dup_f32(&row_sum[1]);
    const float32x4_t lh_row_sum_2 = vld1q_dup_f32(&row_sum[2]);
    const float32x4_t lh_row_sum_3 = vld1q_dup_f32(&row_sum[3]);

    const float32x4_t rh_zero_points_0123 = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_zero_points_4567 = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_zero_points_89AB = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t rh_zero_points_CDEF = vld1q_f32(w);
    w = (const float*)w + 4;

    // Compensation for uint2 compute: -2 row_sum.
    const float32x4_t vtwo = vdupq_n_f32(2.0f);
    const float32x4_t biased_rh_zero_points_0123 = vaddq_f32(rh_zero_points_0123, vtwo);
    const float32x4_t biased_rh_zero_points_4567 = vaddq_f32(rh_zero_points_4567, vtwo);
    const float32x4_t biased_rh_zero_points_89AB = vaddq_f32(rh_zero_points_89AB, vtwo);
    const float32x4_t biased_rh_zero_points_CDEF = vaddq_f32(rh_zero_points_CDEF, vtwo);

    const float32x4_t scaled_lh_row_sum_0x0123 =
        vmulq_f32(biased_rh_zero_points_0123, lh_row_sum_0);
    const float32x4_t scaled_lh_row_sum_0x4567 =
        vmulq_f32(biased_rh_zero_points_4567, lh_row_sum_0);
    const float32x4_t scaled_lh_row_sum_0x89AB =
        vmulq_f32(biased_rh_zero_points_89AB, lh_row_sum_0);
    const float32x4_t scaled_lh_row_sum_0xCDEF =
        vmulq_f32(biased_rh_zero_points_CDEF, lh_row_sum_0);

    const float32x4_t scaled_lh_row_sum_1x0123 =
        vmulq_f32(biased_rh_zero_points_0123, lh_row_sum_1);
    const float32x4_t scaled_lh_row_sum_1x4567 =
        vmulq_f32(biased_rh_zero_points_4567, lh_row_sum_1);
    const float32x4_t scaled_lh_row_sum_1x89AB =
        vmulq_f32(biased_rh_zero_points_89AB, lh_row_sum_1);
    const float32x4_t scaled_lh_row_sum_1xCDEF =
        vmulq_f32(biased_rh_zero_points_CDEF, lh_row_sum_1);

    const float32x4_t scaled_lh_row_sum_2x0123 =
        vmulq_f32(biased_rh_zero_points_0123, lh_row_sum_2);
    const float32x4_t scaled_lh_row_sum_2x4567 =
        vmulq_f32(biased_rh_zero_points_4567, lh_row_sum_2);
    const float32x4_t scaled_lh_row_sum_2x89AB =
        vmulq_f32(biased_rh_zero_points_89AB, lh_row_sum_2);
    const float32x4_t scaled_lh_row_sum_2xCDEF =
        vmulq_f32(biased_rh_zero_points_CDEF, lh_row_sum_2);

    const float32x4_t scaled_lh_row_sum_3x0123 =
        vmulq_f32(biased_rh_zero_points_0123, lh_row_sum_3);
    const float32x4_t scaled_lh_row_sum_3x4567 =
        vmulq_f32(biased_rh_zero_points_4567, lh_row_sum_3);
    const float32x4_t scaled_lh_row_sum_3x89AB =
        vmulq_f32(biased_rh_zero_points_89AB, lh_row_sum_3);
    const float32x4_t scaled_lh_row_sum_3xCDEF =
        vmulq_f32(biased_rh_zero_points_CDEF, lh_row_sum_3);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;

    while (k >= 16 * sizeof(int8_t)) {
      // Load 4x16 activations.
      //
      // Variable names:
      //  * va_MxKK: left-hand row, int8_t, 16 entries for one row.
      //  * vb_NNNNxK: chunk of packed right-hand columns, 2bit packed, 4
      //               columns, 16 entries per column.
      //  * vb_NNNNxKKKK: chunk of unpacked right hand columns, int8_t, four
      //                  columns times four entries.
      const int8x16_t va_0x16 = vld1q_s8(a0);
      a0 += 16;
      const int8x16_t va_1x16 = vld1q_s8(a1);
      a1 += 16;
      const int8x16_t va_2x16 = vld1q_s8(a2);
      a2 += 16;
      const int8x16_t va_3x16 = vld1q_s8(a3);
      a3 += 16;

      // Load a 8x16 block of weights.
      const int8x16_t vb_0123x16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;
      const int8x16_t vb_4567x16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;
      const int8x16_t vb_89ABx16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;
      const int8x16_t vb_CDEFx16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;

      // First crumb.
      const int8x16_t vb_0123x0123 = vandq_s8(vb_0123x16, vmask);
      const int8x16_t vb_4567x0123 = vandq_s8(vb_4567x16, vmask);
      const int8x16_t vb_89ABx0123 = vandq_s8(vb_89ABx16, vmask);
      const int8x16_t vb_CDEFx0123 = vandq_s8(vb_CDEFx16, vmask);

      // Second crumb.
      const int8x16_t vb_0123x4567 = vandq_s8(vshrq_n_s8(vb_0123x16, 2), vmask);
      const int8x16_t vb_4567x4567 = vandq_s8(vshrq_n_s8(vb_4567x16, 2), vmask);
      const int8x16_t vb_89ABx4567 = vandq_s8(vshrq_n_s8(vb_89ABx16, 2), vmask);
      const int8x16_t vb_CDEFx4567 = vandq_s8(vshrq_n_s8(vb_CDEFx16, 2), vmask);

      // Third crumb.
      const int8x16_t vb_0123x89AB = vandq_s8(vshrq_n_s8(vb_0123x16, 4), vmask);
      const int8x16_t vb_4567x89AB = vandq_s8(vshrq_n_s8(vb_4567x16, 4), vmask);
      const int8x16_t vb_89ABx89AB = vandq_s8(vshrq_n_s8(vb_89ABx16, 4), vmask);
      const int8x16_t vb_CDEFx89AB = vandq_s8(vshrq_n_s8(vb_CDEFx16, 4), vmask);

      // Fourth crumb.
      const int8x16_t vb_0123xCDEF = vandq_s8(vshrq_n_s8(vb_0123x16, 6), vmask);
      const int8x16_t vb_4567xCDEF = vandq_s8(vshrq_n_s8(vb_4567x16, 6), vmask);
      const int8x16_t vb_89ABxCDEF = vandq_s8(vshrq_n_s8(vb_89ABx16, 6), vmask);
      const int8x16_t vb_CDEFxCDEF = vandq_s8(vshrq_n_s8(vb_CDEFx16, 6), vmask);

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

      vacc_1x0123 =
          vdotq_lane_s32(vacc_1x0123, vb_0123x0123, vget_low_s8(va_1x16), 0);
      vacc_1x4567 =
          vdotq_lane_s32(vacc_1x4567, vb_4567x0123, vget_low_s8(va_1x16), 0);
      vacc_1x89AB =
          vdotq_lane_s32(vacc_1x89AB, vb_89ABx0123, vget_low_s8(va_1x16), 0);
      vacc_1xCDEF =
          vdotq_lane_s32(vacc_1xCDEF, vb_CDEFx0123, vget_low_s8(va_1x16), 0);
      vacc_1x0123 =
          vdotq_lane_s32(vacc_1x0123, vb_0123x4567, vget_low_s8(va_1x16), 1);
      vacc_1x4567 =
          vdotq_lane_s32(vacc_1x4567, vb_4567x4567, vget_low_s8(va_1x16), 1);
      vacc_1x89AB =
          vdotq_lane_s32(vacc_1x89AB, vb_89ABx4567, vget_low_s8(va_1x16), 1);
      vacc_1xCDEF =
          vdotq_lane_s32(vacc_1xCDEF, vb_CDEFx4567, vget_low_s8(va_1x16), 1);
      vacc_1x0123 =
          vdotq_lane_s32(vacc_1x0123, vb_0123x89AB, vget_high_s8(va_1x16), 0);
      vacc_1x4567 =
          vdotq_lane_s32(vacc_1x4567, vb_4567x89AB, vget_high_s8(va_1x16), 0);
      vacc_1x89AB =
          vdotq_lane_s32(vacc_1x89AB, vb_89ABx89AB, vget_high_s8(va_1x16), 0);
      vacc_1xCDEF =
          vdotq_lane_s32(vacc_1xCDEF, vb_CDEFx89AB, vget_high_s8(va_1x16), 0);
      vacc_1x0123 =
          vdotq_lane_s32(vacc_1x0123, vb_0123xCDEF, vget_high_s8(va_1x16), 1);
      vacc_1x4567 =
          vdotq_lane_s32(vacc_1x4567, vb_4567xCDEF, vget_high_s8(va_1x16), 1);
      vacc_1x89AB =
          vdotq_lane_s32(vacc_1x89AB, vb_89ABxCDEF, vget_high_s8(va_1x16), 1);
      vacc_1xCDEF =
          vdotq_lane_s32(vacc_1xCDEF, vb_CDEFxCDEF, vget_high_s8(va_1x16), 1);

      vacc_2x0123 =
          vdotq_lane_s32(vacc_2x0123, vb_0123x0123, vget_low_s8(va_2x16), 0);
      vacc_2x4567 =
          vdotq_lane_s32(vacc_2x4567, vb_4567x0123, vget_low_s8(va_2x16), 0);
      vacc_2x89AB =
          vdotq_lane_s32(vacc_2x89AB, vb_89ABx0123, vget_low_s8(va_2x16), 0);
      vacc_2xCDEF =
          vdotq_lane_s32(vacc_2xCDEF, vb_CDEFx0123, vget_low_s8(va_2x16), 0);
      vacc_2x0123 =
          vdotq_lane_s32(vacc_2x0123, vb_0123x4567, vget_low_s8(va_2x16), 1);
      vacc_2x4567 =
          vdotq_lane_s32(vacc_2x4567, vb_4567x4567, vget_low_s8(va_2x16), 1);
      vacc_2x89AB =
          vdotq_lane_s32(vacc_2x89AB, vb_89ABx4567, vget_low_s8(va_2x16), 1);
      vacc_2xCDEF =
          vdotq_lane_s32(vacc_2xCDEF, vb_CDEFx4567, vget_low_s8(va_2x16), 1);
      vacc_2x0123 =
          vdotq_lane_s32(vacc_2x0123, vb_0123x89AB, vget_high_s8(va_2x16), 0);
      vacc_2x4567 =
          vdotq_lane_s32(vacc_2x4567, vb_4567x89AB, vget_high_s8(va_2x16), 0);
      vacc_2x89AB =
          vdotq_lane_s32(vacc_2x89AB, vb_89ABx89AB, vget_high_s8(va_2x16), 0);
      vacc_2xCDEF =
          vdotq_lane_s32(vacc_2xCDEF, vb_CDEFx89AB, vget_high_s8(va_2x16), 0);
      vacc_2x0123 =
          vdotq_lane_s32(vacc_2x0123, vb_0123xCDEF, vget_high_s8(va_2x16), 1);
      vacc_2x4567 =
          vdotq_lane_s32(vacc_2x4567, vb_4567xCDEF, vget_high_s8(va_2x16), 1);
      vacc_2x89AB =
          vdotq_lane_s32(vacc_2x89AB, vb_89ABxCDEF, vget_high_s8(va_2x16), 1);
      vacc_2xCDEF =
          vdotq_lane_s32(vacc_2xCDEF, vb_CDEFxCDEF, vget_high_s8(va_2x16), 1);

      vacc_3x0123 =
          vdotq_lane_s32(vacc_3x0123, vb_0123x0123, vget_low_s8(va_3x16), 0);
      vacc_3x4567 =
          vdotq_lane_s32(vacc_3x4567, vb_4567x0123, vget_low_s8(va_3x16), 0);
      vacc_3x89AB =
          vdotq_lane_s32(vacc_3x89AB, vb_89ABx0123, vget_low_s8(va_3x16), 0);
      vacc_3xCDEF =
          vdotq_lane_s32(vacc_3xCDEF, vb_CDEFx0123, vget_low_s8(va_3x16), 0);
      vacc_3x0123 =
          vdotq_lane_s32(vacc_3x0123, vb_0123x4567, vget_low_s8(va_3x16), 1);
      vacc_3x4567 =
          vdotq_lane_s32(vacc_3x4567, vb_4567x4567, vget_low_s8(va_3x16), 1);
      vacc_3x89AB =
          vdotq_lane_s32(vacc_3x89AB, vb_89ABx4567, vget_low_s8(va_3x16), 1);
      vacc_3xCDEF =
          vdotq_lane_s32(vacc_3xCDEF, vb_CDEFx4567, vget_low_s8(va_3x16), 1);
      vacc_3x0123 =
          vdotq_lane_s32(vacc_3x0123, vb_0123x89AB, vget_high_s8(va_3x16), 0);
      vacc_3x4567 =
          vdotq_lane_s32(vacc_3x4567, vb_4567x89AB, vget_high_s8(va_3x16), 0);
      vacc_3x89AB =
          vdotq_lane_s32(vacc_3x89AB, vb_89ABx89AB, vget_high_s8(va_3x16), 0);
      vacc_3xCDEF =
          vdotq_lane_s32(vacc_3xCDEF, vb_CDEFx89AB, vget_high_s8(va_3x16), 0);
      vacc_3x0123 =
          vdotq_lane_s32(vacc_3x0123, vb_0123xCDEF, vget_high_s8(va_3x16), 1);
      vacc_3x4567 =
          vdotq_lane_s32(vacc_3x4567, vb_4567xCDEF, vget_high_s8(va_3x16), 1);
      vacc_3x89AB =
          vdotq_lane_s32(vacc_3x89AB, vb_89ABxCDEF, vget_high_s8(va_3x16), 1);
      vacc_3xCDEF =
          vdotq_lane_s32(vacc_3xCDEF, vb_CDEFxCDEF, vget_high_s8(va_3x16), 1);

      k -= 16 * sizeof(int8_t);
    }

    // Handle up to 8 final positions of `k`.
    if XNN_UNLIKELY (k > 0) {
      // Load a 4x16 block of weights.
      int8x16_t vb_0123x16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;
      int8x16_t vb_4567x16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;
      int8x16_t vb_89ABx16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;
      int8x16_t vb_CDEFx16 = vld1q_s8(w);
      w = (const int8_t*)w + 16;

      if XNN_UNLIKELY (k >= 8 * sizeof(int8_t)) {
        // Load a 4x8 block of activations.
        const int8x8_t va_0x01234567 = vld1_s8(a0);
        a0 += 8;
        const int8x8_t va_1x01234567 = vld1_s8(a1);
        a1 += 8;
        const int8x8_t va_2x01234567 = vld1_s8(a2);
        a2 += 8;
        const int8x8_t va_3x01234567 = vld1_s8(a3);
        a3 += 8;

        // First crumb.
        const int8x16_t vb_0123x0123 = vandq_s8(vb_0123x16, vmask);
        const int8x16_t vb_4567x0123 = vandq_s8(vb_4567x16, vmask);
        const int8x16_t vb_89ABx0123 = vandq_s8(vb_89ABx16, vmask);
        const int8x16_t vb_CDEFx0123 = vandq_s8(vb_CDEFx16, vmask);

        // Second crumb.
        const int8x16_t vb_0123x4567 = vandq_s8(vshrq_n_s8(vb_0123x16, 2), vmask);
        const int8x16_t vb_4567x4567 = vandq_s8(vshrq_n_s8(vb_4567x16, 2), vmask);
        const int8x16_t vb_89ABx4567 = vandq_s8(vshrq_n_s8(vb_89ABx16, 2), vmask);
        const int8x16_t vb_CDEFx4567 = vandq_s8(vshrq_n_s8(vb_CDEFx16, 2), vmask);

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

        vacc_1x0123 =
            vdotq_lane_s32(vacc_1x0123, vb_0123x0123, va_1x01234567, 0);
        vacc_1x4567 =
            vdotq_lane_s32(vacc_1x4567, vb_4567x0123, va_1x01234567, 0);
        vacc_1x89AB =
            vdotq_lane_s32(vacc_1x89AB, vb_89ABx0123, va_1x01234567, 0);
        vacc_1xCDEF =
            vdotq_lane_s32(vacc_1xCDEF, vb_CDEFx0123, va_1x01234567, 0);
        vacc_1x0123 =
            vdotq_lane_s32(vacc_1x0123, vb_0123x4567, va_1x01234567, 1);
        vacc_1x4567 =
            vdotq_lane_s32(vacc_1x4567, vb_4567x4567, va_1x01234567, 1);
        vacc_1x89AB =
            vdotq_lane_s32(vacc_1x89AB, vb_89ABx4567, va_1x01234567, 1);
        vacc_1xCDEF =
            vdotq_lane_s32(vacc_1xCDEF, vb_CDEFx4567, va_1x01234567, 1);

        vacc_2x0123 =
            vdotq_lane_s32(vacc_2x0123, vb_0123x0123, va_2x01234567, 0);
        vacc_2x4567 =
            vdotq_lane_s32(vacc_2x4567, vb_4567x0123, va_2x01234567, 0);
        vacc_2x89AB =
            vdotq_lane_s32(vacc_2x89AB, vb_89ABx0123, va_2x01234567, 0);
        vacc_2xCDEF =
            vdotq_lane_s32(vacc_2xCDEF, vb_CDEFx0123, va_2x01234567, 0);
        vacc_2x0123 =
            vdotq_lane_s32(vacc_2x0123, vb_0123x4567, va_2x01234567, 1);
        vacc_2x4567 =
            vdotq_lane_s32(vacc_2x4567, vb_4567x4567, va_2x01234567, 1);
        vacc_2x89AB =
            vdotq_lane_s32(vacc_2x89AB, vb_89ABx4567, va_2x01234567, 1);
        vacc_2xCDEF =
            vdotq_lane_s32(vacc_2xCDEF, vb_CDEFx4567, va_2x01234567, 1);

        vacc_3x0123 =
            vdotq_lane_s32(vacc_3x0123, vb_0123x0123, va_3x01234567, 0);
        vacc_3x4567 =
            vdotq_lane_s32(vacc_3x4567, vb_4567x0123, va_3x01234567, 0);
        vacc_3x89AB =
            vdotq_lane_s32(vacc_3x89AB, vb_89ABx0123, va_3x01234567, 0);
        vacc_3xCDEF =
            vdotq_lane_s32(vacc_3xCDEF, vb_CDEFx0123, va_3x01234567, 0);
        vacc_3x0123 =
            vdotq_lane_s32(vacc_3x0123, vb_0123x4567, va_3x01234567, 1);
        vacc_3x4567 =
            vdotq_lane_s32(vacc_3x4567, vb_4567x4567, va_3x01234567, 1);
        vacc_3x89AB =
            vdotq_lane_s32(vacc_3x89AB, vb_89ABx4567, va_3x01234567, 1);
        vacc_3xCDEF =
            vdotq_lane_s32(vacc_3xCDEF, vb_CDEFx4567, va_3x01234567, 1);

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
        // Load a 4x4 block of activations.
        const int8x8_t va_0x0123 = vreinterpret_s8_u32(
            vld1_lane_u32((const uint32_t*)a0, vmov_n_u32(0), 0));
        a0 += 4;
        const int8x8_t va_1x0123 = vreinterpret_s8_u32(
            vld1_lane_u32((const uint32_t*)a1, vmov_n_u32(0), 0));
        a1 += 4;
        const int8x8_t va_2x0123 = vreinterpret_s8_u32(
            vld1_lane_u32((const uint32_t*)a2, vmov_n_u32(0), 0));
        a2 += 4;
        const int8x8_t va_3x0123 = vreinterpret_s8_u32(
            vld1_lane_u32((const uint32_t*)a3, vmov_n_u32(0), 0));
        a3 += 4;

        // First crumb.
        const int8x16_t vb_0123x0123 = vandq_s8(vb_0123x16, vmask);
        const int8x16_t vb_4567x0123 = vandq_s8(vb_4567x16, vmask);
        const int8x16_t vb_89ABx0123 = vandq_s8(vb_89ABx16, vmask);
        const int8x16_t vb_CDEFx0123 = vandq_s8(vb_CDEFx16, vmask);

        // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
        vacc_0x0123 = vdotq_lane_s32(vacc_0x0123, vb_0123x0123, va_0x0123, 0);
        vacc_0x4567 = vdotq_lane_s32(vacc_0x4567, vb_4567x0123, va_0x0123, 0);
        vacc_0x89AB = vdotq_lane_s32(vacc_0x89AB, vb_89ABx0123, va_0x0123, 0);
        vacc_0xCDEF = vdotq_lane_s32(vacc_0xCDEF, vb_CDEFx0123, va_0x0123, 0);

        vacc_1x0123 = vdotq_lane_s32(vacc_1x0123, vb_0123x0123, va_1x0123, 0);
        vacc_1x4567 = vdotq_lane_s32(vacc_1x4567, vb_4567x0123, va_1x0123, 0);
        vacc_1x89AB = vdotq_lane_s32(vacc_1x89AB, vb_89ABx0123, va_1x0123, 0);
        vacc_1xCDEF = vdotq_lane_s32(vacc_1xCDEF, vb_CDEFx0123, va_1x0123, 0);

        vacc_2x0123 = vdotq_lane_s32(vacc_2x0123, vb_0123x0123, va_2x0123, 0);
        vacc_2x4567 = vdotq_lane_s32(vacc_2x4567, vb_4567x0123, va_2x0123, 0);
        vacc_2x89AB = vdotq_lane_s32(vacc_2x89AB, vb_89ABx0123, va_2x0123, 0);
        vacc_2xCDEF = vdotq_lane_s32(vacc_2xCDEF, vb_CDEFx0123, va_2x0123, 0);

        vacc_3x0123 = vdotq_lane_s32(vacc_3x0123, vb_0123x0123, va_3x0123, 0);
        vacc_3x4567 = vdotq_lane_s32(vacc_3x4567, vb_4567x0123, va_3x0123, 0);
        vacc_3x89AB = vdotq_lane_s32(vacc_3x89AB, vb_89ABx0123, va_3x0123, 0);
        vacc_3xCDEF = vdotq_lane_s32(vacc_3xCDEF, vb_CDEFx0123, va_3x0123, 0);

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

    float32x4_t vout_1x0123 = vcvtq_f32_s32(vacc_1x0123);
    float32x4_t vout_1x4567 = vcvtq_f32_s32(vacc_1x4567);
    float32x4_t vout_1x89AB = vcvtq_f32_s32(vacc_1x89AB);
    float32x4_t vout_1xCDEF = vcvtq_f32_s32(vacc_1xCDEF);

    float32x4_t vout_2x0123 = vcvtq_f32_s32(vacc_2x0123);
    float32x4_t vout_2x4567 = vcvtq_f32_s32(vacc_2x4567);
    float32x4_t vout_2x89AB = vcvtq_f32_s32(vacc_2x89AB);
    float32x4_t vout_2xCDEF = vcvtq_f32_s32(vacc_2xCDEF);

    float32x4_t vout_3x0123 = vcvtq_f32_s32(vacc_3x0123);
    float32x4_t vout_3x4567 = vcvtq_f32_s32(vacc_3x4567);
    float32x4_t vout_3x89AB = vcvtq_f32_s32(vacc_3x89AB);
    float32x4_t vout_3xCDEF = vcvtq_f32_s32(vacc_3xCDEF);

    // Subtract out the scaled left-hand row sums.
    vout_0x0123 = vsubq_f32(vout_0x0123, scaled_lh_row_sum_0x0123);
    vout_0x4567 = vsubq_f32(vout_0x4567, scaled_lh_row_sum_0x4567);
    vout_0x89AB = vsubq_f32(vout_0x89AB, scaled_lh_row_sum_0x89AB);
    vout_0xCDEF = vsubq_f32(vout_0xCDEF, scaled_lh_row_sum_0xCDEF);

    vout_1x0123 = vsubq_f32(vout_1x0123, scaled_lh_row_sum_1x0123);
    vout_1x4567 = vsubq_f32(vout_1x4567, scaled_lh_row_sum_1x4567);
    vout_1x89AB = vsubq_f32(vout_1x89AB, scaled_lh_row_sum_1x89AB);
    vout_1xCDEF = vsubq_f32(vout_1xCDEF, scaled_lh_row_sum_1xCDEF);

    vout_2x0123 = vsubq_f32(vout_2x0123, scaled_lh_row_sum_2x0123);
    vout_2x4567 = vsubq_f32(vout_2x4567, scaled_lh_row_sum_2x4567);
    vout_2x89AB = vsubq_f32(vout_2x89AB, scaled_lh_row_sum_2x89AB);
    vout_2xCDEF = vsubq_f32(vout_2xCDEF, scaled_lh_row_sum_2xCDEF);

    vout_3x0123 = vsubq_f32(vout_3x0123, scaled_lh_row_sum_3x0123);
    vout_3x4567 = vsubq_f32(vout_3x4567, scaled_lh_row_sum_3x4567);
    vout_3x89AB = vsubq_f32(vout_3x89AB, scaled_lh_row_sum_3x89AB);
    vout_3xCDEF = vsubq_f32(vout_3xCDEF, scaled_lh_row_sum_3xCDEF);

    // Add the product of left/right-hand zero points and `kc`.
    const float32x4_t vscaled_lh_zero_point_0 =
        vdupq_n_f32((float)kc * quantization_params[0].zero_point);
    const float32x4_t vscaled_lh_zero_point_1 =
        vdupq_n_f32((float)kc * quantization_params[1].zero_point);
    const float32x4_t vscaled_lh_zero_point_2 =
        vdupq_n_f32((float)kc * quantization_params[2].zero_point);
    const float32x4_t vscaled_lh_zero_point_3 =
        vdupq_n_f32((float)kc * quantization_params[3].zero_point);

    vout_0x0123 =
        vmlaq_f32(vout_0x0123, rh_zero_points_0123, vscaled_lh_zero_point_0);
    vout_0x4567 =
        vmlaq_f32(vout_0x4567, rh_zero_points_4567, vscaled_lh_zero_point_0);
    vout_0x89AB =
        vmlaq_f32(vout_0x89AB, rh_zero_points_89AB, vscaled_lh_zero_point_0);
    vout_0xCDEF =
        vmlaq_f32(vout_0xCDEF, rh_zero_points_CDEF, vscaled_lh_zero_point_0);

    vout_1x0123 =
        vmlaq_f32(vout_1x0123, rh_zero_points_0123, vscaled_lh_zero_point_1);
    vout_1x4567 =
        vmlaq_f32(vout_1x4567, rh_zero_points_4567, vscaled_lh_zero_point_1);
    vout_1x89AB =
        vmlaq_f32(vout_1x89AB, rh_zero_points_89AB, vscaled_lh_zero_point_1);
    vout_1xCDEF =
        vmlaq_f32(vout_1xCDEF, rh_zero_points_CDEF, vscaled_lh_zero_point_1);

    vout_2x0123 =
        vmlaq_f32(vout_2x0123, rh_zero_points_0123, vscaled_lh_zero_point_2);
    vout_2x4567 =
        vmlaq_f32(vout_2x4567, rh_zero_points_4567, vscaled_lh_zero_point_2);
    vout_2x89AB =
        vmlaq_f32(vout_2x89AB, rh_zero_points_89AB, vscaled_lh_zero_point_2);
    vout_2xCDEF =
        vmlaq_f32(vout_2xCDEF, rh_zero_points_CDEF, vscaled_lh_zero_point_2);

    vout_3x0123 =
        vmlaq_f32(vout_3x0123, rh_zero_points_0123, vscaled_lh_zero_point_3);
    vout_3x4567 =
        vmlaq_f32(vout_3x4567, rh_zero_points_4567, vscaled_lh_zero_point_3);
    vout_3x89AB =
        vmlaq_f32(vout_3x89AB, rh_zero_points_89AB, vscaled_lh_zero_point_3);
    vout_3xCDEF =
        vmlaq_f32(vout_3xCDEF, rh_zero_points_CDEF, vscaled_lh_zero_point_3);

    // Load the left-hand scaling factor.
    const float32x4_t lhs_scale_0 =
        vld1q_dup_f32(&quantization_params[0].inv_scale);
    const float32x4_t lhs_scale_1 =
        vld1q_dup_f32(&quantization_params[1].inv_scale);
    const float32x4_t lhs_scale_2 =
        vld1q_dup_f32(&quantization_params[2].inv_scale);
    const float32x4_t lhs_scale_3 =
        vld1q_dup_f32(&quantization_params[3].inv_scale);

    // Apply the left-hand scaling factor to the accumulated values.
    vout_0x0123 = vmulq_f32(vout_0x0123, lhs_scale_0);
    vout_0x4567 = vmulq_f32(vout_0x4567, lhs_scale_0);
    vout_0x89AB = vmulq_f32(vout_0x89AB, lhs_scale_0);
    vout_0xCDEF = vmulq_f32(vout_0xCDEF, lhs_scale_0);

    vout_1x0123 = vmulq_f32(vout_1x0123, lhs_scale_1);
    vout_1x4567 = vmulq_f32(vout_1x4567, lhs_scale_1);
    vout_1x89AB = vmulq_f32(vout_1x89AB, lhs_scale_1);
    vout_1xCDEF = vmulq_f32(vout_1xCDEF, lhs_scale_1);

    vout_2x0123 = vmulq_f32(vout_2x0123, lhs_scale_2);
    vout_2x4567 = vmulq_f32(vout_2x4567, lhs_scale_2);
    vout_2x89AB = vmulq_f32(vout_2x89AB, lhs_scale_2);
    vout_2xCDEF = vmulq_f32(vout_2xCDEF, lhs_scale_2);

    vout_3x0123 = vmulq_f32(vout_3x0123, lhs_scale_3);
    vout_3x4567 = vmulq_f32(vout_3x4567, lhs_scale_3);
    vout_3x89AB = vmulq_f32(vout_3x89AB, lhs_scale_3);
    vout_3xCDEF = vmulq_f32(vout_3xCDEF, lhs_scale_3);

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
    const float32x4_t vbias4567 = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t vbias89AB = vld1q_f32(w);
    w = (const float*)w + 4;
    const float32x4_t vbiasCDEF = vld1q_f32(w);
    w = (const float*)w + 4;

    vout_0x0123 = vmlaq_f32(vbias0123, vout_0x0123, rh_scale_0123);
    vout_0x4567 = vmlaq_f32(vbias4567, vout_0x4567, rh_scale_4567);
    vout_0x89AB = vmlaq_f32(vbias89AB, vout_0x89AB, rh_scale_89AB);
    vout_0xCDEF = vmlaq_f32(vbiasCDEF, vout_0xCDEF, rh_scale_CDEF);

    vout_1x0123 = vmlaq_f32(vbias0123, vout_1x0123, rh_scale_0123);
    vout_1x4567 = vmlaq_f32(vbias4567, vout_1x4567, rh_scale_4567);
    vout_1x89AB = vmlaq_f32(vbias89AB, vout_1x89AB, rh_scale_89AB);
    vout_1xCDEF = vmlaq_f32(vbiasCDEF, vout_1xCDEF, rh_scale_CDEF);

    vout_2x0123 = vmlaq_f32(vbias0123, vout_2x0123, rh_scale_0123);
    vout_2x4567 = vmlaq_f32(vbias4567, vout_2x4567, rh_scale_4567);
    vout_2x89AB = vmlaq_f32(vbias89AB, vout_2x89AB, rh_scale_89AB);
    vout_2xCDEF = vmlaq_f32(vbiasCDEF, vout_2xCDEF, rh_scale_CDEF);

    vout_3x0123 = vmlaq_f32(vbias0123, vout_3x0123, rh_scale_0123);
    vout_3x4567 = vmlaq_f32(vbias4567, vout_3x4567, rh_scale_4567);
    vout_3x89AB = vmlaq_f32(vbias89AB, vout_3x89AB, rh_scale_89AB);
    vout_3xCDEF = vmlaq_f32(vbiasCDEF, vout_3xCDEF, rh_scale_CDEF);

    // Apply the min/max scaling.
    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout_0x0123 = vmaxq_f32(vout_0x0123, voutput_min);
    vout_0x4567 = vmaxq_f32(vout_0x4567, voutput_min);
    vout_0x89AB = vmaxq_f32(vout_0x89AB, voutput_min);
    vout_0xCDEF = vmaxq_f32(vout_0xCDEF, voutput_min);
    vout_1x0123 = vmaxq_f32(vout_1x0123, voutput_min);
    vout_1x4567 = vmaxq_f32(vout_1x4567, voutput_min);
    vout_1x89AB = vmaxq_f32(vout_1x89AB, voutput_min);
    vout_1xCDEF = vmaxq_f32(vout_1xCDEF, voutput_min);
    vout_2x0123 = vmaxq_f32(vout_2x0123, voutput_min);
    vout_2x4567 = vmaxq_f32(vout_2x4567, voutput_min);
    vout_2x89AB = vmaxq_f32(vout_2x89AB, voutput_min);
    vout_2xCDEF = vmaxq_f32(vout_2xCDEF, voutput_min);
    vout_3x0123 = vmaxq_f32(vout_3x0123, voutput_min);
    vout_3x4567 = vmaxq_f32(vout_3x4567, voutput_min);
    vout_3x89AB = vmaxq_f32(vout_3x89AB, voutput_min);
    vout_3xCDEF = vmaxq_f32(vout_3xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout_0x0123 = vminq_f32(vout_0x0123, voutput_max);
    vout_0x4567 = vminq_f32(vout_0x4567, voutput_max);
    vout_0x89AB = vminq_f32(vout_0x89AB, voutput_max);
    vout_0xCDEF = vminq_f32(vout_0xCDEF, voutput_max);
    vout_1x0123 = vminq_f32(vout_1x0123, voutput_max);
    vout_1x4567 = vminq_f32(vout_1x4567, voutput_max);
    vout_1x89AB = vminq_f32(vout_1x89AB, voutput_max);
    vout_1xCDEF = vminq_f32(vout_1xCDEF, voutput_max);
    vout_2x0123 = vminq_f32(vout_2x0123, voutput_max);
    vout_2x4567 = vminq_f32(vout_2x4567, voutput_max);
    vout_2x89AB = vminq_f32(vout_2x89AB, voutput_max);
    vout_2xCDEF = vminq_f32(vout_2xCDEF, voutput_max);
    vout_3x0123 = vminq_f32(vout_3x0123, voutput_max);
    vout_3x4567 = vminq_f32(vout_3x4567, voutput_max);
    vout_3x89AB = vminq_f32(vout_3x89AB, voutput_max);
    vout_3xCDEF = vminq_f32(vout_3xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout_0x0123);
      vst1q_f32(c0 + 4, vout_0x4567);
      vst1q_f32(c0 + 8, vout_0x89AB);
      vst1q_f32(c0 + 12, vout_0xCDEF);
      vst1q_f32(c1, vout_1x0123);
      vst1q_f32(c1 + 4, vout_1x4567);
      vst1q_f32(c1 + 8, vout_1x89AB);
      vst1q_f32(c1 + 12, vout_1xCDEF);
      vst1q_f32(c2, vout_2x0123);
      vst1q_f32(c2 + 4, vout_2x4567);
      vst1q_f32(c2 + 8, vout_2x89AB);
      vst1q_f32(c2 + 12, vout_2xCDEF);
      vst1q_f32(c3, vout_3x0123);
      vst1q_f32(c3 + 4, vout_3x4567);
      vst1q_f32(c3 + 8, vout_3x89AB);
      vst1q_f32(c3 + 12, vout_3xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f32(c0, vout_0x0123);
        c0 += 4;
        vout_0x0123 = vout_0x89AB;
        vst1q_f32(c1, vout_1x0123);
        c1 += 4;
        vout_1x0123 = vout_1x89AB;
        vst1q_f32(c2, vout_2x0123);
        c2 += 4;
        vout_2x0123 = vout_2x89AB;
        vst1q_f32(c3, vout_3x0123);
        c3 += 4;
        vout_3x0123 = vout_3x89AB;
        vst1q_f32(c0, vout_0x4567);
        c0 += 4;
        vout_0x4567 = vout_0xCDEF;
        vst1q_f32(c1, vout_1x4567);
        c1 += 4;
        vout_1x4567 = vout_1xCDEF;
        vst1q_f32(c2, vout_2x4567);
        c2 += 4;
        vout_2x4567 = vout_2xCDEF;
        vst1q_f32(c3, vout_3x4567);
        c3 += 4;
        vout_3x4567 = vout_3xCDEF;
      }
      if (nc & 4) {
        vst1q_f32(c0, vout_0x0123);
        c0 += 4;
        vout_0x0123 = vout_0x4567;
        vst1q_f32(c1, vout_1x0123);
        c1 += 4;
        vout_1x0123 = vout_1x4567;
        vst1q_f32(c2, vout_2x0123);
        c2 += 4;
        vout_2x0123 = vout_2x4567;
        vst1q_f32(c3, vout_3x0123);
        c3 += 4;
        vout_3x0123 = vout_3x4567;
      }
      float32x2_t vout_0x01 = vget_low_f32(vout_0x0123);
      float32x2_t vout_1x01 = vget_low_f32(vout_1x0123);
      float32x2_t vout_2x01 = vget_low_f32(vout_2x0123);
      float32x2_t vout_3x01 = vget_low_f32(vout_3x0123);
      if (nc & 2) {
        vst1_f32(c0, vout_0x01);
        c0 += 2;
        vst1_f32(c1, vout_1x01);
        c1 += 2;
        vst1_f32(c2, vout_2x01);
        c2 += 2;
        vst1_f32(c3, vout_3x01);
        c3 += 2;
        vout_0x01 = vget_high_f32(vout_0x0123);
        vout_1x01 = vget_high_f32(vout_1x0123);
        vout_2x01 = vget_high_f32(vout_2x0123);
        vout_3x01 = vget_high_f32(vout_3x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vout_0x01, 0);
        vst1_lane_f32(c1, vout_1x01, 0);
        vst1_lane_f32(c2, vout_2x01, 0);
        vst1_lane_f32(c3, vout_3x01, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
