// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_2x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const float* row_sum,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
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
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const int8x16_t vmask = vmovq_n_s8(INT8_C(0x03));
  // Loop over groups of 16 columns.
  do {
    // Initialize the bias with the scaled left-hand weight sums.
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vinput_zero_point0 = vld1q_dup_s32(&quantization_params[0].zero_point);
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point0);
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point0);
    int32x4_t vacc0x89AB = vmulq_s32(vksum89AB, vinput_zero_point0);
    int32x4_t vacc0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point0);
    const int32x4_t vinput_zero_point1 = vld1q_dup_s32(&quantization_params[1].zero_point);
    int32x4_t vacc1x0123 = vmulq_s32(vksum0123, vinput_zero_point1);
    int32x4_t vacc1x4567 = vmulq_s32(vksum4567, vinput_zero_point1);
    int32x4_t vacc1x89AB = vmulq_s32(vksum89AB, vinput_zero_point1);
    int32x4_t vacc1xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point1);
    // TODO: move kernel zero point after weights
    const void* kzp = w;
    w = (const float*)w + 16;

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;

    // 4x partial unrolled loop to load 16 bytes at a time.
    while (k >= 16 * sizeof(int8_t)) {
      // Load a 2x16 block of activations.
      const int8x16_t va_0x16 = vld1q_s8(a0); a0 += 16;
      const int8x16_t va_1x16 = vld1q_s8(a1); a1 += 16;

      // Load a 16x16 block of weights.
      const int8x16_t vb0123x16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89ABx16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDEFx16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      // First crumb.
      const int8x16_t vb0123x0123 = vandq_s8(vb0123x16, vmask);
      const int8x16_t vb4567x0123 = vandq_s8(vb4567x16, vmask);
      const int8x16_t vb89ABx0123 = vandq_s8(vb89ABx16, vmask);
      const int8x16_t vbCDEFx0123 = vandq_s8(vbCDEFx16, vmask);
      // Second crumb.
      const int8x16_t vb0123x4567 = vandq_s8(vshrq_n_s8(vb0123x16, 2), vmask);
      const int8x16_t vb4567x4567 = vandq_s8(vshrq_n_s8(vb4567x16, 2), vmask);
      const int8x16_t vb89ABx4567 = vandq_s8(vshrq_n_s8(vb89ABx16, 2), vmask);
      const int8x16_t vbCDEFx4567 = vandq_s8(vshrq_n_s8(vbCDEFx16, 2), vmask);
      // Third crumb.
      const int8x16_t vb0123x89AB = vandq_s8(vshrq_n_s8(vb0123x16, 4), vmask);
      const int8x16_t vb4567x89AB = vandq_s8(vshrq_n_s8(vb4567x16, 4), vmask);
      const int8x16_t vb89ABx89AB = vandq_s8(vshrq_n_s8(vb89ABx16, 4), vmask);
      const int8x16_t vbCDEFx89AB = vandq_s8(vshrq_n_s8(vbCDEFx16, 4), vmask);
      // Fourth crumb.
      const int8x16_t vb0123xCDEF = vandq_s8(vshrq_n_s8(vb0123x16, 6), vmask);
      const int8x16_t vb4567xCDEF = vandq_s8(vshrq_n_s8(vb4567x16, 6), vmask);
      const int8x16_t vb89ABxCDEF = vandq_s8(vshrq_n_s8(vb89ABx16, 6), vmask);
      const int8x16_t vbCDEFxCDEF = vandq_s8(vshrq_n_s8(vbCDEFx16, 6), vmask);

      // Multiply-accumulate: 2x16 * 16x16 --> 2x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, vget_low_s8(va_0x16), 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x0123, vget_low_s8(va_0x16), 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb89ABx0123, vget_low_s8(va_0x16), 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vbCDEFx0123, vget_low_s8(va_0x16), 0);

      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x4567, vget_low_s8(va_0x16), 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, vget_low_s8(va_0x16), 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb89ABx4567, vget_low_s8(va_0x16), 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vbCDEFx4567, vget_low_s8(va_0x16), 1);

      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x89AB, vget_high_s8(va_0x16), 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x89AB, vget_high_s8(va_0x16), 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb89ABx89AB, vget_high_s8(va_0x16), 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vbCDEFx89AB, vget_high_s8(va_0x16), 0);

      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123xCDEF, vget_high_s8(va_0x16), 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567xCDEF, vget_high_s8(va_0x16), 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb89ABxCDEF, vget_high_s8(va_0x16), 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vbCDEFxCDEF, vget_high_s8(va_0x16), 1);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, vget_low_s8(va_1x16), 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x0123, vget_low_s8(va_1x16), 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb89ABx0123, vget_low_s8(va_1x16), 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vbCDEFx0123, vget_low_s8(va_1x16), 0);

      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x4567, vget_low_s8(va_1x16), 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, vget_low_s8(va_1x16), 1);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb89ABx4567, vget_low_s8(va_1x16), 1);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vbCDEFx4567, vget_low_s8(va_1x16), 1);

      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x89AB, vget_high_s8(va_1x16), 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x89AB, vget_high_s8(va_1x16), 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb89ABx89AB, vget_high_s8(va_1x16), 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vbCDEFx89AB, vget_high_s8(va_1x16), 0);

      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123xCDEF, vget_high_s8(va_1x16), 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567xCDEF, vget_high_s8(va_1x16), 1);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb89ABxCDEF, vget_high_s8(va_1x16), 1);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vbCDEFxCDEF, vget_high_s8(va_1x16), 1);

      k -= 16 * sizeof(int8_t);
    }
    // Handle up to 8 final positions of `k`.
    if XNN_UNLIKELY(k > 0) {
      int8x16_t vb0123x16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      int8x16_t vb4567x16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      int8x16_t vb89ABx16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      int8x16_t vbCDEFx16 = vld1q_s8(w); w = (const int8_t*) w + 16;

      if XNN_UNLIKELY (k >= 8 * sizeof(int8_t)) {
        // Load a 2x8 block of activations.
        const int8x8_t va0x8 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x8 = vld1_s8(a1); a1 += 8;

        // First crumb.
        const int8x16_t vb0123x0123 = vandq_s8(vb0123x16, vmask);
        const int8x16_t vb4567x0123 = vandq_s8(vb4567x16, vmask);
        const int8x16_t vb89ABx0123 = vandq_s8(vb89ABx16, vmask);
        const int8x16_t vbCDEFx0123 = vandq_s8(vbCDEFx16, vmask);
        // Second crumb.
        const int8x16_t vb0123x4567 = vandq_s8(vshrq_n_s8(vb0123x16, 2), vmask);
        const int8x16_t vb4567x4567 = vandq_s8(vshrq_n_s8(vb4567x16, 2), vmask);
        const int8x16_t vb89ABx4567 = vandq_s8(vshrq_n_s8(vb89ABx16, 2), vmask);
        const int8x16_t vbCDEFx4567 = vandq_s8(vshrq_n_s8(vbCDEFx16, 2), vmask);

        // Multiply-accumulate: 2x8 * 8x16 --> 2x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x8, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x0123, va0x8, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb89ABx0123, va0x8, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vbCDEFx0123, va0x8, 0);

        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x4567, va0x8, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x8, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb89ABx4567, va0x8, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vbCDEFx4567, va0x8, 1);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x8, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x0123, va1x8, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb89ABx0123, va1x8, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vbCDEFx0123, va1x8, 0);

        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x4567, va1x8, 1);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x8, 1);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb89ABx4567, va1x8, 1);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vbCDEFx4567, va1x8, 1);

        k -= 8 * sizeof(int8_t);

        vb0123x16 = vshrq_n_s8(vb0123x16, 4);
        vb4567x16 = vshrq_n_s8(vb4567x16, 4);
        vb89ABx16 = vshrq_n_s8(vb89ABx16, 4);
        vbCDEFx16 = vshrq_n_s8(vbCDEFx16, 4);
      }

      // Handle up to 4 final positions of `k`.
      if XNN_UNLIKELY(k >= 4 * sizeof(int8_t)) {
      // Load a 2x4 block of activations.
        const int8x8_t va0x0123 = vreinterpret_s8_s32(vld1_dup_s32((const int32_t*)a0)); a0 += 4;
        const int8x8_t va1x0123 = vreinterpret_s8_s32(vld1_dup_s32((const int32_t*)a1)); a1 += 4;

        // First crumb.
        const int8x16_t vb0123x0123 = vandq_s8(vb0123x16, vmask);
        const int8x16_t vb4567x0123 = vandq_s8(vb4567x16, vmask);
        const int8x16_t vb89ABx0123 = vandq_s8(vb89ABx16, vmask);
        const int8x16_t vbCDEFx0123 = vandq_s8(vbCDEFx16, vmask);

        // Multiply-accumulate: 2x4 * 4x16 --> 2x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x0123, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x0123, va0x0123, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb89ABx0123, va0x0123, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vbCDEFx0123, va0x0123, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x0123, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x0123, va1x0123, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb89ABx0123, va1x0123, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vbCDEFx0123, va1x0123, 0);

        k -= 4 * sizeof(int8_t);
      }
    }
    // Make sure there were no leftovers.
    assert(k == 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    const float32x4_t vtwo = vdupq_n_f32(2.0f);
    const float32x4_t rh_zero_points_0123 = vld1q_f32(kzp); kzp = (const float*)kzp + 4;
    const float32x4_t biased_rh_zero_points_0123 = vaddq_f32(rh_zero_points_0123, vtwo);
    const float32x4_t rh_zero_points_4567 = vld1q_f32(kzp); kzp = (const float*)kzp + 4;
    const float32x4_t biased_rh_zero_points_4567 = vaddq_f32(rh_zero_points_4567, vtwo);
    const float32x4_t rh_zero_points_89AB = vld1q_f32(kzp); kzp = (const float*)kzp + 4;
    const float32x4_t biased_rh_zero_points_89AB = vaddq_f32(rh_zero_points_89AB, vtwo);
    const float32x4_t rh_zero_points_CDEF = vld1q_f32(kzp); kzp = (const float*)kzp + 4;
    const float32x4_t biased_rh_zero_points_CDEF = vaddq_f32(rh_zero_points_CDEF, vtwo);

    // Subtract out the scaled left-hand row sums.
    const float32x4_t lh_row_sum_0 = vld1q_dup_f32(&row_sum[0]);
    vout0x0123 = vfmsq_f32(vout0x0123, biased_rh_zero_points_0123, lh_row_sum_0);
    vout0x4567 = vfmsq_f32(vout0x4567, biased_rh_zero_points_4567, lh_row_sum_0);
    vout0x89AB = vfmsq_f32(vout0x89AB, biased_rh_zero_points_89AB, lh_row_sum_0);
    vout0xCDEF = vfmsq_f32(vout0xCDEF, biased_rh_zero_points_CDEF, lh_row_sum_0);
    const float32x4_t lh_row_sum_1 = vld1q_dup_f32(&row_sum[1]);
    vout1x0123 = vfmsq_f32(vout1x0123, biased_rh_zero_points_0123, lh_row_sum_1);
    vout1x4567 = vfmsq_f32(vout1x4567, biased_rh_zero_points_4567, lh_row_sum_1);
    vout1x89AB = vfmsq_f32(vout1x89AB, biased_rh_zero_points_89AB, lh_row_sum_1);
    vout1xCDEF = vfmsq_f32(vout1xCDEF, biased_rh_zero_points_CDEF, lh_row_sum_1);

    // Add the product of left/right-hand zero points and `kc`.
    const float32x4_t vscaled_lh_zero_point_0 =
      vdupq_n_f32((float)kc * quantization_params[0].zero_point);
    const float32x4_t vscaled_lh_zero_point_1 =
      vdupq_n_f32((float)kc * quantization_params[1].zero_point);
    vout0x0123 =
      vmlaq_f32(vout0x0123, rh_zero_points_0123, vscaled_lh_zero_point_0);
    vout0x4567 =
      vmlaq_f32(vout0x4567, rh_zero_points_4567, vscaled_lh_zero_point_0);
    vout0x89AB =
      vmlaq_f32(vout0x89AB, rh_zero_points_89AB, vscaled_lh_zero_point_0);
    vout0xCDEF =
      vmlaq_f32(vout0xCDEF, rh_zero_points_CDEF, vscaled_lh_zero_point_0);
    vout1x0123 =
      vmlaq_f32(vout1x0123, rh_zero_points_0123, vscaled_lh_zero_point_1);
    vout1x4567 =
      vmlaq_f32(vout1x4567, rh_zero_points_4567, vscaled_lh_zero_point_1);
    vout1x89AB =
      vmlaq_f32(vout1x89AB, rh_zero_points_89AB, vscaled_lh_zero_point_1);
    vout1xCDEF =
      vmlaq_f32(vout1xCDEF, rh_zero_points_CDEF, vscaled_lh_zero_point_1);
    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    vout0x89AB = vmulq_lane_f32(vout0x89AB, vget_low_f32(vinput_scale01), 1);
    vout1x89AB = vmulq_lane_f32(vout1x89AB, vget_high_f32(vinput_scale01), 1);
    vout0xCDEF = vmulq_lane_f32(vout0xCDEF, vget_low_f32(vinput_scale01), 1);
    vout1xCDEF = vmulq_lane_f32(vout1xCDEF, vget_high_f32(vinput_scale01), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
    #endif

    const float32x4_t voutput_min = vdupq_n_f32(params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);

    const float32x4_t voutput_max = vdupq_n_f32(params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c1 + 8, vout1x89AB);
      vst1q_f32(c1 + 12, vout1xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
       vout1x01 = vget_high_f32(vout1x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
