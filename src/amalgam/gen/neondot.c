// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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

  size_t bl = params->scalar.blocksize;
  assert(bl <= kc);
  assert(bl != 0);
  assert(bl % 32 == 0);
  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));
  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    const float32x4_t vinput_zero_point0 = vcvtq_f32_s32(vld1q_dup_s32(&quantization_params[0].zero_point));
    const float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x0123 = vmulq_f32(vksum0123, vinput_zero_point0);
    const float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x4567 = vmulq_f32(vksum4567, vinput_zero_point0);
    const float32x4_t vksum89AB = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x89AB = vmulq_f32(vksum89AB, vinput_zero_point0);
    const float32x4_t vksumCDEF = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0xCDEF = vmulq_f32(vksumCDEF, vinput_zero_point0);

    for (size_t kb=0; kb < kc; kb += bl) {
      int32x4_t vacc0x0123 = vdupq_n_s32(0);
      int32x4_t vacc0x4567 = vdupq_n_s32(0);
      int32x4_t vacc0x89AB = vdupq_n_s32(0);
      int32x4_t vacc0xCDEF = vdupq_n_s32(0);
      // Inner accumulation loop along the 16 columns.
      size_t k = bl;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 1x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

        // Load a 8x16 block of weights.
        const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01234567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01234567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
        const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
        const int8x16_t vb0123x89AB = vshlq_n_s8(vb01234567x89AB, 4);
        const int8x16_t vb0123xCDEF = vshlq_n_s8(vb01234567xCDEF, 4);
        const int8x16_t vb4567x0123 = vandq_s8(vb01234567x0123, vmask);
        const int8x16_t vb4567x4567 = vandq_s8(vb01234567x4567, vmask);
        const int8x16_t vb4567x89AB = vandq_s8(vb01234567x89AB, vmask);
        const int8x16_t vb4567xCDEF = vandq_s8(vb01234567xCDEF, vmask);

        // Multiply-accumulate: 1x8 * 8x16 --> 1x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 1x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;

        // Load a 4x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      }
      const float32x4_t vfilter_output_scale0123 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scale4567 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scale89AB = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scaleCDEF = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;

      float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
      vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
      vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      float32x4_t vf0x89AB = vcvtq_f32_s32(vacc0x89AB);
      vout0x89AB = vfmaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      float32x4_t vf0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
      vout0xCDEF = vfmaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
    }

    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale0);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale0);


    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vaddq_f32(vbias0123, vout0x0123);
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x4567 = vaddq_f32(vbias4567, vout0x4567);
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    vout0x89AB = vaddq_f32(vbias89AB, vout0x89AB);
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    vout0xCDEF = vaddq_f32(vbiasCDEF, vout0xCDEF);

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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

  size_t bl = params->scalar.blocksize;
  assert(bl <= kc);
  assert(bl != 0);
  assert(bl % 32 == 0);
  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));
  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    const float32x4_t vinput_zero_point01 = vcvtq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    const float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vinput_zero_point01), 0);
    float32x4_t vout1x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vinput_zero_point01), 0);
    const float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vinput_zero_point01), 0);
    float32x4_t vout1x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vinput_zero_point01), 0);
    const float32x4_t vksum89AB = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vinput_zero_point01), 0);
    float32x4_t vout1x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vinput_zero_point01), 0);
    const float32x4_t vksumCDEF = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vinput_zero_point01), 0);
    float32x4_t vout1xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vinput_zero_point01), 0);
    const float32x4_t vinput_zero_point23 = vcvtq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    float32x4_t vout2x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vinput_zero_point23), 0);
    float32x4_t vout3x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vinput_zero_point23), 0);
    float32x4_t vout2x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vinput_zero_point23), 0);
    float32x4_t vout3x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vinput_zero_point23), 0);
    float32x4_t vout2x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vinput_zero_point23), 0);
    float32x4_t vout3x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vinput_zero_point23), 0);
    float32x4_t vout2xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vinput_zero_point23), 0);
    float32x4_t vout3xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vinput_zero_point23), 0);

    for (size_t kb=0; kb < kc; kb += bl) {
      int32x4_t vacc0x0123 = vdupq_n_s32(0);
      int32x4_t vacc1x0123 = vdupq_n_s32(0);
      int32x4_t vacc0x4567 = vdupq_n_s32(0);
      int32x4_t vacc1x4567 = vdupq_n_s32(0);
      int32x4_t vacc0x89AB = vdupq_n_s32(0);
      int32x4_t vacc1x89AB = vdupq_n_s32(0);
      int32x4_t vacc0xCDEF = vdupq_n_s32(0);
      int32x4_t vacc1xCDEF = vdupq_n_s32(0);
      int32x4_t vacc2x0123 = vdupq_n_s32(0);
      int32x4_t vacc3x0123 = vdupq_n_s32(0);
      int32x4_t vacc2x4567 = vdupq_n_s32(0);
      int32x4_t vacc3x4567 = vdupq_n_s32(0);
      int32x4_t vacc2x89AB = vdupq_n_s32(0);
      int32x4_t vacc3x89AB = vdupq_n_s32(0);
      int32x4_t vacc2xCDEF = vdupq_n_s32(0);
      int32x4_t vacc3xCDEF = vdupq_n_s32(0);
      // Inner accumulation loop along the 16 columns.
      size_t k = bl;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 4x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

        // Load a 8x16 block of weights.
        const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01234567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01234567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
        const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
        const int8x16_t vb0123x89AB = vshlq_n_s8(vb01234567x89AB, 4);
        const int8x16_t vb0123xCDEF = vshlq_n_s8(vb01234567xCDEF, 4);
        const int8x16_t vb4567x0123 = vandq_s8(vb01234567x0123, vmask);
        const int8x16_t vb4567x4567 = vandq_s8(vb01234567x4567, vmask);
        const int8x16_t vb4567x89AB = vandq_s8(vb01234567x89AB, vmask);
        const int8x16_t vb4567xCDEF = vandq_s8(vb01234567xCDEF, vmask);

        // Multiply-accumulate: 4x8 * 8x16 --> 4x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb4567x89AB, va1x01234567, 1);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb4567x89AB, va2x01234567, 1);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb4567x89AB, va3x01234567, 1);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb4567xCDEF, va3x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 4x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;

        // Load a 4x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x4 * 4x16 --> 4x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      }
      const float32x4_t vfilter_output_scale0123 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scale4567 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scale89AB = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scaleCDEF = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;

      float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
      float32x4_t vf1x0123 = vcvtq_f32_s32(vacc1x0123);
      vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vout1x0123, vf1x0123, vfilter_output_scale0123);
      float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
      float32x4_t vf1x4567 = vcvtq_f32_s32(vacc1x4567);
      vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vout1x4567, vf1x4567, vfilter_output_scale4567);
      float32x4_t vf0x89AB = vcvtq_f32_s32(vacc0x89AB);
      float32x4_t vf1x89AB = vcvtq_f32_s32(vacc1x89AB);
      vout0x89AB = vfmaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vout1x89AB, vf1x89AB, vfilter_output_scale89AB);
      float32x4_t vf0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
      float32x4_t vf1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
      vout0xCDEF = vfmaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vout1xCDEF, vf1xCDEF, vfilter_output_scaleCDEF);
      float32x4_t vf2x0123 = vcvtq_f32_s32(vacc2x0123);
      float32x4_t vf3x0123 = vcvtq_f32_s32(vacc3x0123);
      vout2x0123 = vfmaq_f32(vout2x0123, vf2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vout3x0123, vf3x0123, vfilter_output_scale0123);
      float32x4_t vf2x4567 = vcvtq_f32_s32(vacc2x4567);
      float32x4_t vf3x4567 = vcvtq_f32_s32(vacc3x4567);
      vout2x4567 = vfmaq_f32(vout2x4567, vf2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vout3x4567, vf3x4567, vfilter_output_scale4567);
      float32x4_t vf2x89AB = vcvtq_f32_s32(vacc2x89AB);
      float32x4_t vf3x89AB = vcvtq_f32_s32(vacc3x89AB);
      vout2x89AB = vfmaq_f32(vout2x89AB, vf2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vout3x89AB, vf3x89AB, vfilter_output_scale89AB);
      float32x4_t vf2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
      float32x4_t vf3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
      vout2xCDEF = vfmaq_f32(vout2xCDEF, vf2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vout3xCDEF, vf3xCDEF, vfilter_output_scaleCDEF);
    }

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    vout0x89AB = vmulq_lane_f32(vout0x89AB, vget_low_f32(vinput_scale01), 1);
    vout1x89AB = vmulq_lane_f32(vout1x89AB, vget_high_f32(vinput_scale01), 1);
    vout0xCDEF = vmulq_lane_f32(vout0xCDEF, vget_low_f32(vinput_scale01), 1);
    vout1xCDEF = vmulq_lane_f32(vout1xCDEF, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    vout2x4567 = vmulq_lane_f32(vout2x4567, vget_low_f32(vinput_scale23), 1);
    vout3x4567 = vmulq_lane_f32(vout3x4567, vget_high_f32(vinput_scale23), 1);
    vout2x89AB = vmulq_lane_f32(vout2x89AB, vget_low_f32(vinput_scale23), 1);
    vout3x89AB = vmulq_lane_f32(vout3x89AB, vget_high_f32(vinput_scale23), 1);
    vout2xCDEF = vmulq_lane_f32(vout2xCDEF, vget_low_f32(vinput_scale23), 1);
    vout3xCDEF = vmulq_lane_f32(vout3xCDEF, vget_high_f32(vinput_scale23), 1);


    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vaddq_f32(vbias0123, vout0x0123);
    vout1x0123 = vaddq_f32(vbias0123, vout1x0123);
    vout2x0123 = vaddq_f32(vbias0123, vout2x0123);
    vout3x0123 = vaddq_f32(vbias0123, vout3x0123);
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x4567 = vaddq_f32(vbias4567, vout0x4567);
    vout1x4567 = vaddq_f32(vbias4567, vout1x4567);
    vout2x4567 = vaddq_f32(vbias4567, vout2x4567);
    vout3x4567 = vaddq_f32(vbias4567, vout3x4567);
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    vout0x89AB = vaddq_f32(vbias89AB, vout0x89AB);
    vout1x89AB = vaddq_f32(vbias89AB, vout1x89AB);
    vout2x89AB = vaddq_f32(vbias89AB, vout2x89AB);
    vout3x89AB = vaddq_f32(vbias89AB, vout3x89AB);
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    vout0xCDEF = vaddq_f32(vbiasCDEF, vout0xCDEF);
    vout1xCDEF = vaddq_f32(vbiasCDEF, vout1xCDEF);
    vout2xCDEF = vaddq_f32(vbiasCDEF, vout2xCDEF);
    vout3xCDEF = vaddq_f32(vbiasCDEF, vout3xCDEF);

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c1 + 8, vout1x89AB);
      vst1q_f32(c1 + 12, vout1xCDEF);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c2 + 8, vout2x89AB);
      vst1q_f32(c2 + 12, vout2xCDEF);
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);
      vst1q_f32(c3 + 8, vout3x89AB);
      vst1q_f32(c3 + 12, vout3xCDEF);

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
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x89AB;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x89AB;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xCDEF;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xCDEF;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c3, vout3x01); c3 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout3x01 = vget_high_f32(vout3x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c3, vout3x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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

  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));
  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    const int32x4_t vinput_zero_point0 = vld1q_dup_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point0);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vmulq_s32(vksum89AB, vinput_zero_point0);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 1x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
      const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
      const int8x16_t vb0123x89AB = vshlq_n_s8(vb01234567x89AB, 4);
      const int8x16_t vb0123xCDEF = vshlq_n_s8(vb01234567xCDEF, 4);
      const int8x16_t vb4567x0123 = vandq_s8(vb01234567x0123, vmask);
      const int8x16_t vb4567x4567 = vandq_s8(vb01234567x4567, vmask);
      const int8x16_t vb4567x89AB = vandq_s8(vb01234567x89AB, vmask);
      const int8x16_t vb4567xCDEF = vandq_s8(vb01234567xCDEF, vmask);

      // Multiply-accumulate: 1x8 * 8x16 --> 1x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
      const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
      const int8x16_t vb0123x89AB = vshlq_n_s8(vb01234567x89AB, 4);
      const int8x16_t vb0123xCDEF = vshlq_n_s8(vb01234567xCDEF, 4);

      // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
    }

    float32x4_t vout0x0123 = vcvtq_n_f32_s32(vacc0x0123, 4);
    float32x4_t vout0x4567 = vcvtq_n_f32_s32(vacc0x4567, 4);
    float32x4_t vout0x89AB = vcvtq_n_f32_s32(vacc0x89AB, 4);
    float32x4_t vout0xCDEF = vcvtq_n_f32_s32(vacc0xCDEF, 4);
    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale0);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale0);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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

  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));
  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    const int32x4_t vinput_zero_point01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vinput_zero_point23 = vld1q_s32(&quantization_params[2].zero_point);
    int32x4_t vacc2x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point23), 0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 4x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
      const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
      const int8x16_t vb0123x89AB = vshlq_n_s8(vb01234567x89AB, 4);
      const int8x16_t vb0123xCDEF = vshlq_n_s8(vb01234567xCDEF, 4);
      const int8x16_t vb4567x0123 = vandq_s8(vb01234567x0123, vmask);
      const int8x16_t vb4567x4567 = vandq_s8(vb01234567x4567, vmask);
      const int8x16_t vb4567x89AB = vandq_s8(vb01234567x89AB, vmask);
      const int8x16_t vb4567xCDEF = vandq_s8(vb01234567xCDEF, vmask);

      // Multiply-accumulate: 4x8 * 8x16 --> 4x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb4567x89AB, va1x01234567, 1);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb4567x89AB, va2x01234567, 1);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb4567x89AB, va3x01234567, 1);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb4567xCDEF, va3x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
      const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
      const int8x16_t vb0123x89AB = vshlq_n_s8(vb01234567x89AB, 4);
      const int8x16_t vb0123xCDEF = vshlq_n_s8(vb01234567xCDEF, 4);

      // Multiply-accumulate: 4x4 * 4x16 --> 4x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
    }

    float32x4_t vout0x0123 = vcvtq_n_f32_s32(vacc0x0123, 4);
    float32x4_t vout0x4567 = vcvtq_n_f32_s32(vacc0x4567, 4);
    float32x4_t vout0x89AB = vcvtq_n_f32_s32(vacc0x89AB, 4);
    float32x4_t vout0xCDEF = vcvtq_n_f32_s32(vacc0xCDEF, 4);
    float32x4_t vout1x0123 = vcvtq_n_f32_s32(vacc1x0123, 4);
    float32x4_t vout1x4567 = vcvtq_n_f32_s32(vacc1x4567, 4);
    float32x4_t vout1x89AB = vcvtq_n_f32_s32(vacc1x89AB, 4);
    float32x4_t vout1xCDEF = vcvtq_n_f32_s32(vacc1xCDEF, 4);
    float32x4_t vout2x0123 = vcvtq_n_f32_s32(vacc2x0123, 4);
    float32x4_t vout2x4567 = vcvtq_n_f32_s32(vacc2x4567, 4);
    float32x4_t vout2x89AB = vcvtq_n_f32_s32(vacc2x89AB, 4);
    float32x4_t vout2xCDEF = vcvtq_n_f32_s32(vacc2xCDEF, 4);
    float32x4_t vout3x0123 = vcvtq_n_f32_s32(vacc3x0123, 4);
    float32x4_t vout3x4567 = vcvtq_n_f32_s32(vacc3x4567, 4);
    float32x4_t vout3x89AB = vcvtq_n_f32_s32(vacc3x89AB, 4);
    float32x4_t vout3xCDEF = vcvtq_n_f32_s32(vacc3xCDEF, 4);
    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    vout0x89AB = vmulq_lane_f32(vout0x89AB, vget_low_f32(vinput_scale01), 1);
    vout1x89AB = vmulq_lane_f32(vout1x89AB, vget_high_f32(vinput_scale01), 1);
    vout0xCDEF = vmulq_lane_f32(vout0xCDEF, vget_low_f32(vinput_scale01), 1);
    vout1xCDEF = vmulq_lane_f32(vout1xCDEF, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    vout2x4567 = vmulq_lane_f32(vout2x4567, vget_low_f32(vinput_scale23), 1);
    vout3x4567 = vmulq_lane_f32(vout3x4567, vget_high_f32(vinput_scale23), 1);
    vout2x89AB = vmulq_lane_f32(vout2x89AB, vget_low_f32(vinput_scale23), 1);
    vout3x89AB = vmulq_lane_f32(vout3x89AB, vget_high_f32(vinput_scale23), 1);
    vout2xCDEF = vmulq_lane_f32(vout2xCDEF, vget_low_f32(vinput_scale23), 1);
    vout3xCDEF = vmulq_lane_f32(vout3xCDEF, vget_high_f32(vinput_scale23), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c1 + 8, vout1x89AB);
      vst1q_f32(c1 + 12, vout1xCDEF);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c2 + 8, vout2x89AB);
      vst1q_f32(c2 + 12, vout2xCDEF);
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);
      vst1q_f32(c3 + 8, vout3x89AB);
      vst1q_f32(c3 + 12, vout3xCDEF);

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
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x89AB;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x89AB;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xCDEF;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xCDEF;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c3, vout3x01); c3 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout3x01 = vget_high_f32(vout3x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c3, vout3x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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
    const int32x4_t vinput_zero_point0 = vld1q_dup_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point0);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vmulq_s32(vksum89AB, vinput_zero_point0);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 1x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x8 * 8x16 --> 1x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale0);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale0);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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

  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    const int32x4_t vinput_zero_point0 = vld1q_dup_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point0);

    // Inner accumulation loop along the 8 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 1x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

      // Load a 8x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x8 * 8x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;

      // Load a 4x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x4 * 4x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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

  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    const int32x4_t vinput_zero_point01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vinput_zero_point23 = vld1q_s32(&quantization_params[2].zero_point);
    int32x4_t vacc2x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point23), 0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 4x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x8 * 8x16 --> 4x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb4567x89AB, va1x01234567, 1);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb4567x89AB, va2x01234567, 1);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb4567x89AB, va3x01234567, 1);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb4567xCDEF, va3x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x4 * 4x16 --> 4x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vout2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vout3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    vout0x89AB = vmulq_lane_f32(vout0x89AB, vget_low_f32(vinput_scale01), 1);
    vout1x89AB = vmulq_lane_f32(vout1x89AB, vget_high_f32(vinput_scale01), 1);
    vout0xCDEF = vmulq_lane_f32(vout0xCDEF, vget_low_f32(vinput_scale01), 1);
    vout1xCDEF = vmulq_lane_f32(vout1xCDEF, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    vout2x4567 = vmulq_lane_f32(vout2x4567, vget_low_f32(vinput_scale23), 1);
    vout3x4567 = vmulq_lane_f32(vout3x4567, vget_high_f32(vinput_scale23), 1);
    vout2x89AB = vmulq_lane_f32(vout2x89AB, vget_low_f32(vinput_scale23), 1);
    vout3x89AB = vmulq_lane_f32(vout3x89AB, vget_high_f32(vinput_scale23), 1);
    vout2xCDEF = vmulq_lane_f32(vout2xCDEF, vget_low_f32(vinput_scale23), 1);
    vout3xCDEF = vmulq_lane_f32(vout3xCDEF, vget_high_f32(vinput_scale23), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c1 + 8, vout1x89AB);
      vst1q_f32(c1 + 12, vout1xCDEF);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c2 + 8, vout2x89AB);
      vst1q_f32(c2 + 12, vout2xCDEF);
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);
      vst1q_f32(c3 + 8, vout3x89AB);
      vst1q_f32(c3 + 12, vout3xCDEF);

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
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x89AB;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x89AB;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xCDEF;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xCDEF;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c3, vout3x01); c3 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout3x01 = vget_high_f32(vout3x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c3, vout3x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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

  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    const int32x4_t vinput_zero_point01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vinput_zero_point23 = vld1q_s32(&quantization_params[2].zero_point);
    int32x4_t vacc2x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point23), 0);

    // Inner accumulation loop along the 8 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 4x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

      // Load a 8x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x8 * 8x8 --> 4x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;

      // Load a 4x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x4 * 4x8 --> 4x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    vout2x4567 = vmulq_lane_f32(vout2x4567, vget_low_f32(vinput_scale23), 1);
    vout3x4567 = vmulq_lane_f32(vout3x4567, vget_high_f32(vinput_scale23), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 8;
    } else {
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c3, vout3x01); c3 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout3x01 = vget_high_f32(vout3x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c3, vout3x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;

  do {
    const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vmulq_s32(vksum89AB, vinput_zero_point);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      // Inner accumulation loop along the 16 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 1x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

        // Load a 8x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x8 * 8x16 --> 1x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 1x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);

        // Load a 4x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;

  do {
    const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      // Inner accumulation loop along the 8 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 1x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

        // Load a 8x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x8 * 8x8 --> 1x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 1x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);

        // Load a 4x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x4 * 4x8 --> 1x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vmulq_s32(vksum89AB, vinput_zero_point);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point);
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc2x89AB = vacc0x89AB;
    int32x4_t vacc2xCDEF = vacc0xCDEF;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc3x89AB = vacc0x89AB;
    int32x4_t vacc3xCDEF = vacc0xCDEF;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      // Inner accumulation loop along the 16 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 4x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

        // Load a 8x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x8 * 8x16 --> 4x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb4567x89AB, va1x01234567, 1);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb4567x89AB, va2x01234567, 1);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb4567x89AB, va3x01234567, 1);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb4567xCDEF, va3x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 4x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);
        const int8x8_t va1x01234567 = vld1_s8(a1);
        const int8x8_t va2x01234567 = vld1_s8(a2);
        const int8x8_t va3x01234567 = vld1_s8(a3);

        // Load a 4x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x4 * 4x16 --> 4x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vout2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vout3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale);
    vout1x0123 = vmulq_f32(vout1x0123, vinput_scale);
    vout1x4567 = vmulq_f32(vout1x4567, vinput_scale);
    vout1x89AB = vmulq_f32(vout1x89AB, vinput_scale);
    vout1xCDEF = vmulq_f32(vout1xCDEF, vinput_scale);
    vout2x0123 = vmulq_f32(vout2x0123, vinput_scale);
    vout2x4567 = vmulq_f32(vout2x4567, vinput_scale);
    vout2x89AB = vmulq_f32(vout2x89AB, vinput_scale);
    vout2xCDEF = vmulq_f32(vout2xCDEF, vinput_scale);
    vout3x0123 = vmulq_f32(vout3x0123, vinput_scale);
    vout3x4567 = vmulq_f32(vout3x4567, vinput_scale);
    vout3x89AB = vmulq_f32(vout3x89AB, vinput_scale);
    vout3xCDEF = vmulq_f32(vout3xCDEF, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);
      vst1q_f32(c3 + 8, vout3x89AB);
      vst1q_f32(c3 + 12, vout3xCDEF);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c2 + 8, vout2x89AB);
      vst1q_f32(c2 + 12, vout2xCDEF);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c1 + 8, vout1x89AB);
      vst1q_f32(c1 + 12, vout1xCDEF);
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x89AB;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x89AB;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x89AB;
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xCDEF;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xCDEF;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xCDEF;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c3, vout3x01); c3 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c0, vout0x01); c0 += 2;
       vout3x01 = vget_high_f32(vout3x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c3, vout3x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point);
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      // Inner accumulation loop along the 8 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 4x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

        // Load a 8x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x8 * 8x8 --> 4x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 4x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);
        const int8x8_t va1x01234567 = vld1_s8(a1);
        const int8x8_t va2x01234567 = vld1_s8(a2);
        const int8x8_t va3x01234567 = vld1_s8(a3);

        // Load a 4x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x4 * 4x8 --> 4x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout1x0123 = vmulq_f32(vout1x0123, vinput_scale);
    vout1x4567 = vmulq_f32(vout1x4567, vinput_scale);
    vout2x0123 = vmulq_f32(vout2x0123, vinput_scale);
    vout2x4567 = vmulq_f32(vout2x4567, vinput_scale);
    vout3x0123 = vmulq_f32(vout3x0123, vinput_scale);
    vout3x4567 = vmulq_f32(vout3x4567, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 8;
    } else {
     if (nc & 4) {
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c3, vout3x01); c3 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c0, vout0x01); c0 += 2;
       vout3x01 = vget_high_f32(vout3x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c3, vout3x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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
  int8_t* c0 = c;

  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 1x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x8 * 8x16 --> 1x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale89AB);
    const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscaleCDEF);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    #endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      // Main case where there the 16 columns fit in the destination.
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      // Advance to the next 16 columns.
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      // Final case where not all of the 16 columns fit in the destination.
      int8x8_t vout0x01234567 = vget_low_s8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_s8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_s8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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
  int8_t* c0 = c;

  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

    // Inner accumulation loop along the 8 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 1x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

      // Load a 8x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x8 * 8x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;

      // Load a 4x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x4 * 4x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #endif
    const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neonv8.output_min);
    const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neonv8.output_max);

    vout0x01234567 = vmax_s8(vout0x01234567, voutput_min);

    vout0x01234567 = vmin_s8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      // Main case where there the 8 columns fit in the destination.
      vst1_s8(c0 + 0, vout0x01234567);

      // Advance to the next 8 columns.
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc2x89AB = vacc0x89AB;
    int32x4_t vacc2xCDEF = vacc0xCDEF;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc3x89AB = vacc0x89AB;
    int32x4_t vacc3xCDEF = vacc0xCDEF;

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 4x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x8 * 8x16 --> 4x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb4567x89AB, va1x01234567, 1);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb4567x89AB, va2x01234567, 1);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb4567x89AB, va3x01234567, 1);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb4567xCDEF, va3x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x4 * 4x16 --> 4x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vfpacc1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vfpacc1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vfpacc2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vfpacc2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vfpacc2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vfpacc2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vfpacc3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vfpacc3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vfpacc3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vfpacc3xCDEF = vcvtq_f32_s32(vacc3xCDEF);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    vfpacc2x0123 = vmulq_f32(vfpacc2x0123, vscale0123);
    vfpacc3x0123 = vmulq_f32(vfpacc3x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);
    vfpacc2x4567 = vmulq_f32(vfpacc2x4567, vscale4567);
    vfpacc3x4567 = vmulq_f32(vfpacc3x4567, vscale4567);
    const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale89AB);
    vfpacc1x89AB = vmulq_f32(vfpacc1x89AB, vscale89AB);
    vfpacc2x89AB = vmulq_f32(vfpacc2x89AB, vscale89AB);
    vfpacc3x89AB = vmulq_f32(vfpacc3x89AB, vscale89AB);
    const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscaleCDEF);
    vfpacc1xCDEF = vmulq_f32(vfpacc1xCDEF, vscaleCDEF);
    vfpacc2xCDEF = vmulq_f32(vfpacc2xCDEF, vscaleCDEF);
    vfpacc3xCDEF = vmulq_f32(vfpacc3xCDEF, vscaleCDEF);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);
    vacc1x89AB = vcvtnq_s32_f32(vfpacc1x89AB);
    vacc1xCDEF = vcvtnq_s32_f32(vfpacc1xCDEF);
    vacc2x0123 = vcvtnq_s32_f32(vfpacc2x0123);
    vacc2x4567 = vcvtnq_s32_f32(vfpacc2x4567);
    vacc2x89AB = vcvtnq_s32_f32(vfpacc2x89AB);
    vacc2xCDEF = vcvtnq_s32_f32(vfpacc2xCDEF);
    vacc3x0123 = vcvtnq_s32_f32(vfpacc3x0123);
    vacc3x4567 = vcvtnq_s32_f32(vfpacc3x4567);
    vacc3x89AB = vcvtnq_s32_f32(vfpacc3x89AB);
    vacc3xCDEF = vcvtnq_s32_f32(vfpacc3xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
      const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
      const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);
      const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
      const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF), voutput_zero_point);
      const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
      const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x89AB), vacc3xCDEF), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
      int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
      int8x16_t vout2x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc2x89ABCDEF);
      int8x16_t vout3x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc3x01234567), vacc3x89ABCDEF);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
      const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
      const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);
      const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
      const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF)), voutput_zero_point);
      const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
      const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x89AB), vqmovn_s32(vacc3xCDEF)), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
      int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
      int8x16_t vout2x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc2x89ABCDEF));
      int8x16_t vout3x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc3x01234567), vqmovn_s16(vacc3x89ABCDEF));
    #endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_s8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = vmaxq_s8(vout3x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_s8(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = vminq_s8(vout3x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      // Main case where there the 16 columns fit in the destination.
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_s8(c3 + 0, vout3x0123456789ABCDEF);

      // Advance to the next 16 columns.
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      nc -= 16;
    } else {
      // Final case where not all of the 16 columns fit in the destination.
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vget_low_s8(vout2x0123456789ABCDEF), vget_low_s8(vout3x0123456789ABCDEF));
      if (nc & 8) {
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c2, vget_low_s8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_s8(c3, vget_high_s8(vout2x01234567_3x01234567)); c3 += 8;
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
        vout2x01234567_3x01234567 = vcombine_s8(vget_high_s8(vout2x0123456789ABCDEF), vget_high_s8(vout3x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
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
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;

    // Inner accumulation loop along the 8 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 4x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

      // Load a 8x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x8 * 8x8 --> 4x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;

      // Load a 4x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x4 * 4x8 --> 4x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vfpacc2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vfpacc2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vfpacc3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vfpacc3x4567 = vcvtq_f32_s32(vacc3x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    vfpacc2x0123 = vmulq_f32(vfpacc2x0123, vscale0123);
    vfpacc3x0123 = vmulq_f32(vfpacc3x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);
    vfpacc2x4567 = vmulq_f32(vfpacc2x4567, vscale4567);
    vfpacc3x4567 = vmulq_f32(vfpacc3x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);
    vacc2x0123 = vcvtnq_s32_f32(vfpacc2x0123);
    vacc2x4567 = vcvtnq_s32_f32(vfpacc2x4567);
    vacc3x0123 = vcvtnq_s32_f32(vfpacc3x0123);
    vacc3x4567 = vcvtnq_s32_f32(vfpacc3x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
      const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
      const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
      const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

      int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
      int8x16_t vout2x01234567_3x01234567 = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc3x01234567);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
      const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
      const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
      const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);

      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc3x01234567));
    #endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);

    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_s8(vout2x01234567_3x01234567, voutput_min);

    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_s8(vout2x01234567_3x01234567, voutput_max);

    if (nc >= 8) {
      // Main case where there the 8 columns fit in the destination.
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c2 + 0, vget_low_s8(vout2x01234567_3x01234567));
      vst1_s8(c3 + 0, vget_high_s8(vout2x01234567_3x01234567));

      // Advance to the next 8 columns.
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  int8_t* c0 = c;

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      // Inner accumulation loop along the 16 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 1x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

        // Load a 8x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x8 * 8x16 --> 1x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 1x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);

        // Load a 4x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale89AB);
    const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscaleCDEF);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    #endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      int8x8_t vout0x01234567 = vget_low_s8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_s8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_s8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  int8_t* c0 = c;

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      // Inner accumulation loop along the 8 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 1x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

        // Load a 8x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x8 * 8x8 --> 1x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 1x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);

        // Load a 4x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x4 * 4x8 --> 1x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #endif
    const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neonv8.output_min);
    const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neonv8.output_max);

    vout0x01234567 = vmax_s8(vout0x01234567, voutput_min);

    vout0x01234567 = vmin_s8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vout0x01234567);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc2x89AB = vacc0x89AB;
    int32x4_t vacc2xCDEF = vacc0xCDEF;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc3x89AB = vacc0x89AB;
    int32x4_t vacc3xCDEF = vacc0xCDEF;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      // Inner accumulation loop along the 16 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 4x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

        // Load a 8x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x8 * 8x16 --> 4x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb4567x89AB, va1x01234567, 1);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb4567x89AB, va2x01234567, 1);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb4567x89AB, va3x01234567, 1);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb4567xCDEF, va3x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 4x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);
        const int8x8_t va1x01234567 = vld1_s8(a1);
        const int8x8_t va2x01234567 = vld1_s8(a2);
        const int8x8_t va3x01234567 = vld1_s8(a3);

        // Load a 4x16 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x4 * 4x16 --> 4x16.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vfpacc1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vfpacc1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vfpacc2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vfpacc2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vfpacc2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vfpacc2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vfpacc3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vfpacc3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vfpacc3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vfpacc3xCDEF = vcvtq_f32_s32(vacc3xCDEF);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    vfpacc2x0123 = vmulq_f32(vfpacc2x0123, vscale0123);
    vfpacc3x0123 = vmulq_f32(vfpacc3x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);
    vfpacc2x4567 = vmulq_f32(vfpacc2x4567, vscale4567);
    vfpacc3x4567 = vmulq_f32(vfpacc3x4567, vscale4567);
    const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale89AB);
    vfpacc1x89AB = vmulq_f32(vfpacc1x89AB, vscale89AB);
    vfpacc2x89AB = vmulq_f32(vfpacc2x89AB, vscale89AB);
    vfpacc3x89AB = vmulq_f32(vfpacc3x89AB, vscale89AB);
    const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscaleCDEF);
    vfpacc1xCDEF = vmulq_f32(vfpacc1xCDEF, vscaleCDEF);
    vfpacc2xCDEF = vmulq_f32(vfpacc2xCDEF, vscaleCDEF);
    vfpacc3xCDEF = vmulq_f32(vfpacc3xCDEF, vscaleCDEF);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);
    vacc1x89AB = vcvtnq_s32_f32(vfpacc1x89AB);
    vacc1xCDEF = vcvtnq_s32_f32(vfpacc1xCDEF);
    vacc2x0123 = vcvtnq_s32_f32(vfpacc2x0123);
    vacc2x4567 = vcvtnq_s32_f32(vfpacc2x4567);
    vacc2x89AB = vcvtnq_s32_f32(vfpacc2x89AB);
    vacc2xCDEF = vcvtnq_s32_f32(vfpacc2xCDEF);
    vacc3x0123 = vcvtnq_s32_f32(vfpacc3x0123);
    vacc3x4567 = vcvtnq_s32_f32(vfpacc3x4567);
    vacc3x89AB = vcvtnq_s32_f32(vfpacc3x89AB);
    vacc3xCDEF = vcvtnq_s32_f32(vfpacc3xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
      const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
      const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);
      const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
      const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF), voutput_zero_point);
      const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
      const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x89AB), vacc3xCDEF), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
      int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
      int8x16_t vout2x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc2x89ABCDEF);
      int8x16_t vout3x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc3x01234567), vacc3x89ABCDEF);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
      const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
      const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);
      const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
      const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF)), voutput_zero_point);
      const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
      const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x89AB), vqmovn_s32(vacc3xCDEF)), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
      int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
      int8x16_t vout2x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc2x89ABCDEF));
      int8x16_t vout3x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc3x01234567), vqmovn_s16(vacc3x89ABCDEF));
    #endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);

    vout3x0123456789ABCDEF = vmaxq_s8(vout3x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_s8(vout2x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    vout3x0123456789ABCDEF = vminq_s8(vout3x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_s8(vout2x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c3 + 0, vout3x0123456789ABCDEF);
      vst1q_s8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vget_low_s8(vout2x0123456789ABCDEF), vget_low_s8(vout3x0123456789ABCDEF));
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      if (nc & 8) {
        vst1_s8(c3, vget_high_s8(vout2x01234567_3x01234567)); c3 += 8;
        vst1_s8(c2, vget_low_s8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vout2x01234567_3x01234567 = vcombine_s8(vget_high_s8(vout2x0123456789ABCDEF), vget_high_s8(vout3x0123456789ABCDEF));
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      // Inner accumulation loop along the 8 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 4x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

        // Load a 8x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x8 * 8x8 --> 4x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 4x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);
        const int8x8_t va1x01234567 = vld1_s8(a1);
        const int8x8_t va2x01234567 = vld1_s8(a2);
        const int8x8_t va3x01234567 = vld1_s8(a3);

        // Load a 4x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x4 * 4x8 --> 4x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vfpacc2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vfpacc2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vfpacc3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vfpacc3x4567 = vcvtq_f32_s32(vacc3x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    vfpacc2x0123 = vmulq_f32(vfpacc2x0123, vscale0123);
    vfpacc3x0123 = vmulq_f32(vfpacc3x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);
    vfpacc2x4567 = vmulq_f32(vfpacc2x4567, vscale4567);
    vfpacc3x4567 = vmulq_f32(vfpacc3x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);
    vacc2x0123 = vcvtnq_s32_f32(vfpacc2x0123);
    vacc2x4567 = vcvtnq_s32_f32(vfpacc2x4567);
    vacc3x0123 = vcvtnq_s32_f32(vfpacc3x0123);
    vacc3x4567 = vcvtnq_s32_f32(vfpacc3x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
      const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
      const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
      const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

      int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
      int8x16_t vout2x01234567_3x01234567 = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc3x01234567);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
      const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
      const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
      const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);

      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc3x01234567));
    #endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);

    vout2x01234567_3x01234567 = vmaxq_s8(vout2x01234567_3x01234567, voutput_min);
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);

    vout2x01234567_3x01234567 = vminq_s8(vout2x01234567_3x01234567, voutput_max);
    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c3 + 0, vget_high_s8(vout2x01234567_3x01234567));
      vst1_s8(c2 + 0, vget_low_s8(vout2x01234567_3x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
