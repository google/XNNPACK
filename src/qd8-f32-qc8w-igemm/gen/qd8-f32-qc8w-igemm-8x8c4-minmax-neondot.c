// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/igemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c4__neondot(
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
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (8 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
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
    int32x4_t vacc4x0123 = vacc0x0123;
    int32x4_t vacc4x4567 = vacc0x4567;
    int32x4_t vacc5x0123 = vacc0x0123;
    int32x4_t vacc5x4567 = vacc0x4567;
    int32x4_t vacc6x0123 = vacc0x0123;
    int32x4_t vacc6x4567 = vacc0x4567;
    int32x4_t vacc7x0123 = vacc0x0123;
    int32x4_t vacc7x4567 = vacc0x4567;

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
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      } else {
        a4 = zero_data;
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      } else {
        a5 = zero_data;
      }
      const int8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const int8_t*) ((uintptr_t) a6 + a_offset);
      } else {
        a6 = zero_data;
      }
      const int8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const int8_t*) ((uintptr_t) a7 + a_offset);
      } else {
        a7 = zero_data;
      }
      a += 8;

      // Inner accumulation loop along the 8 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 8x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;
        const int8x8_t va4x01234567 = vld1_s8(a4); a4 += 8;
        const int8x8_t va5x01234567 = vld1_s8(a5); a5 += 8;
        const int8x8_t va6x01234567 = vld1_s8(a6); a6 += 8;
        const int8x8_t va7x01234567 = vld1_s8(a7); a7 += 8;

        // Load a 8x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 8x8 * 8x8 --> 8x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb0123x0123, va4x01234567, 0);
        vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb0123x4567, va4x01234567, 0);
        vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb0123x0123, va5x01234567, 0);
        vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb0123x4567, va5x01234567, 0);
        vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb0123x0123, va6x01234567, 0);
        vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb0123x4567, va6x01234567, 0);
        vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb0123x0123, va7x01234567, 0);
        vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb0123x4567, va7x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
        vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb4567x0123, va4x01234567, 1);
        vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb4567x4567, va4x01234567, 1);
        vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb4567x0123, va5x01234567, 1);
        vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb4567x4567, va5x01234567, 1);
        vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb4567x0123, va6x01234567, 1);
        vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb4567x4567, va6x01234567, 1);
        vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb4567x0123, va7x01234567, 1);
        vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb4567x4567, va7x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 8x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);
        const int8x8_t va1x01234567 = vld1_s8(a1);
        const int8x8_t va2x01234567 = vld1_s8(a2);
        const int8x8_t va3x01234567 = vld1_s8(a3);
        const int8x8_t va4x01234567 = vld1_s8(a4);
        const int8x8_t va5x01234567 = vld1_s8(a5);
        const int8x8_t va6x01234567 = vld1_s8(a6);
        const int8x8_t va7x01234567 = vld1_s8(a7);

        // Load a 4x8 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 8x4 * 4x8 --> 8x8.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb0123x0123, va4x01234567, 0);
        vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb0123x4567, va4x01234567, 0);
        vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb0123x0123, va5x01234567, 0);
        vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb0123x4567, va5x01234567, 0);
        vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb0123x0123, va6x01234567, 0);
        vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb0123x4567, va6x01234567, 0);
        vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb0123x0123, va7x01234567, 0);
        vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb0123x4567, va7x01234567, 0);
      }
      p -= 8 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout4x0123 = vcvtq_f32_s32(vacc4x0123);
    float32x4_t vout4x4567 = vcvtq_f32_s32(vacc4x4567);
    float32x4_t vout5x0123 = vcvtq_f32_s32(vacc5x0123);
    float32x4_t vout5x4567 = vcvtq_f32_s32(vacc5x4567);
    float32x4_t vout6x0123 = vcvtq_f32_s32(vacc6x0123);
    float32x4_t vout6x4567 = vcvtq_f32_s32(vacc6x4567);
    float32x4_t vout7x0123 = vcvtq_f32_s32(vacc7x0123);
    float32x4_t vout7x4567 = vcvtq_f32_s32(vacc7x4567);
    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout1x0123 = vmulq_f32(vout1x0123, vinput_scale);
    vout1x4567 = vmulq_f32(vout1x4567, vinput_scale);
    vout2x0123 = vmulq_f32(vout2x0123, vinput_scale);
    vout2x4567 = vmulq_f32(vout2x4567, vinput_scale);
    vout3x0123 = vmulq_f32(vout3x0123, vinput_scale);
    vout3x4567 = vmulq_f32(vout3x4567, vinput_scale);
    vout4x0123 = vmulq_f32(vout4x0123, vinput_scale);
    vout4x4567 = vmulq_f32(vout4x4567, vinput_scale);
    vout5x0123 = vmulq_f32(vout5x0123, vinput_scale);
    vout5x4567 = vmulq_f32(vout5x4567, vinput_scale);
    vout6x0123 = vmulq_f32(vout6x0123, vinput_scale);
    vout6x4567 = vmulq_f32(vout6x4567, vinput_scale);
    vout7x0123 = vmulq_f32(vout7x0123, vinput_scale);
    vout7x4567 = vmulq_f32(vout7x4567, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      vout4x0123 = vfmaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
      vout5x0123 = vfmaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
      vout6x0123 = vfmaq_f32(vbias0123, vout6x0123, vfilter_output_scale0123);
      vout7x0123 = vfmaq_f32(vbias0123, vout7x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      vout4x0123 = vmlaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
      vout5x0123 = vmlaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
      vout6x0123 = vmlaq_f32(vbias0123, vout6x0123, vfilter_output_scale0123);
      vout7x0123 = vmlaq_f32(vbias0123, vout7x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      vout4x4567 = vfmaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
      vout5x4567 = vfmaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
      vout6x4567 = vfmaq_f32(vbias4567, vout6x4567, vfilter_output_scale4567);
      vout7x4567 = vfmaq_f32(vbias4567, vout7x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      vout4x4567 = vmlaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
      vout5x4567 = vmlaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
      vout6x4567 = vmlaq_f32(vbias4567, vout6x4567, vfilter_output_scale4567);
      vout7x4567 = vmlaq_f32(vbias4567, vout7x4567, vfilter_output_scale4567);
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
    vout4x0123 = vmaxq_f32(vout4x0123, voutput_min);
    vout4x4567 = vmaxq_f32(vout4x4567, voutput_min);
    vout5x0123 = vmaxq_f32(vout5x0123, voutput_min);
    vout5x4567 = vmaxq_f32(vout5x4567, voutput_min);
    vout6x0123 = vmaxq_f32(vout6x0123, voutput_min);
    vout6x4567 = vmaxq_f32(vout6x4567, voutput_min);
    vout7x0123 = vmaxq_f32(vout7x0123, voutput_min);
    vout7x4567 = vmaxq_f32(vout7x4567, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout4x0123 = vminq_f32(vout4x0123, voutput_max);
    vout4x4567 = vminq_f32(vout4x4567, voutput_max);
    vout5x0123 = vminq_f32(vout5x0123, voutput_max);
    vout5x4567 = vminq_f32(vout5x4567, voutput_max);
    vout6x0123 = vminq_f32(vout6x0123, voutput_max);
    vout6x4567 = vminq_f32(vout6x4567, voutput_max);
    vout7x0123 = vminq_f32(vout7x0123, voutput_max);
    vout7x4567 = vminq_f32(vout7x4567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c7, vout7x0123);
      vst1q_f32(c7 + 4, vout7x4567);
      vst1q_f32(c6, vout6x0123);
      vst1q_f32(c6 + 4, vout6x4567);
      vst1q_f32(c5, vout5x0123);
      vst1q_f32(c5 + 4, vout5x4567);
      vst1q_f32(c4, vout4x0123);
      vst1q_f32(c4 + 4, vout4x4567);
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
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);

      nc -= 8;
    } else {
     if (nc & 4) {
       vst1q_f32(c7, vout7x0123); c7 += 4;
       vout7x0123 = vout7x4567;
       vst1q_f32(c6, vout6x0123); c6 += 4;
       vout6x0123 = vout6x4567;
       vst1q_f32(c5, vout5x0123); c5 += 4;
       vout5x0123 = vout5x4567;
       vst1q_f32(c4, vout4x0123); c4 += 4;
       vout4x0123 = vout4x4567;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout7x01 = vget_low_f32(vout7x0123);
     float32x2_t vout6x01 = vget_low_f32(vout6x0123);
     float32x2_t vout5x01 = vget_low_f32(vout5x0123);
     float32x2_t vout4x01 = vget_low_f32(vout4x0123);
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c7, vout7x01); c7 += 2;
       vst1_f32(c6, vout6x01); c6 += 2;
       vst1_f32(c5, vout5x01); c5 += 2;
       vst1_f32(c4, vout4x01); c4 += 2;
       vst1_f32(c3, vout3x01); c3 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c0, vout0x01); c0 += 2;
       vout7x01 = vget_high_f32(vout7x0123);
       vout6x01 = vget_high_f32(vout6x0123);
       vout5x01 = vget_high_f32(vout5x0123);
       vout4x01 = vget_high_f32(vout4x0123);
       vout3x01 = vget_high_f32(vout3x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c7, vout7x01, 0);
       vst1_lane_f32(c6, vout6x01, 0);
       vst1_lane_f32(c5, vout5x01, 0);
       vst1_lane_f32(c4, vout4x01, 0);
       vst1_lane_f32(c3, vout3x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
