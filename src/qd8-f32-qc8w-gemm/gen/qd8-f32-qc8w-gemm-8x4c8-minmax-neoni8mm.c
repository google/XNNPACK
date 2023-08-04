// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x4c8__neoni8mm(
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
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
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
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  // Loop over groups of 4 columns.
  do {
    // Initialize accumulators with bias. 4 bias values are loaded from the
    // weight matrix, at the start of the group of 4 columns.
    const int32x4_t vinput_zero_point01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vinput_zero_point23 = vld1q_s32(&quantization_params[2].zero_point);
    const int32x4_t vksumzp2x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vinput_zero_point45 = vld1q_s32(&quantization_params[4].zero_point);
    const int32x4_t vksumzp4x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point45), 0);
    const int32x4_t vksumzp5x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point45), 0);
    const int32x4_t vinput_zero_point67 = vld1q_s32(&quantization_params[6].zero_point);
    const int32x4_t vksumzp6x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point67), 0);
    const int32x4_t vksumzp7x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point67), 0);
    int32x4_t vacc01x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp1x0123)));
    int32x4_t vacc01x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp1x0123)));
    int32x4_t vacc23x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x0123), vreinterpretq_u64_s32(vksumzp3x0123)));
    int32x4_t vacc23x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x0123), vreinterpretq_u64_s32(vksumzp3x0123)));
    int32x4_t vacc45x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4x0123), vreinterpretq_u64_s32(vksumzp5x0123)));
    int32x4_t vacc45x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4x0123), vreinterpretq_u64_s32(vksumzp5x0123)));
    int32x4_t vacc67x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp6x0123), vreinterpretq_u64_s32(vksumzp7x0123)));
    int32x4_t vacc67x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp6x0123), vreinterpretq_u64_s32(vksumzp7x0123)));

    // Inner accumulation loop along the 4 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 16 * sizeof(int8_t)) {
      // Load a 8x16 block of activations.
      const int8x16_t va0x0123456789ABCDEF = vld1q_s8(a0); a0 += 16;
      const int8x16_t va1x0123456789ABCDEF = vld1q_s8(a1); a1 += 16;
      const int8x16_t va2x0123456789ABCDEF = vld1q_s8(a2); a2 += 16;
      const int8x16_t va3x0123456789ABCDEF = vld1q_s8(a3); a3 += 16;
      const int8x16_t va4x0123456789ABCDEF = vld1q_s8(a4); a4 += 16;
      const int8x16_t va5x0123456789ABCDEF = vld1q_s8(a5); a5 += 16;
      const int8x16_t va6x0123456789ABCDEF = vld1q_s8(a6); a6 += 16;
      const int8x16_t va7x0123456789ABCDEF = vld1q_s8(a7); a7 += 16;
      const int8x16_t va01x01234567 = vreinterpretq_s8_u64(vtrn1q_u64(vreinterpretq_u64_s8(va0x0123456789ABCDEF), vreinterpretq_u64_s8(va1x0123456789ABCDEF)));
      const int8x16_t va01x89ABCDEF = vreinterpretq_s8_u64(vtrn2q_u64(vreinterpretq_u64_s8(va0x0123456789ABCDEF), vreinterpretq_u64_s8(va1x0123456789ABCDEF)));
      const int8x16_t va23x01234567 = vreinterpretq_s8_u64(vtrn1q_u64(vreinterpretq_u64_s8(va2x0123456789ABCDEF), vreinterpretq_u64_s8(va3x0123456789ABCDEF)));
      const int8x16_t va23x89ABCDEF = vreinterpretq_s8_u64(vtrn2q_u64(vreinterpretq_u64_s8(va2x0123456789ABCDEF), vreinterpretq_u64_s8(va3x0123456789ABCDEF)));
      const int8x16_t va45x01234567 = vreinterpretq_s8_u64(vtrn1q_u64(vreinterpretq_u64_s8(va4x0123456789ABCDEF), vreinterpretq_u64_s8(va5x0123456789ABCDEF)));
      const int8x16_t va45x89ABCDEF = vreinterpretq_s8_u64(vtrn2q_u64(vreinterpretq_u64_s8(va4x0123456789ABCDEF), vreinterpretq_u64_s8(va5x0123456789ABCDEF)));
      const int8x16_t va67x01234567 = vreinterpretq_s8_u64(vtrn1q_u64(vreinterpretq_u64_s8(va6x0123456789ABCDEF), vreinterpretq_u64_s8(va7x0123456789ABCDEF)));
      const int8x16_t va67x89ABCDEF = vreinterpretq_s8_u64(vtrn2q_u64(vreinterpretq_u64_s8(va6x0123456789ABCDEF), vreinterpretq_u64_s8(va7x0123456789ABCDEF)));

      // Load a 16x4 block of weights.
      const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 8x8 * 8x4 --> 8x4.
      vacc01x01 = vmmlaq_s32(vacc01x01, va01x01234567, vb01x01234567);
      vacc01x23 = vmmlaq_s32(vacc01x23, va01x01234567, vb23x01234567);
      vacc23x01 = vmmlaq_s32(vacc23x01, va23x01234567, vb01x01234567);
      vacc23x23 = vmmlaq_s32(vacc23x23, va23x01234567, vb23x01234567);
      vacc45x01 = vmmlaq_s32(vacc45x01, va45x01234567, vb01x01234567);
      vacc45x23 = vmmlaq_s32(vacc45x23, va45x01234567, vb23x01234567);
      vacc67x01 = vmmlaq_s32(vacc67x01, va67x01234567, vb01x01234567);
      vacc67x23 = vmmlaq_s32(vacc67x23, va67x01234567, vb23x01234567);
      vacc01x01 = vmmlaq_s32(vacc01x01, va01x89ABCDEF, vb01x89ABCDEF);
      vacc01x23 = vmmlaq_s32(vacc01x23, va01x89ABCDEF, vb23x89ABCDEF);
      vacc23x01 = vmmlaq_s32(vacc23x01, va23x89ABCDEF, vb01x89ABCDEF);
      vacc23x23 = vmmlaq_s32(vacc23x23, va23x89ABCDEF, vb23x89ABCDEF);
      vacc45x01 = vmmlaq_s32(vacc45x01, va45x89ABCDEF, vb01x89ABCDEF);
      vacc45x23 = vmmlaq_s32(vacc45x23, va45x89ABCDEF, vb23x89ABCDEF);
      vacc67x01 = vmmlaq_s32(vacc67x01, va67x89ABCDEF, vb01x89ABCDEF);
      vacc67x23 = vmmlaq_s32(vacc67x23, va67x89ABCDEF, vb23x89ABCDEF);

      k -= 16 * sizeof(int8_t);
    }
    // Handle up to 16 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      do {
        // Load a 8x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;
        const int8x8_t va4x01234567 = vld1_s8(a4); a4 += 8;
        const int8x8_t va5x01234567 = vld1_s8(a5); a5 += 8;
        const int8x8_t va6x01234567 = vld1_s8(a6); a6 += 8;
        const int8x8_t va7x01234567 = vld1_s8(a7); a7 += 8;
        int8x16_t va01x01234567 = vcombine_s8(va0x01234567, va1x01234567);
        int8x16_t va23x01234567 = vcombine_s8(va2x01234567, va3x01234567);
        int8x16_t va45x01234567 = vcombine_s8(va4x01234567, va5x01234567);
        int8x16_t va67x01234567 = vcombine_s8(va6x01234567, va7x01234567);

        // Load a 16x4 block of weights.
        const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 8x4 * 4x4 --> 8x4.
        vacc01x01 = vmmlaq_s32(vacc01x01, va01x01234567, vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, va01x01234567, vb23x01234567);
        vacc23x01 = vmmlaq_s32(vacc23x01, va23x01234567, vb01x01234567);
        vacc23x23 = vmmlaq_s32(vacc23x23, va23x01234567, vb23x01234567);
        vacc45x01 = vmmlaq_s32(vacc45x01, va45x01234567, vb01x01234567);
        vacc45x23 = vmmlaq_s32(vacc45x23, va45x01234567, vb23x01234567);
        vacc67x01 = vmmlaq_s32(vacc67x01, va67x01234567, vb01x01234567);
        vacc67x23 = vmmlaq_s32(vacc67x23, va67x01234567, vb23x01234567);
        k -= 8 * sizeof(int8_t);
      } while (k != 0);
    }

    int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
    int32x4_t vacc1x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
    int32x4_t vacc2x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x01), vreinterpretq_u64_s32(vacc23x23)));
    int32x4_t vacc3x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x01), vreinterpretq_u64_s32(vacc23x23)));
    int32x4_t vacc4x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x01), vreinterpretq_u64_s32(vacc45x23)));
    int32x4_t vacc5x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45x01), vreinterpretq_u64_s32(vacc45x23)));
    int32x4_t vacc6x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc67x01), vreinterpretq_u64_s32(vacc67x23)));
    int32x4_t vacc7x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc67x01), vreinterpretq_u64_s32(vacc67x23)));
    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout4x0123 = vcvtq_f32_s32(vacc4x0123);
    float32x4_t vout5x0123 = vcvtq_f32_s32(vacc5x0123);
    float32x4_t vout6x0123 = vcvtq_f32_s32(vacc6x0123);
    float32x4_t vout7x0123 = vcvtq_f32_s32(vacc7x0123);

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    const float32x4_t vinput_scale45 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[4].zero_point));
    vout4x0123 = vmulq_lane_f32(vout4x0123, vget_low_f32(vinput_scale45), 1);
    vout5x0123 = vmulq_lane_f32(vout5x0123, vget_high_f32(vinput_scale45), 1);
    const float32x4_t vinput_scale67 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[6].zero_point));
    vout6x0123 = vmulq_lane_f32(vout6x0123, vget_low_f32(vinput_scale67), 1);
    vout7x0123 = vmulq_lane_f32(vout7x0123, vget_high_f32(vinput_scale67), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;

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

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout4x0123 = vmaxq_f32(vout4x0123, voutput_min);
    vout5x0123 = vmaxq_f32(vout5x0123, voutput_min);
    vout6x0123 = vmaxq_f32(vout6x0123, voutput_min);
    vout7x0123 = vmaxq_f32(vout7x0123, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout4x0123 = vminq_f32(vout4x0123, voutput_max);
    vout5x0123 = vminq_f32(vout5x0123, voutput_max);
    vout6x0123 = vminq_f32(vout6x0123, voutput_max);
    vout7x0123 = vminq_f32(vout7x0123, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      vst1q_f32(c7, vout7x0123);
      vst1q_f32(c6, vout6x0123);
      vst1q_f32(c5, vout5x0123);
      vst1q_f32(c4, vout4x0123);
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);

      nc -= 4;
    } else {
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
