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

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint16_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  size_t bl = params->fp16arith.blocksize;
  assert(bl <= kc);
  assert(bl != 0);
  size_t n_blocks = kc / bl;
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;
  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));

  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with scaled vksum. 8 scaled vksum values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    const float32x4_t vinput_zero_point01 = vcvtq_f32_s32(vld1q_dup_s32(&quantization_params[0].zero_point));
    const float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x0123 = vmulq_f32(vksum0123, vinput_zero_point01);
    const float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x4567 = vmulq_f32(vksum4567, vinput_zero_point01);


    for (size_t nb=0; nb < n_blocks; ++nb) {
      int32x4_t vacc01x01 = vdupq_n_s32(0);
      int32x4_t vacc01x23 = vdupq_n_s32(0);
      int32x4_t vacc01x45 = vdupq_n_s32(0);
      int32x4_t vacc01x67 = vdupq_n_s32(0);

      size_t k = bl;
      // 2x partial unrolled loop to load 8 bytes at a time.

      uint64x2x2_t va01x0123456789ABCDEF;
      va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

      while (k >= 16 * sizeof(int8_t)) {
        // Load a 1x16 block of activations.
        #if XNN_ARCH_ARM64
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
        #else
          va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[0], 0); a0 += 8;
          va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[1], 0); a0 += 8;
        #endif

        const int8x16_t va01x01234567 = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]);
        const int8x16_t va01x89ABCDEF = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]);

        // Load a 16x8 block of weights.
        const int8x16_t vb01x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01x01234567 = vshlq_n_s8(vb01x0123456789ABCDEF, 4);
        const int8x16_t vb23x01234567 = vshlq_n_s8(vb23x0123456789ABCDEF, 4);
        const int8x16_t vb45x01234567 = vshlq_n_s8(vb45x0123456789ABCDEF, 4);
        const int8x16_t vb67x01234567 = vshlq_n_s8(vb67x0123456789ABCDEF, 4);
        const int8x16_t vb01x89ABCDEF = vandq_s8(vb01x0123456789ABCDEF, vmask);
        const int8x16_t vb23x89ABCDEF = vandq_s8(vb23x0123456789ABCDEF, vmask);
        const int8x16_t vb45x89ABCDEF = vandq_s8(vb45x0123456789ABCDEF, vmask);
        const int8x16_t vb67x89ABCDEF = vandq_s8(vb67x0123456789ABCDEF, vmask);

        // Multiply-accumulate: 1x8 * 8x8 --> 1x8.
        vacc01x01 = vmmlaq_s32(vacc01x01, va01x01234567, vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, va01x01234567, vb23x01234567);
        vacc01x45 = vmmlaq_s32(vacc01x45, va01x01234567, vb45x01234567);
        vacc01x67 = vmmlaq_s32(vacc01x67, va01x01234567, vb67x01234567);
        vacc01x01 = vmmlaq_s32(vacc01x01, va01x89ABCDEF, vb01x89ABCDEF);
        vacc01x23 = vmmlaq_s32(vacc01x23, va01x89ABCDEF, vb23x89ABCDEF);
        vacc01x45 = vmmlaq_s32(vacc01x45, va01x89ABCDEF, vb45x89ABCDEF);
        vacc01x67 = vmmlaq_s32(vacc01x67, va01x89ABCDEF, vb67x89ABCDEF);

        k -= 16 * sizeof(int8_t);
      }
      // Handle up to 8 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 1x8 block of activations.
        uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;

        // Load a 16x8 block of weights.
        const int8x16_t vb01x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01x01234567 = vshlq_n_s8(vb01x0123456789ABCDEF, 4);
        const int8x16_t vb23x01234567 = vshlq_n_s8(vb23x0123456789ABCDEF, 4);
        const int8x16_t vb45x01234567 = vshlq_n_s8(vb45x0123456789ABCDEF, 4);
        const int8x16_t vb67x01234567 = vshlq_n_s8(vb67x0123456789ABCDEF, 4);

        // Multiply-accumulate: 1x4 * 4x8 --> 1x8.
        vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x01234567), vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x01234567), vb23x01234567);
        vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x01234567), vb45x01234567);
        vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x01234567), vb67x01234567);
      }

      #if XNN_ARCH_ARM64
        int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
        int32x4_t vacc0x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
      #else
        int32x4_t vacc0x0123 = vcombine_s32(vget_low_s32(vacc01x01), vget_low_s32(vacc01x23));
        int32x4_t vacc0x4567 = vcombine_s32(vget_low_s32(vacc01x45), vget_low_s32(vacc01x67));
      #endif
      const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
      const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
      float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
      vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
      vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
    }
    const float32x4_t one_sixteenth = vdupq_n_f32(1/16.0);
    vout0x0123 = vmulq_f32(vout0x0123, one_sixteenth);
    vout0x4567 = vmulq_f32(vout0x4567, one_sixteenth);

    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);


    #if XNN_ARCH_ARM64
      const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x0123 = vaddq_f32(vbias0123, vout0x0123);
      const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x4567 = vaddq_f32(vbias4567, vout0x4567);
    #else
      const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x0123 = vaddq_f32(vbias0123, vout0x0123);
      const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x4567 = vaddq_f32(vbias4567, vout0x4567);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));

    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
