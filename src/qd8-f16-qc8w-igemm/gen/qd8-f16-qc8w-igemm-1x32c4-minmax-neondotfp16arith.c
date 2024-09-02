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


void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x32c4__neondotfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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
  uint16_t* c0 = (uint16_t*) c;

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
    const int32x4_t vksumGHIJ = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xGHIJ = vmulq_s32(vksumGHIJ, vinput_zero_point);
    const int32x4_t vksumKLMN = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xKLMN = vmulq_s32(vksumKLMN, vinput_zero_point);
    const int32x4_t vksumOPQR = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xOPQR = vmulq_s32(vksumOPQR, vinput_zero_point);
    const int32x4_t vksumSTUV = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xSTUV = vmulq_s32(vksumSTUV, vinput_zero_point);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      // Inner accumulation loop along the 32 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 1x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

        // Load a 8x32 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xGHIJ = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xKLMN = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xOPQR = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xSTUV = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xGHIJ = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xKLMN = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xOPQR = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xSTUV = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x8 * 8x32 --> 1x32.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc0xGHIJ = vdotq_lane_s32(vacc0xGHIJ, vb0123xGHIJ, va0x01234567, 0);
        vacc0xKLMN = vdotq_lane_s32(vacc0xKLMN, vb0123xKLMN, va0x01234567, 0);
        vacc0xOPQR = vdotq_lane_s32(vacc0xOPQR, vb0123xOPQR, va0x01234567, 0);
        vacc0xSTUV = vdotq_lane_s32(vacc0xSTUV, vb0123xSTUV, va0x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
        vacc0xGHIJ = vdotq_lane_s32(vacc0xGHIJ, vb4567xGHIJ, va0x01234567, 1);
        vacc0xKLMN = vdotq_lane_s32(vacc0xKLMN, vb4567xKLMN, va0x01234567, 1);
        vacc0xOPQR = vdotq_lane_s32(vacc0xOPQR, vb4567xOPQR, va0x01234567, 1);
        vacc0xSTUV = vdotq_lane_s32(vacc0xSTUV, vb4567xSTUV, va0x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 1x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);

        // Load a 4x32 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xGHIJ = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xKLMN = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xOPQR = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xSTUV = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x4 * 4x32 --> 1x32.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc0xGHIJ = vdotq_lane_s32(vacc0xGHIJ, vb0123xGHIJ, va0x01234567, 0);
        vacc0xKLMN = vdotq_lane_s32(vacc0xKLMN, vb0123xKLMN, va0x01234567, 0);
        vacc0xOPQR = vdotq_lane_s32(vacc0xOPQR, vb0123xOPQR, va0x01234567, 0);
        vacc0xSTUV = vdotq_lane_s32(vacc0xSTUV, vb0123xSTUV, va0x01234567, 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout0xGHIJ = vcvtq_f32_s32(vacc0xGHIJ);
    float32x4_t vout0xKLMN = vcvtq_f32_s32(vacc0xKLMN);
    float32x4_t vout0xOPQR = vcvtq_f32_s32(vacc0xOPQR);
    float32x4_t vout0xSTUV = vcvtq_f32_s32(vacc0xSTUV);
    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale);
    vout0xGHIJ = vmulq_f32(vout0xGHIJ, vinput_scale);
    vout0xKLMN = vmulq_f32(vout0xKLMN, vinput_scale);
    vout0xOPQR = vmulq_f32(vout0xOPQR, vinput_scale);
    vout0xSTUV = vmulq_f32(vout0xSTUV, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleKLMN = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleOPQR = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleSTUV = vld1q_f32(w); w = (const float*) w + 4;

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
    const float32x4_t vbiasGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xGHIJ = vfmaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
    #else
      vout0xGHIJ = vmlaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
    #endif
    const float32x4_t vbiasKLMN = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xKLMN = vfmaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
    #else
      vout0xKLMN = vmlaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
    #endif
    const float32x4_t vbiasOPQR = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xOPQR = vfmaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
    #else
      vout0xOPQR = vmlaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
    #endif
    const float32x4_t vbiasSTUV = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xSTUV = vfmaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
    #else
      vout0xSTUV = vmlaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    float16x8_t vfp16out0xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout0xGHIJ), vcvt_f16_f32(vout0xKLMN));
    float16x8_t vfp16out0xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout0xOPQR), vcvt_f16_f32(vout0xSTUV));

    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t*) &params->scalar.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    vfp16out0xGHIJKLMN = vmaxq_f16(vfp16out0xGHIJKLMN, voutput_min);
    vfp16out0xOPQRSTUV = vmaxq_f16(vfp16out0xOPQRSTUV, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t*) &params->scalar.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);
    vfp16out0xGHIJKLMN = vminq_f16(vfp16out0xGHIJKLMN, voutput_max);
    vfp16out0xOPQRSTUV = vminq_f16(vfp16out0xOPQRSTUV, voutput_max);
    if XNN_LIKELY(nc >= 32) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));
      vst1q_u16(c0 + 16, vreinterpretq_u16_f16(vfp16out0xGHIJKLMN));
      vst1q_u16(c0 + 24, vreinterpretq_u16_f16(vfp16out0xOPQRSTUV));

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 32;
    } else {
     if (nc & 16) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0xGHIJKLMN;
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x89ABCDEF)); c0 += 8;
       vfp16out0x89ABCDEF = vfp16out0xOPQRSTUV;
     }
     if (nc & 8) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
     }
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
