// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-neondot.c.in
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


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128(
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

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  do {
    const int32x4_t vinput_zero_point0 = vld1q_dup_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vinit0x0123 = vmulq_s32(vksum0123, vinput_zero_point0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vinit0x4567 = vmulq_s32(vksum4567, vinput_zero_point0);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vinit0x89AB = vmulq_s32(vksum89AB, vinput_zero_point0);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vinit0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point0);
    int32x4_t vacc0x01 = vreinterpretq_s32_u64(vmovl_u32(vget_low_u32(vreinterpretq_u32_s32(vinit0x0123))));
    int32x4_t vacc0x23 = vreinterpretq_s32_u64(vmovl_u32(vget_high_u32(vreinterpretq_u32_s32(vinit0x0123))));
    int32x4_t vacc0x45 = vreinterpretq_s32_u64(vmovl_u32(vget_low_u32(vreinterpretq_u32_s32(vinit0x4567))));
    int32x4_t vacc0x67 = vreinterpretq_s32_u64(vmovl_u32(vget_high_u32(vreinterpretq_u32_s32(vinit0x4567))));
    int32x4_t vacc0x89 = vreinterpretq_s32_u64(vmovl_u32(vget_low_u32(vreinterpretq_u32_s32(vinit0x89AB))));
    int32x4_t vacc0xAB = vreinterpretq_s32_u64(vmovl_u32(vget_high_u32(vreinterpretq_u32_s32(vinit0x89AB))));
    int32x4_t vacc0xCD = vreinterpretq_s32_u64(vmovl_u32(vget_low_u32(vreinterpretq_u32_s32(vinit0xCDEF))));
    int32x4_t vacc0xEF = vreinterpretq_s32_u64(vmovl_u32(vget_high_u32(vreinterpretq_u32_s32(vinit0xCDEF))));

    size_t k = kc;
    for (; k >= 16 * sizeof(int8_t); k -= 16 * sizeof(int8_t)) {
      const uint64x2x2_t va0c0123456789ABCDEF = vld2q_dup_u64((const void*) a0); a0 += 16;

      const int8x16_t vb01c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABc01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDc01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFc01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      vacc0x01 = vdotq_s32(vacc0x01, vb01c01234567, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[0]));
      vacc0x23 = vdotq_s32(vacc0x23, vb23c01234567, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[0]));
      vacc0x45 = vdotq_s32(vacc0x45, vb45c01234567, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[0]));
      vacc0x67 = vdotq_s32(vacc0x67, vb67c01234567, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[0]));
      vacc0x89 = vdotq_s32(vacc0x89, vb89c01234567, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[0]));
      vacc0xAB = vdotq_s32(vacc0xAB, vbABc01234567, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[0]));
      vacc0xCD = vdotq_s32(vacc0xCD, vbCDc01234567, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[0]));
      vacc0xEF = vdotq_s32(vacc0xEF, vbEFc01234567, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[0]));

      const int8x16_t vb01c89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23c89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45c89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67c89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89c89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABc89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDc89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFc89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vacc0x01 = vdotq_s32(vacc0x01, vb01c89ABCDEF, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[1]));
      vacc0x23 = vdotq_s32(vacc0x23, vb23c89ABCDEF, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[1]));
      vacc0x45 = vdotq_s32(vacc0x45, vb45c89ABCDEF, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[1]));
      vacc0x67 = vdotq_s32(vacc0x67, vb67c89ABCDEF, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[1]));
      vacc0x89 = vdotq_s32(vacc0x89, vb89c89ABCDEF, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[1]));
      vacc0xAB = vdotq_s32(vacc0xAB, vbABc89ABCDEF, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[1]));
      vacc0xCD = vdotq_s32(vacc0xCD, vbCDc89ABCDEF, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[1]));
      vacc0xEF = vdotq_s32(vacc0xEF, vbEFc89ABCDEF, vreinterpretq_s8_u64(va0c0123456789ABCDEF.val[1]));
    }
    if XNN_LIKELY(k != 0) {
      assert(k == 8 * sizeof(int8_t));

      const int8x16_t va0c01234567 = vreinterpretq_s8_u64(vld1q_dup_u64((const void*) a0)); a0 += 8;

      const int8x16_t vb01c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89c01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABc01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDc01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFc01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      vacc0x01 = vdotq_s32(vacc0x01, vb01c01234567, va0c01234567);
      vacc0x23 = vdotq_s32(vacc0x23, vb23c01234567, va0c01234567);
      vacc0x45 = vdotq_s32(vacc0x45, vb45c01234567, va0c01234567);
      vacc0x67 = vdotq_s32(vacc0x67, vb67c01234567, va0c01234567);
      vacc0x89 = vdotq_s32(vacc0x89, vb89c01234567, va0c01234567);
      vacc0xAB = vdotq_s32(vacc0xAB, vbABc01234567, va0c01234567);
      vacc0xCD = vdotq_s32(vacc0xCD, vbCDc01234567, va0c01234567);
      vacc0xEF = vdotq_s32(vacc0xEF, vbEFc01234567, va0c01234567);
    }

    #if XNN_ARCH_ARM64
      int32x4_t vacc0x0123 = vpaddq_s32(vacc0x01, vacc0x23);
      int32x4_t vacc0x4567 = vpaddq_s32(vacc0x45, vacc0x67);
      int32x4_t vacc0x89AB = vpaddq_s32(vacc0x89, vacc0xAB);
      int32x4_t vacc0xCDEF = vpaddq_s32(vacc0xCD, vacc0xEF);
    #else
      int32x4_t vacc0x0123 = vcombine_s32(vpadd_s32(vget_low_s32(vacc0x01), vget_high_s32(vacc0x01)), vpadd_s32(vget_low_s32(vacc0x23), vget_high_s32(vacc0x23)));
      int32x4_t vacc0x4567 = vcombine_s32(vpadd_s32(vget_low_s32(vacc0x45), vget_high_s32(vacc0x45)), vpadd_s32(vget_low_s32(vacc0x67), vget_high_s32(vacc0x67)));
      int32x4_t vacc0x89AB = vcombine_s32(vpadd_s32(vget_low_s32(vacc0x89), vget_high_s32(vacc0x89)), vpadd_s32(vget_low_s32(vacc0xAB), vget_high_s32(vacc0xAB)));
      int32x4_t vacc0xCDEF = vcombine_s32(vpadd_s32(vget_low_s32(vacc0xCD), vget_high_s32(vacc0xCD)), vpadd_s32(vget_low_s32(vacc0xEF), vget_high_s32(vacc0xEF)));
    #endif

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
