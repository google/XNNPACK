// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c2-neon-mull-shuffle.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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

  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    const int32x4_t vizp0 = vld1q_dup_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vizp0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vizp0);

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va0x1 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

      k -= 16 * sizeof(int8_t);
    }
    if (k != 0) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);

    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));

    #if XNN_ARCH_ARM64
      const uint16x8x2_t voutput_minmax = vld2q_dup_u16(&params->fp16arith.min);
      const float16x8_t voutput_min = vreinterpretq_f16_u16(voutput_minmax.val[0]);
      const float16x8_t voutput_max = vreinterpretq_f16_u16(voutput_minmax.val[1]);
    #else
      const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
      const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    #endif
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
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
