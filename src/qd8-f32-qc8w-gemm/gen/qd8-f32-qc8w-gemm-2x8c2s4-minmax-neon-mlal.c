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


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal(
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
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    const int32x4_t vizp01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vizp01), 0);
    int32x4_t vacc1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vizp01), 0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vizp01), 0);
    int32x4_t vacc1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vizp01), 0);

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
      int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
      int8x8_t va1x1 = vld1_s8(a1); a1 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
      const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
      vprod1x0123c0 = vmlal_s8(vprod1x0123c0, vb0123c0x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
      const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
      vprod1x4567c0 = vmlal_s8(vprod1x4567c0, vb4567c0x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
      const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
      vprod1x0123c1 = vmlal_s8(vprod1x0123c1, vb0123c1x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
      const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
      vprod1x4567c1 = vmlal_s8(vprod1x4567c1, vb4567c1x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
      const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
      vprod1x0123c2 = vmlal_s8(vprod1x0123c2, vb0123c2x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
      const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
      vprod1x4567c2 = vmlal_s8(vprod1x4567c2, vb4567c2x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
      const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
      vprod1x0123c3 = vmlal_s8(vprod1x0123c3, vb0123c3x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
      const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
      vprod1x4567c3 = vmlal_s8(vprod1x4567c3, vb4567c3x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

      k -= 16 * sizeof(int8_t);
    }
    if (k != 0) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va1x0 = vld1_s8(a1); a1 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

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

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 0, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 8;
    } else {
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
