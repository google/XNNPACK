// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/neon-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"


void xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w = (const float*) w + 4;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;

      const int8x8_t vw01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vw01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int16x8_t vxw01234567c0 = vmovl_s8(vw01234567c0);
      const int16x8_t vxw01234567c1 = vmovl_s8(vw01234567c1);
      const int32x4_t vxw0123c0 = vmovl_s16(vget_low_s16(vxw01234567c0));
      const int32x4_t vxw4567c0 = vmovl_s16(vget_high_s16(vxw01234567c0));
      const int32x4_t vxw0123c1 = vmovl_s16(vget_low_s16(vxw01234567c1));
      const int32x4_t vxw4567c1 = vmovl_s16(vget_high_s16(vxw01234567c1));
      const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
      const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
      const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);
      const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

      const float32x4_t va0c0 = vdupq_lane_f32(va0, 0);
      vacc0x0 = vfmaq_f32(vacc0x0, va0c0, vb0123c0);
      vacc0x1 = vfmaq_f32(vacc0x1, va0c0, vb4567c0);
      const float32x4_t va0c1 = vdupq_lane_f32(va0, 1);
      vacc0x0 = vfmaq_f32(vacc0x0, va0c1, vb0123c1);
      vacc0x1 = vfmaq_f32(vacc0x1, va0c1, vb4567c1);
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

      const int8x8_t vw01230123 = vreinterpret_s8_u32(vld1_dup_u32(w)); w = (const int8_t*) w + 4;
      const int8x8_t vw45674567 = vreinterpret_s8_u32(vld1_dup_u32(w)); w = (const int8_t*) w + 4;
      const int16x8_t vxw01230123 = vmovl_s8(vw01230123);
      const int16x8_t vxw45674567 = vmovl_s8(vw45674567);
      const int32x4_t vxw0123 = vmovl_s16(vget_low_s16(vxw01230123));
      const int32x4_t vxw4567 = vmovl_s16(vget_low_s16(vxw45674567));
      const float32x4_t vb0123 = vcvtq_f32_s32(vxw0123);
      const float32x4_t vb4567 = vcvtq_f32_s32(vxw4567);

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = (const float*) w + 4;
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32(w); w = (const float*) w + 4;
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;

        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
