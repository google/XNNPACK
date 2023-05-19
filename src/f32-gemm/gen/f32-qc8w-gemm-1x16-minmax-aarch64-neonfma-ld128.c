// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/neon-ld128.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>


void xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128(
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
    float32x4_t vacc0x0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x89AB = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0xCDEF = vld1q_f32(w); w = (const float*) w + 4;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const float32x4_t va0 = vld1q_f32(a0); a0 += 4;


      const int8x8_t vw01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vw89ABCDEFc0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int16x8_t vxw01234567c0 = vmovl_s8(vw01234567c0);
      const int16x8_t vxw89ABCDEFc0 = vmovl_s8(vw89ABCDEFc0);
      const int32x4_t vxw0123c0 = vmovl_s16(vget_low_s16(vxw01234567c0));
      const int32x4_t vxw4567c0 = vmovl_s16(vget_high_s16(vxw01234567c0));
      const int32x4_t vxw89ABc0 = vmovl_s16(vget_low_s16(vxw89ABCDEFc0));
      const int32x4_t vxwCDEFc0 = vmovl_s16(vget_high_s16(vxw89ABCDEFc0));
      const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
      const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);
      const float32x4_t vb89ABc0 = vcvtq_f32_s32(vxw89ABc0);
      const float32x4_t vbCDEFc0 = vcvtq_f32_s32(vxwCDEFc0);

      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c0, vget_low_f32(va0), 0);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c0, vget_low_f32(va0), 0);
      vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc0, vget_low_f32(va0), 0);
      vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc0, vget_low_f32(va0), 0);

      const int8x8_t vw01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vw89ABCDEFc1 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int16x8_t vxw01234567c1 = vmovl_s8(vw01234567c1);
      const int16x8_t vxw89ABCDEFc1 = vmovl_s8(vw89ABCDEFc1);
      const int32x4_t vxw0123c1 = vmovl_s16(vget_low_s16(vxw01234567c1));
      const int32x4_t vxw4567c1 = vmovl_s16(vget_high_s16(vxw01234567c1));
      const int32x4_t vxw89ABc1 = vmovl_s16(vget_low_s16(vxw89ABCDEFc1));
      const int32x4_t vxwCDEFc1 = vmovl_s16(vget_high_s16(vxw89ABCDEFc1));
      const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
      const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);
      const float32x4_t vb89ABc1 = vcvtq_f32_s32(vxw89ABc1);
      const float32x4_t vbCDEFc1 = vcvtq_f32_s32(vxwCDEFc1);

      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c1, vget_low_f32(va0), 1);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c1, vget_low_f32(va0), 1);
      vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc1, vget_low_f32(va0), 1);
      vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc1, vget_low_f32(va0), 1);

      const int8x8_t vw01234567c2 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vw89ABCDEFc2 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int16x8_t vxw01234567c2 = vmovl_s8(vw01234567c2);
      const int16x8_t vxw89ABCDEFc2 = vmovl_s8(vw89ABCDEFc2);
      const int32x4_t vxw0123c2 = vmovl_s16(vget_low_s16(vxw01234567c2));
      const int32x4_t vxw4567c2 = vmovl_s16(vget_high_s16(vxw01234567c2));
      const int32x4_t vxw89ABc2 = vmovl_s16(vget_low_s16(vxw89ABCDEFc2));
      const int32x4_t vxwCDEFc2 = vmovl_s16(vget_high_s16(vxw89ABCDEFc2));
      const float32x4_t vb0123c2 = vcvtq_f32_s32(vxw0123c2);
      const float32x4_t vb4567c2 = vcvtq_f32_s32(vxw4567c2);
      const float32x4_t vb89ABc2 = vcvtq_f32_s32(vxw89ABc2);
      const float32x4_t vbCDEFc2 = vcvtq_f32_s32(vxwCDEFc2);

      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c2, vget_high_f32(va0), 0);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c2, vget_high_f32(va0), 0);
      vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc2, vget_high_f32(va0), 0);
      vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc2, vget_high_f32(va0), 0);

      const int8x8_t vw01234567c3 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vw89ABCDEFc3 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int16x8_t vxw01234567c3 = vmovl_s8(vw01234567c3);
      const int16x8_t vxw89ABCDEFc3 = vmovl_s8(vw89ABCDEFc3);
      const int32x4_t vxw0123c3 = vmovl_s16(vget_low_s16(vxw01234567c3));
      const int32x4_t vxw4567c3 = vmovl_s16(vget_high_s16(vxw01234567c3));
      const int32x4_t vxw89ABc3 = vmovl_s16(vget_low_s16(vxw89ABCDEFc3));
      const int32x4_t vxwCDEFc3 = vmovl_s16(vget_high_s16(vxw89ABCDEFc3));
      const float32x4_t vb0123c3 = vcvtq_f32_s32(vxw0123c3);
      const float32x4_t vb4567c3 = vcvtq_f32_s32(vxw4567c3);
      const float32x4_t vb89ABc3 = vcvtq_f32_s32(vxw89ABc3);
      const float32x4_t vbCDEFc3 = vcvtq_f32_s32(vxwCDEFc3);

      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c3, vget_high_f32(va0), 1);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c3, vget_high_f32(va0), 1);
      vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc3, vget_high_f32(va0), 1);
      vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc3, vget_high_f32(va0), 1);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

        const int8x8_t vw01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vw89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxw01234567 = vmovl_s8(vw01234567);
        const int16x8_t vxw89ABCDEF = vmovl_s8(vw89ABCDEF);
        const int32x4_t vxw0123 = vmovl_s16(vget_low_s16(vxw01234567));
        const int32x4_t vxw89AB = vmovl_s16(vget_low_s16(vxw89ABCDEF));
        const int32x4_t vxw4567 = vmovl_s16(vget_high_s16(vxw01234567));
        const int32x4_t vxwCDEF = vmovl_s16(vget_high_s16(vxw89ABCDEF));
        const float32x4_t vb0123 = vcvtq_f32_s32(vxw0123);
        const float32x4_t vb4567 = vcvtq_f32_s32(vxw4567);
        const float32x4_t vb89AB = vcvtq_f32_s32(vxw89AB);
        const float32x4_t vbCDEF = vcvtq_f32_s32(vxwCDEF);

        vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123);
        vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567);
        vacc0x89AB = vfmaq_f32(vacc0x89AB, va0, vb89AB);
        vacc0xCDEF = vfmaq_f32(vacc0xCDEF, va0, vbCDEF);

        k -= sizeof(float);
      } while (k != 0);
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale4567 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale89AB = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscaleCDEF = vld1q_f32(w); w = ((const float*) w + 4);
    vacc0x0123 = vmulq_f32(vacc0x0123, vscale0123);
    vacc0x4567 = vmulq_f32(vacc0x4567, vscale4567);
    vacc0x89AB = vmulq_f32(vacc0x89AB, vscale89AB);
    vacc0xCDEF = vmulq_f32(vacc0xCDEF, vscaleCDEF);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc0x89AB = vminq_f32(vacc0x89AB, vmax);
    vacc0xCDEF = vminq_f32(vacc0xCDEF, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc0x89AB = vmaxq_f32(vacc0x89AB, vmin);
    vacc0xCDEF = vmaxq_f32(vacc0xCDEF, vmin);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vacc0x0123);
      vst1q_f32(c0 + 4, vacc0x4567);
      vst1q_f32(c0 + 8, vacc0x89AB);
      vst1q_f32(c0 + 12, vacc0xCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;

    } else {
      if (nc & 8) {
        vst1q_f32(c0, vacc0x0123); c0 += 4;
        vst1q_f32(c0, vacc0x4567); c0 += 4;

        vacc0x0123 = vacc0x89AB;
        vacc0x4567 = vacc0xCDEF;
      }
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc0x0123 = vacc0x4567;
        vacc0x4567 = vacc0x89AB;
        vacc0x89AB = vacc0xCDEF;
      }
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0x01, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
