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

#include "xnnpack/gemm.h"


void xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }
  const int32x4_t vminus_kernel_zero_point = vld1q_dup_s32(&params->scalar.minus_kernel_zero_point);
  const uint16x8_t vmask = vmovq_n_u16(UINT16_C(0xF));

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        const float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        const float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        const float32x4_t va3 = vld1q_f32(a3); a3 += 4;


        const uint8x16_t vw01234567c0123 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint16x8_t vxw01234567c01 = vmovl_u8(vget_low_u8(vw01234567c0123));
        const uint16x8_t vxw01234567c23 = vmovl_u8(vget_high_u8(vw01234567c0123));
        const uint16x8_t vxw01234567c0 = vandq_u16(vxw01234567c01, vmask);
        const uint16x8_t vxw01234567c1 = vshrq_n_u16(vxw01234567c01, 4);
        const uint16x8_t vxw01234567c2 = vandq_u16(vxw01234567c23, vmask);
        const uint16x8_t vxw01234567c3 = vshrq_n_u16(vxw01234567c23, 4);
        const int32x4_t vxw0123c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c0)));
        const int32x4_t vxw4567c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c0)));
        const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
        const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, vget_low_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, vget_low_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, vget_low_f32(va3), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, vget_low_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, vget_low_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, vget_low_f32(va3), 0);

        const int32x4_t vxw0123c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c1)));
        const int32x4_t vxw4567c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c1)));
        const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
        const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, vget_low_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, vget_low_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, vget_low_f32(va3), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, vget_low_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, vget_low_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, vget_low_f32(va3), 1);

        const int32x4_t vxw0123c2 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c2)));
        const int32x4_t vxw4567c2 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c2)));
        const float32x4_t vb0123c2 = vcvtq_f32_s32(vxw0123c2);
        const float32x4_t vb4567c2 = vcvtq_f32_s32(vxw4567c2);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c2, vget_high_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c2, vget_high_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c2, vget_high_f32(va3), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c2, vget_high_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c2, vget_high_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c2, vget_high_f32(va3), 0);

        const int32x4_t vxw0123c3 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c3)));
        const int32x4_t vxw4567c3 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c3)));
        const float32x4_t vb0123c3 = vcvtq_f32_s32(vxw0123c3);
        const float32x4_t vb4567c3 = vcvtq_f32_s32(vxw4567c3);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c3, vget_high_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c3, vget_high_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c3, vget_high_f32(va3), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c3, vget_high_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c3, vget_high_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c3, vget_high_f32(va3), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;


        const uint8x8_t vw01234567c01 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint16x8_t vxw01234567c01 = vmovl_u8(vw01234567c01);
        const uint16x8_t vxw01234567c0 = vandq_u16(vxw01234567c01, vmask);
        const uint16x8_t vxw01234567c1 = vshrq_n_u16(vxw01234567c01, 4);
        const int32x4_t vxw0123c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c0)));
        const int32x4_t vxw4567c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c0)));
        const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
        const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, va1, 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, va2, 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, va3, 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, va1, 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, va2, 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, va3, 0);

        const int32x4_t vxw0123c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c1)));
        const int32x4_t vxw4567c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c1)));
        const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
        const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, va1, 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, va2, 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, va3, 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, va1, 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, va2, 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, va3, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
        const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
        const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
        const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;

        const uint8x8_t vw01234567 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint16x8_t vxw01234567 = vmovl_u8(vw01234567);
        const int32x4_t vxw0123 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567)));
        const int32x4_t vxw4567 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567)));
        const float32x4_t vb0123 = vcvtq_f32_s32(vxw0123);
        const float32x4_t vb4567 = vcvtq_f32_s32(vxw4567);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567);
      }
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale4567 = vld1q_f32(w); w = ((const float*) w + 4);
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    vacc1x0 = vmulq_f32(vacc1x0, vscale0123);
    vacc2x0 = vmulq_f32(vacc2x0, vscale0123);
    vacc3x0 = vmulq_f32(vacc3x0, vscale0123);
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    vacc1x1 = vmulq_f32(vacc1x1, vscale4567);
    vacc2x1 = vmulq_f32(vacc2x1, vscale4567);
    vacc3x1 = vmulq_f32(vacc3x1, vscale4567);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c3, vacc3); c3 += 2;

        vacc0 = vget_high_f32(vacc0x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc3 = vget_high_f32(vacc3x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c3, vacc3, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
