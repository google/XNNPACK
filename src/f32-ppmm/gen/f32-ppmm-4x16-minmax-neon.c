// Auto-generated file. Do not edit!
//   Template: src/f32-ppmm/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/ppmm.h>


void xnn_f32_ppmm_minmax_ukernel_4x16__neon(
  size_t mr,
  size_t nc,
  size_t kc,
  const float* restrict a,
  const float* restrict w,
  float* restrict c,
  size_t cm_stride,
  size_t cn_stride,
  const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  #if XNN_ARCH_ARM64
    const float32x4x2_t vminmax = vld2q_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vminmax.val[0];
    const float32x4_t vmax = vminmax.val[1];
  #else
    const float32x2x2_t vminmax = vld2_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vcombine_f32(vminmax.val[0], vminmax.val[0]);
    const float32x4_t vmax = vcombine_f32(vminmax.val[1], vminmax.val[1]);
  #endif

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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }


  do {
    float32x4_t vacc0x0123 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x4567 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x89AB = vld1q_f32(w); w += 4;
    float32x4_t vacc0xCDEF = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0123 = vacc0x0123;
    float32x4_t vacc1x4567 = vacc0x4567;
    float32x4_t vacc1x89AB = vacc0x89AB;
    float32x4_t vacc1xCDEF = vacc0xCDEF;
    float32x4_t vacc2x0123 = vacc0x0123;
    float32x4_t vacc2x4567 = vacc0x4567;
    float32x4_t vacc2x89AB = vacc0x89AB;
    float32x4_t vacc2xCDEF = vacc0xCDEF;
    float32x4_t vacc3x0123 = vacc0x0123;
    float32x4_t vacc3x4567 = vacc0x4567;
    float32x4_t vacc3x89AB = vacc0x89AB;
    float32x4_t vacc3xCDEF = vacc0xCDEF;

    size_t k = kc;
    do {
      const float32x4_t va0123 = vld1q_f32(a); a += 4;

      const float32x4_t vb0123 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567 = vld1q_f32(w); w += 4;
      const float32x4_t vb89AB = vld1q_f32(w); w += 4;
      const float32x4_t vbCDEF = vld1q_f32(w); w += 4;

      vacc0x0123 = vmlaq_lane_f32(vacc0x0123, vb0123, vget_low_f32(va0123), 0);
      vacc1x0123 = vmlaq_lane_f32(vacc1x0123, vb0123, vget_low_f32(va0123), 1);
      vacc2x0123 = vmlaq_lane_f32(vacc2x0123, vb0123, vget_high_f32(va0123), 0);
      vacc3x0123 = vmlaq_lane_f32(vacc3x0123, vb0123, vget_high_f32(va0123), 1);
      vacc0x4567 = vmlaq_lane_f32(vacc0x4567, vb4567, vget_low_f32(va0123), 0);
      vacc1x4567 = vmlaq_lane_f32(vacc1x4567, vb4567, vget_low_f32(va0123), 1);
      vacc2x4567 = vmlaq_lane_f32(vacc2x4567, vb4567, vget_high_f32(va0123), 0);
      vacc3x4567 = vmlaq_lane_f32(vacc3x4567, vb4567, vget_high_f32(va0123), 1);
      vacc0x89AB = vmlaq_lane_f32(vacc0x89AB, vb89AB, vget_low_f32(va0123), 0);
      vacc1x89AB = vmlaq_lane_f32(vacc1x89AB, vb89AB, vget_low_f32(va0123), 1);
      vacc2x89AB = vmlaq_lane_f32(vacc2x89AB, vb89AB, vget_high_f32(va0123), 0);
      vacc3x89AB = vmlaq_lane_f32(vacc3x89AB, vb89AB, vget_high_f32(va0123), 1);
      vacc0xCDEF = vmlaq_lane_f32(vacc0xCDEF, vbCDEF, vget_low_f32(va0123), 0);
      vacc1xCDEF = vmlaq_lane_f32(vacc1xCDEF, vbCDEF, vget_low_f32(va0123), 1);
      vacc2xCDEF = vmlaq_lane_f32(vacc2xCDEF, vbCDEF, vget_high_f32(va0123), 0);
      vacc3xCDEF = vmlaq_lane_f32(vacc3xCDEF, vbCDEF, vget_high_f32(va0123), 1);

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc1x4567 = vminq_f32(vacc1x4567, vmax);
    vacc2x4567 = vminq_f32(vacc2x4567, vmax);
    vacc3x4567 = vminq_f32(vacc3x4567, vmax);
    vacc0x89AB = vminq_f32(vacc0x89AB, vmax);
    vacc1x89AB = vminq_f32(vacc1x89AB, vmax);
    vacc2x89AB = vminq_f32(vacc2x89AB, vmax);
    vacc3x89AB = vminq_f32(vacc3x89AB, vmax);
    vacc0xCDEF = vminq_f32(vacc0xCDEF, vmax);
    vacc1xCDEF = vminq_f32(vacc1xCDEF, vmax);
    vacc2xCDEF = vminq_f32(vacc2xCDEF, vmax);
    vacc3xCDEF = vminq_f32(vacc3xCDEF, vmax);

    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
    vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
    vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);
    vacc0x89AB = vmaxq_f32(vacc0x89AB, vmin);
    vacc1x89AB = vmaxq_f32(vacc1x89AB, vmin);
    vacc2x89AB = vmaxq_f32(vacc2x89AB, vmin);
    vacc3x89AB = vmaxq_f32(vacc3x89AB, vmin);
    vacc0xCDEF = vmaxq_f32(vacc0xCDEF, vmin);
    vacc1xCDEF = vmaxq_f32(vacc1xCDEF, vmin);
    vacc2xCDEF = vmaxq_f32(vacc2xCDEF, vmin);
    vacc3xCDEF = vmaxq_f32(vacc3xCDEF, vmin);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c3, vacc3x0123);
      vst1q_f32(c3 + 4, vacc3x4567);
      vst1q_f32(c3 + 8, vacc3x89AB);
      vst1q_f32(c3 + 12, vacc3xCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0123);
      vst1q_f32(c2 + 4, vacc2x4567);
      vst1q_f32(c2 + 8, vacc2x89AB);
      vst1q_f32(c2 + 12, vacc2xCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0123);
      vst1q_f32(c1 + 4, vacc1x4567);
      vst1q_f32(c1 + 8, vacc1x89AB);
      vst1q_f32(c1 + 12, vacc1xCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0123);
      vst1q_f32(c0 + 4, vacc0x4567);
      vst1q_f32(c0 + 8, vacc0x89AB);
      vst1q_f32(c0 + 12, vacc0xCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float*) ((uintptr_t) a - kc * 4);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c3, vacc3x4567); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c2, vacc2x4567); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c1, vacc1x4567); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;
        vst1q_f32(c0, vacc0x4567); c0 += 4;

        vacc3x0123 = vacc3x89AB;
        vacc3x4567 = vacc3xCDEF;
        vacc2x0123 = vacc2x89AB;
        vacc2x4567 = vacc2xCDEF;
        vacc1x0123 = vacc1x89AB;
        vacc1x4567 = vacc1xCDEF;
        vacc0x0123 = vacc0x89AB;
        vacc0x4567 = vacc0xCDEF;
      }
      if (nc & 4) {
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc3x0123 = vacc3x4567;
        vacc3x4567 = vacc3x89AB;
        vacc3x89AB = vacc3xCDEF;
        vacc2x0123 = vacc2x4567;
        vacc2x4567 = vacc2x89AB;
        vacc2x89AB = vacc2xCDEF;
        vacc1x0123 = vacc1x4567;
        vacc1x4567 = vacc1x89AB;
        vacc1x89AB = vacc1xCDEF;
        vacc0x0123 = vacc0x4567;
        vacc0x4567 = vacc0x89AB;
        vacc0x89AB = vacc0xCDEF;
      }
      float32x2_t vacc3x01 = vget_low_f32(vacc3x0123);
      float32x2_t vacc2x01 = vget_low_f32(vacc2x0123);
      float32x2_t vacc1x01 = vget_low_f32(vacc1x0123);
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c3, vacc3x01); c3 += 2;
        vst1_f32(c2, vacc2x01); c2 += 2;
        vst1_f32(c1, vacc1x01); c1 += 2;
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc3x01 = vget_high_f32(vacc3x0123);
        vacc2x01 = vget_high_f32(vacc2x0123);
        vacc1x01 = vget_high_f32(vacc1x0123);
        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c3, vacc3x01, 0);
        vst1_lane_f32(c2, vacc2x01, 0);
        vst1_lane_f32(c1, vacc1x01, 0);
        vst1_lane_f32(c0, vacc0x01, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
