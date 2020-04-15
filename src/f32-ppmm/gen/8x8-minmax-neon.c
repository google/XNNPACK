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


void xnn_f32_ppmm_minmax_ukernel_8x8__neon(
  size_t mr,
  size_t nc,
  size_t kc,
  const float*restrict a,
  const float*restrict w,
  float*restrict c,
  size_t cm_stride,
  size_t cn_stride,
  const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

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
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
  }

  do {
    float32x4_t vacc0x0123 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x4567 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0123 = vacc0x0123;
    float32x4_t vacc1x4567 = vacc0x4567;
    float32x4_t vacc2x0123 = vacc0x0123;
    float32x4_t vacc2x4567 = vacc0x4567;
    float32x4_t vacc3x0123 = vacc0x0123;
    float32x4_t vacc3x4567 = vacc0x4567;
    float32x4_t vacc4x0123 = vacc0x0123;
    float32x4_t vacc4x4567 = vacc0x4567;
    float32x4_t vacc5x0123 = vacc0x0123;
    float32x4_t vacc5x4567 = vacc0x4567;
    float32x4_t vacc6x0123 = vacc0x0123;
    float32x4_t vacc6x4567 = vacc0x4567;
    float32x4_t vacc7x0123 = vacc0x0123;
    float32x4_t vacc7x4567 = vacc0x4567;

    size_t k = kc;
    do {
      const float32x4_t va0123 = vld1q_f32(a); a += 4;
      const float32x4_t va4567 = vld1q_f32(a); a += 4;

      const float32x4_t vb0123 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567 = vld1q_f32(w); w += 4;

      vacc0x0123 = vmlaq_lane_f32(vacc0x0123, vb0123, vget_low_f32(va0123), 0);
      vacc1x0123 = vmlaq_lane_f32(vacc1x0123, vb0123, vget_low_f32(va0123), 1);
      vacc2x0123 = vmlaq_lane_f32(vacc2x0123, vb0123, vget_high_f32(va0123), 0);
      vacc3x0123 = vmlaq_lane_f32(vacc3x0123, vb0123, vget_high_f32(va0123), 1);
      vacc4x0123 = vmlaq_lane_f32(vacc4x0123, vb0123, vget_low_f32(va4567), 0);
      vacc5x0123 = vmlaq_lane_f32(vacc5x0123, vb0123, vget_low_f32(va4567), 1);
      vacc6x0123 = vmlaq_lane_f32(vacc6x0123, vb0123, vget_high_f32(va4567), 0);
      vacc7x0123 = vmlaq_lane_f32(vacc7x0123, vb0123, vget_high_f32(va4567), 1);
      vacc0x4567 = vmlaq_lane_f32(vacc0x4567, vb4567, vget_low_f32(va0123), 0);
      vacc1x4567 = vmlaq_lane_f32(vacc1x4567, vb4567, vget_low_f32(va0123), 1);
      vacc2x4567 = vmlaq_lane_f32(vacc2x4567, vb4567, vget_high_f32(va0123), 0);
      vacc3x4567 = vmlaq_lane_f32(vacc3x4567, vb4567, vget_high_f32(va0123), 1);
      vacc4x4567 = vmlaq_lane_f32(vacc4x4567, vb4567, vget_low_f32(va4567), 0);
      vacc5x4567 = vmlaq_lane_f32(vacc5x4567, vb4567, vget_low_f32(va4567), 1);
      vacc6x4567 = vmlaq_lane_f32(vacc6x4567, vb4567, vget_high_f32(va4567), 0);
      vacc7x4567 = vmlaq_lane_f32(vacc7x4567, vb4567, vget_high_f32(va4567), 1);

      k -= sizeof(float);
    } while (k != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc4x0123 = vminq_f32(vacc4x0123, vmax);
    vacc5x0123 = vminq_f32(vacc5x0123, vmax);
    vacc6x0123 = vminq_f32(vacc6x0123, vmax);
    vacc7x0123 = vminq_f32(vacc7x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc1x4567 = vminq_f32(vacc1x4567, vmax);
    vacc2x4567 = vminq_f32(vacc2x4567, vmax);
    vacc3x4567 = vminq_f32(vacc3x4567, vmax);
    vacc4x4567 = vminq_f32(vacc4x4567, vmax);
    vacc5x4567 = vminq_f32(vacc5x4567, vmax);
    vacc6x4567 = vminq_f32(vacc6x4567, vmax);
    vacc7x4567 = vminq_f32(vacc7x4567, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc4x0123 = vmaxq_f32(vacc4x0123, vmin);
    vacc5x0123 = vmaxq_f32(vacc5x0123, vmin);
    vacc6x0123 = vmaxq_f32(vacc6x0123, vmin);
    vacc7x0123 = vmaxq_f32(vacc7x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
    vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
    vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);
    vacc4x4567 = vmaxq_f32(vacc4x4567, vmin);
    vacc5x4567 = vmaxq_f32(vacc5x4567, vmin);
    vacc6x4567 = vmaxq_f32(vacc6x4567, vmin);
    vacc7x4567 = vmaxq_f32(vacc7x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c7, vacc7x0123);
      vst1q_f32(c7 + 4, vacc7x4567);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      vst1q_f32(c6, vacc6x0123);
      vst1q_f32(c6 + 4, vacc6x4567);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      vst1q_f32(c5, vacc5x0123);
      vst1q_f32(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      vst1q_f32(c4, vacc4x0123);
      vst1q_f32(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c3, vacc3x0123);
      vst1q_f32(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0123);
      vst1q_f32(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0123);
      vst1q_f32(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0123);
      vst1q_f32(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float*) ((uintptr_t) a - kc * 8);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c7, vacc7x0123); c7 += 4;
        vst1q_f32(c6, vacc6x0123); c6 += 4;
        vst1q_f32(c5, vacc5x0123); c5 += 4;
        vst1q_f32(c4, vacc4x0123); c4 += 4;
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc7x0123 = vacc7x4567;
        vacc6x0123 = vacc6x4567;
        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;
      }
      float32x2_t vacc7x01 = vget_low_f32(vacc7x0123);
      float32x2_t vacc6x01 = vget_low_f32(vacc6x0123);
      float32x2_t vacc5x01 = vget_low_f32(vacc5x0123);
      float32x2_t vacc4x01 = vget_low_f32(vacc4x0123);
      float32x2_t vacc3x01 = vget_low_f32(vacc3x0123);
      float32x2_t vacc2x01 = vget_low_f32(vacc2x0123);
      float32x2_t vacc1x01 = vget_low_f32(vacc1x0123);
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c7, vacc7x01); c7 += 2;
        vst1_f32(c6, vacc6x01); c6 += 2;
        vst1_f32(c5, vacc5x01); c5 += 2;
        vst1_f32(c4, vacc4x01); c4 += 2;
        vst1_f32(c3, vacc3x01); c3 += 2;
        vst1_f32(c2, vacc2x01); c2 += 2;
        vst1_f32(c1, vacc1x01); c1 += 2;
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc7x01 = vget_high_f32(vacc7x0123);
        vacc6x01 = vget_high_f32(vacc6x0123);
        vacc5x01 = vget_high_f32(vacc5x0123);
        vacc4x01 = vget_high_f32(vacc4x0123);
        vacc3x01 = vget_high_f32(vacc3x0123);
        vacc2x01 = vget_high_f32(vacc2x0123);
        vacc1x01 = vget_high_f32(vacc1x0123);
        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c7, vacc7x01, 0);
        vst1_lane_f32(c6, vacc6x01, 0);
        vst1_lane_f32(c5, vacc5x01, 0);
        vst1_lane_f32(c4, vacc4x01, 0);
        vst1_lane_f32(c3, vacc3x01, 0);
        vst1_lane_f32(c2, vacc2x01, 0);
        vst1_lane_f32(c1, vacc1x01, 0);
        vst1_lane_f32(c0, vacc0x01, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
