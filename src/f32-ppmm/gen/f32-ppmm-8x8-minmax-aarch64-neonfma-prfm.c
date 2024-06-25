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

#include "xnnpack/common.h"
#include "xnnpack/ppmm.h"
#include "xnnpack/prefetch.h"


void xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm(
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
  assert(mr <= 8);
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

  xnn_prefetch_to_l1((const int8_t*) w + 0);
  xnn_prefetch_to_l1((const int8_t*) w + 64);
  xnn_prefetch_to_l1((const int8_t*) w + 128);
  xnn_prefetch_to_l1((const int8_t*) w + 192);

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

      #if XNN_ARCH_ARM64
        vacc0x0123 = vfmaq_laneq_f32(vacc0x0123, vb0123, va0123, 0);
        vacc1x0123 = vfmaq_laneq_f32(vacc1x0123, vb0123, va0123, 1);
        vacc2x0123 = vfmaq_laneq_f32(vacc2x0123, vb0123, va0123, 2);
        vacc3x0123 = vfmaq_laneq_f32(vacc3x0123, vb0123, va0123, 3);
        vacc4x0123 = vfmaq_laneq_f32(vacc4x0123, vb0123, va4567, 0);
        vacc5x0123 = vfmaq_laneq_f32(vacc5x0123, vb0123, va4567, 1);
        vacc6x0123 = vfmaq_laneq_f32(vacc6x0123, vb0123, va4567, 2);
        vacc7x0123 = vfmaq_laneq_f32(vacc7x0123, vb0123, va4567, 3);
        xnn_prefetch_to_l1((const int8_t*) w + 192);
        vacc0x4567 = vfmaq_laneq_f32(vacc0x4567, vb4567, va0123, 0);
        vacc1x4567 = vfmaq_laneq_f32(vacc1x4567, vb4567, va0123, 1);
        vacc2x4567 = vfmaq_laneq_f32(vacc2x4567, vb4567, va0123, 2);
        vacc3x4567 = vfmaq_laneq_f32(vacc3x4567, vb4567, va0123, 3);
        vacc4x4567 = vfmaq_laneq_f32(vacc4x4567, vb4567, va4567, 0);
        vacc5x4567 = vfmaq_laneq_f32(vacc5x4567, vb4567, va4567, 1);
        vacc6x4567 = vfmaq_laneq_f32(vacc6x4567, vb4567, va4567, 2);
        vacc7x4567 = vfmaq_laneq_f32(vacc7x4567, vb4567, va4567, 3);
      #else
        const float32x4_t va0000 = vdupq_lane_f32(vget_low_f32(va0123), 0);
        const float32x4_t va1111 = vdupq_lane_f32(vget_low_f32(va0123), 1);
        const float32x4_t va2222 = vdupq_lane_f32(vget_high_f32(va0123), 0);
        const float32x4_t va3333 = vdupq_lane_f32(vget_high_f32(va0123), 1);
        const float32x4_t va4444 = vdupq_lane_f32(vget_low_f32(va4567), 0);
        const float32x4_t va5555 = vdupq_lane_f32(vget_low_f32(va4567), 1);
        const float32x4_t va6666 = vdupq_lane_f32(vget_high_f32(va4567), 0);
        const float32x4_t va7777 = vdupq_lane_f32(vget_high_f32(va4567), 1);

        vacc0x0123 = vfmaq_f32(vacc0x0123, va0000, vb0123);
        vacc1x0123 = vfmaq_f32(vacc1x0123, va1111, vb0123);
        vacc2x0123 = vfmaq_f32(vacc2x0123, va2222, vb0123);
        vacc3x0123 = vfmaq_f32(vacc3x0123, va3333, vb0123);
        vacc4x0123 = vfmaq_f32(vacc4x0123, va4444, vb0123);
        vacc5x0123 = vfmaq_f32(vacc5x0123, va5555, vb0123);
        vacc6x0123 = vfmaq_f32(vacc6x0123, va6666, vb0123);
        vacc7x0123 = vfmaq_f32(vacc7x0123, va7777, vb0123);
        xnn_prefetch_to_l1((const int8_t*) w + 192);
        vacc0x4567 = vfmaq_f32(vacc0x4567, va0000, vb4567);
        vacc1x4567 = vfmaq_f32(vacc1x4567, va1111, vb4567);
        vacc2x4567 = vfmaq_f32(vacc2x4567, va2222, vb4567);
        vacc3x4567 = vfmaq_f32(vacc3x4567, va3333, vb4567);
        vacc4x4567 = vfmaq_f32(vacc4x4567, va4444, vb4567);
        vacc5x4567 = vfmaq_f32(vacc5x4567, va5555, vb4567);
        vacc6x4567 = vfmaq_f32(vacc6x4567, va6666, vb4567);
        vacc7x4567 = vfmaq_f32(vacc7x4567, va7777, vb4567);
      #endif

      k -= sizeof(float);
    } while (k != 0);

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
