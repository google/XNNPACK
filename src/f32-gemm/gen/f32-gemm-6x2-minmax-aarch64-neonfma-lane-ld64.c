// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/MRx2-neon-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"


void xnn_f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    float32x2_t vacc0x01 = vld1_f32(w); w += 2;
    float32x2_t vacc1x01 = vacc0x01;
    float32x2_t vacc2x01 = vacc0x01;
    float32x2_t vacc3x01 = vacc0x01;
    float32x2_t vacc4x01 = vacc0x01;
    float32x2_t vacc5x01 = vacc0x01;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;
      const float32x2_t va1 = vld1_f32(a1); a1 += 2;
      const float32x2_t va2 = vld1_f32(a2); a2 += 2;
      const float32x2_t va3 = vld1_f32(a3); a3 += 2;
      const float32x2_t va4 = vld1_f32(a4); a4 += 2;
      const float32x2_t va5 = vld1_f32(a5); a5 += 2;

      const float32x4_t vb01c01 = vld1q_f32(w); w += 4;
      const float32x2_t vb01c0 = vget_low_f32(vb01c01);
      const float32x2_t vb01c1 = vget_high_f32(vb01c01);
      #if XNN_ARCH_ARM64
        vacc0x01 = vfma_lane_f32(vacc0x01, vb01c0, va0, 0);
        vacc1x01 = vfma_lane_f32(vacc1x01, vb01c0, va1, 0);
        vacc2x01 = vfma_lane_f32(vacc2x01, vb01c0, va2, 0);
        vacc3x01 = vfma_lane_f32(vacc3x01, vb01c0, va3, 0);
        vacc4x01 = vfma_lane_f32(vacc4x01, vb01c0, va4, 0);
        vacc5x01 = vfma_lane_f32(vacc5x01, vb01c0, va5, 0);
      #else
        const float32x2_t va0c0 = vdup_lane_f32(va0, 0);
        const float32x2_t va1c0 = vdup_lane_f32(va1, 0);
        const float32x2_t va2c0 = vdup_lane_f32(va2, 0);
        const float32x2_t va3c0 = vdup_lane_f32(va3, 0);
        const float32x2_t va4c0 = vdup_lane_f32(va4, 0);
        const float32x2_t va5c0 = vdup_lane_f32(va5, 0);
        vacc0x01 = vfma_f32(vacc0x01, va0c0, vb01c0);
        vacc1x01 = vfma_f32(vacc1x01, va1c0, vb01c0);
        vacc2x01 = vfma_f32(vacc2x01, va2c0, vb01c0);
        vacc3x01 = vfma_f32(vacc3x01, va3c0, vb01c0);
        vacc4x01 = vfma_f32(vacc4x01, va4c0, vb01c0);
        vacc5x01 = vfma_f32(vacc5x01, va5c0, vb01c0);
      #endif
      #if XNN_ARCH_ARM64
        vacc0x01 = vfma_lane_f32(vacc0x01, vb01c1, va0, 1);
        vacc1x01 = vfma_lane_f32(vacc1x01, vb01c1, va1, 1);
        vacc2x01 = vfma_lane_f32(vacc2x01, vb01c1, va2, 1);
        vacc3x01 = vfma_lane_f32(vacc3x01, vb01c1, va3, 1);
        vacc4x01 = vfma_lane_f32(vacc4x01, vb01c1, va4, 1);
        vacc5x01 = vfma_lane_f32(vacc5x01, vb01c1, va5, 1);
      #else
        const float32x2_t va0c1 = vdup_lane_f32(va0, 1);
        const float32x2_t va1c1 = vdup_lane_f32(va1, 1);
        const float32x2_t va2c1 = vdup_lane_f32(va2, 1);
        const float32x2_t va3c1 = vdup_lane_f32(va3, 1);
        const float32x2_t va4c1 = vdup_lane_f32(va4, 1);
        const float32x2_t va5c1 = vdup_lane_f32(va5, 1);
        vacc0x01 = vfma_f32(vacc0x01, va0c1, vb01c1);
        vacc1x01 = vfma_f32(vacc1x01, va1c1, vb01c1);
        vacc2x01 = vfma_f32(vacc2x01, va2c1, vb01c1);
        vacc3x01 = vfma_f32(vacc3x01, va3c1, vb01c1);
        vacc4x01 = vfma_f32(vacc4x01, va4c1, vb01c1);
        vacc5x01 = vfma_f32(vacc5x01, va5c1, vb01c1);
      #endif
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x2_t va0 = vld1_dup_f32(a0); a0 += 1;
      const float32x2_t va1 = vld1_dup_f32(a1); a1 += 1;
      const float32x2_t va2 = vld1_dup_f32(a2); a2 += 1;
      const float32x2_t va3 = vld1_dup_f32(a3); a3 += 1;
      const float32x2_t va4 = vld1_dup_f32(a4); a4 += 1;
      const float32x2_t va5 = vld1_dup_f32(a5); a5 += 1;

      const float32x2_t vb01 = vld1_f32(w); w += 2;

      vacc0x01 = vfma_f32(vacc0x01, va0, vb01);
      vacc1x01 = vfma_f32(vacc1x01, va1, vb01);
      vacc2x01 = vfma_f32(vacc2x01, va2, vb01);
      vacc3x01 = vfma_f32(vacc3x01, va3, vb01);
      vacc4x01 = vfma_f32(vacc4x01, va4, vb01);
      vacc5x01 = vfma_f32(vacc5x01, va5, vb01);
    }

    const float32x2_t vmax = vld1_dup_f32(&params->scalar.max);
    vacc0x01 = vmin_f32(vacc0x01, vmax);
    vacc1x01 = vmin_f32(vacc1x01, vmax);
    vacc2x01 = vmin_f32(vacc2x01, vmax);
    vacc3x01 = vmin_f32(vacc3x01, vmax);
    vacc4x01 = vmin_f32(vacc4x01, vmax);
    vacc5x01 = vmin_f32(vacc5x01, vmax);
    const float32x2_t vmin = vld1_dup_f32(&params->scalar.min);
    vacc0x01 = vmax_f32(vacc0x01, vmin);
    vacc1x01 = vmax_f32(vacc1x01, vmin);
    vacc2x01 = vmax_f32(vacc2x01, vmin);
    vacc3x01 = vmax_f32(vacc3x01, vmin);
    vacc4x01 = vmax_f32(vacc4x01, vmin);
    vacc5x01 = vmax_f32(vacc5x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      vst1_f32(c0, vacc0x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1_f32(c1, vacc1x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1_f32(c2, vacc2x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1_f32(c3, vacc3x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1_f32(c4, vacc4x01);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1_f32(c5, vacc5x01);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      vst1_lane_f32(c0, vacc0x01, 0);
      vst1_lane_f32(c1, vacc1x01, 0);
      vst1_lane_f32(c2, vacc2x01, 0);
      vst1_lane_f32(c3, vacc3x01, 0);
      vst1_lane_f32(c4, vacc4x01, 0);
      vst1_lane_f32(c5, vacc5x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}
