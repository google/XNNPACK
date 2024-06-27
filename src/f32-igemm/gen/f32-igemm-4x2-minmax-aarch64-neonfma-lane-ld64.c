// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/MRx2-neon-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/igemm.h"


void xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

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
    float32x2_t vacc0x01 = vld1_f32(w); w += 2;
    float32x2_t vacc1x01 = vacc0x01;
    float32x2_t vacc2x01 = vacc0x01;
    float32x2_t vacc3x01 = vacc0x01;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;

        const float32x2_t vb01c0 = vld1_f32(w); w += 2;

        #if XNN_ARCH_ARM64
          vacc0x01 = vfma_lane_f32(vacc0x01, vb01c0, va0, 0);
          vacc1x01 = vfma_lane_f32(vacc1x01, vb01c0, va1, 0);
          vacc2x01 = vfma_lane_f32(vacc2x01, vb01c0, va2, 0);
          vacc3x01 = vfma_lane_f32(vacc3x01, vb01c0, va3, 0);
        #else
          const float32x2_t va0c0 = vdup_lane_f32(va0, 0);
          const float32x2_t va1c0 = vdup_lane_f32(va1, 0);
          const float32x2_t va2c0 = vdup_lane_f32(va2, 0);
          const float32x2_t va3c0 = vdup_lane_f32(va3, 0);
          vacc0x01 = vfma_f32(vacc0x01, va0c0, vb01c0);
          vacc1x01 = vfma_f32(vacc1x01, va1c0, vb01c0);
          vacc2x01 = vfma_f32(vacc2x01, va2c0, vb01c0);
          vacc3x01 = vfma_f32(vacc3x01, va3c0, vb01c0);
        #endif
        const float32x2_t vb01c1 = vld1_f32(w); w += 2;

        #if XNN_ARCH_ARM64
          vacc0x01 = vfma_lane_f32(vacc0x01, vb01c1, va0, 1);
          vacc1x01 = vfma_lane_f32(vacc1x01, vb01c1, va1, 1);
          vacc2x01 = vfma_lane_f32(vacc2x01, vb01c1, va2, 1);
          vacc3x01 = vfma_lane_f32(vacc3x01, vb01c1, va3, 1);
        #else
          const float32x2_t va0c1 = vdup_lane_f32(va0, 1);
          const float32x2_t va1c1 = vdup_lane_f32(va1, 1);
          const float32x2_t va2c1 = vdup_lane_f32(va2, 1);
          const float32x2_t va3c1 = vdup_lane_f32(va3, 1);
          vacc0x01 = vfma_f32(vacc0x01, va0c1, vb01c1);
          vacc1x01 = vfma_f32(vacc1x01, va1c1, vb01c1);
          vacc2x01 = vfma_f32(vacc2x01, va2c1, vb01c1);
          vacc3x01 = vfma_f32(vacc3x01, va3c1, vb01c1);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        const float32x2_t va0 = vld1_dup_f32(a0);
        const float32x2_t va1 = vld1_dup_f32(a1);
        const float32x2_t va2 = vld1_dup_f32(a2);
        const float32x2_t va3 = vld1_dup_f32(a3);

        const float32x2_t vb01 = vld1_f32(w); w += 2;

        vacc0x01 = vfma_f32(vacc0x01, va0, vb01);
        vacc1x01 = vfma_f32(vacc1x01, va1, vb01);
        vacc2x01 = vfma_f32(vacc2x01, va2, vb01);
        vacc3x01 = vfma_f32(vacc3x01, va3, vb01);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const float32x2_t vmax = vld1_dup_f32(&params->scalar.max);
    vacc0x01 = vmin_f32(vacc0x01, vmax);
    vacc1x01 = vmin_f32(vacc1x01, vmax);
    vacc2x01 = vmin_f32(vacc2x01, vmax);
    vacc3x01 = vmin_f32(vacc3x01, vmax);

    const float32x2_t vmin = vld1_dup_f32(&params->scalar.min);
    vacc0x01 = vmax_f32(vacc0x01, vmin);
    vacc1x01 = vmax_f32(vacc1x01, vmin);
    vacc2x01 = vmax_f32(vacc2x01, vmin);
    vacc3x01 = vmax_f32(vacc3x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      vst1_f32(c3, vacc3x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1_f32(c2, vacc2x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1_f32(c1, vacc1x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1_f32(c0, vacc0x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      vst1_lane_f32(c3, vacc3x01, 0);
      vst1_lane_f32(c2, vacc2x01, 0);
      vst1_lane_f32(c1, vacc1x01, 0);
      vst1_lane_f32(c0, vacc0x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}
