// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/neon-ld128.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/igemm.h>


void xnn_f32_igemm_minmax_ukernel_6x8__neonfma_lane_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
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
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        const float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        const float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        const float32x4_t va3 = vld1q_f32(a3); a3 += 4;
        const float32x4_t va4 = vld1q_f32(a4); a4 += 4;
        const float32x4_t va5 = vld1q_f32(a5); a5 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c0, vget_low_f32(va0), 0);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c0, vget_low_f32(va1), 0);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c0, vget_low_f32(va2), 0);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c0, vget_low_f32(va3), 0);
        vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123c0, vget_low_f32(va4), 0);
        vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123c0, vget_low_f32(va5), 0);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c0, vget_low_f32(va0), 0);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c0, vget_low_f32(va1), 0);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c0, vget_low_f32(va2), 0);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c0, vget_low_f32(va3), 0);
        vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567c0, vget_low_f32(va4), 0);
        vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567c0, vget_low_f32(va5), 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c1, vget_low_f32(va0), 1);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c1, vget_low_f32(va1), 1);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c1, vget_low_f32(va2), 1);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c1, vget_low_f32(va3), 1);
        vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123c1, vget_low_f32(va4), 1);
        vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123c1, vget_low_f32(va5), 1);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c1, vget_low_f32(va0), 1);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c1, vget_low_f32(va1), 1);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c1, vget_low_f32(va2), 1);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c1, vget_low_f32(va3), 1);
        vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567c1, vget_low_f32(va4), 1);
        vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567c1, vget_low_f32(va5), 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c2, vget_high_f32(va0), 0);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c2, vget_high_f32(va1), 0);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c2, vget_high_f32(va2), 0);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c2, vget_high_f32(va3), 0);
        vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123c2, vget_high_f32(va4), 0);
        vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123c2, vget_high_f32(va5), 0);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c2, vget_high_f32(va0), 0);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c2, vget_high_f32(va1), 0);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c2, vget_high_f32(va2), 0);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c2, vget_high_f32(va3), 0);
        vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567c2, vget_high_f32(va4), 0);
        vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567c2, vget_high_f32(va5), 0);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c3, vget_high_f32(va0), 1);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c3, vget_high_f32(va1), 1);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c3, vget_high_f32(va2), 1);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c3, vget_high_f32(va3), 1);
        vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123c3, vget_high_f32(va4), 1);
        vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123c3, vget_high_f32(va5), 1);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c3, vget_high_f32(va0), 1);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c3, vget_high_f32(va1), 1);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c3, vget_high_f32(va2), 1);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c3, vget_high_f32(va3), 1);
        vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567c3, vget_high_f32(va4), 1);
        vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567c3, vget_high_f32(va5), 1);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
          const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
          const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
          const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;
          const float32x4_t va4 = vld1q_dup_f32(a4); a4 += 1;
          const float32x4_t va5 = vld1q_dup_f32(a5); a5 += 1;

          const float32x4_t vb0123 = vld1q_f32(w); w += 4;
          const float32x4_t vb4567 = vld1q_f32(w); w += 4;

          vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123);
          vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123);
          vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123);
          vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123);
          vacc4x0123 = vfmaq_f32(vacc4x0123, va4, vb0123);
          vacc5x0123 = vfmaq_f32(vacc5x0123, va5, vb0123);
          vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567);
          vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567);
          vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567);
          vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567);
          vacc4x4567 = vfmaq_f32(vacc4x4567, va4, vb4567);
          vacc5x4567 = vfmaq_f32(vacc5x4567, va5, vb4567);

          k -= sizeof(float);
        } while (k != 0);
      }

      p -= 6 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc4x0123 = vminq_f32(vacc4x0123, vmax);
    vacc5x0123 = vminq_f32(vacc5x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc1x4567 = vminq_f32(vacc1x4567, vmax);
    vacc2x4567 = vminq_f32(vacc2x4567, vmax);
    vacc3x4567 = vminq_f32(vacc3x4567, vmax);
    vacc4x4567 = vminq_f32(vacc4x4567, vmax);
    vacc5x4567 = vminq_f32(vacc5x4567, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc4x0123 = vmaxq_f32(vacc4x0123, vmin);
    vacc5x0123 = vmaxq_f32(vacc5x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
    vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
    vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);
    vacc4x4567 = vmaxq_f32(vacc4x4567, vmin);
    vacc5x4567 = vmaxq_f32(vacc5x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
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

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c5, vacc5x0123); c5 += 4;
        vst1q_f32(c4, vacc4x0123); c4 += 4;
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;
      }
      float32x2_t vacc5x01 = vget_low_f32(vacc5x0123);
      float32x2_t vacc4x01 = vget_low_f32(vacc4x0123);
      float32x2_t vacc3x01 = vget_low_f32(vacc3x0123);
      float32x2_t vacc2x01 = vget_low_f32(vacc2x0123);
      float32x2_t vacc1x01 = vget_low_f32(vacc1x0123);
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c5, vacc5x01); c5 += 2;
        vst1_f32(c4, vacc4x01); c4 += 2;
        vst1_f32(c3, vacc3x01); c3 += 2;
        vst1_f32(c2, vacc2x01); c2 += 2;
        vst1_f32(c1, vacc1x01); c1 += 2;
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc5x01 = vget_high_f32(vacc5x0123);
        vacc4x01 = vget_high_f32(vacc4x0123);
        vacc3x01 = vget_high_f32(vacc3x0123);
        vacc2x01 = vget_high_f32(vacc2x0123);
        vacc1x01 = vget_high_f32(vacc1x0123);
        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
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
