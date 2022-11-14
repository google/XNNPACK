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


void xnn_f32_gemminc_minmax_ukernel_4x8__neon_dup_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const float*restrict acc,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(acc != NULL);

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

  do {
    float32x4_t vacc0x0123 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc0x4567 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc1x0123 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc1x4567 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc2x0123 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc2x4567 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc3x0123 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc3x4567 = vld1q_f32(acc); acc += 4;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const float32x4_t va0 = vld1q_f32(a0); a0 += 4;
      const float32x4_t va1 = vld1q_f32(a1); a1 += 4;
      const float32x4_t va2 = vld1q_f32(a2); a2 += 4;
      const float32x4_t va3 = vld1q_f32(a3); a3 += 4;


      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      const float32x4_t va0c0 = vdupq_lane_f32(vget_low_f32(va0), 0);
      const float32x4_t va1c0 = vdupq_lane_f32(vget_low_f32(va1), 0);
      const float32x4_t va2c0 = vdupq_lane_f32(vget_low_f32(va2), 0);
      const float32x4_t va3c0 = vdupq_lane_f32(vget_low_f32(va3), 0);
      vacc0x0123 = vmlaq_f32(vacc0x0123, va0c0, vb0123c0);
      vacc1x0123 = vmlaq_f32(vacc1x0123, va1c0, vb0123c0);
      vacc2x0123 = vmlaq_f32(vacc2x0123, va2c0, vb0123c0);
      vacc3x0123 = vmlaq_f32(vacc3x0123, va3c0, vb0123c0);
      vacc0x4567 = vmlaq_f32(vacc0x4567, va0c0, vb4567c0);
      vacc1x4567 = vmlaq_f32(vacc1x4567, va1c0, vb4567c0);
      vacc2x4567 = vmlaq_f32(vacc2x4567, va2c0, vb4567c0);
      vacc3x4567 = vmlaq_f32(vacc3x4567, va3c0, vb4567c0);

      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      const float32x4_t va0c1 = vdupq_lane_f32(vget_low_f32(va0), 1);
      const float32x4_t va1c1 = vdupq_lane_f32(vget_low_f32(va1), 1);
      const float32x4_t va2c1 = vdupq_lane_f32(vget_low_f32(va2), 1);
      const float32x4_t va3c1 = vdupq_lane_f32(vget_low_f32(va3), 1);
      vacc0x0123 = vmlaq_f32(vacc0x0123, va0c1, vb0123c1);
      vacc1x0123 = vmlaq_f32(vacc1x0123, va1c1, vb0123c1);
      vacc2x0123 = vmlaq_f32(vacc2x0123, va2c1, vb0123c1);
      vacc3x0123 = vmlaq_f32(vacc3x0123, va3c1, vb0123c1);
      vacc0x4567 = vmlaq_f32(vacc0x4567, va0c1, vb4567c1);
      vacc1x4567 = vmlaq_f32(vacc1x4567, va1c1, vb4567c1);
      vacc2x4567 = vmlaq_f32(vacc2x4567, va2c1, vb4567c1);
      vacc3x4567 = vmlaq_f32(vacc3x4567, va3c1, vb4567c1);

      const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

      const float32x4_t va0c2 = vdupq_lane_f32(vget_high_f32(va0), 0);
      const float32x4_t va1c2 = vdupq_lane_f32(vget_high_f32(va1), 0);
      const float32x4_t va2c2 = vdupq_lane_f32(vget_high_f32(va2), 0);
      const float32x4_t va3c2 = vdupq_lane_f32(vget_high_f32(va3), 0);
      vacc0x0123 = vmlaq_f32(vacc0x0123, va0c2, vb0123c2);
      vacc1x0123 = vmlaq_f32(vacc1x0123, va1c2, vb0123c2);
      vacc2x0123 = vmlaq_f32(vacc2x0123, va2c2, vb0123c2);
      vacc3x0123 = vmlaq_f32(vacc3x0123, va3c2, vb0123c2);
      vacc0x4567 = vmlaq_f32(vacc0x4567, va0c2, vb4567c2);
      vacc1x4567 = vmlaq_f32(vacc1x4567, va1c2, vb4567c2);
      vacc2x4567 = vmlaq_f32(vacc2x4567, va2c2, vb4567c2);
      vacc3x4567 = vmlaq_f32(vacc3x4567, va3c2, vb4567c2);

      const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

      const float32x4_t va0c3 = vdupq_lane_f32(vget_high_f32(va0), 1);
      const float32x4_t va1c3 = vdupq_lane_f32(vget_high_f32(va1), 1);
      const float32x4_t va2c3 = vdupq_lane_f32(vget_high_f32(va2), 1);
      const float32x4_t va3c3 = vdupq_lane_f32(vget_high_f32(va3), 1);
      vacc0x0123 = vmlaq_f32(vacc0x0123, va0c3, vb0123c3);
      vacc1x0123 = vmlaq_f32(vacc1x0123, va1c3, vb0123c3);
      vacc2x0123 = vmlaq_f32(vacc2x0123, va2c3, vb0123c3);
      vacc3x0123 = vmlaq_f32(vacc3x0123, va3c3, vb0123c3);
      vacc0x4567 = vmlaq_f32(vacc0x4567, va0c3, vb4567c3);
      vacc1x4567 = vmlaq_f32(vacc1x4567, va1c3, vb4567c3);
      vacc2x4567 = vmlaq_f32(vacc2x4567, va2c3, vb4567c3);
      vacc3x4567 = vmlaq_f32(vacc3x4567, va3c3, vb4567c3);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
        const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
        const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
        const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;

        const float32x4_t vb0123 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567 = vld1q_f32(w); w += 4;

        vacc0x0123 = vmlaq_f32(vacc0x0123, va0, vb0123);
        vacc1x0123 = vmlaq_f32(vacc1x0123, va1, vb0123);
        vacc2x0123 = vmlaq_f32(vacc2x0123, va2, vb0123);
        vacc3x0123 = vmlaq_f32(vacc3x0123, va3, vb0123);
        vacc0x4567 = vmlaq_f32(vacc0x4567, va0, vb4567);
        vacc1x4567 = vmlaq_f32(vacc1x4567, va1, vb4567);
        vacc2x4567 = vmlaq_f32(vacc2x4567, va2, vb4567);
        vacc3x4567 = vmlaq_f32(vacc3x4567, va3, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc1x4567 = vminq_f32(vacc1x4567, vmax);
    vacc2x4567 = vminq_f32(vacc2x4567, vmax);
    vacc3x4567 = vminq_f32(vacc3x4567, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
    vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
    vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
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

      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;
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
