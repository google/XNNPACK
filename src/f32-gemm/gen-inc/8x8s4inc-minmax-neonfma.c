// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/neon-shuffle.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>


void xnn_f32_gemminc_minmax_ukernel_8x8s4__neonfma(
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
  assert(mr <= 8);
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
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const float* a7 = (const float*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
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
    float32x4_t vacc4x0123 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc4x4567 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc5x0123 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc5x4567 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc6x0123 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc6x4567 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc7x0123 = vld1q_f32(acc); acc += 4;
    float32x4_t vacc7x4567 = vld1q_f32(acc); acc += 4;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      float32x4_t va0 = vld1q_f32(a0); a0 += 4;
      float32x4_t va1 = vld1q_f32(a1); a1 += 4;
      float32x4_t va2 = vld1q_f32(a2); a2 += 4;
      float32x4_t va3 = vld1q_f32(a3); a3 += 4;
      float32x4_t va4 = vld1q_f32(a4); a4 += 4;
      float32x4_t va5 = vld1q_f32(a5); a5 += 4;
      float32x4_t va6 = vld1q_f32(a6); a6 += 4;
      float32x4_t va7 = vld1q_f32(a7); a7 += 4;


      const float32x4_t vb0123c0 = vld1q_f32(w + 0);
      const float32x4_t vb4567c0 = vld1q_f32(w + 4);

      vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123c0);
      vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123c0);
      vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123c0);
      vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123c0);
      vacc4x0123 = vfmaq_f32(vacc4x0123, va4, vb0123c0);
      vacc5x0123 = vfmaq_f32(vacc5x0123, va5, vb0123c0);
      vacc6x0123 = vfmaq_f32(vacc6x0123, va6, vb0123c0);
      vacc7x0123 = vfmaq_f32(vacc7x0123, va7, vb0123c0);
      vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567c0);
      vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567c0);
      vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567c0);
      vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567c0);
      vacc4x4567 = vfmaq_f32(vacc4x4567, va4, vb4567c0);
      vacc5x4567 = vfmaq_f32(vacc5x4567, va5, vb4567c0);
      vacc6x4567 = vfmaq_f32(vacc6x4567, va6, vb4567c0);
      vacc7x4567 = vfmaq_f32(vacc7x4567, va7, vb4567c0);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);
      va6 = vextq_f32(va6, va6, 1);
      va7 = vextq_f32(va7, va7, 1);

      const float32x4_t vb0123c1 = vld1q_f32(w + 8);
      const float32x4_t vb4567c1 = vld1q_f32(w + 12);

      vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123c1);
      vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123c1);
      vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123c1);
      vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123c1);
      vacc4x0123 = vfmaq_f32(vacc4x0123, va4, vb0123c1);
      vacc5x0123 = vfmaq_f32(vacc5x0123, va5, vb0123c1);
      vacc6x0123 = vfmaq_f32(vacc6x0123, va6, vb0123c1);
      vacc7x0123 = vfmaq_f32(vacc7x0123, va7, vb0123c1);
      vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567c1);
      vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567c1);
      vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567c1);
      vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567c1);
      vacc4x4567 = vfmaq_f32(vacc4x4567, va4, vb4567c1);
      vacc5x4567 = vfmaq_f32(vacc5x4567, va5, vb4567c1);
      vacc6x4567 = vfmaq_f32(vacc6x4567, va6, vb4567c1);
      vacc7x4567 = vfmaq_f32(vacc7x4567, va7, vb4567c1);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);
      va6 = vextq_f32(va6, va6, 1);
      va7 = vextq_f32(va7, va7, 1);

      const float32x4_t vb0123c2 = vld1q_f32(w + 16);
      const float32x4_t vb4567c2 = vld1q_f32(w + 20);

      vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123c2);
      vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123c2);
      vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123c2);
      vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123c2);
      vacc4x0123 = vfmaq_f32(vacc4x0123, va4, vb0123c2);
      vacc5x0123 = vfmaq_f32(vacc5x0123, va5, vb0123c2);
      vacc6x0123 = vfmaq_f32(vacc6x0123, va6, vb0123c2);
      vacc7x0123 = vfmaq_f32(vacc7x0123, va7, vb0123c2);
      vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567c2);
      vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567c2);
      vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567c2);
      vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567c2);
      vacc4x4567 = vfmaq_f32(vacc4x4567, va4, vb4567c2);
      vacc5x4567 = vfmaq_f32(vacc5x4567, va5, vb4567c2);
      vacc6x4567 = vfmaq_f32(vacc6x4567, va6, vb4567c2);
      vacc7x4567 = vfmaq_f32(vacc7x4567, va7, vb4567c2);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);
      va6 = vextq_f32(va6, va6, 1);
      va7 = vextq_f32(va7, va7, 1);

      const float32x4_t vb0123c3 = vld1q_f32(w + 24);
      const float32x4_t vb4567c3 = vld1q_f32(w + 28);

      vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123c3);
      vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123c3);
      vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123c3);
      vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123c3);
      vacc4x0123 = vfmaq_f32(vacc4x0123, va4, vb0123c3);
      vacc5x0123 = vfmaq_f32(vacc5x0123, va5, vb0123c3);
      vacc6x0123 = vfmaq_f32(vacc6x0123, va6, vb0123c3);
      vacc7x0123 = vfmaq_f32(vacc7x0123, va7, vb0123c3);
      vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567c3);
      vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567c3);
      vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567c3);
      vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567c3);
      vacc4x4567 = vfmaq_f32(vacc4x4567, va4, vb4567c3);
      vacc5x4567 = vfmaq_f32(vacc5x4567, va5, vb4567c3);
      vacc6x4567 = vfmaq_f32(vacc6x4567, va6, vb4567c3);
      vacc7x4567 = vfmaq_f32(vacc7x4567, va7, vb4567c3);


      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
        const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
        const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
        const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;
        const float32x4_t va4 = vld1q_dup_f32(a4); a4 += 1;
        const float32x4_t va5 = vld1q_dup_f32(a5); a5 += 1;
        const float32x4_t va6 = vld1q_dup_f32(a6); a6 += 1;
        const float32x4_t va7 = vld1q_dup_f32(a7); a7 += 1;

        const float32x4_t vb0123 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123);
        vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123);
        vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123);
        vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123);
        vacc4x0123 = vfmaq_f32(vacc4x0123, va4, vb0123);
        vacc5x0123 = vfmaq_f32(vacc5x0123, va5, vb0123);
        vacc6x0123 = vfmaq_f32(vacc6x0123, va6, vb0123);
        vacc7x0123 = vfmaq_f32(vacc7x0123, va7, vb0123);
        vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567);
        vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567);
        vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567);
        vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567);
        vacc4x4567 = vfmaq_f32(vacc4x4567, va4, vb4567);
        vacc5x4567 = vfmaq_f32(vacc5x4567, va5, vb4567);
        vacc6x4567 = vfmaq_f32(vacc6x4567, va6, vb4567);
        vacc7x4567 = vfmaq_f32(vacc7x4567, va7, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }
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

      a7 = (const float*) ((uintptr_t) a7 - kc);
      a6 = (const float*) ((uintptr_t) a6 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

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
