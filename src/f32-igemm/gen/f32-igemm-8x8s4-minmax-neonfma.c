// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/neon-shuffle.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/igemm.h"


void xnn_f32_igemm_minmax_ukernel_8x8s4__neonfma(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (8 * sizeof(void*)) == 0);
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
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc5x0 = vacc0x0;
    float32x4_t vacc5x1 = vacc0x1;
    float32x4_t vacc6x0 = vacc0x0;
    float32x4_t vacc6x1 = vacc0x1;
    float32x4_t vacc7x0 = vacc0x0;
    float32x4_t vacc7x1 = vacc0x1;

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
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      const float* restrict a7 = a[7];
      assert(a7 != NULL);
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const float*) ((uintptr_t) a7 + a_offset);
      }
      a += 8;

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

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c0);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c0);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c0);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c0);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c0);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c0);
        vacc6x0 = vfmaq_f32(vacc6x0, va6, vb0123c0);
        vacc7x0 = vfmaq_f32(vacc7x0, va7, vb0123c0);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c0);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c0);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c0);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c0);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c0);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c0);
        vacc6x1 = vfmaq_f32(vacc6x1, va6, vb4567c0);
        vacc7x1 = vfmaq_f32(vacc7x1, va7, vb4567c0);

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

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c1);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c1);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c1);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c1);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c1);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c1);
        vacc6x0 = vfmaq_f32(vacc6x0, va6, vb0123c1);
        vacc7x0 = vfmaq_f32(vacc7x0, va7, vb0123c1);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c1);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c1);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c1);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c1);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c1);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c1);
        vacc6x1 = vfmaq_f32(vacc6x1, va6, vb4567c1);
        vacc7x1 = vfmaq_f32(vacc7x1, va7, vb4567c1);

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

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c2);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c2);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c2);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c2);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c2);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c2);
        vacc6x0 = vfmaq_f32(vacc6x0, va6, vb0123c2);
        vacc7x0 = vfmaq_f32(vacc7x0, va7, vb0123c2);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c2);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c2);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c2);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c2);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c2);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c2);
        vacc6x1 = vfmaq_f32(vacc6x1, va6, vb4567c2);
        vacc7x1 = vfmaq_f32(vacc7x1, va7, vb4567c2);

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

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c3);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c3);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c3);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c3);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c3);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c3);
        vacc6x0 = vfmaq_f32(vacc6x0, va6, vb0123c3);
        vacc7x0 = vfmaq_f32(vacc7x0, va7, vb0123c3);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c3);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c3);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c3);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c3);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c3);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c3);
        vacc6x1 = vfmaq_f32(vacc6x1, va6, vb4567c3);
        vacc7x1 = vfmaq_f32(vacc7x1, va7, vb4567c3);


        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        float32x4_t va0 = vld1q_f32(a0); a0 = (const float*) ((uintptr_t) a0 + k);
        float32x4_t va1 = vld1q_f32(a1); a1 = (const float*) ((uintptr_t) a1 + k);
        float32x4_t va2 = vld1q_f32(a2); a2 = (const float*) ((uintptr_t) a2 + k);
        float32x4_t va3 = vld1q_f32(a3); a3 = (const float*) ((uintptr_t) a3 + k);
        float32x4_t va4 = vld1q_f32(a4); a4 = (const float*) ((uintptr_t) a4 + k);
        float32x4_t va5 = vld1q_f32(a5); a5 = (const float*) ((uintptr_t) a5 + k);
        float32x4_t va6 = vld1q_f32(a6); a6 = (const float*) ((uintptr_t) a6 + k);
        float32x4_t va7 = vld1q_f32(a7); a7 = (const float*) ((uintptr_t) a7 + k);


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c0, vb0123c0);
        const float32x4_t vmska1x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c0, vb0123c0);
        const float32x4_t vmska2x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c0, vb0123c0);
        const float32x4_t vmska3x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c0, vb0123c0);
        const float32x4_t vmska4x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c0, vb0123c0);
        const float32x4_t vmska5x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c0, vb0123c0);
        const float32x4_t vmska6x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va6), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc6x0 = vfmaq_f32(vacc6x0, vmska6x0123c0, vb0123c0);
        const float32x4_t vmska7x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va7), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc7x0 = vfmaq_f32(vacc7x0, vmska7x0123c0, vb0123c0);
        const float32x4_t vmska0x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c0, vb4567c0);
        const float32x4_t vmska1x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c0, vb4567c0);
        const float32x4_t vmska2x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c0, vb4567c0);
        const float32x4_t vmska3x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c0, vb4567c0);
        const float32x4_t vmska4x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c0, vb4567c0);
        const float32x4_t vmska5x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c0, vb4567c0);
        const float32x4_t vmska6x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va6), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc6x1 = vfmaq_f32(vacc6x1, vmska6x4567c0, vb4567c0);
        const float32x4_t vmska7x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va7), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc7x1 = vfmaq_f32(vacc7x1, vmska7x4567c0, vb4567c0);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);
        va6 = vextq_f32(va6, va6, 1);
        va7 = vextq_f32(va7, va7, 1);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c1, vb0123c1);
        const float32x4_t vmska1x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c1, vb0123c1);
        const float32x4_t vmska2x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c1, vb0123c1);
        const float32x4_t vmska3x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c1, vb0123c1);
        const float32x4_t vmska4x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c1, vb0123c1);
        const float32x4_t vmska5x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c1, vb0123c1);
        const float32x4_t vmska6x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va6), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc6x0 = vfmaq_f32(vacc6x0, vmska6x0123c1, vb0123c1);
        const float32x4_t vmska7x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va7), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc7x0 = vfmaq_f32(vacc7x0, vmska7x0123c1, vb0123c1);
        const float32x4_t vmska0x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c1, vb4567c1);
        const float32x4_t vmska1x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c1, vb4567c1);
        const float32x4_t vmska2x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c1, vb4567c1);
        const float32x4_t vmska3x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c1, vb4567c1);
        const float32x4_t vmska4x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c1, vb4567c1);
        const float32x4_t vmska5x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c1, vb4567c1);
        const float32x4_t vmska6x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va6), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc6x1 = vfmaq_f32(vacc6x1, vmska6x4567c1, vb4567c1);
        const float32x4_t vmska7x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va7), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc7x1 = vfmaq_f32(vacc7x1, vmska7x4567c1, vb4567c1);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);
        va6 = vextq_f32(va6, va6, 1);
        va7 = vextq_f32(va7, va7, 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c2, vb0123c2);
        const float32x4_t vmska1x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c2, vb0123c2);
        const float32x4_t vmska2x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c2, vb0123c2);
        const float32x4_t vmska3x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c2, vb0123c2);
        const float32x4_t vmska4x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c2, vb0123c2);
        const float32x4_t vmska5x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c2, vb0123c2);
        const float32x4_t vmska6x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va6), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc6x0 = vfmaq_f32(vacc6x0, vmska6x0123c2, vb0123c2);
        const float32x4_t vmska7x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va7), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc7x0 = vfmaq_f32(vacc7x0, vmska7x0123c2, vb0123c2);
        const float32x4_t vmska0x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c2, vb4567c2);
        const float32x4_t vmska1x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c2, vb4567c2);
        const float32x4_t vmska2x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c2, vb4567c2);
        const float32x4_t vmska3x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c2, vb4567c2);
        const float32x4_t vmska4x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c2, vb4567c2);
        const float32x4_t vmska5x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c2, vb4567c2);
        const float32x4_t vmska6x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va6), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc6x1 = vfmaq_f32(vacc6x1, vmska6x4567c2, vb4567c2);
        const float32x4_t vmska7x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va7), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc7x1 = vfmaq_f32(vacc7x1, vmska7x4567c2, vb4567c2);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);
        va6 = vextq_f32(va6, va6, 1);
        va7 = vextq_f32(va7, va7, 1);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c3, vb0123c3);
        const float32x4_t vmska1x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c3, vb0123c3);
        const float32x4_t vmska2x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c3, vb0123c3);
        const float32x4_t vmska3x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c3, vb0123c3);
        const float32x4_t vmska4x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c3, vb0123c3);
        const float32x4_t vmska5x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c3, vb0123c3);
        const float32x4_t vmska6x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va6), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc6x0 = vfmaq_f32(vacc6x0, vmska6x0123c3, vb0123c3);
        const float32x4_t vmska7x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va7), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc7x0 = vfmaq_f32(vacc7x0, vmska7x0123c3, vb0123c3);
        const float32x4_t vmska0x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c3, vb4567c3);
        const float32x4_t vmska1x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c3, vb4567c3);
        const float32x4_t vmska2x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c3, vb4567c3);
        const float32x4_t vmska3x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c3, vb4567c3);
        const float32x4_t vmska4x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c3, vb4567c3);
        const float32x4_t vmska5x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c3, vb4567c3);
        const float32x4_t vmska6x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va6), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc6x1 = vfmaq_f32(vacc6x1, vmska6x4567c3, vb4567c3);
        const float32x4_t vmska7x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va7), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc7x1 = vfmaq_f32(vacc7x1, vmska7x4567c3, vb4567c3);

      }

      p -= 8 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc4x0 = vminq_f32(vacc4x0, vmax);
    vacc5x0 = vminq_f32(vacc5x0, vmax);
    vacc6x0 = vminq_f32(vacc6x0, vmax);
    vacc7x0 = vminq_f32(vacc7x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);
    vacc4x1 = vminq_f32(vacc4x1, vmax);
    vacc5x1 = vminq_f32(vacc5x1, vmax);
    vacc6x1 = vminq_f32(vacc6x1, vmax);
    vacc7x1 = vminq_f32(vacc7x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc4x0 = vmaxq_f32(vacc4x0, vmin);
    vacc5x0 = vmaxq_f32(vacc5x0, vmin);
    vacc6x0 = vmaxq_f32(vacc6x0, vmin);
    vacc7x0 = vmaxq_f32(vacc7x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);
    vacc4x1 = vmaxq_f32(vacc4x1, vmin);
    vacc5x1 = vmaxq_f32(vacc5x1, vmin);
    vacc6x1 = vmaxq_f32(vacc6x1, vmin);
    vacc7x1 = vmaxq_f32(vacc7x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c7, vacc7x0);
      vst1q_f32(c7 + 4, vacc7x1);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      vst1q_f32(c6, vacc6x0);
      vst1q_f32(c6 + 4, vacc6x1);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      vst1q_f32(c5, vacc5x0);
      vst1q_f32(c5 + 4, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      vst1q_f32(c4, vacc4x0);
      vst1q_f32(c4 + 4, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c7, vacc7x0); c7 += 4;
        vst1q_f32(c6, vacc6x0); c6 += 4;
        vst1q_f32(c5, vacc5x0); c5 += 4;
        vst1q_f32(c4, vacc4x0); c4 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc7x0 = vacc7x1;
        vacc6x0 = vacc6x1;
        vacc5x0 = vacc5x1;
        vacc4x0 = vacc4x1;
        vacc3x0 = vacc3x1;
        vacc2x0 = vacc2x1;
        vacc1x0 = vacc1x1;
        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc7 = vget_low_f32(vacc7x0);
      float32x2_t vacc6 = vget_low_f32(vacc6x0);
      float32x2_t vacc5 = vget_low_f32(vacc5x0);
      float32x2_t vacc4 = vget_low_f32(vacc4x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c7, vacc7); c7 += 2;
        vst1_f32(c6, vacc6); c6 += 2;
        vst1_f32(c5, vacc5); c5 += 2;
        vst1_f32(c4, vacc4); c4 += 2;
        vst1_f32(c3, vacc3); c3 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c0, vacc0); c0 += 2;

        vacc7 = vget_high_f32(vacc7x0);
        vacc6 = vget_high_f32(vacc6x0);
        vacc5 = vget_high_f32(vacc5x0);
        vacc4 = vget_high_f32(vacc4x0);
        vacc3 = vget_high_f32(vacc3x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c7, vacc7, 0);
        vst1_lane_f32(c6, vacc6, 0);
        vst1_lane_f32(c5, vacc5, 0);
        vst1_lane_f32(c4, vacc4, 0);
        vst1_lane_f32(c3, vacc3, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
