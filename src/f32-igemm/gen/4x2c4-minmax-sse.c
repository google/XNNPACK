// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/MRx2c4-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/igemm.h>


void xnn_f32_igemm_minmax_ukernel_4x2c4__sse(
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
    __m128 vacc0x0c4 = _mm_load_ss(w);
    __m128 vacc0x1c4 = _mm_load_ss(w + 1);
    __m128 vacc1x0c4 = vacc0x0c4;
    __m128 vacc1x1c4 = vacc0x1c4;
    __m128 vacc2x0c4 = vacc0x0c4;
    __m128 vacc2x1c4 = vacc0x1c4;
    __m128 vacc3x0c4 = vacc0x0c4;
    __m128 vacc3x1c4 = vacc0x1c4;
    w += 2;

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
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const __m128 va0 = _mm_loadu_ps(a0);
        a0 += 4;
        const __m128 va1 = _mm_loadu_ps(a1);
        a1 += 4;
        const __m128 va2 = _mm_loadu_ps(a2);
        a2 += 4;
        const __m128 va3 = _mm_loadu_ps(a3);
        a3 += 4;

        const __m128 vb0 = _mm_loadu_ps(w);
        const __m128 vb1 = _mm_loadu_ps(w + 4);
        w += 8;

        vacc0x0c4 = _mm_add_ps(vacc0x0c4, _mm_mul_ps(va0, vb0));
        vacc0x1c4 = _mm_add_ps(vacc0x1c4, _mm_mul_ps(va0, vb1));
        vacc1x0c4 = _mm_add_ps(vacc1x0c4, _mm_mul_ps(va1, vb0));
        vacc1x1c4 = _mm_add_ps(vacc1x1c4, _mm_mul_ps(va1, vb1));
        vacc2x0c4 = _mm_add_ps(vacc2x0c4, _mm_mul_ps(va2, vb0));
        vacc2x1c4 = _mm_add_ps(vacc2x1c4, _mm_mul_ps(va2, vb1));
        vacc3x0c4 = _mm_add_ps(vacc3x0c4, _mm_mul_ps(va3, vb0));
        vacc3x1c4 = _mm_add_ps(vacc3x1c4, _mm_mul_ps(va3, vb1));
      }
      if XNN_UNLIKELY(k != 0) {
        const __m128 va0 = _mm_loadu_ps(a0);
        const __m128 va1 = _mm_loadu_ps(a1);
        const __m128 va2 = _mm_loadu_ps(a2);
        const __m128 va3 = _mm_loadu_ps(a3);

        const __m128 vb0 = _mm_loadu_ps(w);
        const __m128 vb1 = _mm_loadu_ps(w + 4);
        w += 8;

        const __m128 vmask0 = _mm_cmpeq_ps(_mm_setzero_ps(), vb0);
        const __m128 vmask1 = _mm_cmpeq_ps(_mm_setzero_ps(), vb1);

        vacc0x0c4 = _mm_add_ps(vacc0x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va0), vb0));
        vacc0x1c4 = _mm_add_ps(vacc0x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va0), vb1));
        vacc1x0c4 = _mm_add_ps(vacc1x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va1), vb0));
        vacc1x1c4 = _mm_add_ps(vacc1x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va1), vb1));
        vacc2x0c4 = _mm_add_ps(vacc2x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va2), vb0));
        vacc2x1c4 = _mm_add_ps(vacc2x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va2), vb1));
        vacc3x0c4 = _mm_add_ps(vacc3x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va3), vb0));
        vacc3x1c4 = _mm_add_ps(vacc3x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va3), vb1));
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m128 vacc0x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc0x0c4, vacc0x1c4), _mm_unpackhi_ps(vacc0x0c4, vacc0x1c4));
    const __m128 vacc1x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc1x0c4, vacc1x1c4), _mm_unpackhi_ps(vacc1x0c4, vacc1x1c4));
    const __m128 vacc2x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc2x0c4, vacc2x1c4), _mm_unpackhi_ps(vacc2x0c4, vacc2x1c4));
    const __m128 vacc3x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc3x0c4, vacc3x1c4), _mm_unpackhi_ps(vacc3x0c4, vacc3x1c4));

    __m128 vacc01x01 = _mm_add_ps(_mm_movelh_ps(vacc0x01c2, vacc1x01c2), _mm_movehl_ps(vacc1x01c2, vacc0x01c2));
    __m128 vacc23x01 = _mm_add_ps(_mm_movelh_ps(vacc2x01c2, vacc3x01c2), _mm_movehl_ps(vacc3x01c2, vacc2x01c2));

    const __m128 vmax = _mm_load_ps(params->sse.max);
    vacc01x01 = _mm_min_ps(vacc01x01, vmax);
    vacc23x01 = _mm_min_ps(vacc23x01, vmax);

    const __m128 vmin = _mm_load_ps(params->sse.min);
    vacc01x01 = _mm_max_ps(vacc01x01, vmin);
    vacc23x01 = _mm_max_ps(vacc23x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      _mm_storeh_pi((__m64*) c3, vacc23x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm_storel_pi((__m64*) c2, vacc23x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeh_pi((__m64*) c1, vacc01x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storel_pi((__m64*) c0, vacc01x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      _mm_store_ss(c3, _mm_movehl_ps(vacc23x01, vacc23x01));
      _mm_store_ss(c2, vacc23x01);
      _mm_store_ss(c1, _mm_movehl_ps(vacc01x01, vacc01x01));
      _mm_store_ss(c0, vacc01x01);

      nc = 0;
    }
  } while (nc != 0);
}
