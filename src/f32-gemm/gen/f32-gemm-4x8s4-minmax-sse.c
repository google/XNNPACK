// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/sse-shuffle.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/gemm.h"


void xnn_f32_gemm_minmax_ukernel_4x8s4__sse(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_load_ps(w + 0);
    __m128 vacc0x4567 = _mm_load_ps(w + 4);
    __m128 vacc1x0123 = vacc0x0123;
    __m128 vacc1x4567 = vacc0x4567;
    __m128 vacc2x0123 = vacc0x0123;
    __m128 vacc2x4567 = vacc0x4567;
    __m128 vacc3x0123 = vacc0x0123;
    __m128 vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      __m128 va0 = _mm_loadu_ps(a0);
      a0 += 4;
      __m128 va1 = _mm_loadu_ps(a1);
      a1 += 4;
      __m128 va2 = _mm_loadu_ps(a2);
      a2 += 4;
      __m128 va3 = _mm_loadu_ps(a3);
      a3 += 4;

      const __m128 vb0123c0 = _mm_load_ps(w + 0);
      const __m128 vb4567c0 = _mm_load_ps(w + 4);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123c0));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123c0));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123c0));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567c0));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567c0));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567c0));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567c0));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vb0123c1 = _mm_load_ps(w + 8);
      const __m128 vb4567c1 = _mm_load_ps(w + 12);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123c1));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123c1));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123c1));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567c1));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567c1));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567c1));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567c1));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vb0123c2 = _mm_load_ps(w + 16);
      const __m128 vb4567c2 = _mm_load_ps(w + 20);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123c2));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123c2));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123c2));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567c2));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567c2));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567c2));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567c2));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vb0123c3 = _mm_load_ps(w + 24);
      const __m128 vb4567c3 = _mm_load_ps(w + 28);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123c3));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123c3));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123c3));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567c3));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567c3));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567c3));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567c3));


      w += 32;
    }
    if XNN_UNLIKELY(k != 0) {
      __m128 va0 = _mm_loadu_ps(a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      __m128 va1 = _mm_loadu_ps(a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      __m128 va2 = _mm_loadu_ps(a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      __m128 va3 = _mm_loadu_ps(a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const __m128 vb0123c0 = _mm_load_ps(w + 0);
      const __m128 vb4567c0 = _mm_load_ps(w + 4);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c0), va0), vb0123c0));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c0), va1), vb0123c0));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c0), va2), vb0123c0));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c0), va3), vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c0), va0), vb4567c0));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c0), va1), vb4567c0));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c0), va2), vb4567c0));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c0), va3), vb4567c0));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vb0123c1 = _mm_load_ps(w + 8);
      const __m128 vb4567c1 = _mm_load_ps(w + 12);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c1), va0), vb0123c1));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c1), va1), vb0123c1));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c1), va2), vb0123c1));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c1), va3), vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c1), va0), vb4567c1));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c1), va1), vb4567c1));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c1), va2), vb4567c1));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c1), va3), vb4567c1));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vb0123c2 = _mm_load_ps(w + 16);
      const __m128 vb4567c2 = _mm_load_ps(w + 20);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c2), va0), vb0123c2));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c2), va1), vb0123c2));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c2), va2), vb0123c2));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c2), va3), vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c2), va0), vb4567c2));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c2), va1), vb4567c2));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c2), va2), vb4567c2));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c2), va3), vb4567c2));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vb0123c3 = _mm_load_ps(w + 24);
      const __m128 vb4567c3 = _mm_load_ps(w + 28);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c3), va0), vb0123c3));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c3), va1), vb0123c3));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c3), va2), vb0123c3));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c3), va3), vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c3), va0), vb4567c3));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c3), va1), vb4567c3));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c3), va2), vb4567c3));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c3), va3), vb4567c3));


      w += 32;
    }

    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);
    vacc2x0123 = _mm_min_ps(vacc2x0123, vmax);
    vacc3x0123 = _mm_min_ps(vacc3x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);
    vacc1x4567 = _mm_min_ps(vacc1x4567, vmax);
    vacc2x4567 = _mm_min_ps(vacc2x4567, vmax);
    vacc3x4567 = _mm_min_ps(vacc3x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);
    vacc2x0123 = _mm_max_ps(vacc2x0123, vmin);
    vacc3x0123 = _mm_max_ps(vacc3x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);
    vacc1x4567 = _mm_max_ps(vacc1x4567, vmin);
    vacc2x4567 = _mm_max_ps(vacc2x4567, vmin);
    vacc3x4567 = _mm_max_ps(vacc3x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_ps(c1, vacc1x0123);
      _mm_storeu_ps(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c2, vacc2x0123);
      _mm_storeu_ps(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_ps(c3, vacc3x0123);
      _mm_storeu_ps(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
