// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/sse-load1.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gemm.h"


void xnn_f32_gemminc_minmax_ukernel_6x8__sse_load1(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const float* restrict acc,
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
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }
  const __m128 vmin = _mm_set1_ps(params->scalar.min);
  const __m128 vmax = _mm_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_load_ps(acc + 0);
    __m128 vacc0x4567 = _mm_load_ps(acc + 4);
    __m128 vacc1x0123 = _mm_load_ps(acc + 8);
    __m128 vacc1x4567 = _mm_load_ps(acc + 12);
    __m128 vacc2x0123 = _mm_load_ps(acc + 16);
    __m128 vacc2x4567 = _mm_load_ps(acc + 20);
    __m128 vacc3x0123 = _mm_load_ps(acc + 24);
    __m128 vacc3x4567 = _mm_load_ps(acc + 28);
    __m128 vacc4x0123 = _mm_load_ps(acc + 32);
    __m128 vacc4x4567 = _mm_load_ps(acc + 36);
    __m128 vacc5x0123 = _mm_load_ps(acc + 40);
    __m128 vacc5x4567 = _mm_load_ps(acc + 44);
    acc += 48;

    size_t k = kc;
    do {
      const __m128 va0 = _mm_load1_ps(a0);
      a0 += 1;
      const __m128 va1 = _mm_load1_ps(a1);
      a1 += 1;
      const __m128 va2 = _mm_load1_ps(a2);
      a2 += 1;
      const __m128 va3 = _mm_load1_ps(a3);
      a3 += 1;
      const __m128 va4 = _mm_load1_ps(a4);
      a4 += 1;
      const __m128 va5 = _mm_load1_ps(a5);
      a5 += 1;

      const __m128 vb0123 = _mm_load_ps(w);
      const __m128 vb4567 = _mm_load_ps(w + 4);
      w += 8;

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123));
      vacc4x0123 = _mm_add_ps(vacc4x0123, _mm_mul_ps(va4, vb0123));
      vacc5x0123 = _mm_add_ps(vacc5x0123, _mm_mul_ps(va5, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567));
      vacc4x4567 = _mm_add_ps(vacc4x4567, _mm_mul_ps(va4, vb4567));
      vacc5x4567 = _mm_add_ps(vacc5x4567, _mm_mul_ps(va5, vb4567));

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);
    vacc2x0123 = _mm_min_ps(vacc2x0123, vmax);
    vacc3x0123 = _mm_min_ps(vacc3x0123, vmax);
    vacc4x0123 = _mm_min_ps(vacc4x0123, vmax);
    vacc5x0123 = _mm_min_ps(vacc5x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);
    vacc1x4567 = _mm_min_ps(vacc1x4567, vmax);
    vacc2x4567 = _mm_min_ps(vacc2x4567, vmax);
    vacc3x4567 = _mm_min_ps(vacc3x4567, vmax);
    vacc4x4567 = _mm_min_ps(vacc4x4567, vmax);
    vacc5x4567 = _mm_min_ps(vacc5x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);
    vacc2x0123 = _mm_max_ps(vacc2x0123, vmin);
    vacc3x0123 = _mm_max_ps(vacc3x0123, vmin);
    vacc4x0123 = _mm_max_ps(vacc4x0123, vmin);
    vacc5x0123 = _mm_max_ps(vacc5x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);
    vacc1x4567 = _mm_max_ps(vacc1x4567, vmin);
    vacc2x4567 = _mm_max_ps(vacc2x4567, vmin);
    vacc3x4567 = _mm_max_ps(vacc3x4567, vmin);
    vacc4x4567 = _mm_max_ps(vacc4x4567, vmin);
    vacc5x4567 = _mm_max_ps(vacc5x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c5, vacc5x0123);
      _mm_storeu_ps(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_ps(c4, vacc4x0123);
      _mm_storeu_ps(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_ps(c3, vacc3x0123);
      _mm_storeu_ps(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_ps(c2, vacc2x0123);
      _mm_storeu_ps(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_ps(c1, vacc1x0123);
      _mm_storeu_ps(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c5, vacc5x0123);
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c5 += 4;
        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c5, vacc5x0123);
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc5x0123 = _mm_movehl_ps(vacc5x0123, vacc5x0123);
        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c5, vacc5x0123);
        _mm_store_ss(c4, vacc4x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
