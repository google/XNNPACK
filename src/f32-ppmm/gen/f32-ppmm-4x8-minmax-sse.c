// Auto-generated file. Do not edit!
//   Template: src/f32-ppmm/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/ppmm.h>


void xnn_f32_ppmm_minmax_ukernel_4x8__sse(
  size_t mr,
  size_t nc,
  size_t kc,
  const float*restrict a,
  const float*restrict w,
  float*restrict c,
  size_t cm_stride,
  size_t cn_stride,
  const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

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
    __m128 vacc0x0123 = _mm_load_ps(w);
    __m128 vacc0x4567 = _mm_load_ps(w + 4);
    __m128 vacc1x0123 = vacc0x0123;
    __m128 vacc1x4567 = vacc0x4567;
    __m128 vacc2x0123 = vacc0x0123;
    __m128 vacc2x4567 = vacc0x4567;
    __m128 vacc3x0123 = vacc0x0123;
    __m128 vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    do {
      const __m128 va0123 = _mm_load_ps(a);
      a += 4;

      const __m128 vb0123 = _mm_load_ps(w);
      const __m128 vb4567 = _mm_load_ps(w + 4);
      w += 8;

      const __m128 va0000 = _mm_shuffle_ps(va0123, va0123, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128 va1111 = _mm_shuffle_ps(va0123, va0123, _MM_SHUFFLE(1, 1, 1, 1));
      const __m128 va2222 = _mm_shuffle_ps(va0123, va0123, _MM_SHUFFLE(2, 2, 2, 2));
      const __m128 va3333 = _mm_shuffle_ps(va0123, va0123, _MM_SHUFFLE(3, 3, 3, 3));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0000, vb0123));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1111, vb0123));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2222, vb0123));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3333, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0000, vb4567));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1111, vb4567));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2222, vb4567));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3333, vb4567));

      k -= sizeof(float);
    } while (k != 0);

    const __m128 vmax = _mm_load_ps(params->sse.max);
    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);
    vacc2x0123 = _mm_min_ps(vacc2x0123, vmax);
    vacc3x0123 = _mm_min_ps(vacc3x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);
    vacc1x4567 = _mm_min_ps(vacc1x4567, vmax);
    vacc2x4567 = _mm_min_ps(vacc2x4567, vmax);
    vacc3x4567 = _mm_min_ps(vacc3x4567, vmax);

    const __m128 vmin = _mm_load_ps(params->sse.min);
    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);
    vacc2x0123 = _mm_max_ps(vacc2x0123, vmin);
    vacc3x0123 = _mm_max_ps(vacc3x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);
    vacc1x4567 = _mm_max_ps(vacc1x4567, vmin);
    vacc2x4567 = _mm_max_ps(vacc2x4567, vmin);
    vacc3x4567 = _mm_max_ps(vacc3x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c3, vacc3x0123);
      _mm_storeu_ps(c3 + 4, vacc3x4567);
      _mm_storeu_ps(c2, vacc2x0123);
      _mm_storeu_ps(c2 + 4, vacc2x4567);
      _mm_storeu_ps(c1, vacc1x0123);
      _mm_storeu_ps(c1 + 4, vacc1x4567);
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);

      a = (const float*) ((uintptr_t) a - kc * 4);

      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
