// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/avx-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>


void xnn_f32_gemminc_minmax_ukernel_5x16__fma3_broadcast(
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
  assert(mr <= 5);
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

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(acc + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(acc + 8);
    __m256 vacc1x01234567 = _mm256_load_ps(acc + 16);
    __m256 vacc1x89ABCDEF = _mm256_load_ps(acc + 24);
    __m256 vacc2x01234567 = _mm256_load_ps(acc + 32);
    __m256 vacc2x89ABCDEF = _mm256_load_ps(acc + 40);
    __m256 vacc3x01234567 = _mm256_load_ps(acc + 48);
    __m256 vacc3x89ABCDEF = _mm256_load_ps(acc + 56);
    __m256 vacc4x01234567 = _mm256_load_ps(acc + 64);
    __m256 vacc4x89ABCDEF = _mm256_load_ps(acc + 72);
    acc += 80;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1 = _mm256_broadcast_ss(a1);
      a1 += 1;
      const __m256 va2 = _mm256_broadcast_ss(a2);
      a2 += 1;
      const __m256 va3 = _mm256_broadcast_ss(a3);
      a3 += 1;
      const __m256 va4 = _mm256_broadcast_ss(a4);
      a4 += 1;

      const __m256 vb01234567 = _mm256_load_ps(w);
      const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
      w += 16;

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567, vacc4x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEF, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEF, vacc4x89ABCDEF);

      k -= sizeof(float);
    } while (k != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc4x01234567 = _mm256_max_ps(vmin, vacc4x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_max_ps(vmin, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_max_ps(vmin, vacc4x89ABCDEF);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc4x01234567 = _mm256_min_ps(vmax, vacc4x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_min_ps(vmax, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_min_ps(vmax, vacc4x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c4, vacc4x01234567);
      _mm256_storeu_ps(c4 + 8, vacc4x89ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c4, vacc4x01234567);
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc4x01234567 = vacc4x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c4 += 8;
        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
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
