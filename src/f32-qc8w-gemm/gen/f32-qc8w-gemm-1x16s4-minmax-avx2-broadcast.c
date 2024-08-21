// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/avx-shuffle4.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>
#include <smmintrin.h>

#include "xnnpack/gemm.h"


void xnn_f32_qc8w_gemm_minmax_ukernel_1x16s4__avx2_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_loadu_ps((const float*) w + 0);
    __m256 vacc0x89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    w = (const float*) w + 16;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 += 4;

      const __m256i vbi01234567c0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 0)));
      const __m256i vbi89ABCDEFc0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8)));
      const __m256 vb01234567c0 = _mm256_cvtepi32_ps(vbi01234567c0);
      const __m256 vb89ABCDEFc0 = _mm256_cvtepi32_ps(vbi89ABCDEFc0);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256i vbi01234567c1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16)));
      const __m256i vbi89ABCDEFc1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24)));
      const __m256 vb01234567c1 = _mm256_cvtepi32_ps(vbi01234567c1);
      const __m256 vb89ABCDEFc1 = _mm256_cvtepi32_ps(vbi89ABCDEFc1);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256i vbi01234567c2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 32)));
      const __m256i vbi89ABCDEFc2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 40)));
      const __m256 vb01234567c2 = _mm256_cvtepi32_ps(vbi01234567c2);
      const __m256 vb89ABCDEFc2 = _mm256_cvtepi32_ps(vbi89ABCDEFc2);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256i vbi01234567c3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 48)));
      const __m256i vbi89ABCDEFc3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 56)));
      const __m256 vb01234567c3 = _mm256_cvtepi32_ps(vbi01234567c3);
      const __m256 vb89ABCDEFc3 = _mm256_cvtepi32_ps(vbi89ABCDEFc3);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);


      w = (const int8_t*) w + 64;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 = (const float*) ((uintptr_t) a0 + k);

      const __m256 vzero = _mm256_setzero_ps();
      const __m256i vbi01234567c0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 0)));
      const __m256i vbi89ABCDEFc0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8)));
      const __m256 vb01234567c0 = _mm256_cvtepi32_ps(vbi01234567c0);
      const __m256 vb89ABCDEFc0 = _mm256_cvtepi32_ps(vbi89ABCDEFc0);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256i vbi01234567c1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16)));
      const __m256i vbi89ABCDEFc1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24)));
      const __m256 vb01234567c1 = _mm256_cvtepi32_ps(vbi01234567c1);
      const __m256 vb89ABCDEFc1 = _mm256_cvtepi32_ps(vbi89ABCDEFc1);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256i vbi01234567c2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 32)));
      const __m256i vbi89ABCDEFc2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 40)));
      const __m256 vb01234567c2 = _mm256_cvtepi32_ps(vbi01234567c2);
      const __m256 vb89ABCDEFc2 = _mm256_cvtepi32_ps(vbi89ABCDEFc2);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256i vbi01234567c3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 48)));
      const __m256i vbi89ABCDEFc3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 56)));
      const __m256 vb01234567c3 = _mm256_cvtepi32_ps(vbi01234567c3);
      const __m256 vb89ABCDEFc3 = _mm256_cvtepi32_ps(vbi89ABCDEFc3);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);


      w = (const int8_t*) w + 64;
    }

    const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w + 0);
    vacc0x01234567 = _mm256_mul_ps(vacc0x01234567, vscale01234567);
    const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    vacc0x89ABCDEF = _mm256_mul_ps(vacc0x89ABCDEF, vscale89ABCDEF);
    w = (const float*) w + 16;
    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
