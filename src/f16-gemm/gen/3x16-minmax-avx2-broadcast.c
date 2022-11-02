// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/avx2-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f16_gemm_minmax_ukernel_3x16__avx2_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const void*restrict a,
    size_t a_stride,
    const void*restrict w,
    void*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = a;
  uint16_t* c0 = c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  do {
    __m256 vacc0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
    __m256 vacc0x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    w = (const uint16_t*) w + 16;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a0));
      a0 += 1;
      const __m256 va1 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a1));
      a1 += 1;
      const __m256 va2 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a2));
      a2 += 1;

      const __m256 vb01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      const __m256 vb89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
      w = (const uint16_t*) w + 16;

      vacc0x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb01234567, vacc0x01234567), _MM_FROUND_TO_NEAREST_INT));
      vacc1x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb01234567, vacc1x01234567), _MM_FROUND_TO_NEAREST_INT));
      vacc2x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb01234567, vacc2x01234567), _MM_FROUND_TO_NEAREST_INT));
      vacc0x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));
      vacc1x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));
      vacc2x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));

      k -= sizeof(uint16_t);
    } while (k != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc1x01234567 = _mm256_max_ps(vacc1x01234567, vmin);
    vacc2x01234567 = _mm256_max_ps(vacc2x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = _mm256_max_ps(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = _mm256_max_ps(vacc2x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc1x01234567 = _mm256_min_ps(vacc1x01234567, vmax);
    vacc2x01234567 = _mm256_min_ps(vacc2x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = _mm256_min_ps(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = _mm256_min_ps(vacc2x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c0 + 8), _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_si128((__m128i*) c1, _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c1 + 8), _mm256_cvtps_ph(vacc1x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) c2, _mm256_cvtps_ph(vacc2x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c2 + 8), _mm256_cvtps_ph(vacc2x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);

      nc -= 16;
    } else {
      __m128i vh0x01234567 = _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh1x01234567 = _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh2x01234567 = _mm256_cvtps_ph(vacc2x01234567, _MM_FROUND_TO_NEAREST_INT);
      if (nc & 8) {
        _mm_storeu_si128((__m128i*) c0, vh0x01234567);
        _mm_storeu_si128((__m128i*) c1, vh1x01234567);
        _mm_storeu_si128((__m128i*) c2, vh2x01234567);

        vh0x01234567 = _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);
        vh1x01234567 = _mm256_cvtps_ph(vacc1x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);
        vh2x01234567 = _mm256_cvtps_ph(vacc2x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);

        c0 += 8;
        c1 += 8;
        c2 += 8;
      }
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c0, vh0x01234567);
        _mm_storel_epi64((__m128i*) c1, vh1x01234567);
        _mm_storel_epi64((__m128i*) c2, vh2x01234567);

        vh0x01234567 = _mm_unpackhi_epi64(vh0x01234567, vh0x01234567);
        vh1x01234567 = _mm_unpackhi_epi64(vh1x01234567, vh1x01234567);
        vh2x01234567 = _mm_unpackhi_epi64(vh2x01234567, vh2x01234567);

        c0 += 4;
        c1 += 4;
        c2 += 4;
      }
      if (nc & 2) {
        _mm_storeu_si32(c0, vh0x01234567);
        _mm_storeu_si32(c1, vh1x01234567);
        _mm_storeu_si32(c2, vh2x01234567);

        vh0x01234567 = _mm_srli_epi64(vh0x01234567, 32);
        vh1x01234567 = _mm_srli_epi64(vh1x01234567, 32);
        vh2x01234567 = _mm_srli_epi64(vh2x01234567, 32);

        c0 += 2;
        c1 += 2;
        c2 += 2;
      }
      if (nc & 1) {
        *c0 = (uint16_t) _mm_extract_epi16(vh0x01234567, 0);
        *c1 = (uint16_t) _mm_extract_epi16(vh1x01234567, 0);
        *c2 = (uint16_t) _mm_extract_epi16(vh2x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
