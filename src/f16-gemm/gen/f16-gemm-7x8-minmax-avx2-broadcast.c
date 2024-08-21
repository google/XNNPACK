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

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_gemm_minmax_ukernel_7x8__avx2_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = a;
  uint16_t* c0 = c;

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

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
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint16_t* a5 = (const uint16_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const uint16_t* a6 = (const uint16_t*) ((uintptr_t) a5 + a_stride);
  uint16_t* c6 = (uint16_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  do {
    __m256 vacc0x0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
    __m256 vacc1x0 = vacc0x0;
    __m256 vacc2x0 = vacc0x0;
    __m256 vacc3x0 = vacc0x0;
    __m256 vacc4x0 = vacc0x0;
    __m256 vacc5x0 = vacc0x0;
    __m256 vacc6x0 = vacc0x0;
    w = (const uint16_t*) w + 8;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a0));
      a0 += 1;
      const __m256 va1 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a1));
      a1 += 1;
      const __m256 va2 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a2));
      a2 += 1;
      const __m256 va3 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a3));
      a3 += 1;
      const __m256 va4 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a4));
      a4 += 1;
      const __m256 va5 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a5));
      a5 += 1;
      const __m256 va6 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a6));
      a6 += 1;

      const __m256 vb0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      w = (const uint16_t*) w + 8;

      vacc0x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb0, vacc0x0), _MM_FROUND_TO_NEAREST_INT));
      vacc1x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb0, vacc1x0), _MM_FROUND_TO_NEAREST_INT));
      vacc2x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb0, vacc2x0), _MM_FROUND_TO_NEAREST_INT));
      vacc3x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va3, vb0, vacc3x0), _MM_FROUND_TO_NEAREST_INT));
      vacc4x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va4, vb0, vacc4x0), _MM_FROUND_TO_NEAREST_INT));
      vacc5x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va5, vb0, vacc5x0), _MM_FROUND_TO_NEAREST_INT));
      vacc6x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va6, vb0, vacc6x0), _MM_FROUND_TO_NEAREST_INT));

      k -= sizeof(uint16_t);
    } while (k != 0);

    vacc0x0 = _mm256_max_ps(vacc0x0, vmin);
    vacc1x0 = _mm256_max_ps(vacc1x0, vmin);
    vacc2x0 = _mm256_max_ps(vacc2x0, vmin);
    vacc3x0 = _mm256_max_ps(vacc3x0, vmin);
    vacc4x0 = _mm256_max_ps(vacc4x0, vmin);
    vacc5x0 = _mm256_max_ps(vacc5x0, vmin);
    vacc6x0 = _mm256_max_ps(vacc6x0, vmin);

    vacc0x0 = _mm256_min_ps(vacc0x0, vmax);
    vacc1x0 = _mm256_min_ps(vacc1x0, vmax);
    vacc2x0 = _mm256_min_ps(vacc2x0, vmax);
    vacc3x0 = _mm256_min_ps(vacc3x0, vmax);
    vacc4x0 = _mm256_min_ps(vacc4x0, vmax);
    vacc5x0 = _mm256_min_ps(vacc5x0, vmax);
    vacc6x0 = _mm256_min_ps(vacc6x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_si128((__m128i*) c0, _mm256_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_si128((__m128i*) c1, _mm256_cvtps_ph(vacc1x0, _MM_FROUND_TO_NEAREST_INT));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) c2, _mm256_cvtps_ph(vacc2x0, _MM_FROUND_TO_NEAREST_INT));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) c3, _mm256_cvtps_ph(vacc3x0, _MM_FROUND_TO_NEAREST_INT));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) c4, _mm256_cvtps_ph(vacc4x0, _MM_FROUND_TO_NEAREST_INT));
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) c5, _mm256_cvtps_ph(vacc5x0, _MM_FROUND_TO_NEAREST_INT));
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) c6, _mm256_cvtps_ph(vacc6x0, _MM_FROUND_TO_NEAREST_INT));
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
      a4 = (const uint16_t*) ((uintptr_t) a4 - kc);
      a5 = (const uint16_t*) ((uintptr_t) a5 - kc);
      a6 = (const uint16_t*) ((uintptr_t) a6 - kc);

      nc -= 8;
    } else {
      __m128i vh0x0 = _mm256_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh1x0 = _mm256_cvtps_ph(vacc1x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh2x0 = _mm256_cvtps_ph(vacc2x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh3x0 = _mm256_cvtps_ph(vacc3x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh4x0 = _mm256_cvtps_ph(vacc4x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh5x0 = _mm256_cvtps_ph(vacc5x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh6x0 = _mm256_cvtps_ph(vacc6x0, _MM_FROUND_TO_NEAREST_INT);
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c0, vh0x0);
        _mm_storel_epi64((__m128i*) c1, vh1x0);
        _mm_storel_epi64((__m128i*) c2, vh2x0);
        _mm_storel_epi64((__m128i*) c3, vh3x0);
        _mm_storel_epi64((__m128i*) c4, vh4x0);
        _mm_storel_epi64((__m128i*) c5, vh5x0);
        _mm_storel_epi64((__m128i*) c6, vh6x0);

        vh0x0 = _mm_unpackhi_epi64(vh0x0, vh0x0);
        vh1x0 = _mm_unpackhi_epi64(vh1x0, vh1x0);
        vh2x0 = _mm_unpackhi_epi64(vh2x0, vh2x0);
        vh3x0 = _mm_unpackhi_epi64(vh3x0, vh3x0);
        vh4x0 = _mm_unpackhi_epi64(vh4x0, vh4x0);
        vh5x0 = _mm_unpackhi_epi64(vh5x0, vh5x0);
        vh6x0 = _mm_unpackhi_epi64(vh6x0, vh6x0);

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
        c6 += 4;
      }
      if (nc & 2) {
        _mm_storeu_si32(c0, vh0x0);
        _mm_storeu_si32(c1, vh1x0);
        _mm_storeu_si32(c2, vh2x0);
        _mm_storeu_si32(c3, vh3x0);
        _mm_storeu_si32(c4, vh4x0);
        _mm_storeu_si32(c5, vh5x0);
        _mm_storeu_si32(c6, vh6x0);

        vh0x0 = _mm_srli_epi64(vh0x0, 32);
        vh1x0 = _mm_srli_epi64(vh1x0, 32);
        vh2x0 = _mm_srli_epi64(vh2x0, 32);
        vh3x0 = _mm_srli_epi64(vh3x0, 32);
        vh4x0 = _mm_srli_epi64(vh4x0, 32);
        vh5x0 = _mm_srli_epi64(vh5x0, 32);
        vh6x0 = _mm_srli_epi64(vh6x0, 32);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
        c6 += 2;
      }
      if (nc & 1) {
        *c0 = (uint16_t) _mm_extract_epi16(vh0x0, 0);
        *c1 = (uint16_t) _mm_extract_epi16(vh1x0, 0);
        *c2 = (uint16_t) _mm_extract_epi16(vh2x0, 0);
        *c3 = (uint16_t) _mm_extract_epi16(vh3x0, 0);
        *c4 = (uint16_t) _mm_extract_epi16(vh4x0, 0);
        *c5 = (uint16_t) _mm_extract_epi16(vh5x0, 0);
        *c6 = (uint16_t) _mm_extract_epi16(vh6x0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
