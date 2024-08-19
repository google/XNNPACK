// Auto-generated file. Do not edit!
//   Template: src/f16-igemm/avx2-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (5 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* c0 = c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }

  do {
    __m256 vacc0x0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
    __m256 vacc0x1 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
    __m256 vacc1x0 = vacc0x0;
    __m256 vacc1x1 = vacc0x1;
    __m256 vacc2x0 = vacc0x0;
    __m256 vacc2x1 = vacc0x1;
    __m256 vacc3x0 = vacc0x0;
    __m256 vacc3x1 = vacc0x1;
    __m256 vacc4x0 = vacc0x0;
    __m256 vacc4x1 = vacc0x1;
    w = (const uint16_t*) w + 16;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint16_t* restrict a1 = (const uint16_t*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint16_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint16_t* restrict a2 = (const uint16_t*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint16_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint16_t* restrict a3 = (const uint16_t*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint16_t*) ((uintptr_t) a3 + a_offset);
      }
      const uint16_t* restrict a4 = (const uint16_t*) a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const uint16_t*) ((uintptr_t) a4 + a_offset);
      }
      a += 5;

      size_t k = kc;
      do {
        const __m256 vb0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
        const __m256 vb1 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
        w = (const uint16_t*) w + 16;

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

        vacc0x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb0, vacc0x0), _MM_FROUND_TO_NEAREST_INT));
        vacc0x1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb1, vacc0x1), _MM_FROUND_TO_NEAREST_INT));
        vacc1x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb0, vacc1x0), _MM_FROUND_TO_NEAREST_INT));
        vacc1x1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb1, vacc1x1), _MM_FROUND_TO_NEAREST_INT));
        vacc2x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb0, vacc2x0), _MM_FROUND_TO_NEAREST_INT));
        vacc2x1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb1, vacc2x1), _MM_FROUND_TO_NEAREST_INT));
        vacc3x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va3, vb0, vacc3x0), _MM_FROUND_TO_NEAREST_INT));
        vacc3x1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va3, vb1, vacc3x1), _MM_FROUND_TO_NEAREST_INT));
        vacc4x0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va4, vb0, vacc4x0), _MM_FROUND_TO_NEAREST_INT));
        vacc4x1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va4, vb1, vacc4x1), _MM_FROUND_TO_NEAREST_INT));

        k -= sizeof(uint16_t);
      } while (k != 0);
      p -= 5 * sizeof(void*);
    } while (p != 0);

    vacc0x0 = _mm256_max_ps(vacc0x0, vmin);
    vacc1x0 = _mm256_max_ps(vacc1x0, vmin);
    vacc2x0 = _mm256_max_ps(vacc2x0, vmin);
    vacc3x0 = _mm256_max_ps(vacc3x0, vmin);
    vacc4x0 = _mm256_max_ps(vacc4x0, vmin);
    vacc0x1 = _mm256_max_ps(vacc0x1, vmin);
    vacc1x1 = _mm256_max_ps(vacc1x1, vmin);
    vacc2x1 = _mm256_max_ps(vacc2x1, vmin);
    vacc3x1 = _mm256_max_ps(vacc3x1, vmin);
    vacc4x1 = _mm256_max_ps(vacc4x1, vmin);

    vacc0x0 = _mm256_min_ps(vacc0x0, vmax);
    vacc1x0 = _mm256_min_ps(vacc1x0, vmax);
    vacc2x0 = _mm256_min_ps(vacc2x0, vmax);
    vacc3x0 = _mm256_min_ps(vacc3x0, vmax);
    vacc4x0 = _mm256_min_ps(vacc4x0, vmax);
    vacc0x1 = _mm256_min_ps(vacc0x1, vmax);
    vacc1x1 = _mm256_min_ps(vacc1x1, vmax);
    vacc2x1 = _mm256_min_ps(vacc2x1, vmax);
    vacc3x1 = _mm256_min_ps(vacc3x1, vmax);
    vacc4x1 = _mm256_min_ps(vacc4x1, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm_storeu_si128((__m128i*) c4, _mm256_cvtps_ph(vacc4x0, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c4 + 8), _mm256_cvtps_ph(vacc4x1, _MM_FROUND_TO_NEAREST_INT));
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) c3, _mm256_cvtps_ph(vacc3x0, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c3 + 8), _mm256_cvtps_ph(vacc3x1, _MM_FROUND_TO_NEAREST_INT));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) c2, _mm256_cvtps_ph(vacc2x0, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c2 + 8), _mm256_cvtps_ph(vacc2x1, _MM_FROUND_TO_NEAREST_INT));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) c1, _mm256_cvtps_ph(vacc1x0, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c1 + 8), _mm256_cvtps_ph(vacc1x1, _MM_FROUND_TO_NEAREST_INT));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) c0, _mm256_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c0 + 8), _mm256_cvtps_ph(vacc0x1, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      __m128i vh4x0 = _mm256_cvtps_ph(vacc4x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh3x0 = _mm256_cvtps_ph(vacc3x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh2x0 = _mm256_cvtps_ph(vacc2x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh1x0 = _mm256_cvtps_ph(vacc1x0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh0x0 = _mm256_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT);
      if (nc & 8) {
        _mm_storeu_si128((__m128i*) c4, vh4x0);
        _mm_storeu_si128((__m128i*) c3, vh3x0);
        _mm_storeu_si128((__m128i*) c2, vh2x0);
        _mm_storeu_si128((__m128i*) c1, vh1x0);
        _mm_storeu_si128((__m128i*) c0, vh0x0);

        vh4x0 = _mm256_cvtps_ph(vacc4x1, _MM_FROUND_TO_NEAREST_INT);
        vh3x0 = _mm256_cvtps_ph(vacc3x1, _MM_FROUND_TO_NEAREST_INT);
        vh2x0 = _mm256_cvtps_ph(vacc2x1, _MM_FROUND_TO_NEAREST_INT);
        vh1x0 = _mm256_cvtps_ph(vacc1x1, _MM_FROUND_TO_NEAREST_INT);
        vh0x0 = _mm256_cvtps_ph(vacc0x1, _MM_FROUND_TO_NEAREST_INT);

        c4 += 8;
        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c4, vh4x0);
        _mm_storel_epi64((__m128i*) c3, vh3x0);
        _mm_storel_epi64((__m128i*) c2, vh2x0);
        _mm_storel_epi64((__m128i*) c1, vh1x0);
        _mm_storel_epi64((__m128i*) c0, vh0x0);

        vh4x0 = _mm_unpackhi_epi64(vh4x0, vh4x0);
        vh3x0 = _mm_unpackhi_epi64(vh3x0, vh3x0);
        vh2x0 = _mm_unpackhi_epi64(vh2x0, vh2x0);
        vh1x0 = _mm_unpackhi_epi64(vh1x0, vh1x0);
        vh0x0 = _mm_unpackhi_epi64(vh0x0, vh0x0);

        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storeu_si32(c4, vh4x0);
        _mm_storeu_si32(c3, vh3x0);
        _mm_storeu_si32(c2, vh2x0);
        _mm_storeu_si32(c1, vh1x0);
        _mm_storeu_si32(c0, vh0x0);

        vh4x0 = _mm_srli_epi64(vh4x0, 32);
        vh3x0 = _mm_srli_epi64(vh3x0, 32);
        vh2x0 = _mm_srli_epi64(vh2x0, 32);
        vh1x0 = _mm_srli_epi64(vh1x0, 32);
        vh0x0 = _mm_srli_epi64(vh0x0, 32);

        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        *c4 = _mm_extract_epi16(vh4x0, 0);
        *c3 = _mm_extract_epi16(vh3x0, 0);
        *c2 = _mm_extract_epi16(vh2x0, 0);
        *c1 = _mm_extract_epi16(vh1x0, 0);
        *c0 = _mm_extract_epi16(vh0x0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
