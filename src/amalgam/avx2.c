// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/lut.h>
#include <xnnpack/math.h>
#include <xnnpack/pavgpool.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/transpose.h>
#include <xnnpack/unaligned.h>
#include <xnnpack/vadd.h>
#include <xnnpack/vcvt.h>
#include <xnnpack/vlrelu.h>
#include <xnnpack/vunary.h>


void xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = a;
  uint16_t* c0 = c;

  do {
    __m256 vacc0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
    __m256 vacc0x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
    w = (const uint16_t*) w + 16;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a0));
      a0 += 1;

      const __m256 vb01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      const __m256 vb89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
      w = (const uint16_t*) w + 16;

      vacc0x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb01234567, vacc0x01234567), _MM_FROUND_TO_NEAREST_INT));
      vacc0x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));

      k -= sizeof(uint16_t);
    } while (k != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c0 + 8), _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      __m128i vh0x01234567 = _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT);
      if (nc & 8) {
        _mm_storeu_si128((__m128i*) c0, vh0x01234567);

        vh0x01234567 = _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);

        c0 += 8;
      }
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c0, vh0x01234567);

        vh0x01234567 = _mm_unpackhi_epi64(vh0x01234567, vh0x01234567);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storeu_si32(c0, vh0x01234567);

        vh0x01234567 = _mm_srli_epi64(vh0x01234567, 32);

        c0 += 2;
      }
      if (nc & 1) {
        *c0 = (uint16_t) _mm_extract_epi16(vh0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast(
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
  assert(mr <= 4);
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
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    __m256 vacc0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
    __m256 vacc0x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    w = (const uint16_t*) w + 16;

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

      const __m256 vb01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      const __m256 vb89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
      w = (const uint16_t*) w + 16;

      vacc0x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb01234567, vacc0x01234567), _MM_FROUND_TO_NEAREST_INT));
      vacc1x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb01234567, vacc1x01234567), _MM_FROUND_TO_NEAREST_INT));
      vacc2x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb01234567, vacc2x01234567), _MM_FROUND_TO_NEAREST_INT));
      vacc3x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va3, vb01234567, vacc3x01234567), _MM_FROUND_TO_NEAREST_INT));
      vacc0x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));
      vacc1x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));
      vacc2x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));
      vacc3x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va3, vb89ABCDEF, vacc3x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));

      k -= sizeof(uint16_t);
    } while (k != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc1x01234567 = _mm256_max_ps(vacc1x01234567, vmin);
    vacc2x01234567 = _mm256_max_ps(vacc2x01234567, vmin);
    vacc3x01234567 = _mm256_max_ps(vacc3x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = _mm256_max_ps(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = _mm256_max_ps(vacc2x89ABCDEF, vmin);
    vacc3x89ABCDEF = _mm256_max_ps(vacc3x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc1x01234567 = _mm256_min_ps(vacc1x01234567, vmax);
    vacc2x01234567 = _mm256_min_ps(vacc2x01234567, vmax);
    vacc3x01234567 = _mm256_min_ps(vacc3x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = _mm256_min_ps(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = _mm256_min_ps(vacc2x89ABCDEF, vmax);
    vacc3x89ABCDEF = _mm256_min_ps(vacc3x89ABCDEF, vmax);

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
      _mm_storeu_si128((__m128i*) c3, _mm256_cvtps_ph(vacc3x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c3 + 8), _mm256_cvtps_ph(vacc3x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);

      nc -= 16;
    } else {
      __m128i vh0x01234567 = _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh1x01234567 = _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh2x01234567 = _mm256_cvtps_ph(vacc2x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh3x01234567 = _mm256_cvtps_ph(vacc3x01234567, _MM_FROUND_TO_NEAREST_INT);
      if (nc & 8) {
        _mm_storeu_si128((__m128i*) c0, vh0x01234567);
        _mm_storeu_si128((__m128i*) c1, vh1x01234567);
        _mm_storeu_si128((__m128i*) c2, vh2x01234567);
        _mm_storeu_si128((__m128i*) c3, vh3x01234567);

        vh0x01234567 = _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);
        vh1x01234567 = _mm256_cvtps_ph(vacc1x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);
        vh2x01234567 = _mm256_cvtps_ph(vacc2x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);
        vh3x01234567 = _mm256_cvtps_ph(vacc3x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);

        c0 += 8;
        c1 += 8;
        c2 += 8;
        c3 += 8;
      }
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c0, vh0x01234567);
        _mm_storel_epi64((__m128i*) c1, vh1x01234567);
        _mm_storel_epi64((__m128i*) c2, vh2x01234567);
        _mm_storel_epi64((__m128i*) c3, vh3x01234567);

        vh0x01234567 = _mm_unpackhi_epi64(vh0x01234567, vh0x01234567);
        vh1x01234567 = _mm_unpackhi_epi64(vh1x01234567, vh1x01234567);
        vh2x01234567 = _mm_unpackhi_epi64(vh2x01234567, vh2x01234567);
        vh3x01234567 = _mm_unpackhi_epi64(vh3x01234567, vh3x01234567);

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        _mm_storeu_si32(c0, vh0x01234567);
        _mm_storeu_si32(c1, vh1x01234567);
        _mm_storeu_si32(c2, vh2x01234567);
        _mm_storeu_si32(c3, vh3x01234567);

        vh0x01234567 = _mm_srli_epi64(vh0x01234567, 32);
        vh1x01234567 = _mm_srli_epi64(vh1x01234567, 32);
        vh2x01234567 = _mm_srli_epi64(vh2x01234567, 32);
        vh3x01234567 = _mm_srli_epi64(vh3x01234567, 32);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        *c0 = (uint16_t) _mm_extract_epi16(vh0x01234567, 0);
        *c1 = (uint16_t) _mm_extract_epi16(vh1x01234567, 0);
        *c2 = (uint16_t) _mm_extract_epi16(vh2x01234567, 0);
        *c3 = (uint16_t) _mm_extract_epi16(vh3x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void**restrict a,
    const void*restrict w,
    void*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint16_t* c0 = c;

  do {
    __m256 vacc0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
    __m256 vacc0x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
    w = (const uint16_t*) w + 16;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const __m256 vb01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
        const __m256 vb89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
        w = (const uint16_t*) w + 16;

        const __m256 va0 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a0));
        a0 += 1;

        vacc0x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb01234567, vacc0x01234567), _MM_FROUND_TO_NEAREST_INT));
        vacc0x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));

        k -= sizeof(uint16_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c0 + 8), _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      __m128i vh0x01234567 = _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT);
      if (nc & 8) {
        _mm_storeu_si128((__m128i*) c0, vh0x01234567);

        vh0x01234567 = _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);

        c0 += 8;
      }
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c0, vh0x01234567);

        vh0x01234567 = _mm_unpackhi_epi64(vh0x01234567, vh0x01234567);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storeu_si32(c0, vh0x01234567);

        vh0x01234567 = _mm_srli_epi64(vh0x01234567, 32);

        c0 += 2;
      }
      if (nc & 1) {
        *c0 = _mm_extract_epi16(vh0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void**restrict a,
    const void*restrict w,
    void*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    __m256 vacc0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
    __m256 vacc0x89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
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
      a += 4;

      size_t k = kc;
      do {
        const __m256 vb01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
        const __m256 vb89ABCDEF = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) ((const uint16_t*) w + 8)));
        w = (const uint16_t*) w + 16;

        const __m256 va0 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a0));
        a0 += 1;
        const __m256 va1 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a1));
        a1 += 1;
        const __m256 va2 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a2));
        a2 += 1;
        const __m256 va3 = _mm256_cvtph_ps(_mm_set1_epi16((short) *a3));
        a3 += 1;

        vacc0x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb01234567, vacc0x01234567), _MM_FROUND_TO_NEAREST_INT));
        vacc0x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));
        vacc1x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb01234567, vacc1x01234567), _MM_FROUND_TO_NEAREST_INT));
        vacc1x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));
        vacc2x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb01234567, vacc2x01234567), _MM_FROUND_TO_NEAREST_INT));
        vacc2x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));
        vacc3x01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va3, vb01234567, vacc3x01234567), _MM_FROUND_TO_NEAREST_INT));
        vacc3x89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(va3, vb89ABCDEF, vacc3x89ABCDEF), _MM_FROUND_TO_NEAREST_INT));

        k -= sizeof(uint16_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc1x01234567 = _mm256_max_ps(vacc1x01234567, vmin);
    vacc2x01234567 = _mm256_max_ps(vacc2x01234567, vmin);
    vacc3x01234567 = _mm256_max_ps(vacc3x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = _mm256_max_ps(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = _mm256_max_ps(vacc2x89ABCDEF, vmin);
    vacc3x89ABCDEF = _mm256_max_ps(vacc3x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc1x01234567 = _mm256_min_ps(vacc1x01234567, vmax);
    vacc2x01234567 = _mm256_min_ps(vacc2x01234567, vmax);
    vacc3x01234567 = _mm256_min_ps(vacc3x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = _mm256_min_ps(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = _mm256_min_ps(vacc2x89ABCDEF, vmax);
    vacc3x89ABCDEF = _mm256_min_ps(vacc3x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm_storeu_si128((__m128i*) c3, _mm256_cvtps_ph(vacc3x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c3 + 8), _mm256_cvtps_ph(vacc3x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) c2, _mm256_cvtps_ph(vacc2x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c2 + 8), _mm256_cvtps_ph(vacc2x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) c1, _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c1 + 8), _mm256_cvtps_ph(vacc1x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) c0, _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (c0 + 8), _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      __m128i vh3x01234567 = _mm256_cvtps_ph(vacc3x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh2x01234567 = _mm256_cvtps_ph(vacc2x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh1x01234567 = _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh0x01234567 = _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT);
      if (nc & 8) {
        _mm_storeu_si128((__m128i*) c3, vh3x01234567);
        _mm_storeu_si128((__m128i*) c2, vh2x01234567);
        _mm_storeu_si128((__m128i*) c1, vh1x01234567);
        _mm_storeu_si128((__m128i*) c0, vh0x01234567);

        vh3x01234567 = _mm256_cvtps_ph(vacc3x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);
        vh2x01234567 = _mm256_cvtps_ph(vacc2x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);
        vh1x01234567 = _mm256_cvtps_ph(vacc1x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);
        vh0x01234567 = _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT);

        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c3, vh3x01234567);
        _mm_storel_epi64((__m128i*) c2, vh2x01234567);
        _mm_storel_epi64((__m128i*) c1, vh1x01234567);
        _mm_storel_epi64((__m128i*) c0, vh0x01234567);

        vh3x01234567 = _mm_unpackhi_epi64(vh3x01234567, vh3x01234567);
        vh2x01234567 = _mm_unpackhi_epi64(vh2x01234567, vh2x01234567);
        vh1x01234567 = _mm_unpackhi_epi64(vh1x01234567, vh1x01234567);
        vh0x01234567 = _mm_unpackhi_epi64(vh0x01234567, vh0x01234567);

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storeu_si32(c3, vh3x01234567);
        _mm_storeu_si32(c2, vh2x01234567);
        _mm_storeu_si32(c1, vh1x01234567);
        _mm_storeu_si32(c0, vh0x01234567);

        vh3x01234567 = _mm_srli_epi64(vh3x01234567, 32);
        vh2x01234567 = _mm_srli_epi64(vh2x01234567, 32);
        vh1x01234567 = _mm_srli_epi64(vh1x01234567, 32);
        vh0x01234567 = _mm_srli_epi64(vh0x01234567, 32);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        *c3 = _mm_extract_epi16(vh3x01234567, 0);
        *c2 = _mm_extract_epi16(vh2x01234567, 0);
        *c1 = _mm_extract_epi16(vh1x01234567, 0);
        *c0 = _mm_extract_epi16(vh0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_pavgpool_minmax_ukernel_9p8x__avx2_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const __m256 voutput_min = _mm256_load_ps(params->avx.min);
  const __m256 voutput_max = _mm256_load_ps(params->avx.max);

  uint16_t* o = (uint16_t*) output;
  do {
    {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint16_t* i8 = (const uint16_t*) *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
        i8 += 8;

        const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum018 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vi8), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum01678 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum018, vsum67), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum01678), _MM_FROUND_TO_NEAREST_INT));

        _mm_storeu_si128((__m128i*) b, _mm256_cvtps_ph(vsum, _MM_FROUND_TO_NEAREST_INT));
        b += 8;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

        const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum01a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vacc), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum0167a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01a, vsum67), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum0167a), _MM_FROUND_TO_NEAREST_INT));

        _mm_storeu_si128((__m128i*) b, _mm256_cvtps_ph(vsum, _MM_FROUND_TO_NEAREST_INT));
        b += 8;
      }
    }

    {
      const uint16_t* i0 = (const uint16_t*) input[0];
      assert(i0 != NULL);
      const uint16_t* i1 = (const uint16_t*) input[1];
      const uint16_t* i2 = (const uint16_t*) input[2];
      const uint16_t* i3 = (const uint16_t*) input[3];
      const uint16_t* i4 = (const uint16_t*) input[4];
      const uint16_t* i5 = (const uint16_t*) input[5];
      const uint16_t* i6 = (const uint16_t*) input[6];
      const uint16_t* i7 = (const uint16_t*) input[7];
      input = (const void**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = (const uint16_t*) zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = (const uint16_t*) zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = (const uint16_t*) zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = (const uint16_t*) zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = (const uint16_t*) zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = (const uint16_t*) zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = (const uint16_t*) zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      const __m256 vmultiplier = _mm256_cvtph_ps(_mm_set1_epi16((short) *((const uint16_t*) multiplier)));
      multiplier = (const uint16_t*) multiplier + 1;

      size_t c = channels;
      const uint16_t* b = (const uint16_t*) buffer;
      while (c >= 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
        b += 8;

        const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum01a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vacc), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum0167a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01a, vsum67), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum0167a), _MM_FROUND_TO_NEAREST_INT));

        __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vmultiplier), _MM_FROUND_TO_NEAREST_INT));
        vout = _mm256_max_ps(vout, voutput_min);
        vout = _mm256_min_ps(vout, voutput_max);

        _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT));
        o += 8;

        c -= 8;
      }
      if (c != 0) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

        const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum01a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vacc), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum0167a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01a, vsum67), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum0167a), _MM_FROUND_TO_NEAREST_INT));

        __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vmultiplier), _MM_FROUND_TO_NEAREST_INT));
        vout = _mm256_max_ps(vout, voutput_min);
        vout = _mm256_min_ps(vout, voutput_max);

        __m128i vh = _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT);
        if (c & 4) {
          _mm_storel_epi64((__m128i*) o, vh);
          vh = _mm_unpackhi_epi64(vh, vh);
          o += 4;
        }
        if (c & 2) {
          _mm_storeu_si32(o, vh);
          vh = _mm_srli_epi64(vh, 32);
          o += 2;
        }
        if (c & 1) {
          *o = (uint16_t) _mm_extract_epi16(vh, 0);
          o += 1;
        }
      }
    }
    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_pavgpool_minmax_ukernel_9x__avx2_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const __m256 voutput_min = _mm256_load_ps(params->avx.min);
  const __m256 voutput_max = _mm256_load_ps(params->avx.max);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    const uint16_t* i1 = (const uint16_t*) input[1];
    const uint16_t* i2 = (const uint16_t*) input[2];
    const uint16_t* i3 = (const uint16_t*) input[3];
    const uint16_t* i4 = (const uint16_t*) input[4];
    const uint16_t* i5 = (const uint16_t*) input[5];
    const uint16_t* i6 = (const uint16_t*) input[6];
    const uint16_t* i7 = (const uint16_t*) input[7];
    const uint16_t* i8 = (const uint16_t*) input[8];
    input = (const void**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = (const uint16_t*) zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = (const uint16_t*) zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = (const uint16_t*) zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = (const uint16_t*) zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = (const uint16_t*) zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = (const uint16_t*) zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = (const uint16_t*) zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = (const uint16_t*) zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }

    const __m256 vmultiplier = _mm256_cvtph_ps(_mm_set1_epi16((short) *((const uint16_t*) multiplier)));
    multiplier = (const uint16_t*) multiplier + 1;

    size_t c = channels;
    while (c >= 8) {
      const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;
      const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;
      const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;
      const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;
      const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;
      const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;
      const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      i7 += 8;
      const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      i8 += 8;

      const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum018 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vi8), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum01678 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum018, vsum67), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum01678), _MM_FROUND_TO_NEAREST_INT));

      __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vmultiplier), _MM_FROUND_TO_NEAREST_INT));
      vout = _mm256_max_ps(vout, voutput_min);
      vout = _mm256_min_ps(vout, voutput_max);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT));
      o += 8;

      c -= 8;
    }
    if (c != 0) {
      const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));

      const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum018 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vi8), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum01678 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum018, vsum67), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum01678), _MM_FROUND_TO_NEAREST_INT));

      __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vmultiplier), _MM_FROUND_TO_NEAREST_INT));
      vout = _mm256_max_ps(vout, voutput_min);
      vout = _mm256_min_ps(vout, voutput_max);

      __m128i vh = _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh);
        vh = _mm_unpackhi_epi64(vh, vh);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh);
        vh = _mm_srli_epi64(vh, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh, 0);
        o += 1;
      }
    }
    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x40(
    size_t batch,
    const void* input,
    const void* max,
    void* output,
    void* sum,
    const union xnn_f16_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const __m256 vi_max = _mm256_cvtph_ps(_mm_set1_epi16((short) *((const uint16_t*) max)));
  const __m256 vlog2e = _mm256_load_ps(params->avx2_rr1_p2.log2e);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx2_rr1_p2.magic_bias);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx2_rr1_p2.minus_ln2);
  const __m256 vc2 = _mm256_load_ps(params->avx2_rr1_p2.c2);
  const __m256 vc1 = _mm256_load_ps(params->avx2_rr1_p2.c1);
  const __m256 vdenorm_cutoff = _mm256_load_ps(params->avx2_rr1_p2.denorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  __m256 vacc0 = _mm256_setzero_ps();
  for (; batch >= 40 * sizeof(uint16_t); batch -= 40 * sizeof(uint16_t)) {
    const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 32)));
    i += 40;

    const __m256 vx0 = _mm256_sub_ps(vi0, vi_max);
    const __m256 vx1 = _mm256_sub_ps(vi1, vi_max);
    const __m256 vx2 = _mm256_sub_ps(vi2, vi_max);
    const __m256 vx3 = _mm256_sub_ps(vi3, vi_max);
    const __m256 vx4 = _mm256_sub_ps(vi4, vi_max);

    __m256 vn0 = _mm256_fmadd_ps(vx0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vx1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vx2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vx3, vlog2e, vmagic_bias);
    __m256 vn4 = _mm256_fmadd_ps(vx4, vlog2e, vmagic_bias);

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn2), 23));
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn3), 23));
    const __m256 vs4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn4), 23));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vx0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vx1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vx2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vx3);
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2, vx4);

    const __m256 vp0 = _mm256_fmadd_ps(vc2, vt0, vc1);
    const __m256 vp1 = _mm256_fmadd_ps(vc2, vt1, vc1);
    const __m256 vp2 = _mm256_fmadd_ps(vc2, vt2, vc1);
    const __m256 vp3 = _mm256_fmadd_ps(vc2, vt3, vc1);
    const __m256 vp4 = _mm256_fmadd_ps(vc2, vt4, vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vt4 = _mm256_mul_ps(vt4, vs4);

    __m256 vf0 = _mm256_fmadd_ps(vt0, vp0, vs0);
    __m256 vf1 = _mm256_fmadd_ps(vt1, vp1, vs1);
    __m256 vf2 = _mm256_fmadd_ps(vt2, vp2, vs2);
    __m256 vf3 = _mm256_fmadd_ps(vt3, vp3, vs3);
    __m256 vf4 = _mm256_fmadd_ps(vt4, vp4, vs4);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vx0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vx1, vdenorm_cutoff, _CMP_LT_OS), vf1);
    vf2 = _mm256_andnot_ps(_mm256_cmp_ps(vx2, vdenorm_cutoff, _CMP_LT_OS), vf2);
    vf3 = _mm256_andnot_ps(_mm256_cmp_ps(vx3, vdenorm_cutoff, _CMP_LT_OS), vf3);
    vf4 = _mm256_andnot_ps(_mm256_cmp_ps(vx4, vdenorm_cutoff, _CMP_LT_OS), vf4);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vf1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vf2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vf3, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 32), _mm256_cvtps_ph(vf4, _MM_FROUND_TO_NEAREST_INT));
    o += 40;

    vacc0 = _mm256_add_ps(vacc0, vf0);
    vacc0 = _mm256_add_ps(vacc0, vf1);
    vacc0 = _mm256_add_ps(vacc0, vf2);
    vacc0 = _mm256_add_ps(vacc0, vf3);
    vacc0 = _mm256_add_ps(vacc0, vf4);
  }

  __m256 vacc = vacc0;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    const __m256 vx = _mm256_sub_ps(vi, vi_max);

    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);
    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
    o += 8;

    vacc = _mm256_add_ps(vacc, vf);
  }
  __m128 vacc_lo = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
  if (batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));

    const __m256 vi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    const __m256 vx = _mm256_sub_ps(vi, vi_max);

    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);
    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);

    __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
    __m128 vf_lo = _mm256_castps256_ps128(vf);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      vacc_lo = _mm_add_ps(vacc_lo, vf_lo);
      vf_lo = _mm256_extractf128_ps(vf, 1);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      vacc_lo = _mm_blend_ps(_mm_add_ps(vacc_lo, vf_lo), vacc_lo, 0xC);
      vf_lo = _mm_movehl_ps(vf_lo, vf_lo);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
      vacc_lo = _mm_add_ss(vacc_lo, vf_lo);
    }
  }
  vacc_lo = _mm_add_ps(vacc_lo, _mm_movehl_ps(vacc_lo, vacc_lo));
  vacc_lo = _mm_add_ss(vacc_lo, _mm_movehdup_ps(vacc_lo));
  *((uint16_t*) sum) = (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(vacc_lo, _MM_FROUND_TO_NEAREST_INT), 0);
  _mm256_zeroupper();
}

void xnn_f16_velu_ukernel__avx2_rr1_p3_x16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vprescale = _mm256_load_ps(params->avx2_rr1_p3.prescale);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx2_rr1_p3.sat_cutoff);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx2_rr1_p3.magic_bias);
  const __m256 vlog2e = _mm256_load_ps(params->avx2_rr1_p3.log2e);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx2_rr1_p3.minus_ln2);
  const __m256 vc3 = _mm256_load_ps(params->avx2_rr1_p3.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx2_rr1_p3.c2);
  const __m256 vc1 = _mm256_load_ps(params->avx2_rr1_p3.c1);
  const __m256 valpha = _mm256_load_ps(params->avx2_rr1_p3.alpha);
  const __m256 vbeta = _mm256_load_ps(params->avx2_rr1_p3.beta);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    const __m256 vz0 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx0, vprescale));
    const __m256 vz1 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx1, vprescale));

    __m256 vn0 = _mm256_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vz1, vlog2e, vmagic_bias);

    __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vz0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vz1);

    __m256 vp0 = _mm256_fmadd_ps(vc3, vt0, vc2);
    __m256 vp1 = _mm256_fmadd_ps(vc3, vt1, vc2);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc1);
    vt0 = _mm256_mul_ps(vt0, valpha);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc1);
    vt1 = _mm256_mul_ps(vt1, valpha);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vs0 = _mm256_fmsub_ps(vs0, valpha, valpha);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vs1 = _mm256_fmsub_ps(vs1, valpha, valpha);

    const __m256 ve0 = _mm256_fmadd_ps(vp0, vt0, vs0);
    vx0 = _mm256_mul_ps(vx0, vbeta);
    const __m256 ve1 = _mm256_fmadd_ps(vp1, vt1, vs1);
    vx1 = _mm256_mul_ps(vx1, vbeta);

    const __m256 vy0 = _mm256_blendv_ps(vx0, ve0, vx0);
    const __m256 vy1 = _mm256_blendv_ps(vx1, ve1, vx1);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc3, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);
    vt = _mm256_mul_ps(vt, valpha);
    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_fmsub_ps(vs, valpha, valpha);
    const __m256 ve = _mm256_fmadd_ps(vp, vt, vs);
    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc3, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);
    vt = _mm256_mul_ps(vt, valpha);
    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_fmsub_ps(vs, valpha, valpha);
    const __m256 ve = _mm256_fmadd_ps(vp, vt, vs);
    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vsign_mask = _mm256_load_ps(params->avx2_rr1_p2.sign_mask);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx2_rr1_p2.magic_bias);
  const __m256 vlog2e = _mm256_load_ps(params->avx2_rr1_p2.log2e);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx2_rr1_p2.minus_ln2);
  const __m256 vc2 = _mm256_load_ps(params->avx2_rr1_p2.c2);
  const __m256 vc1 = _mm256_load_ps(params->avx2_rr1_p2.c1);
  const __m256 vone = _mm256_load_ps(params->avx2_rr1_p2.one);
  const __m256 vdenorm_cutoff = _mm256_load_ps(params->avx2_rr1_p2.denorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    const __m256 vx2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    const __m256 vx3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    const __m256 vz0 = _mm256_or_ps(vx0, vsign_mask);
    const __m256 vz1 = _mm256_or_ps(vx1, vsign_mask);
    const __m256 vz2 = _mm256_or_ps(vx2, vsign_mask);
    const __m256 vz3 = _mm256_or_ps(vx3, vsign_mask);

    __m256 vn0 = _mm256_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vz3, vlog2e, vmagic_bias);

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn2), 23));
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn3), 23));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vz0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vz1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vz2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vz3);

    const __m256 vp0 = _mm256_fmadd_ps(vc2, vt0, vc1);
    const __m256 vp1 = _mm256_fmadd_ps(vc2, vt1, vc1);
    const __m256 vp2 = _mm256_fmadd_ps(vc2, vt2, vc1);
    const __m256 vp3 = _mm256_fmadd_ps(vc2, vt3, vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vt3 = _mm256_mul_ps(vt3, vs3);

    const __m256 ve0 = _mm256_fmadd_ps(vt0, vp0, vs0);
    const __m256 ve1 = _mm256_fmadd_ps(vt1, vp1, vs1);
    const __m256 ve2 = _mm256_fmadd_ps(vt2, vp2, vs2);
    const __m256 ve3 = _mm256_fmadd_ps(vt3, vp3, vs3);

    const __m256 vd0 = _mm256_add_ps(ve0, vone);
    const __m256 vd1 = _mm256_add_ps(ve1, vone);
    const __m256 vd2 = _mm256_add_ps(ve2, vone);
    const __m256 vd3 = _mm256_add_ps(ve3, vone);

    const __m256 vr0 = _mm256_rcp_ps(vd0);
    const __m256 vr1 = _mm256_rcp_ps(vd1);
    const __m256 vr2 = _mm256_rcp_ps(vd2);
    const __m256 vr3 = _mm256_rcp_ps(vd3);

    __m256 vf0 = _mm256_mul_ps(ve0, vr0);
    __m256 vf1 = _mm256_mul_ps(ve1, vr1);
    __m256 vf2 = _mm256_mul_ps(ve2, vr2);
    __m256 vf3 = _mm256_mul_ps(ve3, vr3);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vz0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vz1, vdenorm_cutoff, _CMP_LT_OS), vf1);
    vf2 = _mm256_andnot_ps(_mm256_cmp_ps(vz2, vdenorm_cutoff, _CMP_LT_OS), vf2);
    vf3 = _mm256_andnot_ps(_mm256_cmp_ps(vz3, vdenorm_cutoff, _CMP_LT_OS), vf3);

    vf0 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf0), vf0, vx0);
    vf1 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf1), vf1, vx1);
    vf2 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf2), vf2, vx2);
    vf3 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf3), vf3, vx3);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vf1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vf2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vf3, _MM_FROUND_TO_NEAREST_INT));
    o += 32;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);
    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_fmadd_ps(vt, vp, vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    const __m256 vr = _mm256_rcp_ps(vd);
    __m256 vf = _mm256_mul_ps(ve, vr);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);
    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_fmadd_ps(vt, vp, vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    const __m256 vr = _mm256_rcp_ps(vd);
    __m256 vf = _mm256_mul_ps(ve, vr);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f32_qs8_vcvt_ukernel__avx2_x64(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vscale = _mm256_load_ps(params->avx2.scale);
  const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->avx2.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  const __m256i vshuffle_mask = _mm256_load_si256((const __m256i*) params->avx2.shuffle_mask);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->avx2.output_min);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    __m256 vx01 = _mm256_loadu_ps(input);
    __m256 vx23 = _mm256_loadu_ps(input + 8);
    __m256 vx45 = _mm256_loadu_ps(input + 16);
    __m256 vx67 = _mm256_loadu_ps(input + 24);
    __m256 vx89 = _mm256_loadu_ps(input + 32);
    __m256 vxAB = _mm256_loadu_ps(input + 40);
    __m256 vxCD = _mm256_loadu_ps(input + 48);
    __m256 vxEF = _mm256_loadu_ps(input + 56);
    input += 64;

    vx01 = _mm256_mul_ps(vx01, vscale);
    vx23 = _mm256_mul_ps(vx23, vscale);
    vx45 = _mm256_mul_ps(vx45, vscale);
    vx67 = _mm256_mul_ps(vx67, vscale);
    vx89 = _mm256_mul_ps(vx89, vscale);
    vxAB = _mm256_mul_ps(vxAB, vscale);
    vxCD = _mm256_mul_ps(vxCD, vscale);
    vxEF = _mm256_mul_ps(vxEF, vscale);

    vx01 = _mm256_min_ps(vx01, voutput_max_less_zero_point);
    vx23 = _mm256_min_ps(vx23, voutput_max_less_zero_point);
    vx45 = _mm256_min_ps(vx45, voutput_max_less_zero_point);
    vx67 = _mm256_min_ps(vx67, voutput_max_less_zero_point);
    vx89 = _mm256_min_ps(vx89, voutput_max_less_zero_point);
    vxAB = _mm256_min_ps(vxAB, voutput_max_less_zero_point);
    vxCD = _mm256_min_ps(vxCD, voutput_max_less_zero_point);
    vxEF = _mm256_min_ps(vxEF, voutput_max_less_zero_point);

    const __m256i vacc01 = _mm256_cvtps_epi32(vx01);
    const __m256i vacc23 = _mm256_cvtps_epi32(vx23);
    const __m256i vacc45 = _mm256_cvtps_epi32(vx45);
    const __m256i vacc67 = _mm256_cvtps_epi32(vx67);
    const __m256i vacc89 = _mm256_cvtps_epi32(vx89);
    const __m256i vaccAB = _mm256_cvtps_epi32(vxAB);
    const __m256i vaccCD = _mm256_cvtps_epi32(vxCD);
    const __m256i vaccEF = _mm256_cvtps_epi32(vxEF);

    __m256i vacc0213 = _mm256_packs_epi32(vacc01, vacc23);
    __m256i vacc4657 = _mm256_packs_epi32(vacc45, vacc67);
    __m256i vacc8A9B = _mm256_packs_epi32(vacc89, vaccAB);
    __m256i vaccCEDF = _mm256_packs_epi32(vaccCD, vaccEF);

    vacc0213 = _mm256_adds_epi16(vacc0213, voutput_zero_point);
    vacc4657 = _mm256_adds_epi16(vacc4657, voutput_zero_point);
    vacc8A9B = _mm256_adds_epi16(vacc8A9B, voutput_zero_point);
    vaccCEDF = _mm256_adds_epi16(vaccCEDF, voutput_zero_point);

    const __m256i vy02461357 = _mm256_packs_epi16(vacc0213, vacc4657);
    const __m256i vy8ACE9BDF = _mm256_packs_epi16(vacc8A9B, vaccCEDF);

    __m256i vy01234567 = _mm256_permutevar8x32_epi32(vy02461357, vshuffle_mask);
    __m256i vy89ABCDEF = _mm256_permutevar8x32_epi32(vy8ACE9BDF, vshuffle_mask);

    vy01234567 = _mm256_max_epi8(vy01234567, voutput_min);
    vy89ABCDEF = _mm256_max_epi8(vy89ABCDEF, voutput_min);

    _mm256_storeu_si256((__m256i*) output, vy01234567);
    _mm256_storeu_si256((__m256i*) (output + 32), vy89ABCDEF);
    output += 64;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);
    input += 8;

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, _mm256_castsi256_si128(voutput_zero_point));
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, _mm256_castsi256_si128(voutput_min));

    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx2.mask_table[7] - batch));

    __m256 vx = _mm256_maskload_ps(input, vmask);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, _mm256_castsi256_si128(voutput_zero_point));
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, _mm256_castsi256_si128(voutput_min));

    if (batch & (4 * sizeof(float))) {
      _mm_storeu_si32(output, vy);
      output += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storeu_si16(output, vy);
      output += 2;
      vy = _mm_srli_epi32(vy, 16);
    }
    if (batch & (1 * sizeof(float))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_f32_qu8_vcvt_ukernel__avx2_x64(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vscale = _mm256_load_ps(params->avx2.scale);
  const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->avx2.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  const __m256i vshuffle_mask = _mm256_load_si256((const __m256i*) params->avx2.shuffle_mask);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->avx2.output_min);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    __m256 vx01 = _mm256_loadu_ps(input);
    __m256 vx23 = _mm256_loadu_ps(input + 8);
    __m256 vx45 = _mm256_loadu_ps(input + 16);
    __m256 vx67 = _mm256_loadu_ps(input + 24);
    __m256 vx89 = _mm256_loadu_ps(input + 32);
    __m256 vxAB = _mm256_loadu_ps(input + 40);
    __m256 vxCD = _mm256_loadu_ps(input + 48);
    __m256 vxEF = _mm256_loadu_ps(input + 56);
    input += 64;

    vx01 = _mm256_mul_ps(vx01, vscale);
    vx23 = _mm256_mul_ps(vx23, vscale);
    vx45 = _mm256_mul_ps(vx45, vscale);
    vx67 = _mm256_mul_ps(vx67, vscale);
    vx89 = _mm256_mul_ps(vx89, vscale);
    vxAB = _mm256_mul_ps(vxAB, vscale);
    vxCD = _mm256_mul_ps(vxCD, vscale);
    vxEF = _mm256_mul_ps(vxEF, vscale);

    vx01 = _mm256_min_ps(vx01, voutput_max_less_zero_point);
    vx23 = _mm256_min_ps(vx23, voutput_max_less_zero_point);
    vx45 = _mm256_min_ps(vx45, voutput_max_less_zero_point);
    vx67 = _mm256_min_ps(vx67, voutput_max_less_zero_point);
    vx89 = _mm256_min_ps(vx89, voutput_max_less_zero_point);
    vxAB = _mm256_min_ps(vxAB, voutput_max_less_zero_point);
    vxCD = _mm256_min_ps(vxCD, voutput_max_less_zero_point);
    vxEF = _mm256_min_ps(vxEF, voutput_max_less_zero_point);

    const __m256i vacc01 = _mm256_cvtps_epi32(vx01);
    const __m256i vacc23 = _mm256_cvtps_epi32(vx23);
    const __m256i vacc45 = _mm256_cvtps_epi32(vx45);
    const __m256i vacc67 = _mm256_cvtps_epi32(vx67);
    const __m256i vacc89 = _mm256_cvtps_epi32(vx89);
    const __m256i vaccAB = _mm256_cvtps_epi32(vxAB);
    const __m256i vaccCD = _mm256_cvtps_epi32(vxCD);
    const __m256i vaccEF = _mm256_cvtps_epi32(vxEF);

    __m256i vacc0213 = _mm256_packs_epi32(vacc01, vacc23);
    __m256i vacc4657 = _mm256_packs_epi32(vacc45, vacc67);
    __m256i vacc8A9B = _mm256_packs_epi32(vacc89, vaccAB);
    __m256i vaccCEDF = _mm256_packs_epi32(vaccCD, vaccEF);

    vacc0213 = _mm256_adds_epi16(vacc0213, voutput_zero_point);
    vacc4657 = _mm256_adds_epi16(vacc4657, voutput_zero_point);
    vacc8A9B = _mm256_adds_epi16(vacc8A9B, voutput_zero_point);
    vaccCEDF = _mm256_adds_epi16(vaccCEDF, voutput_zero_point);

    const __m256i vy02461357 = _mm256_packus_epi16(vacc0213, vacc4657);
    const __m256i vy8ACE9BDF = _mm256_packus_epi16(vacc8A9B, vaccCEDF);

    __m256i vy01234567 = _mm256_permutevar8x32_epi32(vy02461357, vshuffle_mask);
    __m256i vy89ABCDEF = _mm256_permutevar8x32_epi32(vy8ACE9BDF, vshuffle_mask);

    vy01234567 = _mm256_max_epu8(vy01234567, voutput_min);
    vy89ABCDEF = _mm256_max_epu8(vy89ABCDEF, voutput_min);

    _mm256_storeu_si256((__m256i*) output, vy01234567);
    _mm256_storeu_si256((__m256i*) (output + 32), vy89ABCDEF);
    output += 64;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);
    input += 8;

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, _mm256_castsi256_si128(voutput_zero_point));
    vy = _mm_packus_epi16(vy, vy);
    vy = _mm_max_epu8(vy, _mm256_castsi256_si128(voutput_min));

    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx2.mask_table[7] - batch));

    __m256 vx = _mm256_maskload_ps(input, vmask);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, _mm256_castsi256_si128(voutput_zero_point));
    vy = _mm_packus_epi16(vy, vy);
    vy = _mm_max_epu8(vy, _mm256_castsi256_si128(voutput_min));

    if (batch & (4 * sizeof(float))) {
      _mm_storeu_si32(output, vy);
      output += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storeu_si16(output, vy);
      output += 2;
      vy = _mm_srli_epi32(vy, 16);
    }
    if (batch & (1 * sizeof(float))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vprescale = _mm256_load_ps(params->avx2_rr1_lut4_p4.prescale);
  const __m256 valpha = _mm256_load_ps(params->avx2_rr1_lut4_p4.alpha);
  const __m256 vbeta = _mm256_load_ps(params->avx2_rr1_lut4_p4.beta);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx2_rr1_lut4_p4.sat_cutoff);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx2_rr1_lut4_p4.magic_bias);
  const __m256 vlog2e = _mm256_load_ps(params->avx2_rr1_lut4_p4.log2e);
  const __m256 vtable = _mm256_load_ps(params->avx2_rr1_lut4_p4.table);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx2_rr1_lut4_p4.minus_ln2);
  const __m256 vc4 = _mm256_load_ps(params->avx2_rr1_lut4_p4.c4);
  const __m256 vc3 = _mm256_load_ps(params->avx2_rr1_lut4_p4.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx2_rr1_lut4_p4.c2);

  for (; batch >= 56 * sizeof(float); batch -= 56 * sizeof(float)) {
    __m256 vx0 = _mm256_loadu_ps(input);
    __m256 vx1 = _mm256_loadu_ps(input + 8);
    __m256 vx2 = _mm256_loadu_ps(input + 16);
    __m256 vx3 = _mm256_loadu_ps(input + 24);
    __m256 vx4 = _mm256_loadu_ps(input + 32);
    __m256 vx5 = _mm256_loadu_ps(input + 40);
    __m256 vx6 = _mm256_loadu_ps(input + 48);
    input += 56;

    const __m256 vz0 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx0, vprescale));
    const __m256 vz1 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx1, vprescale));
    const __m256 vz2 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx2, vprescale));
    const __m256 vz3 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx3, vprescale));
    const __m256 vz4 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx4, vprescale));
    const __m256 vz5 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx5, vprescale));
    const __m256 vz6 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx6, vprescale));

    __m256 vn0 = _mm256_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vz3, vlog2e, vmagic_bias);
    __m256 vn4 = _mm256_fmadd_ps(vz4, vlog2e, vmagic_bias);
    __m256 vn5 = _mm256_fmadd_ps(vz5, vlog2e, vmagic_bias);
    __m256 vn6 = _mm256_fmadd_ps(vz6, vlog2e, vmagic_bias);

    const __m256i ven0 = _mm256_slli_epi32(_mm256_castps_si256(vn0), 21);
    const __m256i vl0 = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn0)));
    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    const __m256i ven1 = _mm256_slli_epi32(_mm256_castps_si256(vn1), 21);
    const __m256i vl1 = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn1)));
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    const __m256i ven2 = _mm256_slli_epi32(_mm256_castps_si256(vn2), 21);
    const __m256i vl2 = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn2)));
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    const __m256i ven3 = _mm256_slli_epi32(_mm256_castps_si256(vn3), 21);
    const __m256i vl3 = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn3)));
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    const __m256i ven4 = _mm256_slli_epi32(_mm256_castps_si256(vn4), 21);
    const __m256i vl4 = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn4)));
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    const __m256i ven5 = _mm256_slli_epi32(_mm256_castps_si256(vn5), 21);
    const __m256i vl5 = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn5)));
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);
    const __m256i ven6 = _mm256_slli_epi32(_mm256_castps_si256(vn6), 21);
    const __m256i vl6 = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn6)));
    vn6 = _mm256_sub_ps(vn6, vmagic_bias);

    __m256 vs0 = _mm256_castsi256_ps(_mm256_add_epi32(vl0, ven0));
    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vz0);
    __m256 vs1 = _mm256_castsi256_ps(_mm256_add_epi32(vl1, ven1));
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vz1);
    __m256 vs2 = _mm256_castsi256_ps(_mm256_add_epi32(vl2, ven2));
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vz2);
    __m256 vs3 = _mm256_castsi256_ps(_mm256_add_epi32(vl3, ven3));
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vz3);
    __m256 vs4 = _mm256_castsi256_ps(_mm256_add_epi32(vl4, ven4));
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2, vz4);
    __m256 vs5 = _mm256_castsi256_ps(_mm256_add_epi32(vl5, ven5));
    __m256 vt5 = _mm256_fmadd_ps(vn5, vminus_ln2, vz5);
    __m256 vs6 = _mm256_castsi256_ps(_mm256_add_epi32(vl6, ven6));
    __m256 vt6 = _mm256_fmadd_ps(vn6, vminus_ln2, vz6);

    __m256 vp0 = _mm256_fmadd_ps(vc4, vt0, vc3);
    __m256 vp1 = _mm256_fmadd_ps(vc4, vt1, vc3);
    __m256 vp2 = _mm256_fmadd_ps(vc4, vt2, vc3);
    __m256 vp3 = _mm256_fmadd_ps(vc4, vt3, vc3);
    __m256 vp4 = _mm256_fmadd_ps(vc4, vt4, vc3);
    __m256 vp5 = _mm256_fmadd_ps(vc4, vt5, vc3);
    __m256 vp6 = _mm256_fmadd_ps(vc4, vt6, vc3);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc2);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc2);

    vp0 = _mm256_mul_ps(vp0, vt0);
    vt0 = _mm256_mul_ps(vt0, vs0);
    vp1 = _mm256_mul_ps(vp1, vt1);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vp2 = _mm256_mul_ps(vp2, vt2);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vp3 = _mm256_mul_ps(vp3, vt3);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vp4 = _mm256_mul_ps(vp4, vt4);
    vt4 = _mm256_mul_ps(vt4, vs4);
    vp5 = _mm256_mul_ps(vp5, vt5);
    vt5 = _mm256_mul_ps(vt5, vs5);
    vp6 = _mm256_mul_ps(vp6, vt6);
    vt6 = _mm256_mul_ps(vt6, vs6);

    vs0 = _mm256_fmsub_ps(vs0, valpha, valpha);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vt0);
    vs1 = _mm256_fmsub_ps(vs1, valpha, valpha);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vt1);
    vs2 = _mm256_fmsub_ps(vs2, valpha, valpha);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vt2);
    vs3 = _mm256_fmsub_ps(vs3, valpha, valpha);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vt3);
    vs4 = _mm256_fmsub_ps(vs4, valpha, valpha);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vt4);
    vs5 = _mm256_fmsub_ps(vs5, valpha, valpha);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vt5);
    vs6 = _mm256_fmsub_ps(vs6, valpha, valpha);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vt6);

    const __m256 ve0 = _mm256_fmadd_ps(vp0, valpha, vs0);
    vx0 = _mm256_mul_ps(vx0, vbeta);
    const __m256 ve1 = _mm256_fmadd_ps(vp1, valpha, vs1);
    vx1 = _mm256_mul_ps(vx1, vbeta);
    const __m256 ve2 = _mm256_fmadd_ps(vp2, valpha, vs2);
    vx2 = _mm256_mul_ps(vx2, vbeta);
    const __m256 ve3 = _mm256_fmadd_ps(vp3, valpha, vs3);
    vx3 = _mm256_mul_ps(vx3, vbeta);
    const __m256 ve4 = _mm256_fmadd_ps(vp4, valpha, vs4);
    vx4 = _mm256_mul_ps(vx4, vbeta);
    const __m256 ve5 = _mm256_fmadd_ps(vp5, valpha, vs5);
    vx5 = _mm256_mul_ps(vx5, vbeta);
    const __m256 ve6 = _mm256_fmadd_ps(vp6, valpha, vs6);
    vx6 = _mm256_mul_ps(vx6, vbeta);

    const __m256 vy0 = _mm256_blendv_ps(vx0, ve0, vx0);
    const __m256 vy1 = _mm256_blendv_ps(vx1, ve1, vx1);
    const __m256 vy2 = _mm256_blendv_ps(vx2, ve2, vx2);
    const __m256 vy3 = _mm256_blendv_ps(vx3, ve3, vx3);
    const __m256 vy4 = _mm256_blendv_ps(vx4, ve4, vx4);
    const __m256 vy5 = _mm256_blendv_ps(vx5, ve5, vx5);
    const __m256 vy6 = _mm256_blendv_ps(vx6, ve6, vx6);

    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    _mm256_storeu_ps(output + 16, vy2);
    _mm256_storeu_ps(output + 24, vy3);
    _mm256_storeu_ps(output + 32, vy4);
    _mm256_storeu_ps(output + 40, vy5);
    _mm256_storeu_ps(output + 48, vy6);
    output += 56;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256i ven = _mm256_slli_epi32(_mm256_castps_si256(vn), 21);
    const __m256i vl = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn)));
    __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ven));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc4, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_fmsub_ps(vs, valpha, valpha);
    vp = _mm256_fmadd_ps(vp, vt, vt);
    const __m256 ve = _mm256_fmadd_ps(vp, valpha, vs);

    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx2_rr1_lut4_p4.mask_table[7] - batch));

    __m256 vx = _mm256_maskload_ps(input, vmask);

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256i ven = _mm256_slli_epi32(_mm256_castps_si256(vn), 21);
    const __m256i vl = _mm256_castps_si256(_mm256_permutevar_ps(vtable, _mm256_castps_si256(vn)));
    __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ven));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc4, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_fmsub_ps(vs, valpha, valpha);
    vp = _mm256_fmadd_ps(vp, vt, vt);
    const __m256 ve = _mm256_fmadd_ps(vp, valpha, vs);

    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}

void xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x40(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vsign_mask = _mm256_load_ps(params->avx2_rr1_p5.sign_mask);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx2_rr1_p5.magic_bias);
  const __m256 vlog2e = _mm256_load_ps(params->avx2_rr1_p5.log2e);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx2_rr1_p5.minus_ln2);
  const __m256 vc5 = _mm256_load_ps(params->avx2_rr1_p5.c5);
  const __m256 vc4 = _mm256_load_ps(params->avx2_rr1_p5.c4);
  const __m256 vc3 = _mm256_load_ps(params->avx2_rr1_p5.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx2_rr1_p5.c2);
  const __m256 vc1 = _mm256_load_ps(params->avx2_rr1_p5.c1);
  const __m256 vone = _mm256_load_ps(params->avx2_rr1_p5.one);
  const __m256 vdenorm_cutoff = _mm256_load_ps(params->avx2_rr1_p5.denorm_cutoff);

  for (; batch >= 40 * sizeof(float); batch -= 40 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    const __m256 vx2 = _mm256_loadu_ps(input + 16);
    const __m256 vx3 = _mm256_loadu_ps(input + 24);
    const __m256 vx4 = _mm256_loadu_ps(input + 32);
    input += 40;

    const __m256 vz0 = _mm256_or_ps(vx0, vsign_mask);
    const __m256 vz1 = _mm256_or_ps(vx1, vsign_mask);
    const __m256 vz2 = _mm256_or_ps(vx2, vsign_mask);
    const __m256 vz3 = _mm256_or_ps(vx3, vsign_mask);
    const __m256 vz4 = _mm256_or_ps(vx4, vsign_mask);

    __m256 vn0 = _mm256_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vz3, vlog2e, vmagic_bias);
    __m256 vn4 = _mm256_fmadd_ps(vz4, vlog2e, vmagic_bias);

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn2), 23));
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn3), 23));
    const __m256 vs4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn4), 23));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vz0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vz1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vz2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vz3);
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2, vz4);

    __m256 vp0 = _mm256_fmadd_ps(vc5, vt0, vc4);
    __m256 vp1 = _mm256_fmadd_ps(vc5, vt1, vc4);
    __m256 vp2 = _mm256_fmadd_ps(vc5, vt2, vc4);
    __m256 vp3 = _mm256_fmadd_ps(vc5, vt3, vc4);
    __m256 vp4 = _mm256_fmadd_ps(vc5, vt4, vc4);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc3);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc2);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc1);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc1);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc1);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc1);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vt4 = _mm256_mul_ps(vt4, vs4);

    const __m256 ve0 = _mm256_fmadd_ps(vt0, vp0, vs0);
    const __m256 ve1 = _mm256_fmadd_ps(vt1, vp1, vs1);
    const __m256 ve2 = _mm256_fmadd_ps(vt2, vp2, vs2);
    const __m256 ve3 = _mm256_fmadd_ps(vt3, vp3, vs3);
    const __m256 ve4 = _mm256_fmadd_ps(vt4, vp4, vs4);

    const __m256 vd0 = _mm256_add_ps(ve0, vone);
    const __m256 vd1 = _mm256_add_ps(ve1, vone);
    const __m256 vd2 = _mm256_add_ps(ve2, vone);
    const __m256 vd3 = _mm256_add_ps(ve3, vone);
    const __m256 vd4 = _mm256_add_ps(ve4, vone);

    __m256 vf0 = _mm256_div_ps(ve0, vd0);
    __m256 vf1 = _mm256_div_ps(ve1, vd1);
    __m256 vf2 = _mm256_div_ps(ve2, vd2);
    __m256 vf3 = _mm256_div_ps(ve3, vd3);
    __m256 vf4 = _mm256_div_ps(ve4, vd4);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vz0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vz1, vdenorm_cutoff, _CMP_LT_OS), vf1);
    vf2 = _mm256_andnot_ps(_mm256_cmp_ps(vz2, vdenorm_cutoff, _CMP_LT_OS), vf2);
    vf3 = _mm256_andnot_ps(_mm256_cmp_ps(vz3, vdenorm_cutoff, _CMP_LT_OS), vf3);
    vf4 = _mm256_andnot_ps(_mm256_cmp_ps(vz4, vdenorm_cutoff, _CMP_LT_OS), vf4);

    vf0 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf0), vf0, vx0);
    vf1 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf1), vf1, vx1);
    vf2 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf2), vf2, vx2);
    vf3 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf3), vf3, vx3);
    vf4 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf4), vf4, vx4);

    _mm256_storeu_ps(output, vf0);
    _mm256_storeu_ps(output + 8, vf1);
    _mm256_storeu_ps(output + 16, vf2);
    _mm256_storeu_ps(output + 24, vf3);
    _mm256_storeu_ps(output + 32, vf4);
    output += 40;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_fmadd_ps(vt, vp, vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    __m256 vf = _mm256_div_ps(ve, vd);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm256_storeu_ps(output, vf);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx2_rr1_p5.mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_fmadd_ps(vt, vp, vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    __m256 vf = _mm256_div_ps(ve, vd);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    __m128 vf_lo = _mm256_castps256_ps128(vf);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vf_lo);
      vf_lo = _mm256_extractf128_ps(vf, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vf_lo);
      vf_lo = _mm_movehl_ps(vf_lo, vf_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vf_lo);
    }
  }
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));


      const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m256i vi0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      i0 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      const __m256i vi1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      i1 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m256i vi2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      i2 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      const __m256i vi3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      const __m256i vk3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t))));
      i3 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));

      const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m256i vi4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
      const __m256i vk4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t))));
      i4 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));

      const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      const __m256i vi5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
      const __m256i vk5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t))));
      i5 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi5x89ABCDEF, vk5x89ABCDEF));

      const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m256i vi6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
      const __m256i vk6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t))));
      i6 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi6x89ABCDEF, vk6x89ABCDEF));

      const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      const __m256i vi7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
      const __m256i vk7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t))));
      i7 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi7x89ABCDEF, vk7x89ABCDEF));

      const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m256i vi8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 8)));
      const __m256i vk8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t))));
      i8 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi8x89ABCDEF, vk8x89ABCDEF));

      const __m256i vi9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i9));
      const __m256i vk9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      const __m256i vi9x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i9 + 8)));
      const __m256i vk9x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(int8_t))));
      i9 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi9x89ABCDEF, vk9x89ABCDEF));

      const __m256i vi10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i10));
      const __m256i vk10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m256i vi10x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i10 + 8)));
      const __m256i vk10x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(int8_t))));
      i10 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi10x89ABCDEF, vk10x89ABCDEF));

      const __m256i vi11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i11));
      const __m256i vk11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      const __m256i vi11x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i11 + 8)));
      const __m256i vk11x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(int8_t))));
      i11 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi11x89ABCDEF, vk11x89ABCDEF));

      const __m256i vi12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i12));
      const __m256i vk12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m256i vi12x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i12 + 8)));
      const __m256i vk12x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(int8_t))));
      i12 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi12x89ABCDEF, vk12x89ABCDEF));

      const __m256i vi13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i13));
      const __m256i vk13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      const __m256i vi13x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i13 + 8)));
      const __m256i vk13x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(int8_t))));
      i13 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi13x89ABCDEF, vk13x89ABCDEF));

      const __m256i vi14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i14));
      const __m256i vk14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m256i vi14x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i14 + 8)));
      const __m256i vk14x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(int8_t))));
      i14 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi14x89ABCDEF, vk14x89ABCDEF));

      const __m256i vi15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i15));
      const __m256i vk15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      const __m256i vi15x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i15 + 8)));
      const __m256i vk15x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(int8_t))));
      i15 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi15x89ABCDEF, vk15x89ABCDEF));

      const __m256i vi16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i16));
      const __m256i vk16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m256i vi16x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i16 + 8)));
      const __m256i vk16x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(int8_t))));
      i16 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi16x89ABCDEF, vk16x89ABCDEF));

      const __m256i vi17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i17));
      const __m256i vk17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      const __m256i vi17x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i17 + 8)));
      const __m256i vk17x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(int8_t))));
      i17 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi17x89ABCDEF, vk17x89ABCDEF));

      const __m256i vi18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i18));
      const __m256i vk18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m256i vi18x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i18 + 8)));
      const __m256i vk18x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(int8_t))));
      i18 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi18x89ABCDEF, vk18x89ABCDEF));

      const __m256i vi19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i19));
      const __m256i vk19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      const __m256i vi19x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i19 + 8)));
      const __m256i vk19x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(int8_t))));
      i19 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi19x89ABCDEF, vk19x89ABCDEF));

      const __m256i vi20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i20));
      const __m256i vk20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m256i vi20x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i20 + 8)));
      const __m256i vk20x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(int8_t))));
      i20 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi20x89ABCDEF, vk20x89ABCDEF));

      const __m256i vi21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i21));
      const __m256i vk21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      const __m256i vi21x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i21 + 8)));
      const __m256i vk21x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(int8_t))));
      i21 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi21x89ABCDEF, vk21x89ABCDEF));

      const __m256i vi22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i22));
      const __m256i vk22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m256i vi22x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i22 + 8)));
      const __m256i vk22x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(int8_t))));
      i22 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi22x89ABCDEF, vk22x89ABCDEF));

      const __m256i vi23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i23));
      const __m256i vk23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      const __m256i vi23x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i23 + 8)));
      const __m256i vk23x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(int8_t))));
      i23 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi23x89ABCDEF, vk23x89ABCDEF));

      const __m256i vi24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i24));
      const __m256i vk24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m256i vi24x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i24 + 8)));
      const __m256i vk24x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(int8_t))));
      i24 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi24x89ABCDEF, vk24x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w);
      const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
      w = (const void*) ((const float*) w + 16);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale89ABCDEF);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) k));
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 16)));
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32)));
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 48)));
        i3 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 64)));
        i4 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 80)));
        i5 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 96)));
        i6 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 112)));
        i7 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 128)));
        i8 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        const __m256i vi9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i9));
        const __m256i vk9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 144)));
        i9 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));

        const __m256i vi10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i10));
        const __m256i vk10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 160)));
        i10 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));

        const __m256i vi11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i11));
        const __m256i vk11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 176)));
        i11 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));

        const __m256i vi12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i12));
        const __m256i vk12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 192)));
        i12 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));

        const __m256i vi13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i13));
        const __m256i vk13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 208)));
        i13 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));

        const __m256i vi14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i14));
        const __m256i vk14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 224)));
        i14 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));

        const __m256i vi15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i15));
        const __m256i vk15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 240)));
        i15 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));

        const __m256i vi16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i16));
        const __m256i vk16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 256)));
        i16 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));

        const __m256i vi17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i17));
        const __m256i vk17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 272)));
        i17 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));

        const __m256i vi18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i18));
        const __m256i vk18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 288)));
        i18 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));

        const __m256i vi19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i19));
        const __m256i vk19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 304)));
        i19 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));

        const __m256i vi20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i20));
        const __m256i vk20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 320)));
        i20 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));

        const __m256i vi21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i21));
        const __m256i vk21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 336)));
        i21 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));

        const __m256i vi22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i22));
        const __m256i vk22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 352)));
        i22 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));

        const __m256i vi23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i23));
        const __m256i vk23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 368)));
        i23 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));

        const __m256i vi24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i24));
        const __m256i vk24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 384)));
        i24 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        const __m256 vscale01234567 = _mm256_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t)));
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_3p16c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));


      const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m256i vi0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      i0 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      const __m256i vi1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      i1 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m256i vi2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      i2 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w);
      const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
      w = (const void*) ((const float*) w + 16);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale89ABCDEF);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) k));
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 16)));
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32)));
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        const __m256 vscale01234567 = _mm256_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));


      const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m256i vi0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      i0 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      const __m256i vi1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      i1 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m256i vi2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      i2 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      const __m256i vi3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      const __m256i vk3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t))));
      i3 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));

      const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m256i vi4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
      const __m256i vk4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t))));
      i4 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));

      const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      const __m256i vi5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
      const __m256i vk5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t))));
      i5 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi5x89ABCDEF, vk5x89ABCDEF));

      const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m256i vi6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
      const __m256i vk6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t))));
      i6 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi6x89ABCDEF, vk6x89ABCDEF));

      const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      const __m256i vi7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
      const __m256i vk7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t))));
      i7 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi7x89ABCDEF, vk7x89ABCDEF));

      const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m256i vi8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 8)));
      const __m256i vk8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t))));
      i8 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi8x89ABCDEF, vk8x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w);
      const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
      w = (const void*) ((const float*) w + 16);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale89ABCDEF);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) k));
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 16)));
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32)));
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 48)));
        i3 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 64)));
        i4 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 80)));
        i5 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 96)));
        i6 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 112)));
        i7 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 128)));
        i8 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        const __m256 vscale01234567 = _mm256_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale01234567);
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    w = (const int32_t*) w + 8;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m256i vxb01 = _mm256_cvtepi8_epi16(vb01);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m256i vxb23 = _mm256_cvtepi8_epi16(vb23);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m256i vxb45 = _mm256_cvtepi8_epi16(vb45);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
      const __m256i vxb67 = _mm256_cvtepi8_epi16(vb67);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

      w = (const void*) ((const int8_t*) w + 64);
      k += 8 * sizeof(int8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 vscale01234567 = _mm256_load_ps(w);
    w = (const void*) ((const float*) w + 8);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale01234567);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc00x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc0x01234567), voutput_zero_point);

    vacc00x01234567 = _mm256_permute4x64_epi64(vacc00x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc00x01234567, vacc00x01234567);

    vout = _mm256_max_epi8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);

        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_3x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    __m256i vacc2x01 = vacc0x01;
    __m256i vacc2x23 = vacc0x23;
    __m256i vacc2x45 = vacc0x45;
    __m256i vacc2x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      const __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
      const __m256i vxa2 = _mm256_cvtepi8_epi16(va2);
      a2 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m256i vxb01 = _mm256_cvtepi8_epi16(vb01);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m256i vxb23 = _mm256_cvtepi8_epi16(vb23);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
      const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m256i vxb45 = _mm256_cvtepi8_epi16(vb45);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
      const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
      const __m256i vxb67 = _mm256_cvtepi8_epi16(vb67);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
      vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

      w = (const void*) ((const int8_t*) w + 64);
      k += 8 * sizeof(int8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vscaled1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vscaled2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);

    const __m256 vscale01234567 = _mm256_load_ps(w);
    w = (const void*) ((const float*) w + 8);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale01234567);
    vscaled1x01234567 = _mm256_mul_ps(vscaled1x01234567, vscale01234567);
    vscaled2x01234567 = _mm256_mul_ps(vscaled2x01234567, vscale01234567);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);
    vscaled1x01234567 = _mm256_min_ps(vscaled1x01234567, voutput_max_less_zero_point);
    vscaled2x01234567 = _mm256_min_ps(vscaled2x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vscaled1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vscaled2x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);
    __m256i vacc22x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2x01234567, vacc2x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc22x01234567 = _mm256_permute4x64_epi64(vacc22x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc01x01234567, vacc22x01234567);

    vout = _mm256_max_epi8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);
      _mm_storel_epi64((__m128i*) c1, vout_hi);
      _mm_storeh_pi((__m64*) c2, _mm_castsi128_ps(vout_lo));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);
        _mm_storeu_si32(c1, vout_hi);
        unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout_lo, 2));

        c0 += 4;
        c1 += 4;
        c2 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout_lo, 4));

        c0 += 2;
        c1 += 2;
        c2 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
        *c1 = (int8_t) _mm_extract_epi8(vout_hi, 0);
        *c2 = (int8_t) _mm_extract_epi8(vout_lo, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    w = (const int32_t*) w + 8;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m256i vxb01 = _mm256_cvtepi8_epi16(vb01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m256i vxb23 = _mm256_cvtepi8_epi16(vb23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
        const __m256i vxb45 = _mm256_cvtepi8_epi16(vb45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
        const __m256i vxb67 = _mm256_cvtepi8_epi16(vb67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

        w = (const void*) ((const int8_t*) w + 64);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 vscale01234567 = _mm256_load_ps(w);
    w = (const void*) ((const float*) w + 8);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale01234567);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc00x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc0x01234567), voutput_zero_point);

    vacc00x01234567 = _mm256_permute4x64_epi64(vacc00x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc00x01234567, vacc00x01234567);

    vout = _mm256_max_epi8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);

        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_3x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    __m256i vacc2x01 = vacc0x01;
    __m256i vacc2x23 = vacc0x23;
    __m256i vacc2x45 = vacc0x45;
    __m256i vacc2x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
        const __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
        const __m256i vxa2 = _mm256_cvtepi8_epi16(va2);
        a2 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m256i vxb01 = _mm256_cvtepi8_epi16(vb01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
        vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m256i vxb23 = _mm256_cvtepi8_epi16(vb23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
        vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
        const __m256i vxb45 = _mm256_cvtepi8_epi16(vb45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
        vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
        const __m256i vxb67 = _mm256_cvtepi8_epi16(vb67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
        vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
        vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

        w = (const void*) ((const int8_t*) w + 64);
        k += 8 * sizeof(int8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vscaled1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vscaled2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);

    const __m256 vscale01234567 = _mm256_load_ps(w);
    w = (const void*) ((const float*) w + 8);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale01234567);
    vscaled1x01234567 = _mm256_mul_ps(vscaled1x01234567, vscale01234567);
    vscaled2x01234567 = _mm256_mul_ps(vscaled2x01234567, vscale01234567);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);
    vscaled1x01234567 = _mm256_min_ps(vscaled1x01234567, voutput_max_less_zero_point);
    vscaled2x01234567 = _mm256_min_ps(vscaled2x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vscaled1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vscaled2x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);
    __m256i vacc22x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2x01234567, vacc2x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc22x01234567 = _mm256_permute4x64_epi64(vacc22x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc01x01234567, vacc22x01234567);

    vout = _mm256_max_epi8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storeh_pi((__m64*) c2, _mm_castsi128_ps(vout_lo));
      _mm_storel_epi64((__m128i*) c1, vout_hi);
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout_lo, 2));
        _mm_storeu_si32(c1, vout_hi);
        _mm_storeu_si32(c0, vout_lo);

        c2 += 4;
        c1 += 4;
        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout_lo, 4));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c2 += 2;
        c1 += 2;
        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c2 = (int8_t) _mm_extract_epi8(vout_lo, 8);
        *c1 = (int8_t) _mm_extract_epi8(vout_hi, 0);
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));


      const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m256i vi0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      i0 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      const __m256i vi1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      i1 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m256i vi2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      i2 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      const __m256i vi3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      const __m256i vk3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t))));
      i3 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));

      const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m256i vi4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
      const __m256i vk4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t))));
      i4 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));

      const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      const __m256i vi5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
      const __m256i vk5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t))));
      i5 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi5x89ABCDEF, vk5x89ABCDEF));

      const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m256i vi6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
      const __m256i vk6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t))));
      i6 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi6x89ABCDEF, vk6x89ABCDEF));

      const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      const __m256i vi7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
      const __m256i vk7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t))));
      i7 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi7x89ABCDEF, vk7x89ABCDEF));

      const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m256i vi8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 8)));
      const __m256i vk8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t))));
      i8 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi8x89ABCDEF, vk8x89ABCDEF));

      const __m256i vi9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i9));
      const __m256i vk9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      const __m256i vi9x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i9 + 8)));
      const __m256i vk9x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(int8_t))));
      i9 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi9x89ABCDEF, vk9x89ABCDEF));

      const __m256i vi10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i10));
      const __m256i vk10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m256i vi10x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i10 + 8)));
      const __m256i vk10x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(int8_t))));
      i10 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi10x89ABCDEF, vk10x89ABCDEF));

      const __m256i vi11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i11));
      const __m256i vk11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      const __m256i vi11x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i11 + 8)));
      const __m256i vk11x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(int8_t))));
      i11 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi11x89ABCDEF, vk11x89ABCDEF));

      const __m256i vi12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i12));
      const __m256i vk12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m256i vi12x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i12 + 8)));
      const __m256i vk12x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(int8_t))));
      i12 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi12x89ABCDEF, vk12x89ABCDEF));

      const __m256i vi13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i13));
      const __m256i vk13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      const __m256i vi13x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i13 + 8)));
      const __m256i vk13x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(int8_t))));
      i13 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi13x89ABCDEF, vk13x89ABCDEF));

      const __m256i vi14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i14));
      const __m256i vk14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m256i vi14x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i14 + 8)));
      const __m256i vk14x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(int8_t))));
      i14 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi14x89ABCDEF, vk14x89ABCDEF));

      const __m256i vi15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i15));
      const __m256i vk15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      const __m256i vi15x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i15 + 8)));
      const __m256i vk15x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(int8_t))));
      i15 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi15x89ABCDEF, vk15x89ABCDEF));

      const __m256i vi16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i16));
      const __m256i vk16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m256i vi16x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i16 + 8)));
      const __m256i vk16x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(int8_t))));
      i16 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi16x89ABCDEF, vk16x89ABCDEF));

      const __m256i vi17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i17));
      const __m256i vk17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      const __m256i vi17x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i17 + 8)));
      const __m256i vk17x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(int8_t))));
      i17 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi17x89ABCDEF, vk17x89ABCDEF));

      const __m256i vi18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i18));
      const __m256i vk18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m256i vi18x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i18 + 8)));
      const __m256i vk18x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(int8_t))));
      i18 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi18x89ABCDEF, vk18x89ABCDEF));

      const __m256i vi19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i19));
      const __m256i vk19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      const __m256i vi19x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i19 + 8)));
      const __m256i vk19x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(int8_t))));
      i19 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi19x89ABCDEF, vk19x89ABCDEF));

      const __m256i vi20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i20));
      const __m256i vk20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m256i vi20x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i20 + 8)));
      const __m256i vk20x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(int8_t))));
      i20 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi20x89ABCDEF, vk20x89ABCDEF));

      const __m256i vi21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i21));
      const __m256i vk21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      const __m256i vi21x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i21 + 8)));
      const __m256i vk21x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(int8_t))));
      i21 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi21x89ABCDEF, vk21x89ABCDEF));

      const __m256i vi22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i22));
      const __m256i vk22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m256i vi22x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i22 + 8)));
      const __m256i vk22x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(int8_t))));
      i22 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi22x89ABCDEF, vk22x89ABCDEF));

      const __m256i vi23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i23));
      const __m256i vk23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      const __m256i vi23x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i23 + 8)));
      const __m256i vk23x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(int8_t))));
      i23 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi23x89ABCDEF, vk23x89ABCDEF));

      const __m256i vi24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i24));
      const __m256i vk24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m256i vi24x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i24 + 8)));
      const __m256i vk24x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(int8_t))));
      i24 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi24x89ABCDEF, vk24x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) k));
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 16)));
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32)));
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 48)));
        i3 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 64)));
        i4 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 80)));
        i5 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 96)));
        i6 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 112)));
        i7 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 128)));
        i8 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        const __m256i vi9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i9));
        const __m256i vk9x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 144)));
        i9 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));

        const __m256i vi10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i10));
        const __m256i vk10x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 160)));
        i10 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));

        const __m256i vi11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i11));
        const __m256i vk11x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 176)));
        i11 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));

        const __m256i vi12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i12));
        const __m256i vk12x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 192)));
        i12 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));

        const __m256i vi13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i13));
        const __m256i vk13x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 208)));
        i13 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));

        const __m256i vi14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i14));
        const __m256i vk14x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 224)));
        i14 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));

        const __m256i vi15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i15));
        const __m256i vk15x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 240)));
        i15 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));

        const __m256i vi16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i16));
        const __m256i vk16x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 256)));
        i16 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));

        const __m256i vi17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i17));
        const __m256i vk17x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 272)));
        i17 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));

        const __m256i vi18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i18));
        const __m256i vk18x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 288)));
        i18 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));

        const __m256i vi19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i19));
        const __m256i vk19x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 304)));
        i19 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));

        const __m256i vi20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i20));
        const __m256i vk20x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 320)));
        i20 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));

        const __m256i vi21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i21));
        const __m256i vk21x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 336)));
        i21 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));

        const __m256i vi22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i22));
        const __m256i vk22x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 352)));
        i22 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));

        const __m256i vi23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i23));
        const __m256i vk23x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 368)));
        i23 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));

        const __m256i vi24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i24));
        const __m256i vk24x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 384)));
        i24 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.scale));
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));


      const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m256i vi0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      i0 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      const __m256i vi1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      i1 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m256i vi2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      i2 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      const __m256i vi3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      const __m256i vk3x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t))));
      i3 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));

      const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m256i vi4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
      const __m256i vk4x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t))));
      i4 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));

      const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      const __m256i vi5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
      const __m256i vk5x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t))));
      i5 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi5x89ABCDEF, vk5x89ABCDEF));

      const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m256i vi6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
      const __m256i vk6x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t))));
      i6 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi6x89ABCDEF, vk6x89ABCDEF));

      const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      const __m256i vi7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
      const __m256i vk7x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t))));
      i7 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi7x89ABCDEF, vk7x89ABCDEF));

      const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m256i vi8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 8)));
      const __m256i vk8x89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t))));
      i8 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi8x89ABCDEF, vk8x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) k));
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 16)));
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32)));
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 48)));
        i3 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 64)));
        i4 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 80)));
        i5 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 96)));
        i6 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 112)));
        i7 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (k + 128)));
        i8 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.scale));
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_f32_vcvt_ukernel__avx2_x16(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vminus_zero_point = _mm256_load_si256((const __m256i*) params->avx.minus_zero_point);
  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m256i vx01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input));
    __m256i vx89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input + 8)));
    input += 16;

    vx01234567 = _mm256_add_epi32(vx01234567, vminus_zero_point);
    vx89ABCDEF = _mm256_add_epi32(vx89ABCDEF, vminus_zero_point);

    __m256 vy01234567 = _mm256_cvtepi32_ps(vx01234567);
    __m256 vy89ABCDEF = _mm256_cvtepi32_ps(vx89ABCDEF);

    vy01234567 = _mm256_mul_ps(vy01234567, vscale);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vscale);

    _mm256_storeu_ps(output, vy01234567);
    _mm256_storeu_ps(output + 8, vy89ABCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m256i vx = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input));
    vx = _mm256_add_epi32(vx, vminus_zero_point);
    input += 8;

    __m256 vy = _mm256_cvtepi32_ps(vx);
    vy = _mm256_mul_ps(vy, vscale);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m256i vx = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input));
    vx = _mm256_add_epi32(vx, vminus_zero_point);

    __m256 vy = _mm256_cvtepi32_ps(vx);
    vy = _mm256_mul_ps(vy, vscale);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}

void xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    w = (const int32_t*) w + 8;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m256i vxb01 = _mm256_cvtepi8_epi16(vb01);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m256i vxb23 = _mm256_cvtepi8_epi16(vb23);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m256i vxb45 = _mm256_cvtepi8_epi16(vb45);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
      const __m256i vxb67 = _mm256_cvtepi8_epi16(vb67);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

      w = (const void*) ((const int8_t*) w + 64);
      k += 8 * sizeof(int8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc00x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc0x01234567), voutput_zero_point);

    vacc00x01234567 = _mm256_permute4x64_epi64(vacc00x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc00x01234567, vacc00x01234567);

    vout = _mm256_max_epi8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);

        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    __m256i vacc2x01 = vacc0x01;
    __m256i vacc2x23 = vacc0x23;
    __m256i vacc2x45 = vacc0x45;
    __m256i vacc2x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      const __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
      const __m256i vxa2 = _mm256_cvtepi8_epi16(va2);
      a2 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m256i vxb01 = _mm256_cvtepi8_epi16(vb01);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m256i vxb23 = _mm256_cvtepi8_epi16(vb23);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
      const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m256i vxb45 = _mm256_cvtepi8_epi16(vb45);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
      const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
      const __m256i vxb67 = _mm256_cvtepi8_epi16(vb67);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
      vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

      w = (const void*) ((const int8_t*) w + 64);
      k += 8 * sizeof(int8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vscaled1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vscaled2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);
    vscaled1x01234567 = _mm256_mul_ps(vscaled1x01234567, vscale);
    vscaled2x01234567 = _mm256_mul_ps(vscaled2x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);
    vscaled1x01234567 = _mm256_min_ps(vscaled1x01234567, voutput_max_less_zero_point);
    vscaled2x01234567 = _mm256_min_ps(vscaled2x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vscaled1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vscaled2x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);
    __m256i vacc22x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2x01234567, vacc2x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc22x01234567 = _mm256_permute4x64_epi64(vacc22x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc01x01234567, vacc22x01234567);

    vout = _mm256_max_epi8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);
      _mm_storel_epi64((__m128i*) c1, vout_hi);
      _mm_storeh_pi((__m64*) c2, _mm_castsi128_ps(vout_lo));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);
        _mm_storeu_si32(c1, vout_hi);
        unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout_lo, 2));

        c0 += 4;
        c1 += 4;
        c2 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout_lo, 4));

        c0 += 2;
        c1 += 2;
        c2 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
        *c1 = (int8_t) _mm_extract_epi8(vout_hi, 0);
        *c2 = (int8_t) _mm_extract_epi8(vout_lo, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    w = (const int32_t*) w + 8;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m256i vxb01 = _mm256_cvtepi8_epi16(vb01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m256i vxb23 = _mm256_cvtepi8_epi16(vb23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
        const __m256i vxb45 = _mm256_cvtepi8_epi16(vb45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
        const __m256i vxb67 = _mm256_cvtepi8_epi16(vb67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

        w = (const void*) ((const int8_t*) w + 64);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc00x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc0x01234567), voutput_zero_point);

    vacc00x01234567 = _mm256_permute4x64_epi64(vacc00x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc00x01234567, vacc00x01234567);

    vout = _mm256_max_epi8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);

        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    __m256i vacc2x01 = vacc0x01;
    __m256i vacc2x23 = vacc0x23;
    __m256i vacc2x45 = vacc0x45;
    __m256i vacc2x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
        const __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
        const __m256i vxa2 = _mm256_cvtepi8_epi16(va2);
        a2 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m256i vxb01 = _mm256_cvtepi8_epi16(vb01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
        vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m256i vxb23 = _mm256_cvtepi8_epi16(vb23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
        vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
        const __m256i vxb45 = _mm256_cvtepi8_epi16(vb45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
        vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
        const __m256i vxb67 = _mm256_cvtepi8_epi16(vb67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
        vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
        vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

        w = (const void*) ((const int8_t*) w + 64);
        k += 8 * sizeof(int8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vscaled1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vscaled2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);
    vscaled1x01234567 = _mm256_mul_ps(vscaled1x01234567, vscale);
    vscaled2x01234567 = _mm256_mul_ps(vscaled2x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);
    vscaled1x01234567 = _mm256_min_ps(vscaled1x01234567, voutput_max_less_zero_point);
    vscaled2x01234567 = _mm256_min_ps(vscaled2x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vscaled1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vscaled2x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);
    __m256i vacc22x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2x01234567, vacc2x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc22x01234567 = _mm256_permute4x64_epi64(vacc22x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc01x01234567, vacc22x01234567);

    vout = _mm256_max_epi8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storeh_pi((__m64*) c2, _mm_castsi128_ps(vout_lo));
      _mm_storel_epi64((__m128i*) c1, vout_hi);
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout_lo, 2));
        _mm_storeu_si32(c1, vout_hi);
        _mm_storeu_si32(c0, vout_lo);

        c2 += 4;
        c1 += 4;
        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout_lo, 4));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c2 += 2;
        c1 += 2;
        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c2 = (int8_t) _mm_extract_epi8(vout_lo, 8);
        *c1 = (int8_t) _mm_extract_epi8(vout_hi, 0);
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_x16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m256i vbias = _mm256_load_si256((const __m256i*) params->avx2.bias);
  const __m256i va_multiplier = _mm256_load_si256((const __m256i*) params->avx2.a_multiplier);
  const __m256i vb_multiplier = _mm256_load_si256((const __m256i*) params->avx2.b_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx2.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx2.output_max);

  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m256i va01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
    const __m256i vb01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_b));
    const __m256i va89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m256i vb89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    input_a += 16;
    input_b += 16;

    __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));
    __m256i vacc89ABCDEF = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va89ABCDEF, va_multiplier));

    vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vb01234567, vb_multiplier));
    vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vb89ABCDEF, vb_multiplier));

    vacc01234567 = _mm256_sra_epi32(vacc01234567, vshift);
    vacc89ABCDEF = _mm256_sra_epi32(vacc89ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m256i va01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
      const __m256i vb01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;

      __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vb01234567, vb_multiplier));

      vacc01234567 = _mm256_sra_epi32(vacc01234567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), _mm256_castsi256_si128(voutput_zero_point));
      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(int8_t);
      } else {
        if (batch & (4 * sizeof(int8_t))) {
          _mm_storeu_si32(output, vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(int8_t))) {
          _mm_storeu_si16(output, vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(int8_t))) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}

void xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m256i va_multiplier = _mm256_load_si256((const __m256i*) params->avx2.a_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx2.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx2.output_max);

  const __m256i vbias = _mm256_add_epi32(
    _mm256_broadcastd_epi32(_mm_cvtsi32_si128(params->avx2.b_multiplier[0] * (int32_t) *input_b)),
    _mm256_load_si256((const __m256i*) params->avx2.bias));
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m256i va01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
    const __m256i va89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    input_a += 16;

    __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));
    __m256i vacc89ABCDEF = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va89ABCDEF, va_multiplier));

    vacc01234567 = _mm256_sra_epi32(vacc01234567, vshift);
    vacc89ABCDEF = _mm256_sra_epi32(vacc89ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m256i va01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
      input_a += 8;

      __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));

      vacc01234567 = _mm256_sra_epi32(vacc01234567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), _mm256_castsi256_si128(voutput_zero_point));
      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(int8_t);
      } else {
        if (batch & (4 * sizeof(int8_t))) {
          _mm_storeu_si32(output, vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(int8_t))) {
          _mm_storeu_si16(output, vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(int8_t))) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}

void xnn_qs8_vcvt_ukernel__avx2_x32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vinput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.input_zero_point);
  const __m256i vmultiplier = _mm256_load_si256((const __m256i*) params->avx2.multiplier);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m256i vacc0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vacc1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (input + 16)));
    input += 32;

    vacc0 = _mm256_sub_epi16(vinput_zero_point, vacc0);
    vacc1 = _mm256_sub_epi16(vinput_zero_point, vacc1);

    vacc0 = _mm256_slli_epi16(vacc0, 7);
    vacc1 = _mm256_slli_epi16(vacc1, 7);

    vacc0 = _mm256_mulhrs_epi16(vacc0, vmultiplier);
    vacc1 = _mm256_mulhrs_epi16(vacc1, vmultiplier);

    vacc0 = _mm256_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm256_adds_epi16(vacc1, voutput_zero_point);

    __m256i vy0 = _mm256_packs_epi16(vacc0, vacc1);

    vy0 = _mm256_permute4x64_epi64(vy0, _MM_SHUFFLE(3, 1, 2, 0));

    _mm256_storeu_si256((__m256i*) output, vy0);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m256i vacc = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input));
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);
    input += 16;

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    const __m128i vy = _mm_packs_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    __m256i vacc = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input));
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    __m128i vy = _mm_packs_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    if (batch & (8 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_qs8_vlrelu_ukernel__avx2_x32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vinput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.input_zero_point);
  const __m256i vpositive_multiplier = _mm256_load_si256((const __m256i*) params->avx2.positive_multiplier);
  const __m256i vnegative_multiplier = _mm256_load_si256((const __m256i*) params->avx2.negative_multiplier);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m256i vacc0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vacc1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (input + 16)));
    input += 32;

    __m256i vmultiplier0 = _mm256_cmpgt_epi16(vacc0, vinput_zero_point);
    vacc0 = _mm256_sub_epi16(vinput_zero_point, vacc0);
    __m256i vmultiplier1 = _mm256_cmpgt_epi16(vacc1, vinput_zero_point);
    vacc1 = _mm256_sub_epi16(vinput_zero_point, vacc1);

    vmultiplier0 = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier0);
    vacc0 = _mm256_slli_epi16(vacc0, 7);
    vmultiplier1 = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier1);
    vacc1 = _mm256_slli_epi16(vacc1, 7);

    vacc0 = _mm256_mulhrs_epi16(vacc0, vmultiplier0);
    vacc1 = _mm256_mulhrs_epi16(vacc1, vmultiplier1);

    vacc0 = _mm256_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm256_adds_epi16(vacc1, voutput_zero_point);

    __m256i vy0 = _mm256_packs_epi16(vacc0, vacc1);

    vy0 = _mm256_permute4x64_epi64(vy0, _MM_SHUFFLE(3, 1, 2, 0));

    _mm256_storeu_si256((__m256i*) output, vy0);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m256i vacc = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vmultiplier = _mm256_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);
    input += 16;

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    const __m128i vy = _mm_packs_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    __m256i vacc = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vmultiplier = _mm256_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    __m128i vy = _mm_packs_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    if (batch & (8 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256i vk_zero_point = _mm256_cvtepu16_epi32(_mm_load_si128((const __m128i*) params->fp32_avx2.kernel_zero_point));
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint8_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint8_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint8_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint8_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint8_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint8_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint8_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint8_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint8_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint8_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint8_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint8_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint8_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint8_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint8_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));


      const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi0x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(uint8_t)))), vk_zero_point);
      i0 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi1x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(uint8_t)))), vk_zero_point);
      i1 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi2x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(uint8_t)))), vk_zero_point);
      i2 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi3x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      const __m256i vk3x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(uint8_t)))), vk_zero_point);
      i3 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));

      const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi4x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
      const __m256i vk4x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(uint8_t)))), vk_zero_point);
      i4 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));

      const __m256i vi5x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi5x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
      const __m256i vk5x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(uint8_t)))), vk_zero_point);
      i5 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi5x89ABCDEF, vk5x89ABCDEF));

      const __m256i vi6x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi6x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
      const __m256i vk6x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(uint8_t)))), vk_zero_point);
      i6 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi6x89ABCDEF, vk6x89ABCDEF));

      const __m256i vi7x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi7x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
      const __m256i vk7x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(uint8_t)))), vk_zero_point);
      i7 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi7x89ABCDEF, vk7x89ABCDEF));

      const __m256i vi8x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi8x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 8)));
      const __m256i vk8x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(uint8_t)))), vk_zero_point);
      i8 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi8x89ABCDEF, vk8x89ABCDEF));

      const __m256i vi9x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i9));
      const __m256i vk9x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi9x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i9 + 8)));
      const __m256i vk9x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(uint8_t)))), vk_zero_point);
      i9 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi9x89ABCDEF, vk9x89ABCDEF));

      const __m256i vi10x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i10));
      const __m256i vk10x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi10x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i10 + 8)));
      const __m256i vk10x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(uint8_t)))), vk_zero_point);
      i10 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi10x89ABCDEF, vk10x89ABCDEF));

      const __m256i vi11x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i11));
      const __m256i vk11x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi11x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i11 + 8)));
      const __m256i vk11x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(uint8_t)))), vk_zero_point);
      i11 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi11x89ABCDEF, vk11x89ABCDEF));

      const __m256i vi12x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i12));
      const __m256i vk12x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi12x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i12 + 8)));
      const __m256i vk12x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(uint8_t)))), vk_zero_point);
      i12 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi12x89ABCDEF, vk12x89ABCDEF));

      const __m256i vi13x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i13));
      const __m256i vk13x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi13x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i13 + 8)));
      const __m256i vk13x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(uint8_t)))), vk_zero_point);
      i13 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi13x89ABCDEF, vk13x89ABCDEF));

      const __m256i vi14x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i14));
      const __m256i vk14x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi14x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i14 + 8)));
      const __m256i vk14x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(uint8_t)))), vk_zero_point);
      i14 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi14x89ABCDEF, vk14x89ABCDEF));

      const __m256i vi15x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i15));
      const __m256i vk15x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi15x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i15 + 8)));
      const __m256i vk15x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(uint8_t)))), vk_zero_point);
      i15 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi15x89ABCDEF, vk15x89ABCDEF));

      const __m256i vi16x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i16));
      const __m256i vk16x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi16x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i16 + 8)));
      const __m256i vk16x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(uint8_t)))), vk_zero_point);
      i16 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi16x89ABCDEF, vk16x89ABCDEF));

      const __m256i vi17x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i17));
      const __m256i vk17x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi17x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i17 + 8)));
      const __m256i vk17x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(uint8_t)))), vk_zero_point);
      i17 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi17x89ABCDEF, vk17x89ABCDEF));

      const __m256i vi18x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i18));
      const __m256i vk18x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi18x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i18 + 8)));
      const __m256i vk18x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(uint8_t)))), vk_zero_point);
      i18 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi18x89ABCDEF, vk18x89ABCDEF));

      const __m256i vi19x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i19));
      const __m256i vk19x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi19x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i19 + 8)));
      const __m256i vk19x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(uint8_t)))), vk_zero_point);
      i19 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi19x89ABCDEF, vk19x89ABCDEF));

      const __m256i vi20x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i20));
      const __m256i vk20x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi20x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i20 + 8)));
      const __m256i vk20x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(uint8_t)))), vk_zero_point);
      i20 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi20x89ABCDEF, vk20x89ABCDEF));

      const __m256i vi21x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i21));
      const __m256i vk21x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi21x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i21 + 8)));
      const __m256i vk21x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(uint8_t)))), vk_zero_point);
      i21 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi21x89ABCDEF, vk21x89ABCDEF));

      const __m256i vi22x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i22));
      const __m256i vk22x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi22x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i22 + 8)));
      const __m256i vk22x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(uint8_t)))), vk_zero_point);
      i22 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi22x89ABCDEF, vk22x89ABCDEF));

      const __m256i vi23x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i23));
      const __m256i vk23x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi23x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i23 + 8)));
      const __m256i vk23x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(uint8_t)))), vk_zero_point);
      i23 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi23x89ABCDEF, vk23x89ABCDEF));

      const __m256i vi24x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i24));
      const __m256i vk24x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi24x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i24 + 8)));
      const __m256i vk24x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(uint8_t)))), vk_zero_point);
      i24 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi24x89ABCDEF, vk24x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(uint8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 16);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) k)), vk_zero_point);
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 16))), vk_zero_point);
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32))), vk_zero_point);
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 48))), vk_zero_point);
        i3 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 64))), vk_zero_point);
        i4 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 80))), vk_zero_point);
        i5 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 96))), vk_zero_point);
        i6 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 112))), vk_zero_point);
        i7 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 128))), vk_zero_point);
        i8 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        const __m256i vi9x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i9));
        const __m256i vk9x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 144))), vk_zero_point);
        i9 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi9x01234567, vk9x01234567));

        const __m256i vi10x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i10));
        const __m256i vk10x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 160))), vk_zero_point);
        i10 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi10x01234567, vk10x01234567));

        const __m256i vi11x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i11));
        const __m256i vk11x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 176))), vk_zero_point);
        i11 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi11x01234567, vk11x01234567));

        const __m256i vi12x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i12));
        const __m256i vk12x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 192))), vk_zero_point);
        i12 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi12x01234567, vk12x01234567));

        const __m256i vi13x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i13));
        const __m256i vk13x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 208))), vk_zero_point);
        i13 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi13x01234567, vk13x01234567));

        const __m256i vi14x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i14));
        const __m256i vk14x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 224))), vk_zero_point);
        i14 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi14x01234567, vk14x01234567));

        const __m256i vi15x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i15));
        const __m256i vk15x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 240))), vk_zero_point);
        i15 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi15x01234567, vk15x01234567));

        const __m256i vi16x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i16));
        const __m256i vk16x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 256))), vk_zero_point);
        i16 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi16x01234567, vk16x01234567));

        const __m256i vi17x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i17));
        const __m256i vk17x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 272))), vk_zero_point);
        i17 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi17x01234567, vk17x01234567));

        const __m256i vi18x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i18));
        const __m256i vk18x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 288))), vk_zero_point);
        i18 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi18x01234567, vk18x01234567));

        const __m256i vi19x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i19));
        const __m256i vk19x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 304))), vk_zero_point);
        i19 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi19x01234567, vk19x01234567));

        const __m256i vi20x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i20));
        const __m256i vk20x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 320))), vk_zero_point);
        i20 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi20x01234567, vk20x01234567));

        const __m256i vi21x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i21));
        const __m256i vk21x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 336))), vk_zero_point);
        i21 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi21x01234567, vk21x01234567));

        const __m256i vi22x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i22));
        const __m256i vk22x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 352))), vk_zero_point);
        i22 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi22x01234567, vk22x01234567));

        const __m256i vi23x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i23));
        const __m256i vk23x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 368))), vk_zero_point);
        i23 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi23x01234567, vk23x01234567));

        const __m256i vi24x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i24));
        const __m256i vk24x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 384))), vk_zero_point);
        i24 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi24x01234567, vk24x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.scale));
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256i vk_zero_point = _mm256_cvtepu16_epi32(_mm_load_si128((const __m128i*) params->fp32_avx2.kernel_zero_point));
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));


      const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi0x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(uint8_t)))), vk_zero_point);
      i0 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi1x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(uint8_t)))), vk_zero_point);
      i1 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi2x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(uint8_t)))), vk_zero_point);
      i2 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi3x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      const __m256i vk3x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(uint8_t)))), vk_zero_point);
      i3 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));

      const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi4x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
      const __m256i vk4x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(uint8_t)))), vk_zero_point);
      i4 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));

      const __m256i vi5x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi5x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
      const __m256i vk5x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(uint8_t)))), vk_zero_point);
      i5 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi5x89ABCDEF, vk5x89ABCDEF));

      const __m256i vi6x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi6x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
      const __m256i vk6x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(uint8_t)))), vk_zero_point);
      i6 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi6x89ABCDEF, vk6x89ABCDEF));

      const __m256i vi7x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi7x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
      const __m256i vk7x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(uint8_t)))), vk_zero_point);
      i7 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi7x89ABCDEF, vk7x89ABCDEF));

      const __m256i vi8x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi8x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 8)));
      const __m256i vk8x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(uint8_t)))), vk_zero_point);
      i8 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi8x89ABCDEF, vk8x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(uint8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale);

      const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);

      const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 16);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) k)), vk_zero_point);
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 16))), vk_zero_point);
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32))), vk_zero_point);
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 48))), vk_zero_point);
        i3 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 64))), vk_zero_point);
        i4 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 80))), vk_zero_point);
        i5 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 96))), vk_zero_point);
        i6 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 112))), vk_zero_point);
        i7 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 128))), vk_zero_point);
        i8 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.scale));
        vscaled01234567 = _mm256_min_ps(vscaled01234567, _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point));
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_avx2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx2.output_min);
        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_f32_vcvt_ukernel__avx2_x16(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vminus_zero_point = _mm256_load_si256((const __m256i*) params->avx.minus_zero_point);
  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m256i vx01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input));
    __m256i vx89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (input + 8)));
    input += 16;

    vx01234567 = _mm256_add_epi32(vx01234567, vminus_zero_point);
    vx89ABCDEF = _mm256_add_epi32(vx89ABCDEF, vminus_zero_point);

    __m256 vy01234567 = _mm256_cvtepi32_ps(vx01234567);
    __m256 vy89ABCDEF = _mm256_cvtepi32_ps(vx89ABCDEF);

    vy01234567 = _mm256_mul_ps(vy01234567, vscale);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vscale);

    _mm256_storeu_ps(output, vy01234567);
    _mm256_storeu_ps(output + 8, vy89ABCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m256i vx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input));
    vx = _mm256_add_epi32(vx, vminus_zero_point);
    input += 8;

    __m256 vy = _mm256_cvtepi32_ps(vx);
    vy = _mm256_mul_ps(vy, vscale);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    __m256i vx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input));
    vx = _mm256_add_epi32(vx, vminus_zero_point);

    __m256 vy = _mm256_cvtepi32_ps(vx);
    vy = _mm256_mul_ps(vy, vscale);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    w = (const int32_t*) w + 8;

    size_t k = 0;
    const __m256i vb_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.kernel_zero_point);
    while (k < kc) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepu8_epi16(va0);
      a0 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m256i vxb01 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb01), vb_zero_point);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
      const __m256i vxb23 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb23), vb_zero_point);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      const __m128i vb45 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 32));
      const __m256i vxb45 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb45), vb_zero_point);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      const __m128i vb67 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 48));
      const __m256i vxb67 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb67), vb_zero_point);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

      w = (const void*) ((const uint8_t*) w + 64);
      k += 8 * sizeof(uint8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc00x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc0x01234567), voutput_zero_point);

    vacc00x01234567 = _mm256_permute4x64_epi64(vacc00x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packus_epi16(vacc00x01234567, vacc00x01234567);

    vout = _mm256_max_epu8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);

        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    __m256i vacc2x01 = vacc0x01;
    __m256i vacc2x23 = vacc0x23;
    __m256i vacc2x45 = vacc0x45;
    __m256i vacc2x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t k = 0;
    const __m256i vb_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.kernel_zero_point);
    while (k < kc) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepu8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      const __m256i vxa1 = _mm256_cvtepu8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
      const __m256i vxa2 = _mm256_cvtepu8_epi16(va2);
      a2 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m256i vxb01 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb01), vb_zero_point);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
      const __m256i vxb23 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb23), vb_zero_point);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
      const __m128i vb45 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 32));
      const __m256i vxb45 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb45), vb_zero_point);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
      const __m128i vb67 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 48));
      const __m256i vxb67 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb67), vb_zero_point);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
      vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

      w = (const void*) ((const uint8_t*) w + 64);
      k += 8 * sizeof(uint8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vscaled1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vscaled2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);
    vscaled1x01234567 = _mm256_mul_ps(vscaled1x01234567, vscale);
    vscaled2x01234567 = _mm256_mul_ps(vscaled2x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);
    vscaled1x01234567 = _mm256_min_ps(vscaled1x01234567, voutput_max_less_zero_point);
    vscaled2x01234567 = _mm256_min_ps(vscaled2x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vscaled1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vscaled2x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);
    __m256i vacc22x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2x01234567, vacc2x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc22x01234567 = _mm256_permute4x64_epi64(vacc22x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packus_epi16(vacc01x01234567, vacc22x01234567);

    vout = _mm256_max_epu8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);
      _mm_storel_epi64((__m128i*) c1, vout_hi);
      _mm_storeh_pi((__m64*) c2, _mm_castsi128_ps(vout_lo));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);
        _mm_storeu_si32(c1, vout_hi);
        unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout_lo, 2));

        c0 += 4;
        c1 += 4;
        c2 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout_lo, 4));

        c0 += 2;
        c1 += 2;
        c2 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout_lo, 0);
        *c1 = (uint8_t) _mm_extract_epi8(vout_hi, 0);
        *c2 = (uint8_t) _mm_extract_epi8(vout_lo, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  uint8_t* c0 = c;

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    w = (const int32_t*) w + 8;

    size_t p = ks;
    const __m256i vb_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.kernel_zero_point);
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        const __m256i vxa0 = _mm256_cvtepu8_epi16(va0);
        a0 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m256i vxb01 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb01), vb_zero_point);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
        const __m256i vxb23 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb23), vb_zero_point);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 32));
        const __m256i vxb45 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb45), vb_zero_point);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 48));
        const __m256i vxb67 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb67), vb_zero_point);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

        w = (const void*) ((const uint8_t*) w + 64);
        k += 8 * sizeof(uint8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc00x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc0x01234567), voutput_zero_point);

    vacc00x01234567 = _mm256_permute4x64_epi64(vacc00x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packus_epi16(vacc00x01234567, vacc00x01234567);

    vout = _mm256_max_epu8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);

        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_3x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    __m256i vacc2x01 = vacc0x01;
    __m256i vacc2x23 = vacc0x23;
    __m256i vacc2x45 = vacc0x45;
    __m256i vacc2x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t p = ks;
    const __m256i vb_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.kernel_zero_point);
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        const __m256i vxa0 = _mm256_cvtepu8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
        const __m256i vxa1 = _mm256_cvtepu8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
        const __m256i vxa2 = _mm256_cvtepu8_epi16(va2);
        a2 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m256i vxb01 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb01), vb_zero_point);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
        vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
        const __m256i vxb23 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb23), vb_zero_point);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
        vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 32));
        const __m256i vxb45 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb45), vb_zero_point);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
        vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 48));
        const __m256i vxb67 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb67), vb_zero_point);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
        vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
        vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

        w = (const void*) ((const uint8_t*) w + 64);
        k += 8 * sizeof(uint8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vscaled1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vscaled2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);
    vscaled1x01234567 = _mm256_mul_ps(vscaled1x01234567, vscale);
    vscaled2x01234567 = _mm256_mul_ps(vscaled2x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);
    vscaled1x01234567 = _mm256_min_ps(vscaled1x01234567, voutput_max_less_zero_point);
    vscaled2x01234567 = _mm256_min_ps(vscaled2x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vscaled1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vscaled2x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);
    __m256i vacc22x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2x01234567, vacc2x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc22x01234567 = _mm256_permute4x64_epi64(vacc22x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packus_epi16(vacc01x01234567, vacc22x01234567);

    vout = _mm256_max_epu8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storeh_pi((__m64*) c2, _mm_castsi128_ps(vout_lo));
      _mm_storel_epi64((__m128i*) c1, vout_hi);
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout_lo, 2));
        _mm_storeu_si32(c1, vout_hi);
        _mm_storeu_si32(c0, vout_lo);

        c2 += 4;
        c1 += 4;
        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout_lo, 4));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c2 += 2;
        c1 += 2;
        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c2 = (uint8_t) _mm_extract_epi8(vout_lo, 8);
        *c1 = (uint8_t) _mm_extract_epi8(vout_hi, 0);
        *c0 = (uint8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vadd_minmax_ukernel__avx2_mul32_ld64_x16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m256i vbias = _mm256_load_si256((const __m256i*) params->avx2.bias);
  const __m256i va_multiplier = _mm256_load_si256((const __m256i*) params->avx2.a_multiplier);
  const __m256i vb_multiplier = _mm256_load_si256((const __m256i*) params->avx2.b_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx2.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx2.output_max);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m256i va01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
    const __m256i vb01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_b));
    const __m256i va89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m256i vb89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    input_a += 16;
    input_b += 16;

    __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));
    __m256i vacc89ABCDEF = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va89ABCDEF, va_multiplier));

    vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vb01234567, vb_multiplier));
    vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vb89ABCDEF, vb_multiplier));

    vacc01234567 = _mm256_sra_epi32(vacc01234567, vshift);
    vacc89ABCDEF = _mm256_sra_epi32(vacc89ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m256i va01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
      const __m256i vb01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;

      __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vb01234567, vb_multiplier));

      vacc01234567 = _mm256_sra_epi32(vacc01234567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), _mm256_castsi256_si128(voutput_zero_point));
      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(uint8_t);
      } else {
        if (batch & (4 * sizeof(uint8_t))) {
          _mm_storeu_si32(output, vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(uint8_t))) {
          _mm_storeu_si16(output, vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(uint8_t))) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}

void xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m256i va_multiplier = _mm256_load_si256((const __m256i*) params->avx2.a_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx2.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx2.output_max);

  const __m256i vbias = _mm256_add_epi32(
    _mm256_broadcastd_epi32(_mm_cvtsi32_si128(params->avx2.b_multiplier[0] * (int32_t) *input_b)),
    _mm256_load_si256((const __m256i*) params->avx2.bias));
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m256i va01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
    const __m256i va89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    input_a += 16;

    __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));
    __m256i vacc89ABCDEF = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va89ABCDEF, va_multiplier));

    vacc01234567 = _mm256_sra_epi32(vacc01234567, vshift);
    vacc89ABCDEF = _mm256_sra_epi32(vacc89ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m256i va01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) input_a));
      input_a += 8;

      __m256i vacc01234567 = _mm256_add_epi32(vbias, _mm256_mullo_epi32(va01234567, va_multiplier));

      vacc01234567 = _mm256_sra_epi32(vacc01234567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), _mm256_castsi256_si128(voutput_zero_point));
      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(uint8_t);
      } else {
        if (batch & (4 * sizeof(uint8_t))) {
          _mm_storeu_si32(output, vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(uint8_t))) {
          _mm_storeu_si16(output, vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(uint8_t))) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}

void xnn_qu8_vcvt_ukernel__avx2_x32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vinput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.input_zero_point);
  const __m256i vmultiplier = _mm256_load_si256((const __m256i*) params->avx2.multiplier);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    __m256i vacc0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vacc1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (input + 16)));
    input += 32;

    vacc0 = _mm256_sub_epi16(vinput_zero_point, vacc0);
    vacc1 = _mm256_sub_epi16(vinput_zero_point, vacc1);

    vacc0 = _mm256_slli_epi16(vacc0, 7);
    vacc1 = _mm256_slli_epi16(vacc1, 7);

    vacc0 = _mm256_mulhrs_epi16(vacc0, vmultiplier);
    vacc1 = _mm256_mulhrs_epi16(vacc1, vmultiplier);

    vacc0 = _mm256_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm256_adds_epi16(vacc1, voutput_zero_point);

    __m256i vy0 = _mm256_packus_epi16(vacc0, vacc1);

    vy0 = _mm256_permute4x64_epi64(vy0, _MM_SHUFFLE(3, 1, 2, 0));

    _mm256_storeu_si256((__m256i*) output, vy0);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m256i vacc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);
    input += 16;

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    const __m128i vy = _mm_packus_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    __m256i vacc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    __m128i vy = _mm_packus_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_qu8_vlrelu_ukernel__avx2_x32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vinput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.input_zero_point);
  const __m256i vpositive_multiplier = _mm256_load_si256((const __m256i*) params->avx2.positive_multiplier);
  const __m256i vnegative_multiplier = _mm256_load_si256((const __m256i*) params->avx2.negative_multiplier);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    __m256i vacc0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vacc1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (input + 16)));
    input += 32;

    __m256i vmultiplier0 = _mm256_cmpgt_epi16(vacc0, vinput_zero_point);
    vacc0 = _mm256_sub_epi16(vinput_zero_point, vacc0);
    __m256i vmultiplier1 = _mm256_cmpgt_epi16(vacc1, vinput_zero_point);
    vacc1 = _mm256_sub_epi16(vinput_zero_point, vacc1);

    vmultiplier0 = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier0);
    vacc0 = _mm256_slli_epi16(vacc0, 7);
    vmultiplier1 = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier1);
    vacc1 = _mm256_slli_epi16(vacc1, 7);

    vacc0 = _mm256_mulhrs_epi16(vacc0, vmultiplier0);
    vacc1 = _mm256_mulhrs_epi16(vacc1, vmultiplier1);

    vacc0 = _mm256_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm256_adds_epi16(vacc1, voutput_zero_point);

    __m256i vy0 = _mm256_packus_epi16(vacc0, vacc1);

    vy0 = _mm256_permute4x64_epi64(vy0, _MM_SHUFFLE(3, 1, 2, 0));

    _mm256_storeu_si256((__m256i*) output, vy0);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m256i vacc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vmultiplier = _mm256_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);
    input += 16;

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    const __m128i vy = _mm_packus_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    __m256i vacc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) input));
    __m256i vmultiplier = _mm256_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    __m128i vy = _mm_packus_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2(
    const uint16_t* input,
    uint16_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x16_transpose_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_stride >= block_height * sizeof(uint16_t));
  assert(input_stride >= block_width * sizeof(uint16_t));

  const size_t tile_height = 16;
  const size_t tile_width = 16;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  uint16_t* o = (uint16_t*) output;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t);

  const uint16_t* i0 = (const uint16_t*) input;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 15);
    const size_t oN_stride = rem * output_stride;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx2.mask_table[7 ^ (rem>>1)]));

    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const __m256i v4_0 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_1 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_2 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_3 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_4 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_5 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_6 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_7 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_8 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_9 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_10 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_11 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_12 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_13 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_14 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_15 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      const __m256i v3_0 = _mm256_unpacklo_epi16(v4_0, v4_4);
      const __m256i v3_1 = _mm256_unpackhi_epi16(v4_0, v4_4);
      const __m256i v3_2 = _mm256_unpacklo_epi16(v4_1, v4_5);
      const __m256i v3_3 = _mm256_unpackhi_epi16(v4_1, v4_5);
      const __m256i v3_4 = _mm256_unpacklo_epi16(v4_2, v4_6);
      const __m256i v3_5 = _mm256_unpackhi_epi16(v4_2, v4_6);
      const __m256i v3_6 = _mm256_unpacklo_epi16(v4_3, v4_7);
      const __m256i v3_7 = _mm256_unpackhi_epi16(v4_3, v4_7);
      const __m256i v3_8 = _mm256_unpacklo_epi16(v4_8, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi16(v4_8, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi16(v4_9, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi16(v4_9, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi16(v4_10, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi16(v4_10, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi16(v4_11, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi16(v4_11, v4_15);
      const __m256i v2_0 = _mm256_unpacklo_epi16(v3_0, v3_4);
      const __m256i v2_1 = _mm256_unpackhi_epi16(v3_0, v3_4);
      const __m256i v2_2 = _mm256_unpacklo_epi16(v3_1, v3_5);
      const __m256i v2_3 = _mm256_unpackhi_epi16(v3_1, v3_5);
      const __m256i v2_4 = _mm256_unpacklo_epi16(v3_2, v3_6);
      const __m256i v2_5 = _mm256_unpackhi_epi16(v3_2, v3_6);
      const __m256i v2_6 = _mm256_unpacklo_epi16(v3_3, v3_7);
      const __m256i v2_7 = _mm256_unpackhi_epi16(v3_3, v3_7);
      const __m256i v2_8 = _mm256_unpacklo_epi16(v3_8, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi16(v3_8, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi16(v3_9, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi16(v3_9, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi16(v3_10, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi16(v3_10, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi16(v3_11, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi16(v3_11, v3_15);
      const __m256i v1_0 = _mm256_unpacklo_epi16(v2_0, v2_4);
      const __m256i v1_1 = _mm256_unpackhi_epi16(v2_0, v2_4);
      const __m256i v1_2 = _mm256_unpacklo_epi16(v2_1, v2_5);
      const __m256i v1_3 = _mm256_unpackhi_epi16(v2_1, v2_5);
      const __m256i v1_4 = _mm256_unpacklo_epi16(v2_2, v2_6);
      const __m256i v1_5 = _mm256_unpackhi_epi16(v2_2, v2_6);
      const __m256i v1_6 = _mm256_unpacklo_epi16(v2_3, v2_7);
      const __m256i v1_7 = _mm256_unpackhi_epi16(v2_3, v2_7);
      const __m256i v1_8 = _mm256_unpacklo_epi16(v2_8, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi16(v2_8, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi16(v2_9, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi16(v2_9, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi16(v2_10, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi16(v2_10, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi16(v2_11, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi16(v2_11, v2_15);


      uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        default:
          XNN_UNREACHABLE;
        case 15: {
          const __m256i v0_15 = _mm256_permute2f128_si256(v1_7, v1_15, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_15);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 14: {
          const __m256i v0_14 = _mm256_permute2f128_si256(v1_6, v1_14, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_14);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 13: {
          const __m256i v0_13 = _mm256_permute2f128_si256(v1_5, v1_13, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_13);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 12: {
          const __m256i v0_12 = _mm256_permute2f128_si256(v1_4, v1_12, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_12);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 11: {
          const __m256i v0_11 = _mm256_permute2f128_si256(v1_3, v1_11, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_11);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 10: {
          const __m256i v0_10 = _mm256_permute2f128_si256(v1_2, v1_10, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_10);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 9: {
          const __m256i v0_9 = _mm256_permute2f128_si256(v1_1, v1_9, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_9);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 8: {
          const __m256i v0_8 = _mm256_permute2f128_si256(v1_0, v1_8, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_8);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 7: {
          const __m256i v0_7 = _mm256_insertf128_si256(v1_7, _mm256_castsi256_si128(v1_15), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_7);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 6: {
          const __m256i v0_6 = _mm256_insertf128_si256(v1_6, _mm256_castsi256_si128(v1_14), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_6);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 5: {
          const __m256i v0_5 = _mm256_insertf128_si256(v1_5, _mm256_castsi256_si128(v1_13), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_5);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 4: {
          const __m256i v0_4 = _mm256_insertf128_si256(v1_4, _mm256_castsi256_si128(v1_12), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_4);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 3: {
          const __m256i v0_3 = _mm256_insertf128_si256(v1_3, _mm256_castsi256_si128(v1_11), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_3);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 2: {
          const __m256i v0_2 = _mm256_insertf128_si256(v1_2, _mm256_castsi256_si128(v1_10), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_2);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 1: {
          const __m256i v0_1 = _mm256_insertf128_si256(v1_1, _mm256_castsi256_si128(v1_9), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_1);
        }
        case 0: {
          const __m256i v0_0 = _mm256_insertf128_si256(v1_0, _mm256_castsi256_si128(v1_8), 1);
          _mm256_storeu_si256((__m256i*) o, v0_0);
          o = (uint16_t*) ((uintptr_t) o + tile_hbytes);
        }
      }
    }
    if (bh != 0) {
      const __m256i v4_0 = _mm256_maskload_epi32((const int*) i0, vmask);
      const uint16_t *i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256i v4_1 = _mm256_maskload_epi32((const int*) i1, vmask);
      const uint16_t *i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256i v4_2 = _mm256_maskload_epi32((const int*) i2, vmask);
      const uint16_t *i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m256i v4_3 = _mm256_maskload_epi32((const int*) i3, vmask);
      const uint16_t *i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m256i v4_4 = _mm256_maskload_epi32((const int*) i4, vmask);
      const uint16_t *i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m256i v4_5 = _mm256_maskload_epi32((const int*) i5, vmask);
      const uint16_t *i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m256i v4_6 = _mm256_maskload_epi32((const int*) i6, vmask);
      const uint16_t *i7 = (const uint16_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const __m256i v4_7 = _mm256_maskload_epi32((const int*) i7, vmask);
      const uint16_t *i8 = (const uint16_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const __m256i v4_8 = _mm256_maskload_epi32((const int*) i8, vmask);
      const uint16_t *i9 = (const uint16_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const __m256i v4_9 = _mm256_maskload_epi32((const int*) i9, vmask);
      const uint16_t *i10 = (const uint16_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const __m256i v4_10 = _mm256_maskload_epi32((const int*) i10, vmask);
      const uint16_t *i11 = (const uint16_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const __m256i v4_11 = _mm256_maskload_epi32((const int*) i11, vmask);
      const uint16_t *i12 = (const uint16_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const __m256i v4_12 = _mm256_maskload_epi32((const int*) i12, vmask);
      const uint16_t *i13 = (const uint16_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const __m256i v4_13 = _mm256_maskload_epi32((const int*) i13, vmask);
      const uint16_t *i14 = (const uint16_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const __m256i v4_14 = _mm256_maskload_epi32((const int*) i14, vmask);
      const __m256i v4_15 = _mm256_undefined_si256();

      const __m256i v3_0 = _mm256_unpacklo_epi16(v4_0, v4_4);
      const __m256i v3_1 = _mm256_unpackhi_epi16(v4_0, v4_4);
      const __m256i v3_2 = _mm256_unpacklo_epi16(v4_1, v4_5);
      const __m256i v3_3 = _mm256_unpackhi_epi16(v4_1, v4_5);
      const __m256i v3_4 = _mm256_unpacklo_epi16(v4_2, v4_6);
      const __m256i v3_5 = _mm256_unpackhi_epi16(v4_2, v4_6);
      const __m256i v3_6 = _mm256_unpacklo_epi16(v4_3, v4_7);
      const __m256i v3_7 = _mm256_unpackhi_epi16(v4_3, v4_7);
      const __m256i v3_8 = _mm256_unpacklo_epi16(v4_8, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi16(v4_8, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi16(v4_9, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi16(v4_9, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi16(v4_10, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi16(v4_10, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi16(v4_11, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi16(v4_11, v4_15);
      const __m256i v2_0 = _mm256_unpacklo_epi16(v3_0, v3_4);
      const __m256i v2_1 = _mm256_unpackhi_epi16(v3_0, v3_4);
      const __m256i v2_2 = _mm256_unpacklo_epi16(v3_1, v3_5);
      const __m256i v2_3 = _mm256_unpackhi_epi16(v3_1, v3_5);
      const __m256i v2_4 = _mm256_unpacklo_epi16(v3_2, v3_6);
      const __m256i v2_5 = _mm256_unpackhi_epi16(v3_2, v3_6);
      const __m256i v2_6 = _mm256_unpacklo_epi16(v3_3, v3_7);
      const __m256i v2_7 = _mm256_unpackhi_epi16(v3_3, v3_7);
      const __m256i v2_8 = _mm256_unpacklo_epi16(v3_8, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi16(v3_8, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi16(v3_9, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi16(v3_9, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi16(v3_10, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi16(v3_10, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi16(v3_11, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi16(v3_11, v3_15);
      const __m256i v1_0 = _mm256_unpacklo_epi16(v2_0, v2_4);
      const __m256i v1_1 = _mm256_unpackhi_epi16(v2_0, v2_4);
      const __m256i v1_2 = _mm256_unpacklo_epi16(v2_1, v2_5);
      const __m256i v1_3 = _mm256_unpackhi_epi16(v2_1, v2_5);
      const __m256i v1_4 = _mm256_unpacklo_epi16(v2_2, v2_6);
      const __m256i v1_5 = _mm256_unpackhi_epi16(v2_2, v2_6);
      const __m256i v1_6 = _mm256_unpacklo_epi16(v2_3, v2_7);
      const __m256i v1_7 = _mm256_unpackhi_epi16(v2_3, v2_7);
      const __m256i v1_8 = _mm256_unpacklo_epi16(v2_8, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi16(v2_8, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi16(v2_9, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi16(v2_9, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi16(v2_10, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi16(v2_10, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi16(v2_11, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi16(v2_11, v2_15);

      __m128i v0_0_lo = _mm256_castsi256_si128(v1_0);
      __m128i v0_1_lo = _mm256_castsi256_si128(v1_1);
      __m128i v0_2_lo = _mm256_castsi256_si128(v1_2);
      __m128i v0_3_lo = _mm256_castsi256_si128(v1_3);
      __m128i v0_4_lo = _mm256_castsi256_si128(v1_4);
      __m128i v0_5_lo = _mm256_castsi256_si128(v1_5);
      __m128i v0_6_lo = _mm256_castsi256_si128(v1_6);
      __m128i v0_7_lo = _mm256_castsi256_si128(v1_7);
      __m128i v0_8_lo = _mm256_extractf128_si256(v1_0, 0x1);
      __m128i v0_9_lo = _mm256_extractf128_si256(v1_1, 0x1);
      __m128i v0_10_lo = _mm256_extractf128_si256(v1_2, 0x1);
      __m128i v0_11_lo = _mm256_extractf128_si256(v1_3, 0x1);
      __m128i v0_12_lo = _mm256_extractf128_si256(v1_4, 0x1);
      __m128i v0_13_lo = _mm256_extractf128_si256(v1_5, 0x1);
      __m128i v0_14_lo = _mm256_extractf128_si256(v1_6, 0x1);
      __m128i v0_15_lo = _mm256_extractf128_si256(v1_7, 0x1);

      if (bh & 8) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            _mm_storeu_si128((__m128i*) oN, v0_15_lo);
             v0_15_lo = _mm256_extractf128_si256(v1_15, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            _mm_storeu_si128((__m128i*) oN, v0_14_lo);
             v0_14_lo = _mm256_extractf128_si256(v1_14, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            _mm_storeu_si128((__m128i*) oN, v0_13_lo);
             v0_13_lo = _mm256_extractf128_si256(v1_13, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            _mm_storeu_si128((__m128i*) oN, v0_12_lo);
             v0_12_lo = _mm256_extractf128_si256(v1_12, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            _mm_storeu_si128((__m128i*) oN, v0_11_lo);
             v0_11_lo = _mm256_extractf128_si256(v1_11, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            _mm_storeu_si128((__m128i*) oN, v0_10_lo);
             v0_10_lo = _mm256_extractf128_si256(v1_10, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            _mm_storeu_si128((__m128i*) oN, v0_9_lo);
             v0_9_lo = _mm256_extractf128_si256(v1_9, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            _mm_storeu_si128((__m128i*) oN, v0_8_lo);
             v0_8_lo = _mm256_extractf128_si256(v1_8, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            _mm_storeu_si128((__m128i*) oN, v0_7_lo);
             v0_7_lo = _mm256_castsi256_si128(v1_15);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            _mm_storeu_si128((__m128i*) oN, v0_6_lo);
             v0_6_lo = _mm256_castsi256_si128(v1_14);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            _mm_storeu_si128((__m128i*) oN, v0_5_lo);
             v0_5_lo = _mm256_castsi256_si128(v1_13);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            _mm_storeu_si128((__m128i*) oN, v0_4_lo);
             v0_4_lo = _mm256_castsi256_si128(v1_12);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            _mm_storeu_si128((__m128i*) oN, v0_3_lo);
             v0_3_lo = _mm256_castsi256_si128(v1_11);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            _mm_storeu_si128((__m128i*) oN, v0_2_lo);
             v0_2_lo = _mm256_castsi256_si128(v1_10);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            _mm_storeu_si128((__m128i*) oN, v0_1_lo);
            v0_1_lo = _mm256_castsi256_si128(v1_9);
          case 0:
            _mm_storeu_si128((__m128i*) o, v0_0_lo);
            v0_0_lo = _mm256_castsi256_si128(v1_8);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 8;
      }

      if (bh & 4) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            _mm_storel_epi64((__m128i*) oN, v0_15_lo);
            v0_15_lo = _mm_unpackhi_epi64(v0_15_lo, v0_15_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            _mm_storel_epi64((__m128i*) oN, v0_14_lo);
            v0_14_lo = _mm_unpackhi_epi64(v0_14_lo, v0_14_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            _mm_storel_epi64((__m128i*) oN, v0_13_lo);
            v0_13_lo = _mm_unpackhi_epi64(v0_13_lo, v0_13_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            _mm_storel_epi64((__m128i*) oN, v0_12_lo);
            v0_12_lo = _mm_unpackhi_epi64(v0_12_lo, v0_12_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            _mm_storel_epi64((__m128i*) oN, v0_11_lo);
            v0_11_lo = _mm_unpackhi_epi64(v0_11_lo, v0_11_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            _mm_storel_epi64((__m128i*) oN, v0_10_lo);
            v0_10_lo = _mm_unpackhi_epi64(v0_10_lo, v0_10_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            _mm_storel_epi64((__m128i*) oN, v0_9_lo);
            v0_9_lo = _mm_unpackhi_epi64(v0_9_lo, v0_9_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            _mm_storel_epi64((__m128i*) oN, v0_8_lo);
            v0_8_lo = _mm_unpackhi_epi64(v0_8_lo, v0_8_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            _mm_storel_epi64((__m128i*) oN, v0_7_lo);
            v0_7_lo = _mm_unpackhi_epi64(v0_7_lo, v0_7_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            _mm_storel_epi64((__m128i*) oN, v0_6_lo);
            v0_6_lo = _mm_unpackhi_epi64(v0_6_lo, v0_6_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            _mm_storel_epi64((__m128i*) oN, v0_5_lo);
            v0_5_lo = _mm_unpackhi_epi64(v0_5_lo, v0_5_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            _mm_storel_epi64((__m128i*) oN, v0_4_lo);
            v0_4_lo = _mm_unpackhi_epi64(v0_4_lo, v0_4_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            _mm_storel_epi64((__m128i*) oN, v0_3_lo);
            v0_3_lo = _mm_unpackhi_epi64(v0_3_lo, v0_3_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            _mm_storel_epi64((__m128i*) oN, v0_2_lo);
            v0_2_lo = _mm_unpackhi_epi64(v0_2_lo, v0_2_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            _mm_storel_epi64((__m128i*) oN, v0_1_lo);
            v0_1_lo = _mm_unpackhi_epi64(v0_1_lo, v0_1_lo);
          case 0:
            _mm_storel_epi64((__m128i*) o, v0_0_lo);
            v0_0_lo = _mm_unpackhi_epi64(v0_0_lo, v0_0_lo);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 4;
      }
      if (bh & 2) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            _mm_storeu_si32(oN, v0_15_lo);
            v0_15_lo = _mm_srli_epi64(v0_15_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            _mm_storeu_si32(oN, v0_14_lo);
            v0_14_lo = _mm_srli_epi64(v0_14_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            _mm_storeu_si32(oN, v0_13_lo);
            v0_13_lo = _mm_srli_epi64(v0_13_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            _mm_storeu_si32(oN, v0_12_lo);
            v0_12_lo = _mm_srli_epi64(v0_12_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            _mm_storeu_si32(oN, v0_11_lo);
            v0_11_lo = _mm_srli_epi64(v0_11_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            _mm_storeu_si32(oN, v0_10_lo);
            v0_10_lo = _mm_srli_epi64(v0_10_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            _mm_storeu_si32(oN, v0_9_lo);
            v0_9_lo = _mm_srli_epi64(v0_9_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            _mm_storeu_si32(oN, v0_8_lo);
            v0_8_lo = _mm_srli_epi64(v0_8_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            _mm_storeu_si32(oN, v0_7_lo);
            v0_7_lo = _mm_srli_epi64(v0_7_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            _mm_storeu_si32(oN, v0_6_lo);
            v0_6_lo = _mm_srli_epi64(v0_6_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            _mm_storeu_si32(oN, v0_5_lo);
            v0_5_lo = _mm_srli_epi64(v0_5_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            _mm_storeu_si32(oN, v0_4_lo);
            v0_4_lo = _mm_srli_epi64(v0_4_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            _mm_storeu_si32(oN, v0_3_lo);
            v0_3_lo = _mm_srli_epi64(v0_3_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            _mm_storeu_si32(oN, v0_2_lo);
            v0_2_lo = _mm_srli_epi64(v0_2_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            _mm_storeu_si32(oN, v0_1_lo);
            v0_1_lo = _mm_srli_epi64(v0_1_lo, 32);
          case 0:
            _mm_storeu_si32(o, v0_0_lo);
            v0_0_lo = _mm_srli_epi64(v0_0_lo, 32);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 2;
      }
      if (bh & 1) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_15_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_14_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_13_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_12_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_11_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_10_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_9_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_8_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_7_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_6_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_5_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_4_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_3_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_2_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
             unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_1_lo));
          case 0:
             unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_0_lo));
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    o = (uint16_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x8_lut_ukernel__avx2_x128(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vt0 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) table));
  const __m256i vt1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 16)));
  const __m256i vt2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 32)));
  const __m256i vt3 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 48)));
  const __m256i vt4 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 64)));
  const __m256i vt5 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 80)));
  const __m256i vt6 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 96)));
  const __m256i vt7 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 112)));
  const __m256i vt8 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 128)));
  const __m256i vt9 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 144)));
  const __m256i vtA = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 160)));
  const __m256i vtB = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 176)));
  const __m256i vtC = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 192)));
  const __m256i vtD = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 208)));
  const __m256i vtE = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 224)));
  const __m256i vtF = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) (table + 240)));

  const __m256i vtable0 = vt0;
  const __m256i vtable1 = _mm256_xor_si256(vt0, vt1);
  const __m256i vtable2 = _mm256_xor_si256(vt1, vt2);
  const __m256i vtable3 = _mm256_xor_si256(vt2, vt3);
  const __m256i vtable4 = _mm256_xor_si256(vt3, vt4);
  const __m256i vtable5 = _mm256_xor_si256(vt4, vt5);
  const __m256i vtable6 = _mm256_xor_si256(vt5, vt6);
  const __m256i vtable7 = _mm256_xor_si256(vt6, vt7);
  const __m256i vtable8 = _mm256_xor_si256(_mm256_xor_si256(vt7, vt8), vtable0);
  const __m256i vtable9 = _mm256_xor_si256(_mm256_xor_si256(vt8, vt9), vtable1);
  const __m256i vtableA = _mm256_xor_si256(_mm256_xor_si256(vt9, vtA), vtable2);
  const __m256i vtableB = _mm256_xor_si256(_mm256_xor_si256(vtA, vtB), vtable3);
  const __m256i vtableC = _mm256_xor_si256(_mm256_xor_si256(vtB, vtC), vtable4);
  const __m256i vtableD = _mm256_xor_si256(_mm256_xor_si256(vtC, vtD), vtable5);
  const __m256i vtableE = _mm256_xor_si256(_mm256_xor_si256(vtD, vtE), vtable6);
  const __m256i vtableF = _mm256_xor_si256(_mm256_xor_si256(vtE, vtF), vtable7);

  const __m256i voffset = _mm256_set1_epi8(16);
  for (; batch >= 128 * sizeof(uint8_t); batch -= 128 * sizeof(uint8_t)) {
    __m256i vx0 = _mm256_loadu_si256((const __m256i*) input);
    __m256i vx1 = _mm256_loadu_si256((const __m256i*) (input + 32));
    __m256i vx2 = _mm256_loadu_si256((const __m256i*) (input + 64));
    __m256i vx3 = _mm256_loadu_si256((const __m256i*) (input + 96));
    input += 128;

    __m256i vy0 = _mm256_shuffle_epi8(vtable0, vx0);
    __m256i vy1 = _mm256_shuffle_epi8(vtable0, vx1);
    __m256i vy2 = _mm256_shuffle_epi8(vtable0, vx2);
    __m256i vy3 = _mm256_shuffle_epi8(vtable0, vx3);

    vx0 = _mm256_sub_epi8(vx0, voffset);
    vx1 = _mm256_sub_epi8(vx1, voffset);
    vx2 = _mm256_sub_epi8(vx2, voffset);
    vx3 = _mm256_sub_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable1, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable1, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable1, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable1, vx3));
    vx0 = _mm256_sub_epi8(vx0, voffset);
    vx1 = _mm256_sub_epi8(vx1, voffset);
    vx2 = _mm256_sub_epi8(vx2, voffset);
    vx3 = _mm256_sub_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable2, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable2, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable2, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable2, vx3));
    vx0 = _mm256_sub_epi8(vx0, voffset);
    vx1 = _mm256_sub_epi8(vx1, voffset);
    vx2 = _mm256_sub_epi8(vx2, voffset);
    vx3 = _mm256_sub_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable3, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable3, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable3, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable3, vx3));
    vx0 = _mm256_sub_epi8(vx0, voffset);
    vx1 = _mm256_sub_epi8(vx1, voffset);
    vx2 = _mm256_sub_epi8(vx2, voffset);
    vx3 = _mm256_sub_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable4, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable4, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable4, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable4, vx3));
    vx0 = _mm256_sub_epi8(vx0, voffset);
    vx1 = _mm256_sub_epi8(vx1, voffset);
    vx2 = _mm256_sub_epi8(vx2, voffset);
    vx3 = _mm256_sub_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable5, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable5, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable5, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable5, vx3));
    vx0 = _mm256_sub_epi8(vx0, voffset);
    vx1 = _mm256_sub_epi8(vx1, voffset);
    vx2 = _mm256_sub_epi8(vx2, voffset);
    vx3 = _mm256_sub_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable6, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable6, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable6, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable6, vx3));
    vx0 = _mm256_sub_epi8(vx0, voffset);
    vx1 = _mm256_sub_epi8(vx1, voffset);
    vx2 = _mm256_sub_epi8(vx2, voffset);
    vx3 = _mm256_sub_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable7, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable7, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable7, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable7, vx3));
    vx0 = _mm256_sub_epi8(vx0, voffset);
    vx1 = _mm256_sub_epi8(vx1, voffset);
    vx2 = _mm256_sub_epi8(vx2, voffset);
    vx3 = _mm256_sub_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable8, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable8, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable8, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable8, vx3));

    vx0 = _mm256_subs_epi8(vx0, voffset);
    vx1 = _mm256_subs_epi8(vx1, voffset);
    vx2 = _mm256_subs_epi8(vx2, voffset);
    vx3 = _mm256_subs_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtable9, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtable9, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtable9, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtable9, vx3));
    vx0 = _mm256_subs_epi8(vx0, voffset);
    vx1 = _mm256_subs_epi8(vx1, voffset);
    vx2 = _mm256_subs_epi8(vx2, voffset);
    vx3 = _mm256_subs_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtableA, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtableA, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtableA, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtableA, vx3));
    vx0 = _mm256_subs_epi8(vx0, voffset);
    vx1 = _mm256_subs_epi8(vx1, voffset);
    vx2 = _mm256_subs_epi8(vx2, voffset);
    vx3 = _mm256_subs_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtableB, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtableB, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtableB, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtableB, vx3));
    vx0 = _mm256_subs_epi8(vx0, voffset);
    vx1 = _mm256_subs_epi8(vx1, voffset);
    vx2 = _mm256_subs_epi8(vx2, voffset);
    vx3 = _mm256_subs_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtableC, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtableC, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtableC, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtableC, vx3));
    vx0 = _mm256_subs_epi8(vx0, voffset);
    vx1 = _mm256_subs_epi8(vx1, voffset);
    vx2 = _mm256_subs_epi8(vx2, voffset);
    vx3 = _mm256_subs_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtableD, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtableD, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtableD, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtableD, vx3));
    vx0 = _mm256_subs_epi8(vx0, voffset);
    vx1 = _mm256_subs_epi8(vx1, voffset);
    vx2 = _mm256_subs_epi8(vx2, voffset);
    vx3 = _mm256_subs_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtableE, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtableE, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtableE, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtableE, vx3));
    vx0 = _mm256_subs_epi8(vx0, voffset);
    vx1 = _mm256_subs_epi8(vx1, voffset);
    vx2 = _mm256_subs_epi8(vx2, voffset);
    vx3 = _mm256_subs_epi8(vx3, voffset);
    vy0 = _mm256_xor_si256(vy0, _mm256_shuffle_epi8(vtableF, vx0));
    vy1 = _mm256_xor_si256(vy1, _mm256_shuffle_epi8(vtableF, vx1));
    vy2 = _mm256_xor_si256(vy2, _mm256_shuffle_epi8(vtableF, vx2));
    vy3 = _mm256_xor_si256(vy3, _mm256_shuffle_epi8(vtableF, vx3));

    _mm256_storeu_si256((__m256i*) output, vy0);
    _mm256_storeu_si256((__m256i*) (output + 32), vy1);
    _mm256_storeu_si256((__m256i*) (output + 64), vy2);
    _mm256_storeu_si256((__m256i*) (output + 96), vy3);
    output += 128;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    __m128i vy = _mm_shuffle_epi8(_mm256_castsi256_si128(vtable0), vx);

    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable1), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable2), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable3), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable4), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable5), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable6), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable7), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable8), vx));

    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable9), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableA), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableB), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableC), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableD), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableE), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableF), vx));

    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vx = _mm_loadu_si128((const __m128i*) input);

    __m128i vy = _mm_shuffle_epi8(_mm256_castsi256_si128(vtable0), vx);

    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable1), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable2), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable3), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable4), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable5), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable6), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable7), vx));
    vx = _mm_sub_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable8), vx));

    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtable9), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableA), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableB), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableC), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableD), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableE), vx));
    vx = _mm_subs_epi8(vx, _mm256_castsi256_si128(voffset));
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(_mm256_castsi256_si128(vtableF), vx));

    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_x8_transposec_ukernel__32x32_reuse_switch_avx2(
    const uint8_t* input,
    uint8_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x8_transpose_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_stride >= block_height * sizeof(uint8_t));
  assert(input_stride >= block_width * sizeof(uint8_t));

  const size_t tile_height = 32;
  const size_t tile_width = 32;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  uint8_t* o = (uint8_t*) output;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t);

  const uint8_t* i0 = (const uint8_t*) input;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 31);
    const size_t oN_stride = rem * output_stride;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx2.mask_table[7 ^ (rem>>2)]));

    size_t bh = block_height;
    for (; bh >= 32; bh -= 32) {
      const __m256i v5_0 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_1 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_2 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_3 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_4 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_5 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_6 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_7 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_8 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_9 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_10 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_11 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_12 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_13 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_14 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_15 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_16 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_17 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_18 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_19 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_20 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_21 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_22 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_23 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_24 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_25 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_26 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_27 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_28 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_29 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_30 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_31 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);

      const __m256i v4_0 = _mm256_unpacklo_epi8(v5_0, v5_8);
      const __m256i v4_1 = _mm256_unpackhi_epi8(v5_0, v5_8);
      const __m256i v4_2 = _mm256_unpacklo_epi8(v5_1, v5_9);
      const __m256i v4_3 = _mm256_unpackhi_epi8(v5_1, v5_9);
      const __m256i v4_4 = _mm256_unpacklo_epi8(v5_2, v5_10);
      const __m256i v4_5 = _mm256_unpackhi_epi8(v5_2, v5_10);
      const __m256i v4_6 = _mm256_unpacklo_epi8(v5_3, v5_11);
      const __m256i v4_7 = _mm256_unpackhi_epi8(v5_3, v5_11);
      const __m256i v4_8 = _mm256_unpacklo_epi8(v5_4, v5_12);
      const __m256i v4_9 = _mm256_unpackhi_epi8(v5_4, v5_12);
      const __m256i v4_10 = _mm256_unpacklo_epi8(v5_5, v5_13);
      const __m256i v4_11 = _mm256_unpackhi_epi8(v5_5, v5_13);
      const __m256i v4_12 = _mm256_unpacklo_epi8(v5_6, v5_14);
      const __m256i v4_13 = _mm256_unpackhi_epi8(v5_6, v5_14);
      const __m256i v4_14 = _mm256_unpacklo_epi8(v5_7, v5_15);
      const __m256i v4_15 = _mm256_unpackhi_epi8(v5_7, v5_15);
      const __m256i v4_16 = _mm256_unpacklo_epi8(v5_16, v5_24);
      const __m256i v4_17 = _mm256_unpackhi_epi8(v5_16, v5_24);
      const __m256i v4_18 = _mm256_unpacklo_epi8(v5_17, v5_25);
      const __m256i v4_19 = _mm256_unpackhi_epi8(v5_17, v5_25);
      const __m256i v4_20 = _mm256_unpacklo_epi8(v5_18, v5_26);
      const __m256i v4_21 = _mm256_unpackhi_epi8(v5_18, v5_26);
      const __m256i v4_22 = _mm256_unpacklo_epi8(v5_19, v5_27);
      const __m256i v4_23 = _mm256_unpackhi_epi8(v5_19, v5_27);
      const __m256i v4_24 = _mm256_unpacklo_epi8(v5_20, v5_28);
      const __m256i v4_25 = _mm256_unpackhi_epi8(v5_20, v5_28);
      const __m256i v4_26 = _mm256_unpacklo_epi8(v5_21, v5_29);
      const __m256i v4_27 = _mm256_unpackhi_epi8(v5_21, v5_29);
      const __m256i v4_28 = _mm256_unpacklo_epi8(v5_22, v5_30);
      const __m256i v4_29 = _mm256_unpackhi_epi8(v5_22, v5_30);
      const __m256i v4_30 = _mm256_unpacklo_epi8(v5_23, v5_31);
      const __m256i v4_31 = _mm256_unpackhi_epi8(v5_23, v5_31);
      const __m256i v3_0 = _mm256_unpacklo_epi8(v4_0, v4_8);
      const __m256i v3_1 = _mm256_unpackhi_epi8(v4_0, v4_8);
      const __m256i v3_2 = _mm256_unpacklo_epi8(v4_1, v4_9);
      const __m256i v3_3 = _mm256_unpackhi_epi8(v4_1, v4_9);
      const __m256i v3_4 = _mm256_unpacklo_epi8(v4_2, v4_10);
      const __m256i v3_5 = _mm256_unpackhi_epi8(v4_2, v4_10);
      const __m256i v3_6 = _mm256_unpacklo_epi8(v4_3, v4_11);
      const __m256i v3_7 = _mm256_unpackhi_epi8(v4_3, v4_11);
      const __m256i v3_8 = _mm256_unpacklo_epi8(v4_4, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi8(v4_4, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi8(v4_5, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi8(v4_5, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi8(v4_6, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi8(v4_6, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi8(v4_7, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi8(v4_7, v4_15);
      const __m256i v3_16 = _mm256_unpacklo_epi8(v4_16, v4_24);
      const __m256i v3_17 = _mm256_unpackhi_epi8(v4_16, v4_24);
      const __m256i v3_18 = _mm256_unpacklo_epi8(v4_17, v4_25);
      const __m256i v3_19 = _mm256_unpackhi_epi8(v4_17, v4_25);
      const __m256i v3_20 = _mm256_unpacklo_epi8(v4_18, v4_26);
      const __m256i v3_21 = _mm256_unpackhi_epi8(v4_18, v4_26);
      const __m256i v3_22 = _mm256_unpacklo_epi8(v4_19, v4_27);
      const __m256i v3_23 = _mm256_unpackhi_epi8(v4_19, v4_27);
      const __m256i v3_24 = _mm256_unpacklo_epi8(v4_20, v4_28);
      const __m256i v3_25 = _mm256_unpackhi_epi8(v4_20, v4_28);
      const __m256i v3_26 = _mm256_unpacklo_epi8(v4_21, v4_29);
      const __m256i v3_27 = _mm256_unpackhi_epi8(v4_21, v4_29);
      const __m256i v3_28 = _mm256_unpacklo_epi8(v4_22, v4_30);
      const __m256i v3_29 = _mm256_unpackhi_epi8(v4_22, v4_30);
      const __m256i v3_30 = _mm256_unpacklo_epi8(v4_23, v4_31);
      const __m256i v3_31 = _mm256_unpackhi_epi8(v4_23, v4_31);
      const __m256i v2_0 = _mm256_unpacklo_epi8(v3_0, v3_8);
      const __m256i v2_1 = _mm256_unpackhi_epi8(v3_0, v3_8);
      const __m256i v2_2 = _mm256_unpacklo_epi8(v3_1, v3_9);
      const __m256i v2_3 = _mm256_unpackhi_epi8(v3_1, v3_9);
      const __m256i v2_4 = _mm256_unpacklo_epi8(v3_2, v3_10);
      const __m256i v2_5 = _mm256_unpackhi_epi8(v3_2, v3_10);
      const __m256i v2_6 = _mm256_unpacklo_epi8(v3_3, v3_11);
      const __m256i v2_7 = _mm256_unpackhi_epi8(v3_3, v3_11);
      const __m256i v2_8 = _mm256_unpacklo_epi8(v3_4, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi8(v3_4, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi8(v3_5, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi8(v3_5, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi8(v3_6, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi8(v3_6, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi8(v3_7, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi8(v3_7, v3_15);
      const __m256i v2_16 = _mm256_unpacklo_epi8(v3_16, v3_24);
      const __m256i v2_17 = _mm256_unpackhi_epi8(v3_16, v3_24);
      const __m256i v2_18 = _mm256_unpacklo_epi8(v3_17, v3_25);
      const __m256i v2_19 = _mm256_unpackhi_epi8(v3_17, v3_25);
      const __m256i v2_20 = _mm256_unpacklo_epi8(v3_18, v3_26);
      const __m256i v2_21 = _mm256_unpackhi_epi8(v3_18, v3_26);
      const __m256i v2_22 = _mm256_unpacklo_epi8(v3_19, v3_27);
      const __m256i v2_23 = _mm256_unpackhi_epi8(v3_19, v3_27);
      const __m256i v2_24 = _mm256_unpacklo_epi8(v3_20, v3_28);
      const __m256i v2_25 = _mm256_unpackhi_epi8(v3_20, v3_28);
      const __m256i v2_26 = _mm256_unpacklo_epi8(v3_21, v3_29);
      const __m256i v2_27 = _mm256_unpackhi_epi8(v3_21, v3_29);
      const __m256i v2_28 = _mm256_unpacklo_epi8(v3_22, v3_30);
      const __m256i v2_29 = _mm256_unpackhi_epi8(v3_22, v3_30);
      const __m256i v2_30 = _mm256_unpacklo_epi8(v3_23, v3_31);
      const __m256i v2_31 = _mm256_unpackhi_epi8(v3_23, v3_31);
      const __m256i v1_0 = _mm256_unpacklo_epi8(v2_0, v2_8);
      const __m256i v1_1 = _mm256_unpackhi_epi8(v2_0, v2_8);
      const __m256i v1_2 = _mm256_unpacklo_epi8(v2_1, v2_9);
      const __m256i v1_3 = _mm256_unpackhi_epi8(v2_1, v2_9);
      const __m256i v1_4 = _mm256_unpacklo_epi8(v2_2, v2_10);
      const __m256i v1_5 = _mm256_unpackhi_epi8(v2_2, v2_10);
      const __m256i v1_6 = _mm256_unpacklo_epi8(v2_3, v2_11);
      const __m256i v1_7 = _mm256_unpackhi_epi8(v2_3, v2_11);
      const __m256i v1_8 = _mm256_unpacklo_epi8(v2_4, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi8(v2_4, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi8(v2_5, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi8(v2_5, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi8(v2_6, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi8(v2_6, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi8(v2_7, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi8(v2_7, v2_15);
      const __m256i v1_16 = _mm256_unpacklo_epi8(v2_16, v2_24);
      const __m256i v1_17 = _mm256_unpackhi_epi8(v2_16, v2_24);
      const __m256i v1_18 = _mm256_unpacklo_epi8(v2_17, v2_25);
      const __m256i v1_19 = _mm256_unpackhi_epi8(v2_17, v2_25);
      const __m256i v1_20 = _mm256_unpacklo_epi8(v2_18, v2_26);
      const __m256i v1_21 = _mm256_unpackhi_epi8(v2_18, v2_26);
      const __m256i v1_22 = _mm256_unpacklo_epi8(v2_19, v2_27);
      const __m256i v1_23 = _mm256_unpackhi_epi8(v2_19, v2_27);
      const __m256i v1_24 = _mm256_unpacklo_epi8(v2_20, v2_28);
      const __m256i v1_25 = _mm256_unpackhi_epi8(v2_20, v2_28);
      const __m256i v1_26 = _mm256_unpacklo_epi8(v2_21, v2_29);
      const __m256i v1_27 = _mm256_unpackhi_epi8(v2_21, v2_29);
      const __m256i v1_28 = _mm256_unpacklo_epi8(v2_22, v2_30);
      const __m256i v1_29 = _mm256_unpackhi_epi8(v2_22, v2_30);
      const __m256i v1_30 = _mm256_unpacklo_epi8(v2_23, v2_31);
      const __m256i v1_31 = _mm256_unpackhi_epi8(v2_23, v2_31);


      uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        default:
          XNN_UNREACHABLE;
        case 31: {
          const __m256i v0_31 = _mm256_permute2f128_si256(v1_15, v1_31, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_31);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 30: {
          const __m256i v0_30 = _mm256_permute2f128_si256(v1_14, v1_30, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_30);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 29: {
          const __m256i v0_29 = _mm256_permute2f128_si256(v1_13, v1_29, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_29);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 28: {
          const __m256i v0_28 = _mm256_permute2f128_si256(v1_12, v1_28, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_28);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 27: {
          const __m256i v0_27 = _mm256_permute2f128_si256(v1_11, v1_27, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_27);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 26: {
          const __m256i v0_26 = _mm256_permute2f128_si256(v1_10, v1_26, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_26);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 25: {
          const __m256i v0_25 = _mm256_permute2f128_si256(v1_9, v1_25, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_25);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 24: {
          const __m256i v0_24 = _mm256_permute2f128_si256(v1_8, v1_24, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_24);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 23: {
          const __m256i v0_23 = _mm256_permute2f128_si256(v1_7, v1_23, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_23);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 22: {
          const __m256i v0_22 = _mm256_permute2f128_si256(v1_6, v1_22, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_22);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 21: {
          const __m256i v0_21 = _mm256_permute2f128_si256(v1_5, v1_21, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_21);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 20: {
          const __m256i v0_20 = _mm256_permute2f128_si256(v1_4, v1_20, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_20);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 19: {
          const __m256i v0_19 = _mm256_permute2f128_si256(v1_3, v1_19, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_19);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 18: {
          const __m256i v0_18 = _mm256_permute2f128_si256(v1_2, v1_18, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_18);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 17: {
          const __m256i v0_17 = _mm256_permute2f128_si256(v1_1, v1_17, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_17);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 16: {
          const __m256i v0_16 = _mm256_permute2f128_si256(v1_0, v1_16, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_16);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 15: {
          const __m256i v0_15 = _mm256_insertf128_si256(v1_15, _mm256_castsi256_si128(v1_31), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_15);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 14: {
          const __m256i v0_14 = _mm256_insertf128_si256(v1_14, _mm256_castsi256_si128(v1_30), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_14);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 13: {
          const __m256i v0_13 = _mm256_insertf128_si256(v1_13, _mm256_castsi256_si128(v1_29), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_13);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 12: {
          const __m256i v0_12 = _mm256_insertf128_si256(v1_12, _mm256_castsi256_si128(v1_28), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_12);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 11: {
          const __m256i v0_11 = _mm256_insertf128_si256(v1_11, _mm256_castsi256_si128(v1_27), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_11);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 10: {
          const __m256i v0_10 = _mm256_insertf128_si256(v1_10, _mm256_castsi256_si128(v1_26), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_10);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 9: {
          const __m256i v0_9 = _mm256_insertf128_si256(v1_9, _mm256_castsi256_si128(v1_25), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_9);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 8: {
          const __m256i v0_8 = _mm256_insertf128_si256(v1_8, _mm256_castsi256_si128(v1_24), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_8);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 7: {
          const __m256i v0_7 = _mm256_insertf128_si256(v1_7, _mm256_castsi256_si128(v1_23), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_7);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 6: {
          const __m256i v0_6 = _mm256_insertf128_si256(v1_6, _mm256_castsi256_si128(v1_22), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_6);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 5: {
          const __m256i v0_5 = _mm256_insertf128_si256(v1_5, _mm256_castsi256_si128(v1_21), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_5);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 4: {
          const __m256i v0_4 = _mm256_insertf128_si256(v1_4, _mm256_castsi256_si128(v1_20), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_4);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 3: {
          const __m256i v0_3 = _mm256_insertf128_si256(v1_3, _mm256_castsi256_si128(v1_19), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_3);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 2: {
          const __m256i v0_2 = _mm256_insertf128_si256(v1_2, _mm256_castsi256_si128(v1_18), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_2);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
        }
        case 1: {
          const __m256i v0_1 = _mm256_insertf128_si256(v1_1, _mm256_castsi256_si128(v1_17), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_1);
        }
        case 0: {
          const __m256i v0_0 = _mm256_insertf128_si256(v1_0, _mm256_castsi256_si128(v1_16), 1);
          _mm256_storeu_si256((__m256i*) o, v0_0);
          o = (uint8_t*) ((uintptr_t) o + tile_hbytes);
        }
      }
    }
    if (bh != 0) {
      const __m256i v5_0 = _mm256_maskload_epi32((const int*) i0, vmask);
      const uint8_t *i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256i v5_1 = _mm256_maskload_epi32((const int*) i1, vmask);
      const uint8_t *i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256i v5_2 = _mm256_maskload_epi32((const int*) i2, vmask);
      const uint8_t *i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m256i v5_3 = _mm256_maskload_epi32((const int*) i3, vmask);
      const uint8_t *i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m256i v5_4 = _mm256_maskload_epi32((const int*) i4, vmask);
      const uint8_t *i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m256i v5_5 = _mm256_maskload_epi32((const int*) i5, vmask);
      const uint8_t *i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m256i v5_6 = _mm256_maskload_epi32((const int*) i6, vmask);
      const uint8_t *i7 = (const uint8_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const __m256i v5_7 = _mm256_maskload_epi32((const int*) i7, vmask);
      const uint8_t *i8 = (const uint8_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const __m256i v5_8 = _mm256_maskload_epi32((const int*) i8, vmask);
      const uint8_t *i9 = (const uint8_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const __m256i v5_9 = _mm256_maskload_epi32((const int*) i9, vmask);
      const uint8_t *i10 = (const uint8_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const __m256i v5_10 = _mm256_maskload_epi32((const int*) i10, vmask);
      const uint8_t *i11 = (const uint8_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const __m256i v5_11 = _mm256_maskload_epi32((const int*) i11, vmask);
      const uint8_t *i12 = (const uint8_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const __m256i v5_12 = _mm256_maskload_epi32((const int*) i12, vmask);
      const uint8_t *i13 = (const uint8_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const __m256i v5_13 = _mm256_maskload_epi32((const int*) i13, vmask);
      const uint8_t *i14 = (const uint8_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const __m256i v5_14 = _mm256_maskload_epi32((const int*) i14, vmask);
      const uint8_t *i15 = (const uint8_t*) ((uintptr_t) i14 + input_stride);
      if XNN_UNPREDICTABLE(bh < 16) {
        i15 = i14;
      }
      const __m256i v5_15 = _mm256_maskload_epi32((const int*) i15, vmask);
      const uint8_t *i16 = (const uint8_t*) ((uintptr_t) i15 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 16) {
        i16 = i15;
      }
      const __m256i v5_16 = _mm256_maskload_epi32((const int*) i16, vmask);
      const uint8_t *i17 = (const uint8_t*) ((uintptr_t) i16 + input_stride);
      if XNN_UNPREDICTABLE(bh < 18) {
        i17 = i16;
      }
      const __m256i v5_17 = _mm256_maskload_epi32((const int*) i17, vmask);
      const uint8_t *i18 = (const uint8_t*) ((uintptr_t) i17 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 18) {
        i18 = i17;
      }
      const __m256i v5_18 = _mm256_maskload_epi32((const int*) i18, vmask);
      const uint8_t *i19 = (const uint8_t*) ((uintptr_t) i18 + input_stride);
      if XNN_UNPREDICTABLE(bh < 20) {
        i19 = i18;
      }
      const __m256i v5_19 = _mm256_maskload_epi32((const int*) i19, vmask);
      const uint8_t *i20 = (const uint8_t*) ((uintptr_t) i19 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 20) {
        i20 = i19;
      }
      const __m256i v5_20 = _mm256_maskload_epi32((const int*) i20, vmask);
      const uint8_t *i21 = (const uint8_t*) ((uintptr_t) i20 + input_stride);
      if XNN_UNPREDICTABLE(bh < 22) {
        i21 = i20;
      }
      const __m256i v5_21 = _mm256_maskload_epi32((const int*) i21, vmask);
      const uint8_t *i22 = (const uint8_t*) ((uintptr_t) i21 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 22) {
        i22 = i21;
      }
      const __m256i v5_22 = _mm256_maskload_epi32((const int*) i22, vmask);
      const uint8_t *i23 = (const uint8_t*) ((uintptr_t) i22 + input_stride);
      if XNN_UNPREDICTABLE(bh < 24) {
        i23 = i22;
      }
      const __m256i v5_23 = _mm256_maskload_epi32((const int*) i23, vmask);
      const uint8_t *i24 = (const uint8_t*) ((uintptr_t) i23 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 24) {
        i24 = i23;
      }
      const __m256i v5_24 = _mm256_maskload_epi32((const int*) i24, vmask);
      const uint8_t *i25 = (const uint8_t*) ((uintptr_t) i24 + input_stride);
      if XNN_UNPREDICTABLE(bh < 26) {
        i25 = i24;
      }
      const __m256i v5_25 = _mm256_maskload_epi32((const int*) i25, vmask);
      const uint8_t *i26 = (const uint8_t*) ((uintptr_t) i25 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 26) {
        i26 = i25;
      }
      const __m256i v5_26 = _mm256_maskload_epi32((const int*) i26, vmask);
      const uint8_t *i27 = (const uint8_t*) ((uintptr_t) i26 + input_stride);
      if XNN_UNPREDICTABLE(bh < 28) {
        i27 = i26;
      }
      const __m256i v5_27 = _mm256_maskload_epi32((const int*) i27, vmask);
      const uint8_t *i28 = (const uint8_t*) ((uintptr_t) i27 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 28) {
        i28 = i27;
      }
      const __m256i v5_28 = _mm256_maskload_epi32((const int*) i28, vmask);
      const uint8_t *i29 = (const uint8_t*) ((uintptr_t) i28 + input_stride);
      if XNN_UNPREDICTABLE(bh < 30) {
        i29 = i28;
      }
      const __m256i v5_29 = _mm256_maskload_epi32((const int*) i29, vmask);
      const uint8_t *i30 = (const uint8_t*) ((uintptr_t) i29 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 30) {
        i30 = i29;
      }
      const __m256i v5_30 = _mm256_maskload_epi32((const int*) i30, vmask);
      const __m256i v5_31 = _mm256_undefined_si256();

      const __m256i v4_0 = _mm256_unpacklo_epi8(v5_0, v5_8);
      const __m256i v4_1 = _mm256_unpackhi_epi8(v5_0, v5_8);
      const __m256i v4_2 = _mm256_unpacklo_epi8(v5_1, v5_9);
      const __m256i v4_3 = _mm256_unpackhi_epi8(v5_1, v5_9);
      const __m256i v4_4 = _mm256_unpacklo_epi8(v5_2, v5_10);
      const __m256i v4_5 = _mm256_unpackhi_epi8(v5_2, v5_10);
      const __m256i v4_6 = _mm256_unpacklo_epi8(v5_3, v5_11);
      const __m256i v4_7 = _mm256_unpackhi_epi8(v5_3, v5_11);
      const __m256i v4_8 = _mm256_unpacklo_epi8(v5_4, v5_12);
      const __m256i v4_9 = _mm256_unpackhi_epi8(v5_4, v5_12);
      const __m256i v4_10 = _mm256_unpacklo_epi8(v5_5, v5_13);
      const __m256i v4_11 = _mm256_unpackhi_epi8(v5_5, v5_13);
      const __m256i v4_12 = _mm256_unpacklo_epi8(v5_6, v5_14);
      const __m256i v4_13 = _mm256_unpackhi_epi8(v5_6, v5_14);
      const __m256i v4_14 = _mm256_unpacklo_epi8(v5_7, v5_15);
      const __m256i v4_15 = _mm256_unpackhi_epi8(v5_7, v5_15);
      const __m256i v4_16 = _mm256_unpacklo_epi8(v5_16, v5_24);
      const __m256i v4_17 = _mm256_unpackhi_epi8(v5_16, v5_24);
      const __m256i v4_18 = _mm256_unpacklo_epi8(v5_17, v5_25);
      const __m256i v4_19 = _mm256_unpackhi_epi8(v5_17, v5_25);
      const __m256i v4_20 = _mm256_unpacklo_epi8(v5_18, v5_26);
      const __m256i v4_21 = _mm256_unpackhi_epi8(v5_18, v5_26);
      const __m256i v4_22 = _mm256_unpacklo_epi8(v5_19, v5_27);
      const __m256i v4_23 = _mm256_unpackhi_epi8(v5_19, v5_27);
      const __m256i v4_24 = _mm256_unpacklo_epi8(v5_20, v5_28);
      const __m256i v4_25 = _mm256_unpackhi_epi8(v5_20, v5_28);
      const __m256i v4_26 = _mm256_unpacklo_epi8(v5_21, v5_29);
      const __m256i v4_27 = _mm256_unpackhi_epi8(v5_21, v5_29);
      const __m256i v4_28 = _mm256_unpacklo_epi8(v5_22, v5_30);
      const __m256i v4_29 = _mm256_unpackhi_epi8(v5_22, v5_30);
      const __m256i v4_30 = _mm256_unpacklo_epi8(v5_23, v5_31);
      const __m256i v4_31 = _mm256_unpackhi_epi8(v5_23, v5_31);
      const __m256i v3_0 = _mm256_unpacklo_epi8(v4_0, v4_8);
      const __m256i v3_1 = _mm256_unpackhi_epi8(v4_0, v4_8);
      const __m256i v3_2 = _mm256_unpacklo_epi8(v4_1, v4_9);
      const __m256i v3_3 = _mm256_unpackhi_epi8(v4_1, v4_9);
      const __m256i v3_4 = _mm256_unpacklo_epi8(v4_2, v4_10);
      const __m256i v3_5 = _mm256_unpackhi_epi8(v4_2, v4_10);
      const __m256i v3_6 = _mm256_unpacklo_epi8(v4_3, v4_11);
      const __m256i v3_7 = _mm256_unpackhi_epi8(v4_3, v4_11);
      const __m256i v3_8 = _mm256_unpacklo_epi8(v4_4, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi8(v4_4, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi8(v4_5, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi8(v4_5, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi8(v4_6, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi8(v4_6, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi8(v4_7, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi8(v4_7, v4_15);
      const __m256i v3_16 = _mm256_unpacklo_epi8(v4_16, v4_24);
      const __m256i v3_17 = _mm256_unpackhi_epi8(v4_16, v4_24);
      const __m256i v3_18 = _mm256_unpacklo_epi8(v4_17, v4_25);
      const __m256i v3_19 = _mm256_unpackhi_epi8(v4_17, v4_25);
      const __m256i v3_20 = _mm256_unpacklo_epi8(v4_18, v4_26);
      const __m256i v3_21 = _mm256_unpackhi_epi8(v4_18, v4_26);
      const __m256i v3_22 = _mm256_unpacklo_epi8(v4_19, v4_27);
      const __m256i v3_23 = _mm256_unpackhi_epi8(v4_19, v4_27);
      const __m256i v3_24 = _mm256_unpacklo_epi8(v4_20, v4_28);
      const __m256i v3_25 = _mm256_unpackhi_epi8(v4_20, v4_28);
      const __m256i v3_26 = _mm256_unpacklo_epi8(v4_21, v4_29);
      const __m256i v3_27 = _mm256_unpackhi_epi8(v4_21, v4_29);
      const __m256i v3_28 = _mm256_unpacklo_epi8(v4_22, v4_30);
      const __m256i v3_29 = _mm256_unpackhi_epi8(v4_22, v4_30);
      const __m256i v3_30 = _mm256_unpacklo_epi8(v4_23, v4_31);
      const __m256i v3_31 = _mm256_unpackhi_epi8(v4_23, v4_31);
      const __m256i v2_0 = _mm256_unpacklo_epi8(v3_0, v3_8);
      const __m256i v2_1 = _mm256_unpackhi_epi8(v3_0, v3_8);
      const __m256i v2_2 = _mm256_unpacklo_epi8(v3_1, v3_9);
      const __m256i v2_3 = _mm256_unpackhi_epi8(v3_1, v3_9);
      const __m256i v2_4 = _mm256_unpacklo_epi8(v3_2, v3_10);
      const __m256i v2_5 = _mm256_unpackhi_epi8(v3_2, v3_10);
      const __m256i v2_6 = _mm256_unpacklo_epi8(v3_3, v3_11);
      const __m256i v2_7 = _mm256_unpackhi_epi8(v3_3, v3_11);
      const __m256i v2_8 = _mm256_unpacklo_epi8(v3_4, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi8(v3_4, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi8(v3_5, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi8(v3_5, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi8(v3_6, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi8(v3_6, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi8(v3_7, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi8(v3_7, v3_15);
      const __m256i v2_16 = _mm256_unpacklo_epi8(v3_16, v3_24);
      const __m256i v2_17 = _mm256_unpackhi_epi8(v3_16, v3_24);
      const __m256i v2_18 = _mm256_unpacklo_epi8(v3_17, v3_25);
      const __m256i v2_19 = _mm256_unpackhi_epi8(v3_17, v3_25);
      const __m256i v2_20 = _mm256_unpacklo_epi8(v3_18, v3_26);
      const __m256i v2_21 = _mm256_unpackhi_epi8(v3_18, v3_26);
      const __m256i v2_22 = _mm256_unpacklo_epi8(v3_19, v3_27);
      const __m256i v2_23 = _mm256_unpackhi_epi8(v3_19, v3_27);
      const __m256i v2_24 = _mm256_unpacklo_epi8(v3_20, v3_28);
      const __m256i v2_25 = _mm256_unpackhi_epi8(v3_20, v3_28);
      const __m256i v2_26 = _mm256_unpacklo_epi8(v3_21, v3_29);
      const __m256i v2_27 = _mm256_unpackhi_epi8(v3_21, v3_29);
      const __m256i v2_28 = _mm256_unpacklo_epi8(v3_22, v3_30);
      const __m256i v2_29 = _mm256_unpackhi_epi8(v3_22, v3_30);
      const __m256i v2_30 = _mm256_unpacklo_epi8(v3_23, v3_31);
      const __m256i v2_31 = _mm256_unpackhi_epi8(v3_23, v3_31);
      const __m256i v1_0 = _mm256_unpacklo_epi8(v2_0, v2_8);
      const __m256i v1_1 = _mm256_unpackhi_epi8(v2_0, v2_8);
      const __m256i v1_2 = _mm256_unpacklo_epi8(v2_1, v2_9);
      const __m256i v1_3 = _mm256_unpackhi_epi8(v2_1, v2_9);
      const __m256i v1_4 = _mm256_unpacklo_epi8(v2_2, v2_10);
      const __m256i v1_5 = _mm256_unpackhi_epi8(v2_2, v2_10);
      const __m256i v1_6 = _mm256_unpacklo_epi8(v2_3, v2_11);
      const __m256i v1_7 = _mm256_unpackhi_epi8(v2_3, v2_11);
      const __m256i v1_8 = _mm256_unpacklo_epi8(v2_4, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi8(v2_4, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi8(v2_5, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi8(v2_5, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi8(v2_6, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi8(v2_6, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi8(v2_7, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi8(v2_7, v2_15);
      const __m256i v1_16 = _mm256_unpacklo_epi8(v2_16, v2_24);
      const __m256i v1_17 = _mm256_unpackhi_epi8(v2_16, v2_24);
      const __m256i v1_18 = _mm256_unpacklo_epi8(v2_17, v2_25);
      const __m256i v1_19 = _mm256_unpackhi_epi8(v2_17, v2_25);
      const __m256i v1_20 = _mm256_unpacklo_epi8(v2_18, v2_26);
      const __m256i v1_21 = _mm256_unpackhi_epi8(v2_18, v2_26);
      const __m256i v1_22 = _mm256_unpacklo_epi8(v2_19, v2_27);
      const __m256i v1_23 = _mm256_unpackhi_epi8(v2_19, v2_27);
      const __m256i v1_24 = _mm256_unpacklo_epi8(v2_20, v2_28);
      const __m256i v1_25 = _mm256_unpackhi_epi8(v2_20, v2_28);
      const __m256i v1_26 = _mm256_unpacklo_epi8(v2_21, v2_29);
      const __m256i v1_27 = _mm256_unpackhi_epi8(v2_21, v2_29);
      const __m256i v1_28 = _mm256_unpacklo_epi8(v2_22, v2_30);
      const __m256i v1_29 = _mm256_unpackhi_epi8(v2_22, v2_30);
      const __m256i v1_30 = _mm256_unpacklo_epi8(v2_23, v2_31);
      const __m256i v1_31 = _mm256_unpackhi_epi8(v2_23, v2_31);

      __m128i v0_0_lo = _mm256_castsi256_si128(v1_0);
      __m128i v0_1_lo = _mm256_castsi256_si128(v1_1);
      __m128i v0_2_lo = _mm256_castsi256_si128(v1_2);
      __m128i v0_3_lo = _mm256_castsi256_si128(v1_3);
      __m128i v0_4_lo = _mm256_castsi256_si128(v1_4);
      __m128i v0_5_lo = _mm256_castsi256_si128(v1_5);
      __m128i v0_6_lo = _mm256_castsi256_si128(v1_6);
      __m128i v0_7_lo = _mm256_castsi256_si128(v1_7);
      __m128i v0_8_lo = _mm256_castsi256_si128(v1_8);
      __m128i v0_9_lo = _mm256_castsi256_si128(v1_9);
      __m128i v0_10_lo = _mm256_castsi256_si128(v1_10);
      __m128i v0_11_lo = _mm256_castsi256_si128(v1_11);
      __m128i v0_12_lo = _mm256_castsi256_si128(v1_12);
      __m128i v0_13_lo = _mm256_castsi256_si128(v1_13);
      __m128i v0_14_lo = _mm256_castsi256_si128(v1_14);
      __m128i v0_15_lo = _mm256_castsi256_si128(v1_15);
      __m128i v0_16_lo = _mm256_extractf128_si256(v1_0, 0x1);
      __m128i v0_17_lo = _mm256_extractf128_si256(v1_1, 0x1);
      __m128i v0_18_lo = _mm256_extractf128_si256(v1_2, 0x1);
      __m128i v0_19_lo = _mm256_extractf128_si256(v1_3, 0x1);
      __m128i v0_20_lo = _mm256_extractf128_si256(v1_4, 0x1);
      __m128i v0_21_lo = _mm256_extractf128_si256(v1_5, 0x1);
      __m128i v0_22_lo = _mm256_extractf128_si256(v1_6, 0x1);
      __m128i v0_23_lo = _mm256_extractf128_si256(v1_7, 0x1);
      __m128i v0_24_lo = _mm256_extractf128_si256(v1_8, 0x1);
      __m128i v0_25_lo = _mm256_extractf128_si256(v1_9, 0x1);
      __m128i v0_26_lo = _mm256_extractf128_si256(v1_10, 0x1);
      __m128i v0_27_lo = _mm256_extractf128_si256(v1_11, 0x1);
      __m128i v0_28_lo = _mm256_extractf128_si256(v1_12, 0x1);
      __m128i v0_29_lo = _mm256_extractf128_si256(v1_13, 0x1);
      __m128i v0_30_lo = _mm256_extractf128_si256(v1_14, 0x1);
      __m128i v0_31_lo = _mm256_extractf128_si256(v1_15, 0x1);

      if (bh & 16) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 31:
            _mm_storeu_si128((__m128i*) oN, v0_31_lo);
             v0_31_lo = _mm256_extractf128_si256(v1_31, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 30:
            _mm_storeu_si128((__m128i*) oN, v0_30_lo);
             v0_30_lo = _mm256_extractf128_si256(v1_30, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 29:
            _mm_storeu_si128((__m128i*) oN, v0_29_lo);
             v0_29_lo = _mm256_extractf128_si256(v1_29, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 28:
            _mm_storeu_si128((__m128i*) oN, v0_28_lo);
             v0_28_lo = _mm256_extractf128_si256(v1_28, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 27:
            _mm_storeu_si128((__m128i*) oN, v0_27_lo);
             v0_27_lo = _mm256_extractf128_si256(v1_27, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 26:
            _mm_storeu_si128((__m128i*) oN, v0_26_lo);
             v0_26_lo = _mm256_extractf128_si256(v1_26, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 25:
            _mm_storeu_si128((__m128i*) oN, v0_25_lo);
             v0_25_lo = _mm256_extractf128_si256(v1_25, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 24:
            _mm_storeu_si128((__m128i*) oN, v0_24_lo);
             v0_24_lo = _mm256_extractf128_si256(v1_24, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 23:
            _mm_storeu_si128((__m128i*) oN, v0_23_lo);
             v0_23_lo = _mm256_extractf128_si256(v1_23, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 22:
            _mm_storeu_si128((__m128i*) oN, v0_22_lo);
             v0_22_lo = _mm256_extractf128_si256(v1_22, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 21:
            _mm_storeu_si128((__m128i*) oN, v0_21_lo);
             v0_21_lo = _mm256_extractf128_si256(v1_21, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 20:
            _mm_storeu_si128((__m128i*) oN, v0_20_lo);
             v0_20_lo = _mm256_extractf128_si256(v1_20, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 19:
            _mm_storeu_si128((__m128i*) oN, v0_19_lo);
             v0_19_lo = _mm256_extractf128_si256(v1_19, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 18:
            _mm_storeu_si128((__m128i*) oN, v0_18_lo);
             v0_18_lo = _mm256_extractf128_si256(v1_18, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 17:
            _mm_storeu_si128((__m128i*) oN, v0_17_lo);
             v0_17_lo = _mm256_extractf128_si256(v1_17, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 16:
            _mm_storeu_si128((__m128i*) oN, v0_16_lo);
             v0_16_lo = _mm256_extractf128_si256(v1_16, 0x1);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 15:
            _mm_storeu_si128((__m128i*) oN, v0_15_lo);
             v0_15_lo = _mm256_castsi256_si128(v1_31);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            _mm_storeu_si128((__m128i*) oN, v0_14_lo);
             v0_14_lo = _mm256_castsi256_si128(v1_30);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            _mm_storeu_si128((__m128i*) oN, v0_13_lo);
             v0_13_lo = _mm256_castsi256_si128(v1_29);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            _mm_storeu_si128((__m128i*) oN, v0_12_lo);
             v0_12_lo = _mm256_castsi256_si128(v1_28);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            _mm_storeu_si128((__m128i*) oN, v0_11_lo);
             v0_11_lo = _mm256_castsi256_si128(v1_27);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            _mm_storeu_si128((__m128i*) oN, v0_10_lo);
             v0_10_lo = _mm256_castsi256_si128(v1_26);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            _mm_storeu_si128((__m128i*) oN, v0_9_lo);
             v0_9_lo = _mm256_castsi256_si128(v1_25);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            _mm_storeu_si128((__m128i*) oN, v0_8_lo);
             v0_8_lo = _mm256_castsi256_si128(v1_24);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            _mm_storeu_si128((__m128i*) oN, v0_7_lo);
             v0_7_lo = _mm256_castsi256_si128(v1_23);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            _mm_storeu_si128((__m128i*) oN, v0_6_lo);
             v0_6_lo = _mm256_castsi256_si128(v1_22);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            _mm_storeu_si128((__m128i*) oN, v0_5_lo);
             v0_5_lo = _mm256_castsi256_si128(v1_21);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            _mm_storeu_si128((__m128i*) oN, v0_4_lo);
             v0_4_lo = _mm256_castsi256_si128(v1_20);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            _mm_storeu_si128((__m128i*) oN, v0_3_lo);
             v0_3_lo = _mm256_castsi256_si128(v1_19);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            _mm_storeu_si128((__m128i*) oN, v0_2_lo);
             v0_2_lo = _mm256_castsi256_si128(v1_18);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            _mm_storeu_si128((__m128i*) oN, v0_1_lo);
            v0_1_lo = _mm256_castsi256_si128(v1_17);
          case 0:
            _mm_storeu_si128((__m128i*) o, v0_0_lo);
            v0_0_lo = _mm256_castsi256_si128(v1_16);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 16;
      }

      if (bh & 8) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 31:
            _mm_storel_epi64((__m128i*) oN, v0_31_lo);
            v0_31_lo = _mm_unpackhi_epi64(v0_31_lo, v0_31_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 30:
            _mm_storel_epi64((__m128i*) oN, v0_30_lo);
            v0_30_lo = _mm_unpackhi_epi64(v0_30_lo, v0_30_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 29:
            _mm_storel_epi64((__m128i*) oN, v0_29_lo);
            v0_29_lo = _mm_unpackhi_epi64(v0_29_lo, v0_29_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 28:
            _mm_storel_epi64((__m128i*) oN, v0_28_lo);
            v0_28_lo = _mm_unpackhi_epi64(v0_28_lo, v0_28_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 27:
            _mm_storel_epi64((__m128i*) oN, v0_27_lo);
            v0_27_lo = _mm_unpackhi_epi64(v0_27_lo, v0_27_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 26:
            _mm_storel_epi64((__m128i*) oN, v0_26_lo);
            v0_26_lo = _mm_unpackhi_epi64(v0_26_lo, v0_26_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 25:
            _mm_storel_epi64((__m128i*) oN, v0_25_lo);
            v0_25_lo = _mm_unpackhi_epi64(v0_25_lo, v0_25_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 24:
            _mm_storel_epi64((__m128i*) oN, v0_24_lo);
            v0_24_lo = _mm_unpackhi_epi64(v0_24_lo, v0_24_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 23:
            _mm_storel_epi64((__m128i*) oN, v0_23_lo);
            v0_23_lo = _mm_unpackhi_epi64(v0_23_lo, v0_23_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 22:
            _mm_storel_epi64((__m128i*) oN, v0_22_lo);
            v0_22_lo = _mm_unpackhi_epi64(v0_22_lo, v0_22_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 21:
            _mm_storel_epi64((__m128i*) oN, v0_21_lo);
            v0_21_lo = _mm_unpackhi_epi64(v0_21_lo, v0_21_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 20:
            _mm_storel_epi64((__m128i*) oN, v0_20_lo);
            v0_20_lo = _mm_unpackhi_epi64(v0_20_lo, v0_20_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 19:
            _mm_storel_epi64((__m128i*) oN, v0_19_lo);
            v0_19_lo = _mm_unpackhi_epi64(v0_19_lo, v0_19_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 18:
            _mm_storel_epi64((__m128i*) oN, v0_18_lo);
            v0_18_lo = _mm_unpackhi_epi64(v0_18_lo, v0_18_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 17:
            _mm_storel_epi64((__m128i*) oN, v0_17_lo);
            v0_17_lo = _mm_unpackhi_epi64(v0_17_lo, v0_17_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 16:
            _mm_storel_epi64((__m128i*) oN, v0_16_lo);
            v0_16_lo = _mm_unpackhi_epi64(v0_16_lo, v0_16_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 15:
            _mm_storel_epi64((__m128i*) oN, v0_15_lo);
            v0_15_lo = _mm_unpackhi_epi64(v0_15_lo, v0_15_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            _mm_storel_epi64((__m128i*) oN, v0_14_lo);
            v0_14_lo = _mm_unpackhi_epi64(v0_14_lo, v0_14_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            _mm_storel_epi64((__m128i*) oN, v0_13_lo);
            v0_13_lo = _mm_unpackhi_epi64(v0_13_lo, v0_13_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            _mm_storel_epi64((__m128i*) oN, v0_12_lo);
            v0_12_lo = _mm_unpackhi_epi64(v0_12_lo, v0_12_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            _mm_storel_epi64((__m128i*) oN, v0_11_lo);
            v0_11_lo = _mm_unpackhi_epi64(v0_11_lo, v0_11_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            _mm_storel_epi64((__m128i*) oN, v0_10_lo);
            v0_10_lo = _mm_unpackhi_epi64(v0_10_lo, v0_10_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            _mm_storel_epi64((__m128i*) oN, v0_9_lo);
            v0_9_lo = _mm_unpackhi_epi64(v0_9_lo, v0_9_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            _mm_storel_epi64((__m128i*) oN, v0_8_lo);
            v0_8_lo = _mm_unpackhi_epi64(v0_8_lo, v0_8_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            _mm_storel_epi64((__m128i*) oN, v0_7_lo);
            v0_7_lo = _mm_unpackhi_epi64(v0_7_lo, v0_7_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            _mm_storel_epi64((__m128i*) oN, v0_6_lo);
            v0_6_lo = _mm_unpackhi_epi64(v0_6_lo, v0_6_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            _mm_storel_epi64((__m128i*) oN, v0_5_lo);
            v0_5_lo = _mm_unpackhi_epi64(v0_5_lo, v0_5_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            _mm_storel_epi64((__m128i*) oN, v0_4_lo);
            v0_4_lo = _mm_unpackhi_epi64(v0_4_lo, v0_4_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            _mm_storel_epi64((__m128i*) oN, v0_3_lo);
            v0_3_lo = _mm_unpackhi_epi64(v0_3_lo, v0_3_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            _mm_storel_epi64((__m128i*) oN, v0_2_lo);
            v0_2_lo = _mm_unpackhi_epi64(v0_2_lo, v0_2_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            _mm_storel_epi64((__m128i*) oN, v0_1_lo);
            v0_1_lo = _mm_unpackhi_epi64(v0_1_lo, v0_1_lo);
          case 0:
            _mm_storel_epi64((__m128i*) o, v0_0_lo);
            v0_0_lo = _mm_unpackhi_epi64(v0_0_lo, v0_0_lo);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 8;
      }
      if (bh & 4) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 31:
            _mm_storeu_si32(oN, v0_31_lo);
            v0_31_lo = _mm_srli_epi64(v0_31_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 30:
            _mm_storeu_si32(oN, v0_30_lo);
            v0_30_lo = _mm_srli_epi64(v0_30_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 29:
            _mm_storeu_si32(oN, v0_29_lo);
            v0_29_lo = _mm_srli_epi64(v0_29_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 28:
            _mm_storeu_si32(oN, v0_28_lo);
            v0_28_lo = _mm_srli_epi64(v0_28_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 27:
            _mm_storeu_si32(oN, v0_27_lo);
            v0_27_lo = _mm_srli_epi64(v0_27_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 26:
            _mm_storeu_si32(oN, v0_26_lo);
            v0_26_lo = _mm_srli_epi64(v0_26_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 25:
            _mm_storeu_si32(oN, v0_25_lo);
            v0_25_lo = _mm_srli_epi64(v0_25_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 24:
            _mm_storeu_si32(oN, v0_24_lo);
            v0_24_lo = _mm_srli_epi64(v0_24_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 23:
            _mm_storeu_si32(oN, v0_23_lo);
            v0_23_lo = _mm_srli_epi64(v0_23_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 22:
            _mm_storeu_si32(oN, v0_22_lo);
            v0_22_lo = _mm_srli_epi64(v0_22_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 21:
            _mm_storeu_si32(oN, v0_21_lo);
            v0_21_lo = _mm_srli_epi64(v0_21_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 20:
            _mm_storeu_si32(oN, v0_20_lo);
            v0_20_lo = _mm_srli_epi64(v0_20_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 19:
            _mm_storeu_si32(oN, v0_19_lo);
            v0_19_lo = _mm_srli_epi64(v0_19_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 18:
            _mm_storeu_si32(oN, v0_18_lo);
            v0_18_lo = _mm_srli_epi64(v0_18_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 17:
            _mm_storeu_si32(oN, v0_17_lo);
            v0_17_lo = _mm_srli_epi64(v0_17_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 16:
            _mm_storeu_si32(oN, v0_16_lo);
            v0_16_lo = _mm_srli_epi64(v0_16_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 15:
            _mm_storeu_si32(oN, v0_15_lo);
            v0_15_lo = _mm_srli_epi64(v0_15_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            _mm_storeu_si32(oN, v0_14_lo);
            v0_14_lo = _mm_srli_epi64(v0_14_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            _mm_storeu_si32(oN, v0_13_lo);
            v0_13_lo = _mm_srli_epi64(v0_13_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            _mm_storeu_si32(oN, v0_12_lo);
            v0_12_lo = _mm_srli_epi64(v0_12_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            _mm_storeu_si32(oN, v0_11_lo);
            v0_11_lo = _mm_srli_epi64(v0_11_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            _mm_storeu_si32(oN, v0_10_lo);
            v0_10_lo = _mm_srli_epi64(v0_10_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            _mm_storeu_si32(oN, v0_9_lo);
            v0_9_lo = _mm_srli_epi64(v0_9_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            _mm_storeu_si32(oN, v0_8_lo);
            v0_8_lo = _mm_srli_epi64(v0_8_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            _mm_storeu_si32(oN, v0_7_lo);
            v0_7_lo = _mm_srli_epi64(v0_7_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            _mm_storeu_si32(oN, v0_6_lo);
            v0_6_lo = _mm_srli_epi64(v0_6_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            _mm_storeu_si32(oN, v0_5_lo);
            v0_5_lo = _mm_srli_epi64(v0_5_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            _mm_storeu_si32(oN, v0_4_lo);
            v0_4_lo = _mm_srli_epi64(v0_4_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            _mm_storeu_si32(oN, v0_3_lo);
            v0_3_lo = _mm_srli_epi64(v0_3_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            _mm_storeu_si32(oN, v0_2_lo);
            v0_2_lo = _mm_srli_epi64(v0_2_lo, 32);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            _mm_storeu_si32(oN, v0_1_lo);
            v0_1_lo = _mm_srli_epi64(v0_1_lo, 32);
          case 0:
            _mm_storeu_si32(o, v0_0_lo);
            v0_0_lo = _mm_srli_epi64(v0_0_lo, 32);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 4;
      }
      if (bh & 2) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 31:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_31_lo));
             v0_31_lo = _mm_srli_epi32(v0_31_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 30:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_30_lo));
             v0_30_lo = _mm_srli_epi32(v0_30_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 29:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_29_lo));
             v0_29_lo = _mm_srli_epi32(v0_29_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 28:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_28_lo));
             v0_28_lo = _mm_srli_epi32(v0_28_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 27:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_27_lo));
             v0_27_lo = _mm_srli_epi32(v0_27_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 26:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_26_lo));
             v0_26_lo = _mm_srli_epi32(v0_26_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 25:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_25_lo));
             v0_25_lo = _mm_srli_epi32(v0_25_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 24:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_24_lo));
             v0_24_lo = _mm_srli_epi32(v0_24_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 23:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_23_lo));
             v0_23_lo = _mm_srli_epi32(v0_23_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 22:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_22_lo));
             v0_22_lo = _mm_srli_epi32(v0_22_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 21:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_21_lo));
             v0_21_lo = _mm_srli_epi32(v0_21_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 20:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_20_lo));
             v0_20_lo = _mm_srli_epi32(v0_20_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 19:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_19_lo));
             v0_19_lo = _mm_srli_epi32(v0_19_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 18:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_18_lo));
             v0_18_lo = _mm_srli_epi32(v0_18_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 17:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_17_lo));
             v0_17_lo = _mm_srli_epi32(v0_17_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 16:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_16_lo));
             v0_16_lo = _mm_srli_epi32(v0_16_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 15:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_15_lo));
             v0_15_lo = _mm_srli_epi32(v0_15_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_14_lo));
             v0_14_lo = _mm_srli_epi32(v0_14_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_13_lo));
             v0_13_lo = _mm_srli_epi32(v0_13_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_12_lo));
             v0_12_lo = _mm_srli_epi32(v0_12_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_11_lo));
             v0_11_lo = _mm_srli_epi32(v0_11_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_10_lo));
             v0_10_lo = _mm_srli_epi32(v0_10_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_9_lo));
             v0_9_lo = _mm_srli_epi32(v0_9_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_8_lo));
             v0_8_lo = _mm_srli_epi32(v0_8_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_7_lo));
             v0_7_lo = _mm_srli_epi32(v0_7_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_6_lo));
             v0_6_lo = _mm_srli_epi32(v0_6_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_5_lo));
             v0_5_lo = _mm_srli_epi32(v0_5_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_4_lo));
             v0_4_lo = _mm_srli_epi32(v0_4_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_3_lo));
             v0_3_lo = _mm_srli_epi32(v0_3_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_2_lo));
             v0_2_lo = _mm_srli_epi32(v0_2_lo, 16);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
             unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_1_lo));
             v0_1_lo = _mm_srli_epi32(v0_1_lo, 16);
          case 0:
             unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_0_lo));
             v0_0_lo = _mm_srli_epi32(v0_0_lo, 16);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 2;
      }
      if (bh & 1) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 31:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_31_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 30:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_30_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 29:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_29_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 28:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_28_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 27:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_27_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 26:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_26_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 25:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_25_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 24:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_24_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 23:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_23_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 22:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_22_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 21:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_21_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 20:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_20_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 19:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_19_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 18:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_18_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 17:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_17_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 16:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_16_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 15:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_15_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_14_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_13_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_12_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_11_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_10_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_9_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_8_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_7_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_6_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_5_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_4_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_3_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            *oN = (uint8_t) _mm_cvtsi128_si32(v0_2_lo);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
              *oN = (uint8_t) _mm_cvtsi128_si32(v0_1_lo);
          case 0:
            *o = (uint8_t) _mm_cvtsi128_si32(v0_0_lo);
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o = (uint8_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
