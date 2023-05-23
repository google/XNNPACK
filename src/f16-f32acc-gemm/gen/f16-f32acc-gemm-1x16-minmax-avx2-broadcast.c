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


void xnn_f16_f32acc_gemm_minmax_ukernel_1x16__avx2_broadcast(
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

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);

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
