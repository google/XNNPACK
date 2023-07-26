// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/sse-shuffle.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/gemm.h>


void xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__sse2(
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

  do {
    __m128 vacc0x0123 = _mm_loadu_ps((const float*) w + 0);
    __m128 vacc0x4567 = _mm_loadu_ps((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      __m128 va0 = _mm_loadu_ps(a0);
      a0 += 4;

      const __m128i vb01234567c0 = _mm_loadl_epi64((const __m128i *) ((const int8_t*) w + 0));
      const __m128i vbw01234567c0 = _mm_unpacklo_epi8(vb01234567c0, vb01234567c0);
      const __m128 vb0123c0 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vbw01234567c0, vbw01234567c0), 24));
      const __m128 vb4567c0 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vbw01234567c0, vbw01234567c0), 24));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567c0));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128i vb01234567c1 = _mm_loadl_epi64((const __m128i *) ((const int8_t*) w + 8));
      const __m128i vbw01234567c1 = _mm_unpacklo_epi8(vb01234567c1, vb01234567c1);
      const __m128 vb0123c1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vbw01234567c1, vbw01234567c1), 24));
      const __m128 vb4567c1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vbw01234567c1, vbw01234567c1), 24));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567c1));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128i vb01234567c2 = _mm_loadl_epi64((const __m128i *) ((const int8_t*) w + 16));
      const __m128i vbw01234567c2 = _mm_unpacklo_epi8(vb01234567c2, vb01234567c2);
      const __m128 vb0123c2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vbw01234567c2, vbw01234567c2), 24));
      const __m128 vb4567c2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vbw01234567c2, vbw01234567c2), 24));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567c2));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128i vb01234567c3 = _mm_loadl_epi64((const __m128i *) ((const int8_t*) w + 24));
      const __m128i vbw01234567c3 = _mm_unpacklo_epi8(vb01234567c3, vb01234567c3);
      const __m128 vb0123c3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vbw01234567c3, vbw01234567c3), 24));
      const __m128 vb4567c3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vbw01234567c3, vbw01234567c3), 24));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567c3));


      w = (const int8_t*) w + 32;
    }
    if XNN_UNLIKELY(k != 0) {
      __m128 va0 = _mm_loadu_ps(a0);
      a0 = (const float*) ((uintptr_t) a0 + k);

      const __m128i vb01234567c0 = _mm_loadl_epi64((const __m128i *) ((const int8_t*) w + 0));
      const __m128i vbw01234567c0 = _mm_unpacklo_epi8(vb01234567c0, vb01234567c0);
      const __m128 vb0123c0 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vbw01234567c0, vbw01234567c0), 24));
      const __m128 vb4567c0 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vbw01234567c0, vbw01234567c0), 24));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c0), va0), vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c0), va0), vb4567c0));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128i vb01234567c1 = _mm_loadl_epi64((const __m128i *) ((const int8_t*) w + 8));
      const __m128i vbw01234567c1 = _mm_unpacklo_epi8(vb01234567c1, vb01234567c1);
      const __m128 vb0123c1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vbw01234567c1, vbw01234567c1), 24));
      const __m128 vb4567c1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vbw01234567c1, vbw01234567c1), 24));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c1), va0), vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c1), va0), vb4567c1));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128i vb01234567c2 = _mm_loadl_epi64((const __m128i *) ((const int8_t*) w + 16));
      const __m128i vbw01234567c2 = _mm_unpacklo_epi8(vb01234567c2, vb01234567c2);
      const __m128 vb0123c2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vbw01234567c2, vbw01234567c2), 24));
      const __m128 vb4567c2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vbw01234567c2, vbw01234567c2), 24));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c2), va0), vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c2), va0), vb4567c2));

      va0 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128i vb01234567c3 = _mm_loadl_epi64((const __m128i *) ((const int8_t*) w + 24));
      const __m128i vbw01234567c3 = _mm_unpacklo_epi8(vb01234567c3, vb01234567c3);
      const __m128 vb0123c3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vbw01234567c3, vbw01234567c3), 24));
      const __m128 vb4567c3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vbw01234567c3, vbw01234567c3), 24));

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb0123c3), va0), vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(_mm_andnot_ps(_mm_cmpeq_ps(_mm_setzero_ps(), vb4567c3), va0), vb4567c3));


      w = (const int8_t*) w + 32;
    }

    const __m128 vscale0123 = _mm_loadu_ps((const float*) w + 0);
    vacc0x0123 = _mm_mul_ps(vacc0x0123, vscale0123);
    const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
    vacc0x4567 = _mm_mul_ps(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    const __m128 vmax = _mm_load_ps(params->sse.max);
    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);

    const __m128 vmin = _mm_load_ps(params->sse.min);
    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

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
