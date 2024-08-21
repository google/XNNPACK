// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c8-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/igemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  __m128i vinput_zero_point = _mm_cvtsi32_si128(*((const int*) &quantization_params->zero_point));
  vinput_zero_point = _mm_shuffle_epi32(vinput_zero_point, _MM_SHUFFLE(0, 0, 0, 0));
  const __m128 vinput_scale = _mm_load1_ps(&quantization_params->inv_scale);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vksum_lo = _mm_srli_epi32(_mm_slli_epi32(vksum, 16), 16);
    const __m128i vksum_hi = _mm_srli_epi32(vksum, 16);

    __m128i vzpprodksumhi0 = _mm_mulhi_epu16(vinput_zero_point, vksum_lo);
    const __m128i vzpprodksumlo0 = _mm_mullo_epi16(vinput_zero_point, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
