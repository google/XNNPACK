// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vinput_zero_point01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128i vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point1 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(2, 2, 2, 2));
    __m128i vinput_zero_point2 = _mm_cvtsi32_si128(*((const int*) &quantization_params[2].zero_point));
    vinput_zero_point2 = _mm_shuffle_epi32(vinput_zero_point2, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vksum_lo = _mm_srli_epi32(_mm_slli_epi32(vksum, 16), 16);
    const __m128i vksum_hi = _mm_srli_epi32(vksum, 16);

    __m128i vzpprodksumhi0 = _mm_mulhi_epu16(vinput_zero_point0, vksum_lo);
    const __m128i vzpprodksumlo0 = _mm_mullo_epi16(vinput_zero_point0, vksum_lo);
    __m128i vzpprodksumhi1 = _mm_mulhi_epu16(vinput_zero_point1, vksum_lo);
    const __m128i vzpprodksumlo1 = _mm_mullo_epi16(vinput_zero_point1, vksum_lo);
    __m128i vzpprodksumhi2 = _mm_mulhi_epu16(vinput_zero_point2, vksum_lo);
    const __m128i vzpprodksumlo2 = _mm_mullo_epi16(vinput_zero_point2, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point0, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point0, 15), vksum_lo));
    vzpprodksumhi1 = _mm_add_epi16(vzpprodksumhi1, _mm_mullo_epi16(vinput_zero_point1, vksum_hi));
    vzpprodksumhi1 = _mm_sub_epi16(vzpprodksumhi1, _mm_and_si128(_mm_srai_epi16(vinput_zero_point1, 15), vksum_lo));
    vzpprodksumhi2 = _mm_add_epi16(vzpprodksumhi2, _mm_mullo_epi16(vinput_zero_point2, vksum_hi));
    vzpprodksumhi2 = _mm_sub_epi16(vzpprodksumhi2, _mm_and_si128(_mm_srai_epi16(vinput_zero_point2, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);
    vzpprodksumhi1 = _mm_slli_si128(vzpprodksumhi1, 2);
    vzpprodksumhi2 = _mm_slli_si128(vzpprodksumhi2, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);
    const __m128i vksumzp1 = _mm_or_si128(vzpprodksumhi1, vzpprodksumlo1);
    const __m128i vksumzp2 = _mm_or_si128(vzpprodksumhi2, vzpprodksumlo2);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);
    const __m128i vksum011 = _mm_unpacklo_epi32(vksumzp1, vzero);
    const __m128i vksum231 = _mm_unpackhi_epi32(vksumzp1, vzero);
    const __m128i vksum012 = _mm_unpacklo_epi32(vksumzp2, vzero);
    const __m128i vksum232 = _mm_unpackhi_epi32(vksumzp2, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);
    __m128i vacc1x0 = _mm_unpacklo_epi64(vksum011, vzero);
    __m128i vacc1x1 = _mm_unpackhi_epi64(vksum011, vzero);
    __m128i vacc2x0 = _mm_unpacklo_epi64(vksum012, vzero);
    __m128i vacc2x1 = _mm_unpackhi_epi64(vksum012, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    __m128i vacc1x2 = _mm_unpacklo_epi64(vksum231, vzero);
    __m128i vacc1x3 = _mm_unpackhi_epi64(vksum231, vzero);
    __m128i vacc2x2 = _mm_unpacklo_epi64(vksum232, vzero);
    __m128i vacc2x3 = _mm_unpackhi_epi64(vksum232, vzero);
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_srai_epi16(_mm_unpacklo_epi8(va2, va2), 8);
      a2 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128i vinput_scale01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128 vinput_scale0 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale1 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(3, 3, 3, 3)));
    const __m128 vinput_scale2 = _mm_load1_ps(&quantization_params[2].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale1);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale2);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c2, vout2x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
