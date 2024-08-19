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


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse2_ld128(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

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
    __m128i vzpprodksumhi1 = _mm_mulhi_epu16(vinput_zero_point, vksum_lo);
    const __m128i vzpprodksumlo1 = _mm_mullo_epi16(vinput_zero_point, vksum_lo);
    __m128i vzpprodksumhi2 = _mm_mulhi_epu16(vinput_zero_point, vksum_lo);
    const __m128i vzpprodksumlo2 = _mm_mullo_epi16(vinput_zero_point, vksum_lo);
    __m128i vzpprodksumhi3 = _mm_mulhi_epu16(vinput_zero_point, vksum_lo);
    const __m128i vzpprodksumlo3 = _mm_mullo_epi16(vinput_zero_point, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));
    vzpprodksumhi1 = _mm_add_epi16(vzpprodksumhi1, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi1 = _mm_sub_epi16(vzpprodksumhi1, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));
    vzpprodksumhi2 = _mm_add_epi16(vzpprodksumhi2, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi2 = _mm_sub_epi16(vzpprodksumhi2, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));
    vzpprodksumhi3 = _mm_add_epi16(vzpprodksumhi3, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi3 = _mm_sub_epi16(vzpprodksumhi3, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);
    vzpprodksumhi1 = _mm_slli_si128(vzpprodksumhi1, 2);
    vzpprodksumhi2 = _mm_slli_si128(vzpprodksumhi2, 2);
    vzpprodksumhi3 = _mm_slli_si128(vzpprodksumhi3, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);
    const __m128i vksumzp1 = _mm_or_si128(vzpprodksumhi1, vzpprodksumlo1);
    const __m128i vksumzp2 = _mm_or_si128(vzpprodksumhi2, vzpprodksumlo2);
    const __m128i vksumzp3 = _mm_or_si128(vzpprodksumhi3, vzpprodksumlo3);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);
    const __m128i vksum011 = _mm_unpacklo_epi32(vksumzp1, vzero);
    const __m128i vksum231 = _mm_unpackhi_epi32(vksumzp1, vzero);
    const __m128i vksum012 = _mm_unpacklo_epi32(vksumzp2, vzero);
    const __m128i vksum232 = _mm_unpackhi_epi32(vksumzp2, vzero);
    const __m128i vksum013 = _mm_unpacklo_epi32(vksumzp3, vzero);
    const __m128i vksum233 = _mm_unpackhi_epi32(vksumzp3, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);
    __m128i vacc1x0 = _mm_unpacklo_epi64(vksum011, vzero);
    __m128i vacc1x1 = _mm_unpackhi_epi64(vksum011, vzero);
    __m128i vacc2x0 = _mm_unpacklo_epi64(vksum012, vzero);
    __m128i vacc2x1 = _mm_unpackhi_epi64(vksum012, vzero);
    __m128i vacc3x0 = _mm_unpacklo_epi64(vksum013, vzero);
    __m128i vacc3x1 = _mm_unpackhi_epi64(vksum013, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    __m128i vacc1x2 = _mm_unpacklo_epi64(vksum231, vzero);
    __m128i vacc1x3 = _mm_unpackhi_epi64(vksum231, vzero);
    __m128i vacc2x2 = _mm_unpacklo_epi64(vksum232, vzero);
    __m128i vacc2x3 = _mm_unpackhi_epi64(vksum232, vzero);
    __m128i vacc3x2 = _mm_unpacklo_epi64(vksum233, vzero);
    __m128i vacc3x3 = _mm_unpackhi_epi64(vksum233, vzero);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_srai_epi16(_mm_unpacklo_epi8(va2, va2), 8);
        a2 += 8;
        const __m128i va3 = _mm_loadl_epi64((const __m128i*) a3);
        const __m128i vxa3 = _mm_srai_epi16(_mm_unpacklo_epi8(va3, va3), 8);
        a3 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vsb01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01);
        const __m128i vxb0 = _mm_unpacklo_epi8(vb01, vsb01);
        const __m128i vxb1 = _mm_unpackhi_epi8(vb01, vsb01);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3, vxb0));
        vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3, vxb1));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vsb23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23);
        const __m128i vxb2 = _mm_unpacklo_epi8(vb23, vsb23);
        const __m128i vxb3 = _mm_unpackhi_epi8(vb23, vsb23);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));
        vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3, vxb2));
        vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));
    const __m128i vacc3x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x0, vacc3x2), _mm_unpackhi_epi32(vacc3x0, vacc3x2));
    const __m128i vacc3x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x1, vacc3x3), _mm_unpackhi_epi32(vacc3x1, vacc3x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));
    __m128i vacc3x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x02, vacc3x13), _mm_unpackhi_epi32(vacc3x02, vacc3x13));

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);
    __m128 vout3x0123 = _mm_cvtepi32_ps(vacc3x0123);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale);
    vout3x0123 = _mm_mul_ps(vout3x0123, vinput_scale);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);
    vout3x0123 = _mm_mul_ps(vout3x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);
    vout3x0123 = _mm_add_ps(vout3x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);
    vout3x0123 = _mm_max_ps(vout3x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);
    vout3x0123 = _mm_min_ps(vout3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c3, vout3x0123);
      _mm_storeu_ps(c2, vout2x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c0, vout0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c3, vout3x0123);
        vout3x0123 = _mm_unpackhi_ps(vout3x0123, vout3x0123);
        c3 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c3, vout3x0123);
        _mm_store_ss(c2, vout2x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
