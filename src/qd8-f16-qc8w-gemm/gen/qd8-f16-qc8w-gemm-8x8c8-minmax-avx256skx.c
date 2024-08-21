// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx8c8-avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  uint16_t* c6 = (uint16_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  uint16_t* c7 = (uint16_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const __m128i vinit0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vinit1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    const __m256i vinit01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vinit0), vinit1, 1);
    const __m128i vinit2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vinit3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    const __m256i vinit23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vinit2), vinit3, 1);
    const __m128i vinit4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vinit5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    const __m256i vinit45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vinit4), vinit5, 1);
    const __m128i vinit6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vinit7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    const __m256i vinit67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vinit6), vinit7, 1);
    const __m256i vinput_zero_point0 = _mm256_set1_epi32((int) quantization_params[0].zero_point);
    __m256i vacc0x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point0);
    __m256i vacc0x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point0);
    __m256i vacc0x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point0);
    __m256i vacc0x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point0);
    const __m256i vinput_zero_point1 = _mm256_set1_epi32((int) quantization_params[1].zero_point);
    __m256i vacc1x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point1);
    __m256i vacc1x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point1);
    __m256i vacc1x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point1);
    __m256i vacc1x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point1);
    const __m256i vinput_zero_point2 = _mm256_set1_epi32((int) quantization_params[2].zero_point);
    __m256i vacc2x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point2);
    __m256i vacc2x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point2);
    __m256i vacc2x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point2);
    __m256i vacc2x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point2);
    const __m256i vinput_zero_point3 = _mm256_set1_epi32((int) quantization_params[3].zero_point);
    __m256i vacc3x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point3);
    __m256i vacc3x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point3);
    __m256i vacc3x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point3);
    __m256i vacc3x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point3);
    const __m256i vinput_zero_point4 = _mm256_set1_epi32((int) quantization_params[4].zero_point);
    __m256i vacc4x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point4);
    __m256i vacc4x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point4);
    __m256i vacc4x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point4);
    __m256i vacc4x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point4);
    const __m256i vinput_zero_point5 = _mm256_set1_epi32((int) quantization_params[5].zero_point);
    __m256i vacc5x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point5);
    __m256i vacc5x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point5);
    __m256i vacc5x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point5);
    __m256i vacc5x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point5);
    const __m256i vinput_zero_point6 = _mm256_set1_epi32((int) quantization_params[6].zero_point);
    __m256i vacc6x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point6);
    __m256i vacc6x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point6);
    __m256i vacc6x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point6);
    __m256i vacc6x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point6);
    const __m256i vinput_zero_point7 = _mm256_set1_epi32((int) quantization_params[7].zero_point);
    __m256i vacc7x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point7);
    __m256i vacc7x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point7);
    __m256i vacc7x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point7);
    __m256i vacc7x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point7);
    w = (const int32_t*) w + 8;

    size_t k = kc;

    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      const __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
      const __m256i vxa2 = _mm256_cvtepi8_epi16(va2);
      a2 += 8;
      const __m128i va3 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a3));
      const __m256i vxa3 = _mm256_cvtepi8_epi16(va3);
      a3 += 8;
      const __m128i va4 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a4));
      const __m256i vxa4 = _mm256_cvtepi8_epi16(va4);
      a4 += 8;
      const __m128i va5 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a5));
      const __m256i vxa5 = _mm256_cvtepi8_epi16(va5);
      a5 += 8;
      const __m128i va6 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a6));
      const __m256i vxa6 = _mm256_cvtepi8_epi16(va6);
      a6 += 8;
      const __m128i va7 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a7));
      const __m256i vxa7 = _mm256_cvtepi8_epi16(va7);
      a7 += 8;

      const __m256i vxb01 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*) w));

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
      vacc3x01 = _mm256_add_epi32(vacc3x01, _mm256_madd_epi16(vxa3, vxb01));
      vacc4x01 = _mm256_add_epi32(vacc4x01, _mm256_madd_epi16(vxa4, vxb01));
      vacc5x01 = _mm256_add_epi32(vacc5x01, _mm256_madd_epi16(vxa5, vxb01));
      vacc6x01 = _mm256_add_epi32(vacc6x01, _mm256_madd_epi16(vxa6, vxb01));
      vacc7x01 = _mm256_add_epi32(vacc7x01, _mm256_madd_epi16(vxa7, vxb01));
      const __m256i vxb23 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*) ((const int8_t*) w + 16)));

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
      vacc3x23 = _mm256_add_epi32(vacc3x23, _mm256_madd_epi16(vxa3, vxb23));
      vacc4x23 = _mm256_add_epi32(vacc4x23, _mm256_madd_epi16(vxa4, vxb23));
      vacc5x23 = _mm256_add_epi32(vacc5x23, _mm256_madd_epi16(vxa5, vxb23));
      vacc6x23 = _mm256_add_epi32(vacc6x23, _mm256_madd_epi16(vxa6, vxb23));
      vacc7x23 = _mm256_add_epi32(vacc7x23, _mm256_madd_epi16(vxa7, vxb23));
      const __m256i vxb45 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*) ((const int8_t*) w + 32)));

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
      vacc3x45 = _mm256_add_epi32(vacc3x45, _mm256_madd_epi16(vxa3, vxb45));
      vacc4x45 = _mm256_add_epi32(vacc4x45, _mm256_madd_epi16(vxa4, vxb45));
      vacc5x45 = _mm256_add_epi32(vacc5x45, _mm256_madd_epi16(vxa5, vxb45));
      vacc6x45 = _mm256_add_epi32(vacc6x45, _mm256_madd_epi16(vxa6, vxb45));
      vacc7x45 = _mm256_add_epi32(vacc7x45, _mm256_madd_epi16(vxa7, vxb45));
      const __m256i vxb67 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*) ((const int8_t*) w + 48)));

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
      vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));
      vacc3x67 = _mm256_add_epi32(vacc3x67, _mm256_madd_epi16(vxa3, vxb67));
      vacc4x67 = _mm256_add_epi32(vacc4x67, _mm256_madd_epi16(vxa4, vxb67));
      vacc5x67 = _mm256_add_epi32(vacc5x67, _mm256_madd_epi16(vxa5, vxb67));
      vacc6x67 = _mm256_add_epi32(vacc6x67, _mm256_madd_epi16(vxa6, vxb67));
      vacc7x67 = _mm256_add_epi32(vacc7x67, _mm256_madd_epi16(vxa7, vxb67));

      w = (const int8_t*) w + 64;
      k -= 8 * sizeof(int8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);
    const __m256i vacc3x0213 = _mm256_hadd_epi32(vacc3x01, vacc3x23);
    const __m256i vacc3x4657 = _mm256_hadd_epi32(vacc3x45, vacc3x67);
    const __m256i vacc4x0213 = _mm256_hadd_epi32(vacc4x01, vacc4x23);
    const __m256i vacc4x4657 = _mm256_hadd_epi32(vacc4x45, vacc4x67);
    const __m256i vacc5x0213 = _mm256_hadd_epi32(vacc5x01, vacc5x23);
    const __m256i vacc5x4657 = _mm256_hadd_epi32(vacc5x45, vacc5x67);
    const __m256i vacc6x0213 = _mm256_hadd_epi32(vacc6x01, vacc6x23);
    const __m256i vacc6x4657 = _mm256_hadd_epi32(vacc6x45, vacc6x67);
    const __m256i vacc7x0213 = _mm256_hadd_epi32(vacc7x01, vacc7x23);
    const __m256i vacc7x4657 = _mm256_hadd_epi32(vacc7x45, vacc7x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);
    const __m256i vacc3x02461357 = _mm256_hadd_epi32(vacc3x0213, vacc3x4657);
    const __m256i vacc4x02461357 = _mm256_hadd_epi32(vacc4x0213, vacc4x4657);
    const __m256i vacc5x02461357 = _mm256_hadd_epi32(vacc5x0213, vacc5x4657);
    const __m256i vacc6x02461357 = _mm256_hadd_epi32(vacc6x0213, vacc6x4657);
    const __m256i vacc7x02461357 = _mm256_hadd_epi32(vacc7x0213, vacc7x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);
    __m256i vacc3x01234567 = _mm256_permutevar8x32_epi32(vacc3x02461357, vpermute_mask);
    __m256i vacc4x01234567 = _mm256_permutevar8x32_epi32(vacc4x02461357, vpermute_mask);
    __m256i vacc5x01234567 = _mm256_permutevar8x32_epi32(vacc5x02461357, vpermute_mask);
    __m256i vacc6x01234567 = _mm256_permutevar8x32_epi32(vacc6x02461357, vpermute_mask);
    __m256i vacc7x01234567 = _mm256_permutevar8x32_epi32(vacc7x02461357, vpermute_mask);

    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    const __m256 vinput_scale0 = _mm256_broadcast_ss(&quantization_params[0].inv_scale);
    __m256 vout1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    const __m256 vinput_scale1 = _mm256_broadcast_ss(&quantization_params[1].inv_scale);
    __m256 vout2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);
    const __m256 vinput_scale2 = _mm256_broadcast_ss(&quantization_params[2].inv_scale);
    __m256 vout3x01234567 = _mm256_cvtepi32_ps(vacc3x01234567);
    const __m256 vinput_scale3 = _mm256_broadcast_ss(&quantization_params[3].inv_scale);
    __m256 vout4x01234567 = _mm256_cvtepi32_ps(vacc4x01234567);
    const __m256 vinput_scale4 = _mm256_broadcast_ss(&quantization_params[4].inv_scale);
    __m256 vout5x01234567 = _mm256_cvtepi32_ps(vacc5x01234567);
    const __m256 vinput_scale5 = _mm256_broadcast_ss(&quantization_params[5].inv_scale);
    __m256 vout6x01234567 = _mm256_cvtepi32_ps(vacc6x01234567);
    const __m256 vinput_scale6 = _mm256_broadcast_ss(&quantization_params[6].inv_scale);
    __m256 vout7x01234567 = _mm256_cvtepi32_ps(vacc7x01234567);
    const __m256 vinput_scale7 = _mm256_broadcast_ss(&quantization_params[7].inv_scale);

    vout0x01234567 = _mm256_mul_ps(vout0x01234567, vinput_scale0);
    vout1x01234567 = _mm256_mul_ps(vout1x01234567, vinput_scale1);
    vout2x01234567 = _mm256_mul_ps(vout2x01234567, vinput_scale2);
    vout3x01234567 = _mm256_mul_ps(vout3x01234567, vinput_scale3);
    vout4x01234567 = _mm256_mul_ps(vout4x01234567, vinput_scale4);
    vout5x01234567 = _mm256_mul_ps(vout5x01234567, vinput_scale5);
    vout6x01234567 = _mm256_mul_ps(vout6x01234567, vinput_scale6);
    vout7x01234567 = _mm256_mul_ps(vout7x01234567, vinput_scale7);

    const __m256 vfilter_output_scale01234567 = _mm256_load_ps((const float*) w);
    const __m256 vbias01234567 = _mm256_load_ps((const float*) w + 8);
    w = (const float*) w + 16;
    vout0x01234567 = _mm256_fmadd_ps(vout0x01234567, vfilter_output_scale01234567, vbias01234567);
    vout1x01234567 = _mm256_fmadd_ps(vout1x01234567, vfilter_output_scale01234567, vbias01234567);
    vout2x01234567 = _mm256_fmadd_ps(vout2x01234567, vfilter_output_scale01234567, vbias01234567);
    vout3x01234567 = _mm256_fmadd_ps(vout3x01234567, vfilter_output_scale01234567, vbias01234567);
    vout4x01234567 = _mm256_fmadd_ps(vout4x01234567, vfilter_output_scale01234567, vbias01234567);
    vout5x01234567 = _mm256_fmadd_ps(vout5x01234567, vfilter_output_scale01234567, vbias01234567);
    vout6x01234567 = _mm256_fmadd_ps(vout6x01234567, vfilter_output_scale01234567, vbias01234567);
    vout7x01234567 = _mm256_fmadd_ps(vout7x01234567, vfilter_output_scale01234567, vbias01234567);

    vout0x01234567 = _mm256_max_ps(vout0x01234567, vmin);
    vout1x01234567 = _mm256_max_ps(vout1x01234567, vmin);
    vout2x01234567 = _mm256_max_ps(vout2x01234567, vmin);
    vout3x01234567 = _mm256_max_ps(vout3x01234567, vmin);
    vout4x01234567 = _mm256_max_ps(vout4x01234567, vmin);
    vout5x01234567 = _mm256_max_ps(vout5x01234567, vmin);
    vout6x01234567 = _mm256_max_ps(vout6x01234567, vmin);
    vout7x01234567 = _mm256_max_ps(vout7x01234567, vmin);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, vmax);
    vout1x01234567 = _mm256_min_ps(vout1x01234567, vmax);
    vout2x01234567 = _mm256_min_ps(vout2x01234567, vmax);
    vout3x01234567 = _mm256_min_ps(vout3x01234567, vmax);
    vout4x01234567 = _mm256_min_ps(vout4x01234567, vmax);
    vout5x01234567 = _mm256_min_ps(vout5x01234567, vmax);
    vout6x01234567 = _mm256_min_ps(vout6x01234567, vmax);
    vout7x01234567 = _mm256_min_ps(vout7x01234567, vmax);
    __m128i vfp16out0x01234567 = _mm256_cvtps_ph(vout0x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out1x01234567 = _mm256_cvtps_ph(vout1x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out2x01234567 = _mm256_cvtps_ph(vout2x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out3x01234567 = _mm256_cvtps_ph(vout3x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out4x01234567 = _mm256_cvtps_ph(vout4x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out5x01234567 = _mm256_cvtps_ph(vout5x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out6x01234567 = _mm256_cvtps_ph(vout6x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out7x01234567 = _mm256_cvtps_ph(vout7x01234567, _MM_FROUND_TO_NEAREST_INT);
    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_si128((__m128i*) c0, vfp16out0x01234567);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_si128((__m128i*) c1, vfp16out1x01234567);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) c2, vfp16out2x01234567);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) c3, vfp16out3x01234567);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) c4, vfp16out4x01234567);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) c5, vfp16out5x01234567);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) c6, vfp16out6x01234567);
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);
      _mm_storeu_si128((__m128i*) c7, vfp16out7x01234567);
      c7 = (uint16_t*) ((uintptr_t) c7 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);

      nc -= 8;
    } else {
      // Prepare mask for valid 16-bit elements (depends on nc).
      const __mmask8 vmask = _cvtu32_mask8((UINT32_C(1) << nc) - 1);
      _mm_mask_storeu_epi16(c0, vmask, vfp16out0x01234567);
      _mm_mask_storeu_epi16(c1, vmask, vfp16out1x01234567);
      _mm_mask_storeu_epi16(c2, vmask, vfp16out2x01234567);
      _mm_mask_storeu_epi16(c3, vmask, vfp16out3x01234567);
      _mm_mask_storeu_epi16(c4, vmask, vfp16out4x01234567);
      _mm_mask_storeu_epi16(c5, vmask, vfp16out5x01234567);
      _mm_mask_storeu_epi16(c6, vmask, vfp16out6x01234567);
      _mm_mask_storeu_epi16(c7, vmask, vfp16out7x01234567);
      nc = 0;
    }
  } while (nc != 0);
}
