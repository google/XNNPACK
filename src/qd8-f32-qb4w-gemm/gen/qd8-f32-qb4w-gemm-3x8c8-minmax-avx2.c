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


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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
  size_t bl = params->avx.blocksize;
  assert(bl <= round_up_po2(kc, 16));
  assert(bl != 0);
  assert(bl % 32 == 0);
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

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const __m128 vinit0 = _mm_load_ss(&((const float*) w)[0]);
    const __m128 vinit1 = _mm_load_ss(&((const float*) w)[1]);
    const __m256 vinit01 = _mm256_insertf128_ps(_mm256_castps128_ps256(vinit0), vinit1, 1);
    const __m128 vinit2 = _mm_load_ss(&((const float*) w)[2]);
    const __m128 vinit3 = _mm_load_ss(&((const float*) w)[3]);
    const __m256 vinit23 = _mm256_insertf128_ps(_mm256_castps128_ps256(vinit2), vinit3, 1);
    const __m128 vinit4 = _mm_load_ss(&((const float*) w)[4]);
    const __m128 vinit5 = _mm_load_ss(&((const float*) w)[5]);
    const __m256 vinit45 = _mm256_insertf128_ps(_mm256_castps128_ps256(vinit4), vinit5, 1);
    const __m128 vinit6 = _mm_load_ss(&((const float*) w)[6]);
    const __m128 vinit7 = _mm_load_ss(&((const float*) w)[7]);
    const __m256 vinit67 = _mm256_insertf128_ps(_mm256_castps128_ps256(vinit6), vinit7, 1);
    const __m256 vinput_zero_point0 = _mm256_set1_ps((float) quantization_params[0].zero_point);
    __m256 vout0x01 = _mm256_mul_ps(vinit01, vinput_zero_point0);
    __m256 vout0x23 = _mm256_mul_ps(vinit23, vinput_zero_point0);
    __m256 vout0x45 = _mm256_mul_ps(vinit45, vinput_zero_point0);
    __m256 vout0x67 = _mm256_mul_ps(vinit67, vinput_zero_point0);
    const __m256 vinput_zero_point1 = _mm256_set1_ps((float) quantization_params[1].zero_point);
    __m256 vout1x01 = _mm256_mul_ps(vinit01, vinput_zero_point1);
    __m256 vout1x23 = _mm256_mul_ps(vinit23, vinput_zero_point1);
    __m256 vout1x45 = _mm256_mul_ps(vinit45, vinput_zero_point1);
    __m256 vout1x67 = _mm256_mul_ps(vinit67, vinput_zero_point1);
    const __m256 vinput_zero_point2 = _mm256_set1_ps((float) quantization_params[2].zero_point);
    __m256 vout2x01 = _mm256_mul_ps(vinit01, vinput_zero_point2);
    __m256 vout2x23 = _mm256_mul_ps(vinit23, vinput_zero_point2);
    __m256 vout2x45 = _mm256_mul_ps(vinit45, vinput_zero_point2);
    __m256 vout2x67 = _mm256_mul_ps(vinit67, vinput_zero_point2);
    w = (const int32_t*) w + 8;

    for (size_t kb=0; kb < kc; kb += bl) {
      __m256i vacc0x01 = _mm256_setzero_si256();
      __m256i vacc0x23 = _mm256_setzero_si256();
      __m256i vacc0x45 = _mm256_setzero_si256();
      __m256i vacc0x67 = _mm256_setzero_si256();
      __m256i vacc1x01 = _mm256_setzero_si256();
      __m256i vacc1x23 = _mm256_setzero_si256();
      __m256i vacc1x45 = _mm256_setzero_si256();
      __m256i vacc1x67 = _mm256_setzero_si256();
      __m256i vacc2x01 = _mm256_setzero_si256();
      __m256i vacc2x23 = _mm256_setzero_si256();
      __m256i vacc2x45 = _mm256_setzero_si256();
      __m256i vacc2x67 = _mm256_setzero_si256();

      size_t k = bl;
      while (k >= 16 * sizeof(int8_t)) {
        __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;
        __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
        __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
        a1 += 8;
        __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
        __m256i vxa2 = _mm256_cvtepi8_epi16(va2);
        a2 += 8;

        __m128i vb01 = _mm_load_si128((const __m128i*) w);
        __m128i vbs01 = _mm_slli_epi32(vb01, 4);
        __m128i vbm01 = _mm_and_si128(vbs01, vmask);
        __m256i vxb01 = _mm256_cvtepi8_epi16(vbm01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
        vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
        __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        __m128i vbs23 = _mm_slli_epi32(vb23, 4);
        __m128i vbm23 = _mm_and_si128(vbs23, vmask);
        __m256i vxb23 = _mm256_cvtepi8_epi16(vbm23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
        vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
        __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
        __m128i vbs45 = _mm_slli_epi32(vb45, 4);
        __m128i vbm45 = _mm_and_si128(vbs45, vmask);
        __m256i vxb45 = _mm256_cvtepi8_epi16(vbm45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
        vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
        __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
        __m128i vbs67 = _mm_slli_epi32(vb67, 4);
        __m128i vbm67 = _mm_and_si128(vbs67, vmask);
        __m256i vxb67 = _mm256_cvtepi8_epi16(vbm67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
        vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
        vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

        va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;
        va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
        vxa1 = _mm256_cvtepi8_epi16(va1);
        a1 += 8;
        va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
        vxa2 = _mm256_cvtepi8_epi16(va2);
        a2 += 8;

        vbm01 = _mm_and_si128(vb01, vmask);
        vxb01 = _mm256_cvtepi8_epi16(vbm01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
        vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
        vbm23 = _mm_and_si128(vb23, vmask);
        vxb23 = _mm256_cvtepi8_epi16(vbm23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
        vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
        vbm45 = _mm_and_si128(vb45, vmask);
        vxb45 = _mm256_cvtepi8_epi16(vbm45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
        vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
        vbm67 = _mm_and_si128(vb67, vmask);
        vxb67 = _mm256_cvtepi8_epi16(vbm67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
        vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
        vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      }

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

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vbs01 = _mm_slli_epi32(vb01, 4);
        const __m128i vbm01 = _mm_and_si128(vbs01, vmask);
        const __m256i vxb01 = _mm256_cvtepi8_epi16(vbm01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
        vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vbs23 = _mm_slli_epi32(vb23, 4);
        const __m128i vbm23 = _mm_and_si128(vbs23, vmask);
        const __m256i vxb23 = _mm256_cvtepi8_epi16(vbm23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
        vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
        const __m128i vbs45 = _mm_slli_epi32(vb45, 4);
        const __m128i vbm45 = _mm_and_si128(vbs45, vmask);
        const __m256i vxb45 = _mm256_cvtepi8_epi16(vbm45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
        vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
        const __m128i vbs67 = _mm_slli_epi32(vb67, 4);
        const __m128i vbm67 = _mm_and_si128(vbs67, vmask);
        const __m256i vxb67 = _mm256_cvtepi8_epi16(vbm67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
        vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
        vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));

        w = (const int8_t*) w + 64;
        k -= 8 * sizeof(int8_t);
      }

      const __m128 vfilter_output_scale0 = _mm_castsi128_ps(_mm_set1_epi32((uint32_t) ((const uint16_t*) w)[0] << 16));
      const __m128 vfilter_output_scale1 = _mm_castsi128_ps(_mm_set1_epi32((uint32_t) ((const uint16_t*) w)[1] << 16));
      const __m256 vfilter_output_scale01 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(vfilter_output_scale0), vfilter_output_scale1, 1);
      vout0x01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc0x01), vfilter_output_scale01, vout0x01);
      vout1x01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc1x01), vfilter_output_scale01, vout1x01);
      vout2x01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc2x01), vfilter_output_scale01, vout2x01);
      const __m128 vfilter_output_scale2 = _mm_castsi128_ps(_mm_set1_epi32((uint32_t) ((const uint16_t*) w)[2] << 16));
      const __m128 vfilter_output_scale3 = _mm_castsi128_ps(_mm_set1_epi32((uint32_t) ((const uint16_t*) w)[3] << 16));
      const __m256 vfilter_output_scale23 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(vfilter_output_scale2), vfilter_output_scale3, 1);
      vout0x23 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc0x23), vfilter_output_scale23, vout0x23);
      vout1x23 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc1x23), vfilter_output_scale23, vout1x23);
      vout2x23 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc2x23), vfilter_output_scale23, vout2x23);
      const __m128 vfilter_output_scale4 = _mm_castsi128_ps(_mm_set1_epi32((uint32_t) ((const uint16_t*) w)[4] << 16));
      const __m128 vfilter_output_scale5 = _mm_castsi128_ps(_mm_set1_epi32((uint32_t) ((const uint16_t*) w)[5] << 16));
      const __m256 vfilter_output_scale45 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(vfilter_output_scale4), vfilter_output_scale5, 1);
      vout0x45 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc0x45), vfilter_output_scale45, vout0x45);
      vout1x45 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc1x45), vfilter_output_scale45, vout1x45);
      vout2x45 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc2x45), vfilter_output_scale45, vout2x45);
      const __m128 vfilter_output_scale6 = _mm_castsi128_ps(_mm_set1_epi32((uint32_t) ((const uint16_t*) w)[6] << 16));
      const __m128 vfilter_output_scale7 = _mm_castsi128_ps(_mm_set1_epi32((uint32_t) ((const uint16_t*) w)[7] << 16));
      const __m256 vfilter_output_scale67 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(vfilter_output_scale6), vfilter_output_scale7, 1);
      vout0x67 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc0x67), vfilter_output_scale67, vout0x67);
      vout1x67 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc1x67), vfilter_output_scale67, vout1x67);
      vout2x67 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc2x67), vfilter_output_scale67, vout2x67);

      w = (const uint16_t*) w + 8;
    }

    const __m256 vout0x0213 = _mm256_hadd_ps(vout0x01, vout0x23);
    const __m256 vout0x4657 = _mm256_hadd_ps(vout0x45, vout0x67);
    const __m256 vout1x0213 = _mm256_hadd_ps(vout1x01, vout1x23);
    const __m256 vout1x4657 = _mm256_hadd_ps(vout1x45, vout1x67);
    const __m256 vout2x0213 = _mm256_hadd_ps(vout2x01, vout2x23);
    const __m256 vout2x4657 = _mm256_hadd_ps(vout2x45, vout2x67);

    const __m256 vout0x02461357 = _mm256_hadd_ps(vout0x0213, vout0x4657);
    const __m256 vout1x02461357 = _mm256_hadd_ps(vout1x0213, vout1x4657);
    const __m256 vout2x02461357 = _mm256_hadd_ps(vout2x0213, vout2x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256 vout0x01234567 = _mm256_permutevar8x32_ps(vout0x02461357, vpermute_mask);
    __m256 vout1x01234567 = _mm256_permutevar8x32_ps(vout1x02461357, vpermute_mask);
    __m256 vout2x01234567 = _mm256_permutevar8x32_ps(vout2x02461357, vpermute_mask);

    const __m256 vinput_scale0 = _mm256_broadcast_ss(&quantization_params[0].inv_scale);
    const __m256 vinput_scale1 = _mm256_broadcast_ss(&quantization_params[1].inv_scale);
    const __m256 vinput_scale2 = _mm256_broadcast_ss(&quantization_params[2].inv_scale);

    const __m256 vbias01234567 = _mm256_loadu_ps((const float*) w);
    w = (const float*) w + 8;
    vout0x01234567 = _mm256_fmadd_ps(vout0x01234567, vinput_scale0, vbias01234567);
    vout1x01234567 = _mm256_fmadd_ps(vout1x01234567, vinput_scale1, vbias01234567);
    vout2x01234567 = _mm256_fmadd_ps(vout2x01234567, vinput_scale2, vbias01234567);

    

    vout0x01234567 = _mm256_max_ps(vout0x01234567, vmin);
    vout1x01234567 = _mm256_max_ps(vout1x01234567, vmin);
    vout2x01234567 = _mm256_max_ps(vout2x01234567, vmin);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, vmax);
    vout1x01234567 = _mm256_min_ps(vout1x01234567, vmax);
    vout2x01234567 = _mm256_min_ps(vout2x01234567, vmax);

    if XNN_LIKELY(nc >= 8) {
      _mm256_storeu_ps(c0, vout0x01234567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm256_storeu_ps(c1, vout1x01234567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c2, vout2x01234567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 8;
    } else {
      __m128 vout0x0123 = _mm256_castps256_ps128(vout0x01234567);
      __m128 vout1x0123 = _mm256_castps256_ps128(vout1x01234567);
      __m128 vout2x0123 = _mm256_castps256_ps128(vout2x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vout0x0123);
        _mm_storeu_ps(c1, vout1x0123);
        _mm_storeu_ps(c2, vout2x0123);

        vout0x0123 = _mm256_extractf128_ps(vout0x01234567, 1);
        vout1x0123 = _mm256_extractf128_ps(vout1x01234567, 1);
        vout2x0123 = _mm256_extractf128_ps(vout2x01234567, 1);

        c0 += 4;
        c1 += 4;
        c2 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        _mm_storel_pi((__m64*) c1, vout1x0123);
        _mm_storel_pi((__m64*) c2, vout2x0123);

        vout0x0123 = _mm_movehl_ps(vout0x0123, vout0x0123);
        vout1x0123 = _mm_movehl_ps(vout1x0123, vout1x0123);
        vout2x0123 = _mm_movehl_ps(vout2x0123, vout2x0123);

        c0 += 2;
        c1 += 2;
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
