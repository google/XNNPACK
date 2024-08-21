// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/prefetch.h"

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
  const __m512i vinput_zero_point1 = _mm512_set1_epi32((int) quantization_params[1].zero_point);
  const __m512i vinput_zero_point2 = _mm512_set1_epi32((int) quantization_params[2].zero_point);
  const __m512i vinput_zero_point3 = _mm512_set1_epi32((int) quantization_params[3].zero_point);
  const __m512i vinput_zero_point4 = _mm512_set1_epi32((int) quantization_params[4].zero_point);
  const __m512i vinput_zero_point5 = _mm512_set1_epi32((int) quantization_params[5].zero_point);
  const __m512i vinput_zero_point6 = _mm512_set1_epi32((int) quantization_params[6].zero_point);
  const __m512i vinput_zero_point7 = _mm512_set1_epi32((int) quantization_params[7].zero_point);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m256i vmask = _mm256_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);

  do {
    const __m512i vksum0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    const __m512i vksum4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    const __m512i vksum89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    const __m512i vksumCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);

    __m512i vacc0x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point0);
    __m512i vacc0x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point0);
    __m512i vacc0x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point0);
    __m512i vacc0xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point0);
    __m512i vacc1x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point1);
    __m512i vacc1x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point1);
    __m512i vacc1x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point1);
    __m512i vacc1xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point1);
    __m512i vacc2x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point2);
    __m512i vacc2x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point2);
    __m512i vacc2x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point2);
    __m512i vacc2xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point2);
    __m512i vacc3x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point3);
    __m512i vacc3x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point3);
    __m512i vacc3x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point3);
    __m512i vacc3xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point3);
    __m512i vacc4x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point4);
    __m512i vacc4x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point4);
    __m512i vacc4x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point4);
    __m512i vacc4xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point4);
    __m512i vacc5x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point5);
    __m512i vacc5x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point5);
    __m512i vacc5x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point5);
    __m512i vacc5xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point5);
    __m512i vacc6x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point6);
    __m512i vacc6x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point6);
    __m512i vacc6x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point6);
    __m512i vacc6xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point6);
    __m512i vacc7x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point7);
    __m512i vacc7x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point7);
    __m512i vacc7x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point7);
    __m512i vacc7xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point7);
    w = (const int32_t*) w + 16;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;
      __m512i va4 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a4)));
      a4 += 8;
      __m512i va5 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a5)));
      a5 += 8;
      __m512i va6 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a6)));
      a6 += 8;
      __m512i va7 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a7)));
      a7 += 8;

      __m256i vbb0123 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 0));
      __m256i vbs0123 = _mm256_slli_epi32(vbb0123, 4);
      __m256i vbm0123 = _mm256_and_si256(vbs0123, vmask);
      __m512i vb0123 = _mm512_cvtepi8_epi16(vbm0123);
      xnn_prefetch_to_l1((const int8_t*) w + 896);

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      vacc4x0123 = _mm512_add_epi32(vacc4x0123, _mm512_madd_epi16(va4, vb0123));
      vacc5x0123 = _mm512_add_epi32(vacc5x0123, _mm512_madd_epi16(va5, vb0123));
      vacc6x0123 = _mm512_add_epi32(vacc6x0123, _mm512_madd_epi16(va6, vb0123));
      vacc7x0123 = _mm512_add_epi32(vacc7x0123, _mm512_madd_epi16(va7, vb0123));
      __m256i vbb4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      __m256i vbs4567 = _mm256_slli_epi32(vbb4567, 4);
      __m256i vbm4567 = _mm256_and_si256(vbs4567, vmask);
      __m512i vb4567 = _mm512_cvtepi8_epi16(vbm4567);

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      vacc4x4567 = _mm512_add_epi32(vacc4x4567, _mm512_madd_epi16(va4, vb4567));
      vacc5x4567 = _mm512_add_epi32(vacc5x4567, _mm512_madd_epi16(va5, vb4567));
      vacc6x4567 = _mm512_add_epi32(vacc6x4567, _mm512_madd_epi16(va6, vb4567));
      vacc7x4567 = _mm512_add_epi32(vacc7x4567, _mm512_madd_epi16(va7, vb4567));
      __m256i vbb89AB = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 64));
      __m256i vbs89AB = _mm256_slli_epi32(vbb89AB, 4);
      __m256i vbm89AB = _mm256_and_si256(vbs89AB, vmask);
      __m512i vb89AB = _mm512_cvtepi8_epi16(vbm89AB);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      vacc4x89AB = _mm512_add_epi32(vacc4x89AB, _mm512_madd_epi16(va4, vb89AB));
      vacc5x89AB = _mm512_add_epi32(vacc5x89AB, _mm512_madd_epi16(va5, vb89AB));
      vacc6x89AB = _mm512_add_epi32(vacc6x89AB, _mm512_madd_epi16(va6, vb89AB));
      vacc7x89AB = _mm512_add_epi32(vacc7x89AB, _mm512_madd_epi16(va7, vb89AB));
      __m256i vbbCDEF = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 96));
      __m256i vbsCDEF = _mm256_slli_epi32(vbbCDEF, 4);
      __m256i vbmCDEF = _mm256_and_si256(vbsCDEF, vmask);
      __m512i vbCDEF = _mm512_cvtepi8_epi16(vbmCDEF);

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));
      vacc4xCDEF = _mm512_add_epi32(vacc4xCDEF, _mm512_madd_epi16(va4, vbCDEF));
      vacc5xCDEF = _mm512_add_epi32(vacc5xCDEF, _mm512_madd_epi16(va5, vbCDEF));
      vacc6xCDEF = _mm512_add_epi32(vacc6xCDEF, _mm512_madd_epi16(va6, vbCDEF));
      vacc7xCDEF = _mm512_add_epi32(vacc7xCDEF, _mm512_madd_epi16(va7, vbCDEF));

      va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;
      va4 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a4)));
      a4 += 8;
      va5 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a5)));
      a5 += 8;
      va6 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a6)));
      a6 += 8;
      va7 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a7)));
      a7 += 8;

      vbm0123 = _mm256_and_si256(vbb0123, vmask);
      vb0123 = _mm512_cvtepi8_epi16(vbm0123);

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      vacc4x0123 = _mm512_add_epi32(vacc4x0123, _mm512_madd_epi16(va4, vb0123));
      vacc5x0123 = _mm512_add_epi32(vacc5x0123, _mm512_madd_epi16(va5, vb0123));
      vacc6x0123 = _mm512_add_epi32(vacc6x0123, _mm512_madd_epi16(va6, vb0123));
      vacc7x0123 = _mm512_add_epi32(vacc7x0123, _mm512_madd_epi16(va7, vb0123));
      vbm4567 = _mm256_and_si256(vbb4567, vmask);
      vb4567 = _mm512_cvtepi8_epi16(vbm4567);

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      vacc4x4567 = _mm512_add_epi32(vacc4x4567, _mm512_madd_epi16(va4, vb4567));
      vacc5x4567 = _mm512_add_epi32(vacc5x4567, _mm512_madd_epi16(va5, vb4567));
      vacc6x4567 = _mm512_add_epi32(vacc6x4567, _mm512_madd_epi16(va6, vb4567));
      vacc7x4567 = _mm512_add_epi32(vacc7x4567, _mm512_madd_epi16(va7, vb4567));
      vbm89AB = _mm256_and_si256(vbb89AB, vmask);
      vb89AB = _mm512_cvtepi8_epi16(vbm89AB);

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      vacc4x89AB = _mm512_add_epi32(vacc4x89AB, _mm512_madd_epi16(va4, vb89AB));
      vacc5x89AB = _mm512_add_epi32(vacc5x89AB, _mm512_madd_epi16(va5, vb89AB));
      vacc6x89AB = _mm512_add_epi32(vacc6x89AB, _mm512_madd_epi16(va6, vb89AB));
      vacc7x89AB = _mm512_add_epi32(vacc7x89AB, _mm512_madd_epi16(va7, vb89AB));
      vbmCDEF = _mm256_and_si256(vbbCDEF, vmask);
      vbCDEF = _mm512_cvtepi8_epi16(vbmCDEF);

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));
      vacc4xCDEF = _mm512_add_epi32(vacc4xCDEF, _mm512_madd_epi16(va4, vbCDEF));
      vacc5xCDEF = _mm512_add_epi32(vacc5xCDEF, _mm512_madd_epi16(va5, vbCDEF));
      vacc6xCDEF = _mm512_add_epi32(vacc6xCDEF, _mm512_madd_epi16(va6, vbCDEF));
      vacc7xCDEF = _mm512_add_epi32(vacc7xCDEF, _mm512_madd_epi16(va7, vbCDEF));

      w = (const int8_t*) w + 128;
      k -= 16 * sizeof(int8_t);
    }

    while (k >= 8 * sizeof(int8_t)) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;
      const __m512i va4 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a4)));
      a4 += 8;
      const __m512i va5 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a5)));
      a5 += 8;
      const __m512i va6 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a6)));
      a6 += 8;
      const __m512i va7 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a7)));
      a7 += 8;

      const __m256i vbb0123 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 0));
      const __m256i vbs0123 = _mm256_slli_epi32(vbb0123, 4);
      const __m256i vbm0123 = _mm256_and_si256(vbs0123, vmask);
      const __m512i vb0123 = _mm512_cvtepi8_epi16(vbm0123);
      xnn_prefetch_to_l1((const int8_t*) w + 896);

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      vacc4x0123 = _mm512_add_epi32(vacc4x0123, _mm512_madd_epi16(va4, vb0123));
      vacc5x0123 = _mm512_add_epi32(vacc5x0123, _mm512_madd_epi16(va5, vb0123));
      vacc6x0123 = _mm512_add_epi32(vacc6x0123, _mm512_madd_epi16(va6, vb0123));
      vacc7x0123 = _mm512_add_epi32(vacc7x0123, _mm512_madd_epi16(va7, vb0123));
      const __m256i vbb4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vbs4567 = _mm256_slli_epi32(vbb4567, 4);
      const __m256i vbm4567 = _mm256_and_si256(vbs4567, vmask);
      const __m512i vb4567 = _mm512_cvtepi8_epi16(vbm4567);

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      vacc4x4567 = _mm512_add_epi32(vacc4x4567, _mm512_madd_epi16(va4, vb4567));
      vacc5x4567 = _mm512_add_epi32(vacc5x4567, _mm512_madd_epi16(va5, vb4567));
      vacc6x4567 = _mm512_add_epi32(vacc6x4567, _mm512_madd_epi16(va6, vb4567));
      vacc7x4567 = _mm512_add_epi32(vacc7x4567, _mm512_madd_epi16(va7, vb4567));
      const __m256i vbb89AB = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 64));
      const __m256i vbs89AB = _mm256_slli_epi32(vbb89AB, 4);
      const __m256i vbm89AB = _mm256_and_si256(vbs89AB, vmask);
      const __m512i vb89AB = _mm512_cvtepi8_epi16(vbm89AB);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      vacc4x89AB = _mm512_add_epi32(vacc4x89AB, _mm512_madd_epi16(va4, vb89AB));
      vacc5x89AB = _mm512_add_epi32(vacc5x89AB, _mm512_madd_epi16(va5, vb89AB));
      vacc6x89AB = _mm512_add_epi32(vacc6x89AB, _mm512_madd_epi16(va6, vb89AB));
      vacc7x89AB = _mm512_add_epi32(vacc7x89AB, _mm512_madd_epi16(va7, vb89AB));
      const __m256i vbbCDEF = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 96));
      const __m256i vbsCDEF = _mm256_slli_epi32(vbbCDEF, 4);
      const __m256i vbmCDEF = _mm256_and_si256(vbsCDEF, vmask);
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(vbmCDEF);

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));
      vacc4xCDEF = _mm512_add_epi32(vacc4xCDEF, _mm512_madd_epi16(va4, vbCDEF));
      vacc5xCDEF = _mm512_add_epi32(vacc5xCDEF, _mm512_madd_epi16(va5, vbCDEF));
      vacc6xCDEF = _mm512_add_epi32(vacc6xCDEF, _mm512_madd_epi16(va6, vbCDEF));
      vacc7xCDEF = _mm512_add_epi32(vacc7xCDEF, _mm512_madd_epi16(va7, vbCDEF));

      w = (const int8_t*) w + 128;
      k -= 8 * sizeof(int8_t);
    }

    // Add 4 adjacent sums
    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));
    const __m512i vacc4x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x0123, vacc4x4567), _mm512_unpackhi_epi32(vacc4x0123, vacc4x4567));
    const __m512i vacc4x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x89AB, vacc4xCDEF), _mm512_unpackhi_epi32(vacc4x89AB, vacc4xCDEF));
    const __m512i vacc5x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc5x0123, vacc5x4567), _mm512_unpackhi_epi32(vacc5x0123, vacc5x4567));
    const __m512i vacc5x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc5x89AB, vacc5xCDEF), _mm512_unpackhi_epi32(vacc5x89AB, vacc5xCDEF));
    const __m512i vacc6x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc6x0123, vacc6x4567), _mm512_unpackhi_epi32(vacc6x0123, vacc6x4567));
    const __m512i vacc6x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc6x89AB, vacc6xCDEF), _mm512_unpackhi_epi32(vacc6x89AB, vacc6xCDEF));
    const __m512i vacc7x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc7x0123, vacc7x4567), _mm512_unpackhi_epi32(vacc7x0123, vacc7x4567));
    const __m512i vacc7x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc7x89AB, vacc7xCDEF), _mm512_unpackhi_epi32(vacc7x89AB, vacc7xCDEF));

    const __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    const __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    const __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    const __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));
    const __m512i vacc4x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x04152637, vacc4x8C9DAEBF), _mm512_unpackhi_epi32(vacc4x04152637, vacc4x8C9DAEBF));
    const __m512i vacc5x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc5x04152637, vacc5x8C9DAEBF), _mm512_unpackhi_epi32(vacc5x04152637, vacc5x8C9DAEBF));
    const __m512i vacc6x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc6x04152637, vacc6x8C9DAEBF), _mm512_unpackhi_epi32(vacc6x04152637, vacc6x8C9DAEBF));
    const __m512i vacc7x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc7x04152637, vacc7x8C9DAEBF), _mm512_unpackhi_epi32(vacc7x04152637, vacc7x8C9DAEBF));

    const __m512i vidx = _mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0);
    __m512i vacc0x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc0x084C195D2A6E3B7F);
    __m512i vacc1x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc1x084C195D2A6E3B7F);
    __m512i vacc2x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc2x084C195D2A6E3B7F);
    __m512i vacc3x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc3x084C195D2A6E3B7F);
    __m512i vacc4x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc4x084C195D2A6E3B7F);
    __m512i vacc5x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc5x084C195D2A6E3B7F);
    __m512i vacc6x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc6x084C195D2A6E3B7F);
    __m512i vacc7x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc7x084C195D2A6E3B7F);

    vacc0x0123456789ABCDEF = _mm512_srai_epi32(vacc0x0123456789ABCDEF, 4);
    vacc1x0123456789ABCDEF = _mm512_srai_epi32(vacc1x0123456789ABCDEF, 4);
    vacc2x0123456789ABCDEF = _mm512_srai_epi32(vacc2x0123456789ABCDEF, 4);
    vacc3x0123456789ABCDEF = _mm512_srai_epi32(vacc3x0123456789ABCDEF, 4);
    vacc4x0123456789ABCDEF = _mm512_srai_epi32(vacc4x0123456789ABCDEF, 4);
    vacc5x0123456789ABCDEF = _mm512_srai_epi32(vacc5x0123456789ABCDEF, 4);
    vacc6x0123456789ABCDEF = _mm512_srai_epi32(vacc6x0123456789ABCDEF, 4);
    vacc7x0123456789ABCDEF = _mm512_srai_epi32(vacc7x0123456789ABCDEF, 4);
    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
    __m512 vscaled7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, _mm512_set1_ps(quantization_params[7].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vscaled4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_fmadd_ps(vscaled5x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_fmadd_ps(vscaled6x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_fmadd_ps(vscaled7x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);
    vscaled5x0123456789ABCDEF = _mm512_max_ps(vscaled5x0123456789ABCDEF, voutput_min);
    vscaled6x0123456789ABCDEF = _mm512_max_ps(vscaled6x0123456789ABCDEF, voutput_min);
    vscaled7x0123456789ABCDEF = _mm512_max_ps(vscaled7x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);
      _mm512_storeu_ps(c1, vscaled1x0123456789ABCDEF);
      _mm512_storeu_ps(c2, vscaled2x0123456789ABCDEF);
      _mm512_storeu_ps(c3, vscaled3x0123456789ABCDEF);
      _mm512_storeu_ps(c4, vscaled4x0123456789ABCDEF);
      _mm512_storeu_ps(c5, vscaled5x0123456789ABCDEF);
      _mm512_storeu_ps(c6, vscaled6x0123456789ABCDEF);
      _mm512_storeu_ps(c7, vscaled7x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3, vmask, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4, vmask, vscaled4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c5, vmask, vscaled5x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c6, vmask, vscaled6x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c7, vmask, vscaled7x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
