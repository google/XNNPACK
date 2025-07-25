// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
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
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);

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

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      vacc4x0123 = _mm512_add_epi32(vacc4x0123, _mm512_madd_epi16(va4, vb0123));
      vacc5x0123 = _mm512_add_epi32(vacc5x0123, _mm512_madd_epi16(va5, vb0123));
      vacc6x0123 = _mm512_add_epi32(vacc6x0123, _mm512_madd_epi16(va6, vb0123));
      vacc7x0123 = _mm512_add_epi32(vacc7x0123, _mm512_madd_epi16(va7, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      vacc4x4567 = _mm512_add_epi32(vacc4x4567, _mm512_madd_epi16(va4, vb4567));
      vacc5x4567 = _mm512_add_epi32(vacc5x4567, _mm512_madd_epi16(va5, vb4567));
      vacc6x4567 = _mm512_add_epi32(vacc6x4567, _mm512_madd_epi16(va6, vb4567));
      vacc7x4567 = _mm512_add_epi32(vacc7x4567, _mm512_madd_epi16(va7, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      vacc4x89AB = _mm512_add_epi32(vacc4x89AB, _mm512_madd_epi16(va4, vb89AB));
      vacc5x89AB = _mm512_add_epi32(vacc5x89AB, _mm512_madd_epi16(va5, vb89AB));
      vacc6x89AB = _mm512_add_epi32(vacc6x89AB, _mm512_madd_epi16(va6, vb89AB));
      vacc7x89AB = _mm512_add_epi32(vacc7x89AB, _mm512_madd_epi16(va7, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

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
