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
#include "src/xnnpack/prefetch.h"

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 5);
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

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
  const __m512i vinput_zero_point1 = _mm512_set1_epi32((int) quantization_params[1].zero_point);
  const __m512i vinput_zero_point2 = _mm512_set1_epi32((int) quantization_params[2].zero_point);
  const __m512i vinput_zero_point3 = _mm512_set1_epi32((int) quantization_params[3].zero_point);
  const __m512i vinput_zero_point4 = _mm512_set1_epi32((int) quantization_params[4].zero_point);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  const __m256i vmask = _mm256_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);

  do {
    const __m512i vksum0 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    const __m512i vksum2 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    const __m512i vksum3 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);

    __m512i vacc0x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point0);
    __m512i vacc0x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point0);
    __m512i vacc0x2 = _mm512_mullo_epi32(vksum2, vinput_zero_point0);
    __m512i vacc0x3 = _mm512_mullo_epi32(vksum3, vinput_zero_point0);
    __m512i vacc1x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point1);
    __m512i vacc1x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point1);
    __m512i vacc1x2 = _mm512_mullo_epi32(vksum2, vinput_zero_point1);
    __m512i vacc1x3 = _mm512_mullo_epi32(vksum3, vinput_zero_point1);
    __m512i vacc2x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point2);
    __m512i vacc2x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point2);
    __m512i vacc2x2 = _mm512_mullo_epi32(vksum2, vinput_zero_point2);
    __m512i vacc2x3 = _mm512_mullo_epi32(vksum3, vinput_zero_point2);
    __m512i vacc3x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point3);
    __m512i vacc3x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point3);
    __m512i vacc3x2 = _mm512_mullo_epi32(vksum2, vinput_zero_point3);
    __m512i vacc3x3 = _mm512_mullo_epi32(vksum3, vinput_zero_point3);
    __m512i vacc4x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point4);
    __m512i vacc4x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point4);
    __m512i vacc4x2 = _mm512_mullo_epi32(vksum2, vinput_zero_point4);
    __m512i vacc4x3 = _mm512_mullo_epi32(vksum3, vinput_zero_point4);
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

      __m256i vbb0 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 0));
      __m256i vbs0 = _mm256_slli_epi32(vbb0, 4);
      __m256i vbm0 = _mm256_and_si256(vbs0, vmask);
      __m512i vb0 = _mm512_cvtepi8_epi16(vbm0);
      xnn_prefetch_to_l1((const int8_t*) w + 896);

      vacc0x0 = _mm512_add_epi32(vacc0x0, _mm512_madd_epi16(va0, vb0));
      vacc1x0 = _mm512_add_epi32(vacc1x0, _mm512_madd_epi16(va1, vb0));
      vacc2x0 = _mm512_add_epi32(vacc2x0, _mm512_madd_epi16(va2, vb0));
      vacc3x0 = _mm512_add_epi32(vacc3x0, _mm512_madd_epi16(va3, vb0));
      vacc4x0 = _mm512_add_epi32(vacc4x0, _mm512_madd_epi16(va4, vb0));
      __m256i vbb1 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      __m256i vbs1 = _mm256_slli_epi32(vbb1, 4);
      __m256i vbm1 = _mm256_and_si256(vbs1, vmask);
      __m512i vb1 = _mm512_cvtepi8_epi16(vbm1);

      vacc0x1 = _mm512_add_epi32(vacc0x1, _mm512_madd_epi16(va0, vb1));
      vacc1x1 = _mm512_add_epi32(vacc1x1, _mm512_madd_epi16(va1, vb1));
      vacc2x1 = _mm512_add_epi32(vacc2x1, _mm512_madd_epi16(va2, vb1));
      vacc3x1 = _mm512_add_epi32(vacc3x1, _mm512_madd_epi16(va3, vb1));
      vacc4x1 = _mm512_add_epi32(vacc4x1, _mm512_madd_epi16(va4, vb1));
      __m256i vbb2 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 64));
      __m256i vbs2 = _mm256_slli_epi32(vbb2, 4);
      __m256i vbm2 = _mm256_and_si256(vbs2, vmask);
      __m512i vb2 = _mm512_cvtepi8_epi16(vbm2);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      vacc0x2 = _mm512_add_epi32(vacc0x2, _mm512_madd_epi16(va0, vb2));
      vacc1x2 = _mm512_add_epi32(vacc1x2, _mm512_madd_epi16(va1, vb2));
      vacc2x2 = _mm512_add_epi32(vacc2x2, _mm512_madd_epi16(va2, vb2));
      vacc3x2 = _mm512_add_epi32(vacc3x2, _mm512_madd_epi16(va3, vb2));
      vacc4x2 = _mm512_add_epi32(vacc4x2, _mm512_madd_epi16(va4, vb2));
      __m256i vbb3 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 96));
      __m256i vbs3 = _mm256_slli_epi32(vbb3, 4);
      __m256i vbm3 = _mm256_and_si256(vbs3, vmask);
      __m512i vb3 = _mm512_cvtepi8_epi16(vbm3);

      vacc0x3 = _mm512_add_epi32(vacc0x3, _mm512_madd_epi16(va0, vb3));
      vacc1x3 = _mm512_add_epi32(vacc1x3, _mm512_madd_epi16(va1, vb3));
      vacc2x3 = _mm512_add_epi32(vacc2x3, _mm512_madd_epi16(va2, vb3));
      vacc3x3 = _mm512_add_epi32(vacc3x3, _mm512_madd_epi16(va3, vb3));
      vacc4x3 = _mm512_add_epi32(vacc4x3, _mm512_madd_epi16(va4, vb3));

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

      vbm0 = _mm256_and_si256(vbb0, vmask);
      vb0 = _mm512_cvtepi8_epi16(vbm0);

      vacc0x0 = _mm512_add_epi32(vacc0x0, _mm512_madd_epi16(va0, vb0));
      vacc1x0 = _mm512_add_epi32(vacc1x0, _mm512_madd_epi16(va1, vb0));
      vacc2x0 = _mm512_add_epi32(vacc2x0, _mm512_madd_epi16(va2, vb0));
      vacc3x0 = _mm512_add_epi32(vacc3x0, _mm512_madd_epi16(va3, vb0));
      vacc4x0 = _mm512_add_epi32(vacc4x0, _mm512_madd_epi16(va4, vb0));
      vbm1 = _mm256_and_si256(vbb1, vmask);
      vb1 = _mm512_cvtepi8_epi16(vbm1);

      vacc0x1 = _mm512_add_epi32(vacc0x1, _mm512_madd_epi16(va0, vb1));
      vacc1x1 = _mm512_add_epi32(vacc1x1, _mm512_madd_epi16(va1, vb1));
      vacc2x1 = _mm512_add_epi32(vacc2x1, _mm512_madd_epi16(va2, vb1));
      vacc3x1 = _mm512_add_epi32(vacc3x1, _mm512_madd_epi16(va3, vb1));
      vacc4x1 = _mm512_add_epi32(vacc4x1, _mm512_madd_epi16(va4, vb1));
      vbm2 = _mm256_and_si256(vbb2, vmask);
      vb2 = _mm512_cvtepi8_epi16(vbm2);

      vacc0x2 = _mm512_add_epi32(vacc0x2, _mm512_madd_epi16(va0, vb2));
      vacc1x2 = _mm512_add_epi32(vacc1x2, _mm512_madd_epi16(va1, vb2));
      vacc2x2 = _mm512_add_epi32(vacc2x2, _mm512_madd_epi16(va2, vb2));
      vacc3x2 = _mm512_add_epi32(vacc3x2, _mm512_madd_epi16(va3, vb2));
      vacc4x2 = _mm512_add_epi32(vacc4x2, _mm512_madd_epi16(va4, vb2));
      vbm3 = _mm256_and_si256(vbb3, vmask);
      vb3 = _mm512_cvtepi8_epi16(vbm3);

      vacc0x3 = _mm512_add_epi32(vacc0x3, _mm512_madd_epi16(va0, vb3));
      vacc1x3 = _mm512_add_epi32(vacc1x3, _mm512_madd_epi16(va1, vb3));
      vacc2x3 = _mm512_add_epi32(vacc2x3, _mm512_madd_epi16(va2, vb3));
      vacc3x3 = _mm512_add_epi32(vacc3x3, _mm512_madd_epi16(va3, vb3));
      vacc4x3 = _mm512_add_epi32(vacc4x3, _mm512_madd_epi16(va4, vb3));

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

      const __m256i vbb0 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 0));
      const __m256i vbs0 = _mm256_slli_epi32(vbb0, 4);
      const __m256i vbm0 = _mm256_and_si256(vbs0, vmask);
      const __m512i vb0 = _mm512_cvtepi8_epi16(vbm0);
      xnn_prefetch_to_l1((const int8_t*) w + 896);

      vacc0x0 = _mm512_add_epi32(vacc0x0, _mm512_madd_epi16(va0, vb0));
      vacc1x0 = _mm512_add_epi32(vacc1x0, _mm512_madd_epi16(va1, vb0));
      vacc2x0 = _mm512_add_epi32(vacc2x0, _mm512_madd_epi16(va2, vb0));
      vacc3x0 = _mm512_add_epi32(vacc3x0, _mm512_madd_epi16(va3, vb0));
      vacc4x0 = _mm512_add_epi32(vacc4x0, _mm512_madd_epi16(va4, vb0));
      const __m256i vbb1 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vbs1 = _mm256_slli_epi32(vbb1, 4);
      const __m256i vbm1 = _mm256_and_si256(vbs1, vmask);
      const __m512i vb1 = _mm512_cvtepi8_epi16(vbm1);

      vacc0x1 = _mm512_add_epi32(vacc0x1, _mm512_madd_epi16(va0, vb1));
      vacc1x1 = _mm512_add_epi32(vacc1x1, _mm512_madd_epi16(va1, vb1));
      vacc2x1 = _mm512_add_epi32(vacc2x1, _mm512_madd_epi16(va2, vb1));
      vacc3x1 = _mm512_add_epi32(vacc3x1, _mm512_madd_epi16(va3, vb1));
      vacc4x1 = _mm512_add_epi32(vacc4x1, _mm512_madd_epi16(va4, vb1));
      const __m256i vbb2 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 64));
      const __m256i vbs2 = _mm256_slli_epi32(vbb2, 4);
      const __m256i vbm2 = _mm256_and_si256(vbs2, vmask);
      const __m512i vb2 = _mm512_cvtepi8_epi16(vbm2);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      vacc0x2 = _mm512_add_epi32(vacc0x2, _mm512_madd_epi16(va0, vb2));
      vacc1x2 = _mm512_add_epi32(vacc1x2, _mm512_madd_epi16(va1, vb2));
      vacc2x2 = _mm512_add_epi32(vacc2x2, _mm512_madd_epi16(va2, vb2));
      vacc3x2 = _mm512_add_epi32(vacc3x2, _mm512_madd_epi16(va3, vb2));
      vacc4x2 = _mm512_add_epi32(vacc4x2, _mm512_madd_epi16(va4, vb2));
      const __m256i vbb3 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 96));
      const __m256i vbs3 = _mm256_slli_epi32(vbb3, 4);
      const __m256i vbm3 = _mm256_and_si256(vbs3, vmask);
      const __m512i vb3 = _mm512_cvtepi8_epi16(vbm3);

      vacc0x3 = _mm512_add_epi32(vacc0x3, _mm512_madd_epi16(va0, vb3));
      vacc1x3 = _mm512_add_epi32(vacc1x3, _mm512_madd_epi16(va1, vb3));
      vacc2x3 = _mm512_add_epi32(vacc2x3, _mm512_madd_epi16(va2, vb3));
      vacc3x3 = _mm512_add_epi32(vacc3x3, _mm512_madd_epi16(va3, vb3));
      vacc4x3 = _mm512_add_epi32(vacc4x3, _mm512_madd_epi16(va4, vb3));

      w = (const int8_t*) w + 128;
      k -= 8 * sizeof(int8_t);
    }

    // Add 4 adjacent sums
    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0, vacc0x1), _mm512_unpackhi_epi32(vacc0x0, vacc0x1));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x2, vacc0x3), _mm512_unpackhi_epi32(vacc0x2, vacc0x3));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0, vacc1x1), _mm512_unpackhi_epi32(vacc1x0, vacc1x1));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x2, vacc1x3), _mm512_unpackhi_epi32(vacc1x2, vacc1x3));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0, vacc2x1), _mm512_unpackhi_epi32(vacc2x0, vacc2x1));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x2, vacc2x3), _mm512_unpackhi_epi32(vacc2x2, vacc2x3));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0, vacc3x1), _mm512_unpackhi_epi32(vacc3x0, vacc3x1));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x2, vacc3x3), _mm512_unpackhi_epi32(vacc3x2, vacc3x3));
    const __m512i vacc4x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x0, vacc4x1), _mm512_unpackhi_epi32(vacc4x0, vacc4x1));
    const __m512i vacc4x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x2, vacc4x3), _mm512_unpackhi_epi32(vacc4x2, vacc4x3));

    const __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    const __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    const __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    const __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));
    const __m512i vacc4x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x04152637, vacc4x8C9DAEBF), _mm512_unpackhi_epi32(vacc4x04152637, vacc4x8C9DAEBF));

    const __m512i vidx = _mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0);
    __m512i vacc0x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc0x084C195D2A6E3B7F);
    __m512i vacc1x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc1x084C195D2A6E3B7F);
    __m512i vacc2x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc2x084C195D2A6E3B7F);
    __m512i vacc3x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc3x084C195D2A6E3B7F);
    __m512i vacc4x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc4x084C195D2A6E3B7F);

    vacc0x0123456789ABCDEF = _mm512_srai_epi32(vacc0x0123456789ABCDEF, 4);
    vacc1x0123456789ABCDEF = _mm512_srai_epi32(vacc1x0123456789ABCDEF, 4);
    vacc2x0123456789ABCDEF = _mm512_srai_epi32(vacc2x0123456789ABCDEF, 4);
    vacc3x0123456789ABCDEF = _mm512_srai_epi32(vacc3x0123456789ABCDEF, 4);
    vacc4x0123456789ABCDEF = _mm512_srai_epi32(vacc4x0123456789ABCDEF, 4);
    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, _mm512_set1_ps(quantization_params[4].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vscaled4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);
      _mm512_storeu_ps(c1, vscaled1x0123456789ABCDEF);
      _mm512_storeu_ps(c2, vscaled2x0123456789ABCDEF);
      _mm512_storeu_ps(c3, vscaled3x0123456789ABCDEF);
      _mm512_storeu_ps(c4, vscaled4x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3, vmask, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4, vmask, vscaled4x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
