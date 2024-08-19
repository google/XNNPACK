// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
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


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x16c8__avx512vnni(
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
  assert(mr <= 9);
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
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }

  const __m512i vsign_mask = _mm512_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point + 128);
  const __m512i vinput_zero_point1 = _mm512_set1_epi32((int) quantization_params[1].zero_point + 128);
  const __m512i vinput_zero_point2 = _mm512_set1_epi32((int) quantization_params[2].zero_point + 128);
  const __m512i vinput_zero_point3 = _mm512_set1_epi32((int) quantization_params[3].zero_point + 128);
  const __m512i vinput_zero_point4 = _mm512_set1_epi32((int) quantization_params[4].zero_point + 128);
  const __m512i vinput_zero_point5 = _mm512_set1_epi32((int) quantization_params[5].zero_point + 128);
  const __m512i vinput_zero_point6 = _mm512_set1_epi32((int) quantization_params[6].zero_point + 128);
  const __m512i vinput_zero_point7 = _mm512_set1_epi32((int) quantization_params[7].zero_point + 128);
  const __m512i vinput_zero_point8 = _mm512_set1_epi32((int) quantization_params[8].zero_point + 128);
  const __m512 voutput_min = _mm512_set1_ps(params->avx512vnni.min);
  const __m512 voutput_max = _mm512_set1_ps(params->avx512vnni.max);
  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32(w);
    __m512i vsum0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point0);
    __m512i vacc0x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum0x0123456789ABCDEF, 0));
    __m512i vacc0x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum0x0123456789ABCDEF, 1));
    __m512i vsum1x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point1);
    __m512i vacc1x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum1x0123456789ABCDEF, 0));
    __m512i vacc1x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum1x0123456789ABCDEF, 1));
    __m512i vsum2x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point2);
    __m512i vacc2x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum2x0123456789ABCDEF, 0));
    __m512i vacc2x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum2x0123456789ABCDEF, 1));
    __m512i vsum3x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point3);
    __m512i vacc3x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum3x0123456789ABCDEF, 0));
    __m512i vacc3x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum3x0123456789ABCDEF, 1));
    __m512i vsum4x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point4);
    __m512i vacc4x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum4x0123456789ABCDEF, 0));
    __m512i vacc4x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum4x0123456789ABCDEF, 1));
    __m512i vsum5x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point5);
    __m512i vacc5x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum5x0123456789ABCDEF, 0));
    __m512i vacc5x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum5x0123456789ABCDEF, 1));
    __m512i vsum6x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point6);
    __m512i vacc6x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum6x0123456789ABCDEF, 0));
    __m512i vacc6x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum6x0123456789ABCDEF, 1));
    __m512i vsum7x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point7);
    __m512i vacc7x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum7x0123456789ABCDEF, 0));
    __m512i vacc7x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum7x0123456789ABCDEF, 1));
    __m512i vsum8x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point8);
    __m512i vacc8x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum8x0123456789ABCDEF, 0));
    __m512i vacc8x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum8x0123456789ABCDEF, 1));
    w = (const int32_t*) w + 16;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0)), vsign_mask);
      const __m512i va0x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8)), vsign_mask);
      a0 += 16;
      const __m512i va1x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a1)), vsign_mask);
      const __m512i va1x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a1 + 8)), vsign_mask);
      a1 += 16;
      const __m512i va2x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a2)), vsign_mask);
      const __m512i va2x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a2 + 8)), vsign_mask);
      a2 += 16;
      const __m512i va3x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a3)), vsign_mask);
      const __m512i va3x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a3 + 8)), vsign_mask);
      a3 += 16;
      const __m512i va4x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a4)), vsign_mask);
      const __m512i va4x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a4 + 8)), vsign_mask);
      a4 += 16;
      const __m512i va5x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a5)), vsign_mask);
      const __m512i va5x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a5 + 8)), vsign_mask);
      a5 += 16;
      const __m512i va6x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a6)), vsign_mask);
      const __m512i va6x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a6 + 8)), vsign_mask);
      a6 += 16;
      const __m512i va7x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a7)), vsign_mask);
      const __m512i va7x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a7 + 8)), vsign_mask);
      a7 += 16;
      const __m512i va8x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a8)), vsign_mask);
      const __m512i va8x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a8 + 8)), vsign_mask);
      a8 += 16;

      const __m512i vb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vb01234567x89ABCDEF = _mm512_load_si512((const int8_t*) w + 128);
      const __m512i vb89ABCDEFx89ABCDEF = _mm512_load_si512((const int8_t*) w + 192);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x01234567, vb01234567x01234567);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x01234567, vb89ABCDEFx01234567);
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x01234567, vb01234567x01234567);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x01234567, vb89ABCDEFx01234567);
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x01234567, vb01234567x01234567);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x01234567, vb89ABCDEFx01234567);
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x01234567, vb01234567x01234567);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x01234567, vb89ABCDEFx01234567);
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x01234567, vb01234567x01234567);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x01234567, vb89ABCDEFx01234567);
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, va6x01234567, vb01234567x01234567);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, va6x01234567, vb89ABCDEFx01234567);
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, va7x01234567, vb01234567x01234567);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, va7x01234567, vb89ABCDEFx01234567);
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, va8x01234567, vb01234567x01234567);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, va8x01234567, vb89ABCDEFx01234567);
      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x89ABCDEF, vb01234567x89ABCDEF);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x89ABCDEF, vb01234567x89ABCDEF);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x89ABCDEF, vb01234567x89ABCDEF);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x89ABCDEF, vb01234567x89ABCDEF);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x89ABCDEF, vb01234567x89ABCDEF);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x89ABCDEF, vb01234567x89ABCDEF);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, va6x89ABCDEF, vb01234567x89ABCDEF);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, va6x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, va7x89ABCDEF, vb01234567x89ABCDEF);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, va7x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, va8x89ABCDEF, vb01234567x89ABCDEF);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, va8x89ABCDEF, vb89ABCDEFx89ABCDEF);

      w = (const int8_t*) w + 256;
      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0)), vsign_mask);
      a0 += 8;
      const __m512i va1x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a1)), vsign_mask);
      a1 += 8;
      const __m512i va2x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a2)), vsign_mask);
      a2 += 8;
      const __m512i va3x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a3)), vsign_mask);
      a3 += 8;
      const __m512i va4x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a4)), vsign_mask);
      a4 += 8;
      const __m512i va5x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a5)), vsign_mask);
      a5 += 8;
      const __m512i va6x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a6)), vsign_mask);
      a6 += 8;
      const __m512i va7x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a7)), vsign_mask);
      a7 += 8;
      const __m512i va8x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a8)), vsign_mask);
      a8 += 8;

      const __m512i vb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x01234567, vb01234567x01234567);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x01234567, vb89ABCDEFx01234567);
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x01234567, vb01234567x01234567);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x01234567, vb89ABCDEFx01234567);
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x01234567, vb01234567x01234567);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x01234567, vb89ABCDEFx01234567);
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x01234567, vb01234567x01234567);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x01234567, vb89ABCDEFx01234567);
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x01234567, vb01234567x01234567);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x01234567, vb89ABCDEFx01234567);
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, va6x01234567, vb01234567x01234567);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, va6x01234567, vb89ABCDEFx01234567);
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, va7x01234567, vb01234567x01234567);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, va7x01234567, vb89ABCDEFx01234567);
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, va8x01234567, vb01234567x01234567);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, va8x01234567, vb89ABCDEFx01234567);

      w = (const int8_t*) w + 128;
      k -= 8 * sizeof(int8_t);
    }

    // Add adjacent pairs
    const __m512i vidx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i vsum0x01234567 = _mm512_add_epi32(vacc0x01234567, _mm512_srli_epi64(vacc0x01234567, 32));
    const __m512i vsum0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, _mm512_srli_epi64(vacc0x89ABCDEF, 32));
    __m512i vacc0x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum0x01234567, vidx, vsum0x89ABCDEF);
    const __m512i vsum1x01234567 = _mm512_add_epi32(vacc1x01234567, _mm512_srli_epi64(vacc1x01234567, 32));
    const __m512i vsum1x89ABCDEF = _mm512_add_epi32(vacc1x89ABCDEF, _mm512_srli_epi64(vacc1x89ABCDEF, 32));
    __m512i vacc1x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum1x01234567, vidx, vsum1x89ABCDEF);
    const __m512i vsum2x01234567 = _mm512_add_epi32(vacc2x01234567, _mm512_srli_epi64(vacc2x01234567, 32));
    const __m512i vsum2x89ABCDEF = _mm512_add_epi32(vacc2x89ABCDEF, _mm512_srli_epi64(vacc2x89ABCDEF, 32));
    __m512i vacc2x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum2x01234567, vidx, vsum2x89ABCDEF);
    const __m512i vsum3x01234567 = _mm512_add_epi32(vacc3x01234567, _mm512_srli_epi64(vacc3x01234567, 32));
    const __m512i vsum3x89ABCDEF = _mm512_add_epi32(vacc3x89ABCDEF, _mm512_srli_epi64(vacc3x89ABCDEF, 32));
    __m512i vacc3x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum3x01234567, vidx, vsum3x89ABCDEF);
    const __m512i vsum4x01234567 = _mm512_add_epi32(vacc4x01234567, _mm512_srli_epi64(vacc4x01234567, 32));
    const __m512i vsum4x89ABCDEF = _mm512_add_epi32(vacc4x89ABCDEF, _mm512_srli_epi64(vacc4x89ABCDEF, 32));
    __m512i vacc4x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum4x01234567, vidx, vsum4x89ABCDEF);
    const __m512i vsum5x01234567 = _mm512_add_epi32(vacc5x01234567, _mm512_srli_epi64(vacc5x01234567, 32));
    const __m512i vsum5x89ABCDEF = _mm512_add_epi32(vacc5x89ABCDEF, _mm512_srli_epi64(vacc5x89ABCDEF, 32));
    __m512i vacc5x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum5x01234567, vidx, vsum5x89ABCDEF);
    const __m512i vsum6x01234567 = _mm512_add_epi32(vacc6x01234567, _mm512_srli_epi64(vacc6x01234567, 32));
    const __m512i vsum6x89ABCDEF = _mm512_add_epi32(vacc6x89ABCDEF, _mm512_srli_epi64(vacc6x89ABCDEF, 32));
    __m512i vacc6x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum6x01234567, vidx, vsum6x89ABCDEF);
    const __m512i vsum7x01234567 = _mm512_add_epi32(vacc7x01234567, _mm512_srli_epi64(vacc7x01234567, 32));
    const __m512i vsum7x89ABCDEF = _mm512_add_epi32(vacc7x89ABCDEF, _mm512_srli_epi64(vacc7x89ABCDEF, 32));
    __m512i vacc7x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum7x01234567, vidx, vsum7x89ABCDEF);
    const __m512i vsum8x01234567 = _mm512_add_epi32(vacc8x01234567, _mm512_srli_epi64(vacc8x01234567, 32));
    const __m512i vsum8x89ABCDEF = _mm512_add_epi32(vacc8x89ABCDEF, _mm512_srli_epi64(vacc8x89ABCDEF, 32));
    __m512i vacc8x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum8x01234567, vidx, vsum8x89ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
    __m512 vscaled7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);
    __m512 vscaled8x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc8x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, _mm512_set1_ps(quantization_params[8].inv_scale));

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
    vscaled8x0123456789ABCDEF = _mm512_fmadd_ps(vscaled8x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);
    vscaled5x0123456789ABCDEF = _mm512_max_ps(vscaled5x0123456789ABCDEF, voutput_min);
    vscaled6x0123456789ABCDEF = _mm512_max_ps(vscaled6x0123456789ABCDEF, voutput_min);
    vscaled7x0123456789ABCDEF = _mm512_max_ps(vscaled7x0123456789ABCDEF, voutput_min);
    vscaled8x0123456789ABCDEF = _mm512_max_ps(vscaled8x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);
      _mm512_storeu_ps(c1, vscaled1x0123456789ABCDEF);
      _mm512_storeu_ps(c2, vscaled2x0123456789ABCDEF);
      _mm512_storeu_ps(c3, vscaled3x0123456789ABCDEF);
      _mm512_storeu_ps(c4, vscaled4x0123456789ABCDEF);
      _mm512_storeu_ps(c5, vscaled5x0123456789ABCDEF);
      _mm512_storeu_ps(c6, vscaled6x0123456789ABCDEF);
      _mm512_storeu_ps(c7, vscaled7x0123456789ABCDEF);
      _mm512_storeu_ps(c8, vscaled8x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);
      a8 = (const int8_t*) ((uintptr_t) a8 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);

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
      _mm512_mask_storeu_ps(c8, vmask, vscaled8x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
