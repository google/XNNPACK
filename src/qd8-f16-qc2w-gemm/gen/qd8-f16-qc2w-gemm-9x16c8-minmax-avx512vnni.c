// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
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
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f16_qc2w_gemm_minmax_ukernel_9x16c8__avx512vnni(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params,
    const float* row_sum,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 9);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const size_t original_kc = kc;
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
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  uint16_t* c8 = (uint16_t*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }

  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
  const __m512i vinput_zero_point1 = _mm512_set1_epi32((int) quantization_params[1].zero_point);
  const __m512i vinput_zero_point2 = _mm512_set1_epi32((int) quantization_params[2].zero_point);
  const __m512i vinput_zero_point3 = _mm512_set1_epi32((int) quantization_params[3].zero_point);
  const __m512i vinput_zero_point4 = _mm512_set1_epi32((int) quantization_params[4].zero_point);
  const __m512i vinput_zero_point5 = _mm512_set1_epi32((int) quantization_params[5].zero_point);
  const __m512i vinput_zero_point6 = _mm512_set1_epi32((int) quantization_params[6].zero_point);
  const __m512i vinput_zero_point7 = _mm512_set1_epi32((int) quantization_params[7].zero_point);
  const __m512i vinput_zero_point8 = _mm512_set1_epi32((int) quantization_params[8].zero_point);
  const __m512 voutput_min = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512 voutput_max = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.max));
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  const __m512i vmask = _mm512_set1_epi8(0x03);
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
    const void* kzp = w;
    w = (const float*) w + 16;

    size_t k = kc;
    while (k >= 32 * sizeof(int8_t)) {
      const __m512i vbb01234567x0123456789ABCDEF = _mm512_load_si512(w);
      const __m512i vbb89ABCDEFx0123456789ABCDEF = _mm512_load_si512((const int8_t*) w + 64);

      const __m512i vb01234567x0 = _mm512_and_si512(vbb01234567x0123456789ABCDEF, vmask);
      const __m512i vb89ABCDEFx0 = _mm512_and_si512(vbb89ABCDEFx0123456789ABCDEF, vmask);
      const __m512i va0x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x0, va0x0);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx0, va0x0);
      const __m512i va1x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a1));
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, vb01234567x0, va1x0);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, vb89ABCDEFx0, va1x0);
      const __m512i va2x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a2));
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, vb01234567x0, va2x0);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, vb89ABCDEFx0, va2x0);
      const __m512i va3x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a3));
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, vb01234567x0, va3x0);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, vb89ABCDEFx0, va3x0);
      const __m512i va4x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a4));
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, vb01234567x0, va4x0);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, vb89ABCDEFx0, va4x0);
      const __m512i va5x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a5));
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, vb01234567x0, va5x0);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, vb89ABCDEFx0, va5x0);
      const __m512i va6x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a6));
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, vb01234567x0, va6x0);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, vb89ABCDEFx0, va6x0);
      const __m512i va7x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a7));
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, vb01234567x0, va7x0);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, vb89ABCDEFx0, va7x0);
      const __m512i va8x0 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a8));
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, vb01234567x0, va8x0);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, vb89ABCDEFx0, va8x0);

      const __m512i vb01234567x1 = _mm512_and_si512(_mm512_srli_epi32(vbb01234567x0123456789ABCDEF, 2), vmask);
      const __m512i vb89ABCDEFx1 = _mm512_and_si512(_mm512_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 2), vmask);
      const __m512i va0x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8));
      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x1, va0x1);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx1, va0x1);
      const __m512i va1x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a1 + 8));
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, vb01234567x1, va1x1);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, vb89ABCDEFx1, va1x1);
      const __m512i va2x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a2 + 8));
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, vb01234567x1, va2x1);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, vb89ABCDEFx1, va2x1);
      const __m512i va3x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a3 + 8));
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, vb01234567x1, va3x1);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, vb89ABCDEFx1, va3x1);
      const __m512i va4x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a4 + 8));
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, vb01234567x1, va4x1);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, vb89ABCDEFx1, va4x1);
      const __m512i va5x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a5 + 8));
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, vb01234567x1, va5x1);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, vb89ABCDEFx1, va5x1);
      const __m512i va6x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a6 + 8));
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, vb01234567x1, va6x1);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, vb89ABCDEFx1, va6x1);
      const __m512i va7x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a7 + 8));
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, vb01234567x1, va7x1);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, vb89ABCDEFx1, va7x1);
      const __m512i va8x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a8 + 8));
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, vb01234567x1, va8x1);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, vb89ABCDEFx1, va8x1);


      const __m512i vb01234567x2 = _mm512_and_si512(_mm512_srli_epi32(vbb01234567x0123456789ABCDEF, 4), vmask);
      const __m512i vb89ABCDEFx2 = _mm512_and_si512(_mm512_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 4), vmask);
      const __m512i va0x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 16));
      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x2, va0x2);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx2, va0x2);
      const __m512i va1x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a1 + 16));
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, vb01234567x2, va1x2);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, vb89ABCDEFx2, va1x2);
      const __m512i va2x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a2 + 16));
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, vb01234567x2, va2x2);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, vb89ABCDEFx2, va2x2);
      const __m512i va3x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a3 + 16));
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, vb01234567x2, va3x2);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, vb89ABCDEFx2, va3x2);
      const __m512i va4x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a4 + 16));
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, vb01234567x2, va4x2);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, vb89ABCDEFx2, va4x2);
      const __m512i va5x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a5 + 16));
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, vb01234567x2, va5x2);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, vb89ABCDEFx2, va5x2);
      const __m512i va6x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a6 + 16));
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, vb01234567x2, va6x2);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, vb89ABCDEFx2, va6x2);
      const __m512i va7x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a7 + 16));
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, vb01234567x2, va7x2);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, vb89ABCDEFx2, va7x2);
      const __m512i va8x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a8 + 16));
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, vb01234567x2, va8x2);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, vb89ABCDEFx2, va8x2);

      const __m512i vb01234567x3 = _mm512_and_si512(_mm512_srli_epi32(vbb01234567x0123456789ABCDEF, 6), vmask);
      const __m512i vb89ABCDEFx3 = _mm512_and_si512(_mm512_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 6), vmask);
      const __m512i va0x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 24));
      a0 += 32;
      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x3, va0x3);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx3, va0x3);
      const __m512i va1x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a1 + 24));
      a1 += 32;
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, vb01234567x3, va1x3);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, vb89ABCDEFx3, va1x3);
      const __m512i va2x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a2 + 24));
      a2 += 32;
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, vb01234567x3, va2x3);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, vb89ABCDEFx3, va2x3);
      const __m512i va3x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a3 + 24));
      a3 += 32;
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, vb01234567x3, va3x3);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, vb89ABCDEFx3, va3x3);
      const __m512i va4x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a4 + 24));
      a4 += 32;
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, vb01234567x3, va4x3);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, vb89ABCDEFx3, va4x3);
      const __m512i va5x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a5 + 24));
      a5 += 32;
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, vb01234567x3, va5x3);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, vb89ABCDEFx3, va5x3);
      const __m512i va6x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a6 + 24));
      a6 += 32;
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, vb01234567x3, va6x3);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, vb89ABCDEFx3, va6x3);
      const __m512i va7x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a7 + 24));
      a7 += 32;
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, vb01234567x3, va7x3);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, vb89ABCDEFx3, va7x3);
      const __m512i va8x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a8 + 24));
      a8 += 32;
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, vb01234567x3, va8x3);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, vb89ABCDEFx3, va8x3);

      w = (const int8_t*) w + 128;
      k -= 32 * sizeof(int8_t);
    }
    while (k >= 16 * sizeof(int8_t)) {
      const __m512i va0x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
      const __m512i va0x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8));
      a0 += 16;
      const __m512i va1x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a1));
      const __m512i va1x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a1 + 8));
      a1 += 16;
      const __m512i va2x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a2));
      const __m512i va2x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a2 + 8));
      a2 += 16;
      const __m512i va3x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a3));
      const __m512i va3x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a3 + 8));
      a3 += 16;
      const __m512i va4x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a4));
      const __m512i va4x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a4 + 8));
      a4 += 16;
      const __m512i va5x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a5));
      const __m512i va5x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a5 + 8));
      a5 += 16;
      const __m512i va6x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a6));
      const __m512i va6x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a6 + 8));
      a6 += 16;
      const __m512i va7x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a7));
      const __m512i va7x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a7 + 8));
      a7 += 16;
      const __m512i va8x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a8));
      const __m512i va8x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a8 + 8));
      a8 += 16;

      const __m512i vbb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vbb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vb01234567x01234567 = _mm512_and_si512(vbb01234567x01234567, vmask);
      const __m512i vb89ABCDEFx01234567 = _mm512_and_si512(vbb89ABCDEFx01234567, vmask);
      const __m512i vbs01234567x89ABCDEF = _mm512_srli_epi32(vbb01234567x01234567, 2);
      const __m512i vbs89ABCDEFx89ABCDEF = _mm512_srli_epi32(vbb89ABCDEFx01234567, 2);
      const __m512i vb01234567x89ABCDEF = _mm512_and_si512(vbs01234567x89ABCDEF, vmask);
      const __m512i vb89ABCDEFx89ABCDEF = _mm512_and_si512(vbs89ABCDEFx89ABCDEF, vmask);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x01234567, va0x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx01234567, va0x01234567);
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, vb01234567x01234567, va1x01234567);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, vb89ABCDEFx01234567, va1x01234567);
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, vb01234567x01234567, va2x01234567);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, vb89ABCDEFx01234567, va2x01234567);
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, vb01234567x01234567, va3x01234567);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, vb89ABCDEFx01234567, va3x01234567);
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, vb01234567x01234567, va4x01234567);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, vb89ABCDEFx01234567, va4x01234567);
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, vb01234567x01234567, va5x01234567);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, vb89ABCDEFx01234567, va5x01234567);
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, vb01234567x01234567, va6x01234567);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, vb89ABCDEFx01234567, va6x01234567);
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, vb01234567x01234567, va7x01234567);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, vb89ABCDEFx01234567, va7x01234567);
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, vb01234567x01234567, va8x01234567);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, vb89ABCDEFx01234567, va8x01234567);
      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x89ABCDEF, va0x89ABCDEF);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx89ABCDEF, va0x89ABCDEF);
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, vb01234567x89ABCDEF, va1x89ABCDEF);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, vb89ABCDEFx89ABCDEF, va1x89ABCDEF);
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, vb01234567x89ABCDEF, va2x89ABCDEF);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, vb89ABCDEFx89ABCDEF, va2x89ABCDEF);
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, vb01234567x89ABCDEF, va3x89ABCDEF);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, vb89ABCDEFx89ABCDEF, va3x89ABCDEF);
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, vb01234567x89ABCDEF, va4x89ABCDEF);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, vb89ABCDEFx89ABCDEF, va4x89ABCDEF);
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, vb01234567x89ABCDEF, va5x89ABCDEF);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, vb89ABCDEFx89ABCDEF, va5x89ABCDEF);
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, vb01234567x89ABCDEF, va6x89ABCDEF);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, vb89ABCDEFx89ABCDEF, va6x89ABCDEF);
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, vb01234567x89ABCDEF, va7x89ABCDEF);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, vb89ABCDEFx89ABCDEF, va7x89ABCDEF);
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, vb01234567x89ABCDEF, va8x89ABCDEF);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, vb89ABCDEFx89ABCDEF, va8x89ABCDEF);

      w = (const int8_t*) w + 128;
      k -= 16 * sizeof(int8_t);
      if (k != 0) {
        const __m512i vb01234567x2 = _mm512_and_si512(_mm512_srli_epi32(vbb01234567x01234567, 4), vmask);
        const __m512i vb89ABCDEFx2 = _mm512_and_si512(_mm512_srli_epi32(vbb89ABCDEFx01234567, 4), vmask);
        const __m512i va0x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
        a0 += 8;
        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x2, va0x2);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx2, va0x2);
        const __m512i va1x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a1));
        a1 += 8;
        vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, vb01234567x2, va1x2);
        vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, vb89ABCDEFx2, va1x2);
        const __m512i va2x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a2));
        a2 += 8;
        vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, vb01234567x2, va2x2);
        vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, vb89ABCDEFx2, va2x2);
        const __m512i va3x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a3));
        a3 += 8;
        vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, vb01234567x2, va3x2);
        vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, vb89ABCDEFx2, va3x2);
        const __m512i va4x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a4));
        a4 += 8;
        vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, vb01234567x2, va4x2);
        vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, vb89ABCDEFx2, va4x2);
        const __m512i va5x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a5));
        a5 += 8;
        vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, vb01234567x2, va5x2);
        vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, vb89ABCDEFx2, va5x2);
        const __m512i va6x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a6));
        a6 += 8;
        vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, vb01234567x2, va6x2);
        vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, vb89ABCDEFx2, va6x2);
        const __m512i va7x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a7));
        a7 += 8;
        vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, vb01234567x2, va7x2);
        vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, vb89ABCDEFx2, va7x2);
        const __m512i va8x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a8));
        a8 += 8;
        vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, vb01234567x2, va8x2);
        vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, vb89ABCDEFx2, va8x2);
        k -= 8 * sizeof(int8_t);
      }
    }

    if (k != 0) {
      const __m512i va0x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
      a0 += 8;
      const __m512i va1x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a1));
      a1 += 8;
      const __m512i va2x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a2));
      a2 += 8;
      const __m512i va3x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a3));
      a3 += 8;
      const __m512i va4x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a4));
      a4 += 8;
      const __m512i va5x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a5));
      a5 += 8;
      const __m512i va6x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a6));
      a6 += 8;
      const __m512i va7x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a7));
      a7 += 8;
      const __m512i va8x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a8));
      a8 += 8;

      const __m512i vbb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vbb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vb01234567x01234567 = _mm512_and_si512(vbb01234567x01234567, vmask);
      const __m512i vb89ABCDEFx01234567 = _mm512_and_si512(vbb89ABCDEFx01234567, vmask);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x01234567, va0x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx01234567, va0x01234567);
      vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, vb01234567x01234567, va1x01234567);
      vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, vb89ABCDEFx01234567, va1x01234567);
      vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, vb01234567x01234567, va2x01234567);
      vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, vb89ABCDEFx01234567, va2x01234567);
      vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, vb01234567x01234567, va3x01234567);
      vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, vb89ABCDEFx01234567, va3x01234567);
      vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, vb01234567x01234567, va4x01234567);
      vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, vb89ABCDEFx01234567, va4x01234567);
      vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, vb01234567x01234567, va5x01234567);
      vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, vb89ABCDEFx01234567, va5x01234567);
      vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, vb01234567x01234567, va6x01234567);
      vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, vb89ABCDEFx01234567, va6x01234567);
      vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, vb01234567x01234567, va7x01234567);
      vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, vb89ABCDEFx01234567, va7x01234567);
      vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, vb01234567x01234567, va8x01234567);
      vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, vb89ABCDEFx01234567, va8x01234567);

      w = (const int8_t*) w + 128;
      k -= 8 * sizeof(int8_t);
    }
    assert(k == 0);

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

    const __m512 rh_zero_points_0123456789ABCDEF = _mm512_load_ps((const float*) kzp);
    const __m512 biased_rh_zero_points_0123456789ABCDEF = _mm512_add_ps(rh_zero_points_0123456789ABCDEF, _mm512_set1_ps(2.0f));

    // Subtract out the scaled left-hand row sums.
    const __m512 lh_row_sum_0 = _mm512_set1_ps(row_sum[0]);
    vscaled0x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_0, vscaled0x0123456789ABCDEF);
    const __m512 lh_row_sum_1 = _mm512_set1_ps(row_sum[1]);
    vscaled1x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_1, vscaled1x0123456789ABCDEF);
    const __m512 lh_row_sum_2 = _mm512_set1_ps(row_sum[2]);
    vscaled2x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_2, vscaled2x0123456789ABCDEF);
    const __m512 lh_row_sum_3 = _mm512_set1_ps(row_sum[3]);
    vscaled3x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_3, vscaled3x0123456789ABCDEF);
    const __m512 lh_row_sum_4 = _mm512_set1_ps(row_sum[4]);
    vscaled4x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_4, vscaled4x0123456789ABCDEF);
    const __m512 lh_row_sum_5 = _mm512_set1_ps(row_sum[5]);
    vscaled5x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_5, vscaled5x0123456789ABCDEF);
    const __m512 lh_row_sum_6 = _mm512_set1_ps(row_sum[6]);
    vscaled6x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_6, vscaled6x0123456789ABCDEF);
    const __m512 lh_row_sum_7 = _mm512_set1_ps(row_sum[7]);
    vscaled7x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_7, vscaled7x0123456789ABCDEF);
    const __m512 lh_row_sum_8 = _mm512_set1_ps(row_sum[8]);
    vscaled8x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_8, vscaled8x0123456789ABCDEF);
    // Add the product of left/right-hand zero points and `kc`.
    const __m512 vscaled_lh_zero_point_0 = _mm512_set1_ps((float) original_kc * quantization_params[0].zero_point);
    const __m512 vscaled_lh_zero_point_1 = _mm512_set1_ps((float) original_kc * quantization_params[1].zero_point);
    const __m512 vscaled_lh_zero_point_2 = _mm512_set1_ps((float) original_kc * quantization_params[2].zero_point);
    const __m512 vscaled_lh_zero_point_3 = _mm512_set1_ps((float) original_kc * quantization_params[3].zero_point);
    const __m512 vscaled_lh_zero_point_4 = _mm512_set1_ps((float) original_kc * quantization_params[4].zero_point);
    const __m512 vscaled_lh_zero_point_5 = _mm512_set1_ps((float) original_kc * quantization_params[5].zero_point);
    const __m512 vscaled_lh_zero_point_6 = _mm512_set1_ps((float) original_kc * quantization_params[6].zero_point);
    const __m512 vscaled_lh_zero_point_7 = _mm512_set1_ps((float) original_kc * quantization_params[7].zero_point);
    const __m512 vscaled_lh_zero_point_8 = _mm512_set1_ps((float) original_kc * quantization_params[8].zero_point);
    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_0, vscaled0x0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_1, vscaled1x0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_2, vscaled2x0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_3, vscaled3x0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_4, vscaled4x0123456789ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_5, vscaled5x0123456789ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_6, vscaled6x0123456789ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_7, vscaled7x0123456789ABCDEF);
    vscaled8x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_8, vscaled8x0123456789ABCDEF);
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
      _mm256_storeu_si256((__m256i*) c0, _mm512_cvtps_ph(vscaled0x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) c1, _mm512_cvtps_ph(vscaled1x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) c2, _mm512_cvtps_ph(vscaled2x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) c3, _mm512_cvtps_ph(vscaled3x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) c4, _mm512_cvtps_ph(vscaled4x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) c5, _mm512_cvtps_ph(vscaled5x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) c6, _mm512_cvtps_ph(vscaled6x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) c7, _mm512_cvtps_ph(vscaled7x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) c8, _mm512_cvtps_ph(vscaled8x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);
      a8 = (const int8_t*) ((uintptr_t) a8 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);
      c7 = (uint16_t*) ((uintptr_t) c7 + cn_stride);
      c8 = (uint16_t*) ((uintptr_t) c8 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 16-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm256_mask_storeu_epi16(c0, vmask, _mm512_cvtps_ph(vscaled0x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c1, vmask, _mm512_cvtps_ph(vscaled1x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c2, vmask, _mm512_cvtps_ph(vscaled2x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c3, vmask, _mm512_cvtps_ph(vscaled3x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c4, vmask, _mm512_cvtps_ph(vscaled4x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c5, vmask, _mm512_cvtps_ph(vscaled5x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c6, vmask, _mm512_cvtps_ph(vscaled6x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c7, vmask, _mm512_cvtps_ph(vscaled7x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c8, vmask, _mm512_cvtps_ph(vscaled8x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      nc = 0;
    }
  } while (nc != 0);
}
