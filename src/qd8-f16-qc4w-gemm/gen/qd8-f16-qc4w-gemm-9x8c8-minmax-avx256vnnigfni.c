// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx8c8-avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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

  const __m256i vsign_mask = _mm256_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m256i vinput_zero_point0 = _mm256_set1_epi32((int) quantization_params[0].zero_point + 128);
  const __m256i vinput_zero_point1 = _mm256_set1_epi32((int) quantization_params[1].zero_point + 128);
  const __m256i vinput_zero_point2 = _mm256_set1_epi32((int) quantization_params[2].zero_point + 128);
  const __m256i vinput_zero_point3 = _mm256_set1_epi32((int) quantization_params[3].zero_point + 128);
  const __m256i vinput_zero_point4 = _mm256_set1_epi32((int) quantization_params[4].zero_point + 128);
  const __m256i vinput_zero_point5 = _mm256_set1_epi32((int) quantization_params[5].zero_point + 128);
  const __m256i vinput_zero_point6 = _mm256_set1_epi32((int) quantization_params[6].zero_point + 128);
  const __m256i vinput_zero_point7 = _mm256_set1_epi32((int) quantization_params[7].zero_point + 128);
  const __m256i vinput_zero_point8 = _mm256_set1_epi32((int) quantization_params[8].zero_point + 128);
  const __m256 voutput_min = _mm256_set1_ps(params->avxvnni.min);
  const __m256 voutput_max = _mm256_set1_ps(params->avxvnni.max);
  const __m256i vmask = _mm256_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  const __m256i vshl4 = _mm256_set1_epi64x(0x01020408);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m256i vksum01234567 = _mm256_load_si256(w);
    __m256i vsum0x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point0);
    __m256i vacc0x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum0x01234567, 0));
    __m256i vacc0x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum0x01234567, 1));
    __m256i vsum1x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point1);
    __m256i vacc1x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum1x01234567, 0));
    __m256i vacc1x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum1x01234567, 1));
    __m256i vsum2x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point2);
    __m256i vacc2x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum2x01234567, 0));
    __m256i vacc2x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum2x01234567, 1));
    __m256i vsum3x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point3);
    __m256i vacc3x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum3x01234567, 0));
    __m256i vacc3x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum3x01234567, 1));
    __m256i vsum4x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point4);
    __m256i vacc4x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum4x01234567, 0));
    __m256i vacc4x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum4x01234567, 1));
    __m256i vsum5x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point5);
    __m256i vacc5x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum5x01234567, 0));
    __m256i vacc5x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum5x01234567, 1));
    __m256i vsum6x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point6);
    __m256i vacc6x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum6x01234567, 0));
    __m256i vacc6x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum6x01234567, 1));
    __m256i vsum7x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point7);
    __m256i vacc7x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum7x01234567, 0));
    __m256i vacc7x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum7x01234567, 1));
    __m256i vsum8x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point8);
    __m256i vacc8x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum8x01234567, 0));
    __m256i vacc8x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum8x01234567, 1));
    w = (const int32_t*) w + 8;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m256i va0x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a0)), vsign_mask);
      const __m256i va0x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8)), vsign_mask);
      a0 += 16;
      const __m256i va1x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a1)), vsign_mask);
      const __m256i va1x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8)), vsign_mask);
      a1 += 16;
      const __m256i va2x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a2)), vsign_mask);
      const __m256i va2x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8)), vsign_mask);
      a2 += 16;
      const __m256i va3x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a3)), vsign_mask);
      const __m256i va3x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a3 + 8)), vsign_mask);
      a3 += 16;
      const __m256i va4x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a4)), vsign_mask);
      const __m256i va4x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a4 + 8)), vsign_mask);
      a4 += 16;
      const __m256i va5x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a5)), vsign_mask);
      const __m256i va5x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a5 + 8)), vsign_mask);
      a5 += 16;
      const __m256i va6x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a6)), vsign_mask);
      const __m256i va6x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a6 + 8)), vsign_mask);
      a6 += 16;
      const __m256i va7x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a7)), vsign_mask);
      const __m256i va7x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a7 + 8)), vsign_mask);
      a7 += 16;
      const __m256i va8x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a8)), vsign_mask);
      const __m256i va8x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a8 + 8)), vsign_mask);
      a8 += 16;

      const __m256i vbb01234567x01234567 = _mm256_load_si256(w);
      const __m256i vbb89ABCDEFx01234567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vb01234567x0123 = _mm256_gf2p8affine_epi64_epi8(vbb01234567x01234567, vshl4, 0);
      const __m256i vb89ABCDEFx0123 = _mm256_gf2p8affine_epi64_epi8(vbb89ABCDEFx01234567, vshl4, 0);
      const __m256i vb01234567x4567 = _mm256_and_si256(vbb01234567x01234567, vmask);
      const __m256i vb89ABCDEFx4567 = _mm256_and_si256(vbb89ABCDEFx01234567, vmask);

      vacc0x0123 = _mm256_dpbusd_epi32(vacc0x0123, va0x01234567, vb01234567x0123);
      vacc0x4567 = _mm256_dpbusd_epi32(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
      vacc1x0123 = _mm256_dpbusd_epi32(vacc1x0123, va1x01234567, vb01234567x0123);
      vacc1x4567 = _mm256_dpbusd_epi32(vacc1x4567, va1x01234567, vb89ABCDEFx0123);
      vacc2x0123 = _mm256_dpbusd_epi32(vacc2x0123, va2x01234567, vb01234567x0123);
      vacc2x4567 = _mm256_dpbusd_epi32(vacc2x4567, va2x01234567, vb89ABCDEFx0123);
      vacc3x0123 = _mm256_dpbusd_epi32(vacc3x0123, va3x01234567, vb01234567x0123);
      vacc3x4567 = _mm256_dpbusd_epi32(vacc3x4567, va3x01234567, vb89ABCDEFx0123);
      vacc4x0123 = _mm256_dpbusd_epi32(vacc4x0123, va4x01234567, vb01234567x0123);
      vacc4x4567 = _mm256_dpbusd_epi32(vacc4x4567, va4x01234567, vb89ABCDEFx0123);
      vacc5x0123 = _mm256_dpbusd_epi32(vacc5x0123, va5x01234567, vb01234567x0123);
      vacc5x4567 = _mm256_dpbusd_epi32(vacc5x4567, va5x01234567, vb89ABCDEFx0123);
      vacc6x0123 = _mm256_dpbusd_epi32(vacc6x0123, va6x01234567, vb01234567x0123);
      vacc6x4567 = _mm256_dpbusd_epi32(vacc6x4567, va6x01234567, vb89ABCDEFx0123);
      vacc7x0123 = _mm256_dpbusd_epi32(vacc7x0123, va7x01234567, vb01234567x0123);
      vacc7x4567 = _mm256_dpbusd_epi32(vacc7x4567, va7x01234567, vb89ABCDEFx0123);
      vacc8x0123 = _mm256_dpbusd_epi32(vacc8x0123, va8x01234567, vb01234567x0123);
      vacc8x4567 = _mm256_dpbusd_epi32(vacc8x4567, va8x01234567, vb89ABCDEFx0123);
      vacc0x0123 = _mm256_dpbusd_epi32(vacc0x0123, va0x89ABCDEF, vb01234567x4567);
      vacc0x4567 = _mm256_dpbusd_epi32(vacc0x4567, va0x89ABCDEF, vb89ABCDEFx4567);
      vacc1x0123 = _mm256_dpbusd_epi32(vacc1x0123, va1x89ABCDEF, vb01234567x4567);
      vacc1x4567 = _mm256_dpbusd_epi32(vacc1x4567, va1x89ABCDEF, vb89ABCDEFx4567);
      vacc2x0123 = _mm256_dpbusd_epi32(vacc2x0123, va2x89ABCDEF, vb01234567x4567);
      vacc2x4567 = _mm256_dpbusd_epi32(vacc2x4567, va2x89ABCDEF, vb89ABCDEFx4567);
      vacc3x0123 = _mm256_dpbusd_epi32(vacc3x0123, va3x89ABCDEF, vb01234567x4567);
      vacc3x4567 = _mm256_dpbusd_epi32(vacc3x4567, va3x89ABCDEF, vb89ABCDEFx4567);
      vacc4x0123 = _mm256_dpbusd_epi32(vacc4x0123, va4x89ABCDEF, vb01234567x4567);
      vacc4x4567 = _mm256_dpbusd_epi32(vacc4x4567, va4x89ABCDEF, vb89ABCDEFx4567);
      vacc5x0123 = _mm256_dpbusd_epi32(vacc5x0123, va5x89ABCDEF, vb01234567x4567);
      vacc5x4567 = _mm256_dpbusd_epi32(vacc5x4567, va5x89ABCDEF, vb89ABCDEFx4567);
      vacc6x0123 = _mm256_dpbusd_epi32(vacc6x0123, va6x89ABCDEF, vb01234567x4567);
      vacc6x4567 = _mm256_dpbusd_epi32(vacc6x4567, va6x89ABCDEF, vb89ABCDEFx4567);
      vacc7x0123 = _mm256_dpbusd_epi32(vacc7x0123, va7x89ABCDEF, vb01234567x4567);
      vacc7x4567 = _mm256_dpbusd_epi32(vacc7x4567, va7x89ABCDEF, vb89ABCDEFx4567);
      vacc8x0123 = _mm256_dpbusd_epi32(vacc8x0123, va8x89ABCDEF, vb01234567x4567);
      vacc8x4567 = _mm256_dpbusd_epi32(vacc8x4567, va8x89ABCDEF, vb89ABCDEFx4567);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const __m256i va0x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a0)), vsign_mask);
      a0 += 8;
      const __m256i va1x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a1)), vsign_mask);
      a1 += 8;
      const __m256i va2x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a2)), vsign_mask);
      a2 += 8;
      const __m256i va3x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a3)), vsign_mask);
      a3 += 8;
      const __m256i va4x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a4)), vsign_mask);
      a4 += 8;
      const __m256i va5x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a5)), vsign_mask);
      a5 += 8;
      const __m256i va6x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a6)), vsign_mask);
      a6 += 8;
      const __m256i va7x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a7)), vsign_mask);
      a7 += 8;
      const __m256i va8x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a8)), vsign_mask);
      a8 += 8;

      const __m256i vbb01234567x01234567 = _mm256_load_si256(w);
      const __m256i vbb89ABCDEFx01234567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vb01234567x0123 = _mm256_gf2p8affine_epi64_epi8(vbb01234567x01234567, vshl4, 0);
      const __m256i vb89ABCDEFx0123 = _mm256_gf2p8affine_epi64_epi8(vbb89ABCDEFx01234567, vshl4, 0);

      vacc0x0123 = _mm256_dpbusd_epi32(vacc0x0123, va0x01234567, vb01234567x0123);
      vacc0x4567 = _mm256_dpbusd_epi32(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
      vacc1x0123 = _mm256_dpbusd_epi32(vacc1x0123, va1x01234567, vb01234567x0123);
      vacc1x4567 = _mm256_dpbusd_epi32(vacc1x4567, va1x01234567, vb89ABCDEFx0123);
      vacc2x0123 = _mm256_dpbusd_epi32(vacc2x0123, va2x01234567, vb01234567x0123);
      vacc2x4567 = _mm256_dpbusd_epi32(vacc2x4567, va2x01234567, vb89ABCDEFx0123);
      vacc3x0123 = _mm256_dpbusd_epi32(vacc3x0123, va3x01234567, vb01234567x0123);
      vacc3x4567 = _mm256_dpbusd_epi32(vacc3x4567, va3x01234567, vb89ABCDEFx0123);
      vacc4x0123 = _mm256_dpbusd_epi32(vacc4x0123, va4x01234567, vb01234567x0123);
      vacc4x4567 = _mm256_dpbusd_epi32(vacc4x4567, va4x01234567, vb89ABCDEFx0123);
      vacc5x0123 = _mm256_dpbusd_epi32(vacc5x0123, va5x01234567, vb01234567x0123);
      vacc5x4567 = _mm256_dpbusd_epi32(vacc5x4567, va5x01234567, vb89ABCDEFx0123);
      vacc6x0123 = _mm256_dpbusd_epi32(vacc6x0123, va6x01234567, vb01234567x0123);
      vacc6x4567 = _mm256_dpbusd_epi32(vacc6x4567, va6x01234567, vb89ABCDEFx0123);
      vacc7x0123 = _mm256_dpbusd_epi32(vacc7x0123, va7x01234567, vb01234567x0123);
      vacc7x4567 = _mm256_dpbusd_epi32(vacc7x4567, va7x01234567, vb89ABCDEFx0123);
      vacc8x0123 = _mm256_dpbusd_epi32(vacc8x0123, va8x01234567, vb01234567x0123);
      vacc8x4567 = _mm256_dpbusd_epi32(vacc8x4567, va8x01234567, vb89ABCDEFx0123);

      w = (const int8_t*) w + 64;
      k -= 8 * sizeof(int8_t);
    }

    // Add adjacent pairs
    const __m256i vsum0x02134657 = _mm256_hadd_epi32(vacc0x0123, vacc0x4567);
    __m256i vacc0x01234567 = _mm256_permute4x64_epi64(vsum0x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum1x02134657 = _mm256_hadd_epi32(vacc1x0123, vacc1x4567);
    __m256i vacc1x01234567 = _mm256_permute4x64_epi64(vsum1x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum2x02134657 = _mm256_hadd_epi32(vacc2x0123, vacc2x4567);
    __m256i vacc2x01234567 = _mm256_permute4x64_epi64(vsum2x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum3x02134657 = _mm256_hadd_epi32(vacc3x0123, vacc3x4567);
    __m256i vacc3x01234567 = _mm256_permute4x64_epi64(vsum3x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum4x02134657 = _mm256_hadd_epi32(vacc4x0123, vacc4x4567);
    __m256i vacc4x01234567 = _mm256_permute4x64_epi64(vsum4x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum5x02134657 = _mm256_hadd_epi32(vacc5x0123, vacc5x4567);
    __m256i vacc5x01234567 = _mm256_permute4x64_epi64(vsum5x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum6x02134657 = _mm256_hadd_epi32(vacc6x0123, vacc6x4567);
    __m256i vacc6x01234567 = _mm256_permute4x64_epi64(vsum6x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum7x02134657 = _mm256_hadd_epi32(vacc7x0123, vacc7x4567);
    __m256i vacc7x01234567 = _mm256_permute4x64_epi64(vsum7x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum8x02134657 = _mm256_hadd_epi32(vacc8x0123, vacc8x4567);
    __m256i vacc8x01234567 = _mm256_permute4x64_epi64(vsum8x02134657, _MM_SHUFFLE(3, 1, 2, 0));

    vacc0x01234567 = _mm256_srai_epi32(vacc0x01234567, 4);
    vacc1x01234567 = _mm256_srai_epi32(vacc1x01234567, 4);
    vacc2x01234567 = _mm256_srai_epi32(vacc2x01234567, 4);
    vacc3x01234567 = _mm256_srai_epi32(vacc3x01234567, 4);
    vacc4x01234567 = _mm256_srai_epi32(vacc4x01234567, 4);
    vacc5x01234567 = _mm256_srai_epi32(vacc5x01234567, 4);
    vacc6x01234567 = _mm256_srai_epi32(vacc6x01234567, 4);
    vacc7x01234567 = _mm256_srai_epi32(vacc7x01234567, 4);
    vacc8x01234567 = _mm256_srai_epi32(vacc8x01234567, 4);
    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vout1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vout2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);
    __m256 vout3x01234567 = _mm256_cvtepi32_ps(vacc3x01234567);
    __m256 vout4x01234567 = _mm256_cvtepi32_ps(vacc4x01234567);
    __m256 vout5x01234567 = _mm256_cvtepi32_ps(vacc5x01234567);
    __m256 vout6x01234567 = _mm256_cvtepi32_ps(vacc6x01234567);
    __m256 vout7x01234567 = _mm256_cvtepi32_ps(vacc7x01234567);
    __m256 vout8x01234567 = _mm256_cvtepi32_ps(vacc8x01234567);

    vout0x01234567 = _mm256_mul_ps(vout0x01234567, _mm256_set1_ps(quantization_params[0].inv_scale));
    vout1x01234567 = _mm256_mul_ps(vout1x01234567, _mm256_set1_ps(quantization_params[1].inv_scale));
    vout2x01234567 = _mm256_mul_ps(vout2x01234567, _mm256_set1_ps(quantization_params[2].inv_scale));
    vout3x01234567 = _mm256_mul_ps(vout3x01234567, _mm256_set1_ps(quantization_params[3].inv_scale));
    vout4x01234567 = _mm256_mul_ps(vout4x01234567, _mm256_set1_ps(quantization_params[4].inv_scale));
    vout5x01234567 = _mm256_mul_ps(vout5x01234567, _mm256_set1_ps(quantization_params[5].inv_scale));
    vout6x01234567 = _mm256_mul_ps(vout6x01234567, _mm256_set1_ps(quantization_params[6].inv_scale));
    vout7x01234567 = _mm256_mul_ps(vout7x01234567, _mm256_set1_ps(quantization_params[7].inv_scale));
    vout8x01234567 = _mm256_mul_ps(vout8x01234567, _mm256_set1_ps(quantization_params[8].inv_scale));

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
    vout8x01234567 = _mm256_fmadd_ps(vout8x01234567, vfilter_output_scale01234567, vbias01234567);

    vout0x01234567 = _mm256_max_ps(vout0x01234567, voutput_min);
    vout1x01234567 = _mm256_max_ps(vout1x01234567, voutput_min);
    vout2x01234567 = _mm256_max_ps(vout2x01234567, voutput_min);
    vout3x01234567 = _mm256_max_ps(vout3x01234567, voutput_min);
    vout4x01234567 = _mm256_max_ps(vout4x01234567, voutput_min);
    vout5x01234567 = _mm256_max_ps(vout5x01234567, voutput_min);
    vout6x01234567 = _mm256_max_ps(vout6x01234567, voutput_min);
    vout7x01234567 = _mm256_max_ps(vout7x01234567, voutput_min);
    vout8x01234567 = _mm256_max_ps(vout8x01234567, voutput_min);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, voutput_max);
    vout1x01234567 = _mm256_min_ps(vout1x01234567, voutput_max);
    vout2x01234567 = _mm256_min_ps(vout2x01234567, voutput_max);
    vout3x01234567 = _mm256_min_ps(vout3x01234567, voutput_max);
    vout4x01234567 = _mm256_min_ps(vout4x01234567, voutput_max);
    vout5x01234567 = _mm256_min_ps(vout5x01234567, voutput_max);
    vout6x01234567 = _mm256_min_ps(vout6x01234567, voutput_max);
    vout7x01234567 = _mm256_min_ps(vout7x01234567, voutput_max);
    vout8x01234567 = _mm256_min_ps(vout8x01234567, voutput_max);

    __m128i vfp16out0x01234567 = _mm256_cvtps_ph(vout0x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out1x01234567 = _mm256_cvtps_ph(vout1x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out2x01234567 = _mm256_cvtps_ph(vout2x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out3x01234567 = _mm256_cvtps_ph(vout3x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out4x01234567 = _mm256_cvtps_ph(vout4x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out5x01234567 = _mm256_cvtps_ph(vout5x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out6x01234567 = _mm256_cvtps_ph(vout6x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out7x01234567 = _mm256_cvtps_ph(vout7x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out8x01234567 = _mm256_cvtps_ph(vout8x01234567, _MM_FROUND_TO_NEAREST_INT);
    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_si128((__m128i*) c0, vfp16out0x01234567);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_si128((__m128i*) c1, vfp16out1x01234567);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) c2, vfp16out2x01234567);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) c3, vfp16out3x01234567);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) c4, vfp16out4x01234567);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) c5, vfp16out5x01234567);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) c6, vfp16out6x01234567);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);
      _mm_storeu_si128((__m128i*) c7, vfp16out7x01234567);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);
      c7 = (uint16_t*) ((uintptr_t) c7 + cn_stride);
      _mm_storeu_si128((__m128i*) c8, vfp16out8x01234567);
      a8 = (const int8_t*) ((uintptr_t) a8 - kc);
      c8 = (uint16_t*) ((uintptr_t) c8 + cn_stride);
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
      _mm_mask_storeu_epi16(c8, vmask, vfp16out8x01234567);
      nc = 0;
    }
  } while (nc != 0);
}
