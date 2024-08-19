// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx8c8-avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
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


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x8c8__avx256vnni(
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
  assert(mr <= 14);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    c7 = c6;
  }
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    c8 = c7;
  }
  float* c9 = (float*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    c9 = c8;
  }
  float* c10 = (float*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    c10 = c9;
  }
  float* c11 = (float*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    c11 = c10;
  }
  float* c12 = (float*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    c12 = c11;
  }
  float* c13 = (float*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 14) {
    c13 = c12;
  }

  const __m256i vsign_mask = _mm256_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m256i vinput_zero_point = _mm256_set1_epi32((int) quantization_params->zero_point + 128);
  const __m256 vinput_inv_scale = _mm256_set1_ps(quantization_params->inv_scale);
  const __m256 voutput_min = _mm256_set1_ps(params->avxvnni.min);
  const __m256 voutput_max = _mm256_set1_ps(params->avxvnni.max);
  do {
    const __m256i vksum01234567 = _mm256_load_si256(w);
    const __m256i vsum0x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point);
    __m256i vacc0x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum0x01234567, 0));
    __m256i vacc0x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum0x01234567, 1));
    __m256i vacc1x0123 = vacc0x0123;
    __m256i vacc1x4567 = vacc0x4567;
    __m256i vacc2x0123 = vacc0x0123;
    __m256i vacc2x4567 = vacc0x4567;
    __m256i vacc3x0123 = vacc0x0123;
    __m256i vacc3x4567 = vacc0x4567;
    __m256i vacc4x0123 = vacc0x0123;
    __m256i vacc4x4567 = vacc0x4567;
    __m256i vacc5x0123 = vacc0x0123;
    __m256i vacc5x4567 = vacc0x4567;
    __m256i vacc6x0123 = vacc0x0123;
    __m256i vacc6x4567 = vacc0x4567;
    __m256i vacc7x0123 = vacc0x0123;
    __m256i vacc7x4567 = vacc0x4567;
    __m256i vacc8x0123 = vacc0x0123;
    __m256i vacc8x4567 = vacc0x4567;
    __m256i vacc9x0123 = vacc0x0123;
    __m256i vacc9x4567 = vacc0x4567;
    __m256i vacc10x0123 = vacc0x0123;
    __m256i vacc10x4567 = vacc0x4567;
    __m256i vacc11x0123 = vacc0x0123;
    __m256i vacc11x4567 = vacc0x4567;
    __m256i vacc12x0123 = vacc0x0123;
    __m256i vacc12x4567 = vacc0x4567;
    __m256i vacc13x0123 = vacc0x0123;
    __m256i vacc13x4567 = vacc0x4567;
    w = (const int32_t*) w + 8;

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
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      } else {
        a4 = zero_data;
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      } else {
        a5 = zero_data;
      }
      const int8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const int8_t*) ((uintptr_t) a6 + a_offset);
      } else {
        a6 = zero_data;
      }
      const int8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const int8_t*) ((uintptr_t) a7 + a_offset);
      } else {
        a7 = zero_data;
      }
      const int8_t* restrict a8 = a[8];
      if XNN_UNPREDICTABLE(a8 != zero) {
        a8 = (const int8_t*) ((uintptr_t) a8 + a_offset);
      } else {
        a8 = zero_data;
      }
      const int8_t* restrict a9 = a[9];
      if XNN_UNPREDICTABLE(a9 != zero) {
        a9 = (const int8_t*) ((uintptr_t) a9 + a_offset);
      } else {
        a9 = zero_data;
      }
      const int8_t* restrict a10 = a[10];
      if XNN_UNPREDICTABLE(a10 != zero) {
        a10 = (const int8_t*) ((uintptr_t) a10 + a_offset);
      } else {
        a10 = zero_data;
      }
      const int8_t* restrict a11 = a[11];
      if XNN_UNPREDICTABLE(a11 != zero) {
        a11 = (const int8_t*) ((uintptr_t) a11 + a_offset);
      } else {
        a11 = zero_data;
      }
      const int8_t* restrict a12 = a[12];
      if XNN_UNPREDICTABLE(a12 != zero) {
        a12 = (const int8_t*) ((uintptr_t) a12 + a_offset);
      } else {
        a12 = zero_data;
      }
      const int8_t* restrict a13 = a[13];
      if XNN_UNPREDICTABLE(a13 != zero) {
        a13 = (const int8_t*) ((uintptr_t) a13 + a_offset);
      } else {
        a13 = zero_data;
      }
      a += 14;

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
        const __m256i va9x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a9)), vsign_mask);
        const __m256i va9x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a9 + 8)), vsign_mask);
        a9 += 16;
        const __m256i va10x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a10)), vsign_mask);
        const __m256i va10x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a10 + 8)), vsign_mask);
        a10 += 16;
        const __m256i va11x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a11)), vsign_mask);
        const __m256i va11x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a11 + 8)), vsign_mask);
        a11 += 16;
        const __m256i va12x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a12)), vsign_mask);
        const __m256i va12x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a12 + 8)), vsign_mask);
        a12 += 16;
        const __m256i va13x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a13)), vsign_mask);
        const __m256i va13x89ABCDEF = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a13 + 8)), vsign_mask);
        a13 += 16;

        const __m256i vb01234567x0123 = _mm256_load_si256(w);
        const __m256i vb89ABCDEFx0123 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
        const __m256i vb01234567x4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 64));
        const __m256i vb89ABCDEFx4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 96));

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
        vacc9x0123 = _mm256_dpbusd_epi32(vacc9x0123, va9x01234567, vb01234567x0123);
        vacc9x4567 = _mm256_dpbusd_epi32(vacc9x4567, va9x01234567, vb89ABCDEFx0123);
        vacc10x0123 = _mm256_dpbusd_epi32(vacc10x0123, va10x01234567, vb01234567x0123);
        vacc10x4567 = _mm256_dpbusd_epi32(vacc10x4567, va10x01234567, vb89ABCDEFx0123);
        vacc11x0123 = _mm256_dpbusd_epi32(vacc11x0123, va11x01234567, vb01234567x0123);
        vacc11x4567 = _mm256_dpbusd_epi32(vacc11x4567, va11x01234567, vb89ABCDEFx0123);
        vacc12x0123 = _mm256_dpbusd_epi32(vacc12x0123, va12x01234567, vb01234567x0123);
        vacc12x4567 = _mm256_dpbusd_epi32(vacc12x4567, va12x01234567, vb89ABCDEFx0123);
        vacc13x0123 = _mm256_dpbusd_epi32(vacc13x0123, va13x01234567, vb01234567x0123);
        vacc13x4567 = _mm256_dpbusd_epi32(vacc13x4567, va13x01234567, vb89ABCDEFx0123);
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
        vacc9x0123 = _mm256_dpbusd_epi32(vacc9x0123, va9x89ABCDEF, vb01234567x4567);
        vacc9x4567 = _mm256_dpbusd_epi32(vacc9x4567, va9x89ABCDEF, vb89ABCDEFx4567);
        vacc10x0123 = _mm256_dpbusd_epi32(vacc10x0123, va10x89ABCDEF, vb01234567x4567);
        vacc10x4567 = _mm256_dpbusd_epi32(vacc10x4567, va10x89ABCDEF, vb89ABCDEFx4567);
        vacc11x0123 = _mm256_dpbusd_epi32(vacc11x0123, va11x89ABCDEF, vb01234567x4567);
        vacc11x4567 = _mm256_dpbusd_epi32(vacc11x4567, va11x89ABCDEF, vb89ABCDEFx4567);
        vacc12x0123 = _mm256_dpbusd_epi32(vacc12x0123, va12x89ABCDEF, vb01234567x4567);
        vacc12x4567 = _mm256_dpbusd_epi32(vacc12x4567, va12x89ABCDEF, vb89ABCDEFx4567);
        vacc13x0123 = _mm256_dpbusd_epi32(vacc13x0123, va13x89ABCDEF, vb01234567x4567);
        vacc13x4567 = _mm256_dpbusd_epi32(vacc13x4567, va13x89ABCDEF, vb89ABCDEFx4567);

        w = (const int8_t*) w + 128;
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
        const __m256i va9x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a9)), vsign_mask);
        a9 += 8;
        const __m256i va10x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a10)), vsign_mask);
        a10 += 8;
        const __m256i va11x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a11)), vsign_mask);
        a11 += 8;
        const __m256i va12x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a12)), vsign_mask);
        a12 += 8;
        const __m256i va13x01234567 = _mm256_xor_si256(_mm256_set1_epi64x((int64_t) unaligned_load_u64(a13)), vsign_mask);
        a13 += 8;

        const __m256i vb01234567x0123 = _mm256_load_si256(w);
        const __m256i vb89ABCDEFx0123 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));

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
        vacc9x0123 = _mm256_dpbusd_epi32(vacc9x0123, va9x01234567, vb01234567x0123);
        vacc9x4567 = _mm256_dpbusd_epi32(vacc9x4567, va9x01234567, vb89ABCDEFx0123);
        vacc10x0123 = _mm256_dpbusd_epi32(vacc10x0123, va10x01234567, vb01234567x0123);
        vacc10x4567 = _mm256_dpbusd_epi32(vacc10x4567, va10x01234567, vb89ABCDEFx0123);
        vacc11x0123 = _mm256_dpbusd_epi32(vacc11x0123, va11x01234567, vb01234567x0123);
        vacc11x4567 = _mm256_dpbusd_epi32(vacc11x4567, va11x01234567, vb89ABCDEFx0123);
        vacc12x0123 = _mm256_dpbusd_epi32(vacc12x0123, va12x01234567, vb01234567x0123);
        vacc12x4567 = _mm256_dpbusd_epi32(vacc12x4567, va12x01234567, vb89ABCDEFx0123);
        vacc13x0123 = _mm256_dpbusd_epi32(vacc13x0123, va13x01234567, vb01234567x0123);
        vacc13x4567 = _mm256_dpbusd_epi32(vacc13x4567, va13x01234567, vb89ABCDEFx0123);

        w = (const int8_t*) w + 64;
        k -= 8 * sizeof(int8_t);
      }

      p -= 14 * sizeof(void*);
    } while (p != 0);


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
    const __m256i vsum9x02134657 = _mm256_hadd_epi32(vacc9x0123, vacc9x4567);
    __m256i vacc9x01234567 = _mm256_permute4x64_epi64(vsum9x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum10x02134657 = _mm256_hadd_epi32(vacc10x0123, vacc10x4567);
    __m256i vacc10x01234567 = _mm256_permute4x64_epi64(vsum10x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum11x02134657 = _mm256_hadd_epi32(vacc11x0123, vacc11x4567);
    __m256i vacc11x01234567 = _mm256_permute4x64_epi64(vsum11x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum12x02134657 = _mm256_hadd_epi32(vacc12x0123, vacc12x4567);
    __m256i vacc12x01234567 = _mm256_permute4x64_epi64(vsum12x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum13x02134657 = _mm256_hadd_epi32(vacc13x0123, vacc13x4567);
    __m256i vacc13x01234567 = _mm256_permute4x64_epi64(vsum13x02134657, _MM_SHUFFLE(3, 1, 2, 0));

    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vout1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vout2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);
    __m256 vout3x01234567 = _mm256_cvtepi32_ps(vacc3x01234567);
    __m256 vout4x01234567 = _mm256_cvtepi32_ps(vacc4x01234567);
    __m256 vout5x01234567 = _mm256_cvtepi32_ps(vacc5x01234567);
    __m256 vout6x01234567 = _mm256_cvtepi32_ps(vacc6x01234567);
    __m256 vout7x01234567 = _mm256_cvtepi32_ps(vacc7x01234567);
    __m256 vout8x01234567 = _mm256_cvtepi32_ps(vacc8x01234567);
    __m256 vout9x01234567 = _mm256_cvtepi32_ps(vacc9x01234567);
    __m256 vout10x01234567 = _mm256_cvtepi32_ps(vacc10x01234567);
    __m256 vout11x01234567 = _mm256_cvtepi32_ps(vacc11x01234567);
    __m256 vout12x01234567 = _mm256_cvtepi32_ps(vacc12x01234567);
    __m256 vout13x01234567 = _mm256_cvtepi32_ps(vacc13x01234567);

    vout0x01234567 = _mm256_mul_ps(vout0x01234567, vinput_inv_scale);
    vout1x01234567 = _mm256_mul_ps(vout1x01234567, vinput_inv_scale);
    vout2x01234567 = _mm256_mul_ps(vout2x01234567, vinput_inv_scale);
    vout3x01234567 = _mm256_mul_ps(vout3x01234567, vinput_inv_scale);
    vout4x01234567 = _mm256_mul_ps(vout4x01234567, vinput_inv_scale);
    vout5x01234567 = _mm256_mul_ps(vout5x01234567, vinput_inv_scale);
    vout6x01234567 = _mm256_mul_ps(vout6x01234567, vinput_inv_scale);
    vout7x01234567 = _mm256_mul_ps(vout7x01234567, vinput_inv_scale);
    vout8x01234567 = _mm256_mul_ps(vout8x01234567, vinput_inv_scale);
    vout9x01234567 = _mm256_mul_ps(vout9x01234567, vinput_inv_scale);
    vout10x01234567 = _mm256_mul_ps(vout10x01234567, vinput_inv_scale);
    vout11x01234567 = _mm256_mul_ps(vout11x01234567, vinput_inv_scale);
    vout12x01234567 = _mm256_mul_ps(vout12x01234567, vinput_inv_scale);
    vout13x01234567 = _mm256_mul_ps(vout13x01234567, vinput_inv_scale);

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
    vout9x01234567 = _mm256_fmadd_ps(vout9x01234567, vfilter_output_scale01234567, vbias01234567);
    vout10x01234567 = _mm256_fmadd_ps(vout10x01234567, vfilter_output_scale01234567, vbias01234567);
    vout11x01234567 = _mm256_fmadd_ps(vout11x01234567, vfilter_output_scale01234567, vbias01234567);
    vout12x01234567 = _mm256_fmadd_ps(vout12x01234567, vfilter_output_scale01234567, vbias01234567);
    vout13x01234567 = _mm256_fmadd_ps(vout13x01234567, vfilter_output_scale01234567, vbias01234567);

    vout0x01234567 = _mm256_max_ps(vout0x01234567, voutput_min);
    vout1x01234567 = _mm256_max_ps(vout1x01234567, voutput_min);
    vout2x01234567 = _mm256_max_ps(vout2x01234567, voutput_min);
    vout3x01234567 = _mm256_max_ps(vout3x01234567, voutput_min);
    vout4x01234567 = _mm256_max_ps(vout4x01234567, voutput_min);
    vout5x01234567 = _mm256_max_ps(vout5x01234567, voutput_min);
    vout6x01234567 = _mm256_max_ps(vout6x01234567, voutput_min);
    vout7x01234567 = _mm256_max_ps(vout7x01234567, voutput_min);
    vout8x01234567 = _mm256_max_ps(vout8x01234567, voutput_min);
    vout9x01234567 = _mm256_max_ps(vout9x01234567, voutput_min);
    vout10x01234567 = _mm256_max_ps(vout10x01234567, voutput_min);
    vout11x01234567 = _mm256_max_ps(vout11x01234567, voutput_min);
    vout12x01234567 = _mm256_max_ps(vout12x01234567, voutput_min);
    vout13x01234567 = _mm256_max_ps(vout13x01234567, voutput_min);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, voutput_max);
    vout1x01234567 = _mm256_min_ps(vout1x01234567, voutput_max);
    vout2x01234567 = _mm256_min_ps(vout2x01234567, voutput_max);
    vout3x01234567 = _mm256_min_ps(vout3x01234567, voutput_max);
    vout4x01234567 = _mm256_min_ps(vout4x01234567, voutput_max);
    vout5x01234567 = _mm256_min_ps(vout5x01234567, voutput_max);
    vout6x01234567 = _mm256_min_ps(vout6x01234567, voutput_max);
    vout7x01234567 = _mm256_min_ps(vout7x01234567, voutput_max);
    vout8x01234567 = _mm256_min_ps(vout8x01234567, voutput_max);
    vout9x01234567 = _mm256_min_ps(vout9x01234567, voutput_max);
    vout10x01234567 = _mm256_min_ps(vout10x01234567, voutput_max);
    vout11x01234567 = _mm256_min_ps(vout11x01234567, voutput_max);
    vout12x01234567 = _mm256_min_ps(vout12x01234567, voutput_max);
    vout13x01234567 = _mm256_min_ps(vout13x01234567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      _mm256_storeu_ps(c13, vout13x01234567);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      _mm256_storeu_ps(c12, vout12x01234567);
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      _mm256_storeu_ps(c11, vout11x01234567);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      _mm256_storeu_ps(c10, vout10x01234567);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm256_storeu_ps(c9, vout9x01234567);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm256_storeu_ps(c8, vout8x01234567);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm256_storeu_ps(c7, vout7x01234567);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm256_storeu_ps(c6, vout6x01234567);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm256_storeu_ps(c5, vout5x01234567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm256_storeu_ps(c4, vout4x01234567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_ps(c3, vout3x01234567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c2, vout2x01234567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vout1x01234567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vout0x01234567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask8 vmask = _cvtu32_mask8((UINT32_C(1) << nc) - 1);
      _mm256_mask_storeu_ps(c13, vmask, vout13x01234567);
      _mm256_mask_storeu_ps(c12, vmask, vout12x01234567);
      _mm256_mask_storeu_ps(c11, vmask, vout11x01234567);
      _mm256_mask_storeu_ps(c10, vmask, vout10x01234567);
      _mm256_mask_storeu_ps(c9, vmask, vout9x01234567);
      _mm256_mask_storeu_ps(c8, vmask, vout8x01234567);
      _mm256_mask_storeu_ps(c7, vmask, vout7x01234567);
      _mm256_mask_storeu_ps(c6, vmask, vout6x01234567);
      _mm256_mask_storeu_ps(c5, vmask, vout5x01234567);
      _mm256_mask_storeu_ps(c4, vmask, vout4x01234567);
      _mm256_mask_storeu_ps(c3, vmask, vout3x01234567);
      _mm256_mask_storeu_ps(c2, vmask, vout2x01234567);
      _mm256_mask_storeu_ps(c1, vmask, vout1x01234567);
      _mm256_mask_storeu_ps(c0, vmask, vout0x01234567);
      nc = 0;
    }
  } while (nc != 0);
}
