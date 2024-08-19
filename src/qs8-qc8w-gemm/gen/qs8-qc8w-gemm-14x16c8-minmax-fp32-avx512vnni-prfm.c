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
#include "xnnpack/prefetch.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  int8_t* c7 = (int8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  int8_t* c8 = (int8_t*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const int8_t* a9 = (const int8_t*) ((uintptr_t) a8 + a_stride);
  int8_t* c9 = (int8_t*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
    c9 = c8;
  }
  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9 + a_stride);
  int8_t* c10 = (int8_t*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
    c10 = c9;
  }
  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  int8_t* c11 = (int8_t*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    a11 = a10;
    c11 = c10;
  }
  const int8_t* a12 = (const int8_t*) ((uintptr_t) a11 + a_stride);
  int8_t* c12 = (int8_t*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    a12 = a11;
    c12 = c11;
  }
  const int8_t* a13 = (const int8_t*) ((uintptr_t) a12 + a_stride);
  int8_t* c13 = (int8_t*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 14) {
    a13 = a12;
    c13 = c12;
  }

  const __m512i vsign_mask = _mm512_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m512 voutput_max_less_zero_point = _mm512_set1_ps(params->fp32_avx512vnni.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_avx512vnni.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512vnni.output_min);
  do {
    __m512i vacc0x01234567 = _mm512_cvtepu32_epi64(_mm256_load_si256((const __m256i*) w));
    __m512i vacc0x89ABCDEF = _mm512_cvtepu32_epi64(_mm256_load_si256((const __m256i*) ((const int32_t*) w + 8)));
    __m512i vacc1x01234567 = vacc0x01234567;
    __m512i vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc2x01234567 = vacc0x01234567;
    __m512i vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc3x01234567 = vacc0x01234567;
    __m512i vacc3x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc4x01234567 = vacc0x01234567;
    __m512i vacc4x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc5x01234567 = vacc0x01234567;
    __m512i vacc5x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc6x01234567 = vacc0x01234567;
    __m512i vacc6x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc7x01234567 = vacc0x01234567;
    __m512i vacc7x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc8x01234567 = vacc0x01234567;
    __m512i vacc8x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc9x01234567 = vacc0x01234567;
    __m512i vacc9x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc10x01234567 = vacc0x01234567;
    __m512i vacc10x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc11x01234567 = vacc0x01234567;
    __m512i vacc11x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc12x01234567 = vacc0x01234567;
    __m512i vacc12x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc13x01234567 = vacc0x01234567;
    __m512i vacc13x89ABCDEF = vacc0x89ABCDEF;
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
      const __m512i va9x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a9)), vsign_mask);
      const __m512i va9x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a9 + 8)), vsign_mask);
      a9 += 16;
      const __m512i va10x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a10)), vsign_mask);
      const __m512i va10x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a10 + 8)), vsign_mask);
      a10 += 16;
      const __m512i va11x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a11)), vsign_mask);
      const __m512i va11x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a11 + 8)), vsign_mask);
      a11 += 16;
      const __m512i va12x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a12)), vsign_mask);
      const __m512i va12x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a12 + 8)), vsign_mask);
      a12 += 16;
      const __m512i va13x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a13)), vsign_mask);
      const __m512i va13x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a13 + 8)), vsign_mask);
      a13 += 16;

      const __m512i vb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vb01234567x89ABCDEF = _mm512_load_si512((const int8_t*) w + 128);
      const __m512i vb89ABCDEFx89ABCDEF = _mm512_load_si512((const int8_t*) w + 192);
      xnn_prefetch_to_l1((const int8_t*) w + 768);
      xnn_prefetch_to_l1((const int8_t*) w + 832);

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
      vacc9x01234567 = _mm512_dpbusd_epi32(vacc9x01234567, va9x01234567, vb01234567x01234567);
      vacc9x89ABCDEF = _mm512_dpbusd_epi32(vacc9x89ABCDEF, va9x01234567, vb89ABCDEFx01234567);
      vacc10x01234567 = _mm512_dpbusd_epi32(vacc10x01234567, va10x01234567, vb01234567x01234567);
      vacc10x89ABCDEF = _mm512_dpbusd_epi32(vacc10x89ABCDEF, va10x01234567, vb89ABCDEFx01234567);
      vacc11x01234567 = _mm512_dpbusd_epi32(vacc11x01234567, va11x01234567, vb01234567x01234567);
      vacc11x89ABCDEF = _mm512_dpbusd_epi32(vacc11x89ABCDEF, va11x01234567, vb89ABCDEFx01234567);
      vacc12x01234567 = _mm512_dpbusd_epi32(vacc12x01234567, va12x01234567, vb01234567x01234567);
      vacc12x89ABCDEF = _mm512_dpbusd_epi32(vacc12x89ABCDEF, va12x01234567, vb89ABCDEFx01234567);
      vacc13x01234567 = _mm512_dpbusd_epi32(vacc13x01234567, va13x01234567, vb01234567x01234567);
      vacc13x89ABCDEF = _mm512_dpbusd_epi32(vacc13x89ABCDEF, va13x01234567, vb89ABCDEFx01234567);
      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);
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
      vacc9x01234567 = _mm512_dpbusd_epi32(vacc9x01234567, va9x89ABCDEF, vb01234567x89ABCDEF);
      vacc9x89ABCDEF = _mm512_dpbusd_epi32(vacc9x89ABCDEF, va9x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc10x01234567 = _mm512_dpbusd_epi32(vacc10x01234567, va10x89ABCDEF, vb01234567x89ABCDEF);
      vacc10x89ABCDEF = _mm512_dpbusd_epi32(vacc10x89ABCDEF, va10x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc11x01234567 = _mm512_dpbusd_epi32(vacc11x01234567, va11x89ABCDEF, vb01234567x89ABCDEF);
      vacc11x89ABCDEF = _mm512_dpbusd_epi32(vacc11x89ABCDEF, va11x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc12x01234567 = _mm512_dpbusd_epi32(vacc12x01234567, va12x89ABCDEF, vb01234567x89ABCDEF);
      vacc12x89ABCDEF = _mm512_dpbusd_epi32(vacc12x89ABCDEF, va12x89ABCDEF, vb89ABCDEFx89ABCDEF);
      vacc13x01234567 = _mm512_dpbusd_epi32(vacc13x01234567, va13x89ABCDEF, vb01234567x89ABCDEF);
      vacc13x89ABCDEF = _mm512_dpbusd_epi32(vacc13x89ABCDEF, va13x89ABCDEF, vb89ABCDEFx89ABCDEF);

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
      const __m512i va9x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a9)), vsign_mask);
      a9 += 8;
      const __m512i va10x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a10)), vsign_mask);
      a10 += 8;
      const __m512i va11x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a11)), vsign_mask);
      a11 += 8;
      const __m512i va12x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a12)), vsign_mask);
      a12 += 8;
      const __m512i va13x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a13)), vsign_mask);
      a13 += 8;

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
      vacc9x01234567 = _mm512_dpbusd_epi32(vacc9x01234567, va9x01234567, vb01234567x01234567);
      vacc9x89ABCDEF = _mm512_dpbusd_epi32(vacc9x89ABCDEF, va9x01234567, vb89ABCDEFx01234567);
      vacc10x01234567 = _mm512_dpbusd_epi32(vacc10x01234567, va10x01234567, vb01234567x01234567);
      vacc10x89ABCDEF = _mm512_dpbusd_epi32(vacc10x89ABCDEF, va10x01234567, vb89ABCDEFx01234567);
      vacc11x01234567 = _mm512_dpbusd_epi32(vacc11x01234567, va11x01234567, vb01234567x01234567);
      vacc11x89ABCDEF = _mm512_dpbusd_epi32(vacc11x89ABCDEF, va11x01234567, vb89ABCDEFx01234567);
      vacc12x01234567 = _mm512_dpbusd_epi32(vacc12x01234567, va12x01234567, vb01234567x01234567);
      vacc12x89ABCDEF = _mm512_dpbusd_epi32(vacc12x89ABCDEF, va12x01234567, vb89ABCDEFx01234567);
      vacc13x01234567 = _mm512_dpbusd_epi32(vacc13x01234567, va13x01234567, vb01234567x01234567);
      vacc13x89ABCDEF = _mm512_dpbusd_epi32(vacc13x89ABCDEF, va13x01234567, vb89ABCDEFx01234567);
      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

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
    const __m512i vsum9x01234567 = _mm512_add_epi32(vacc9x01234567, _mm512_srli_epi64(vacc9x01234567, 32));
    const __m512i vsum9x89ABCDEF = _mm512_add_epi32(vacc9x89ABCDEF, _mm512_srli_epi64(vacc9x89ABCDEF, 32));
    __m512i vacc9x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum9x01234567, vidx, vsum9x89ABCDEF);
    const __m512i vsum10x01234567 = _mm512_add_epi32(vacc10x01234567, _mm512_srli_epi64(vacc10x01234567, 32));
    const __m512i vsum10x89ABCDEF = _mm512_add_epi32(vacc10x89ABCDEF, _mm512_srli_epi64(vacc10x89ABCDEF, 32));
    __m512i vacc10x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum10x01234567, vidx, vsum10x89ABCDEF);
    const __m512i vsum11x01234567 = _mm512_add_epi32(vacc11x01234567, _mm512_srli_epi64(vacc11x01234567, 32));
    const __m512i vsum11x89ABCDEF = _mm512_add_epi32(vacc11x89ABCDEF, _mm512_srli_epi64(vacc11x89ABCDEF, 32));
    __m512i vacc11x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum11x01234567, vidx, vsum11x89ABCDEF);
    const __m512i vsum12x01234567 = _mm512_add_epi32(vacc12x01234567, _mm512_srli_epi64(vacc12x01234567, 32));
    const __m512i vsum12x89ABCDEF = _mm512_add_epi32(vacc12x89ABCDEF, _mm512_srli_epi64(vacc12x89ABCDEF, 32));
    __m512i vacc12x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum12x01234567, vidx, vsum12x89ABCDEF);
    const __m512i vsum13x01234567 = _mm512_add_epi32(vacc13x01234567, _mm512_srli_epi64(vacc13x01234567, 32));
    const __m512i vsum13x89ABCDEF = _mm512_add_epi32(vacc13x89ABCDEF, _mm512_srli_epi64(vacc13x89ABCDEF, 32));
    __m512i vacc13x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum13x01234567, vidx, vsum13x89ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
    __m512 vscaled7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);
    __m512 vscaled8x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc8x0123456789ABCDEF);
    __m512 vscaled9x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc9x0123456789ABCDEF);
    __m512 vscaled10x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc10x0123456789ABCDEF);
    __m512 vscaled11x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc11x0123456789ABCDEF);
    __m512 vscaled12x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc12x0123456789ABCDEF);
    __m512 vscaled13x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc13x0123456789ABCDEF);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const float*) w + 16;
    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, vscale012345678ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled9x0123456789ABCDEF = _mm512_min_ps(vscaled9x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled10x0123456789ABCDEF = _mm512_min_ps(vscaled10x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled11x0123456789ABCDEF = _mm512_min_ps(vscaled11x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled12x0123456789ABCDEF = _mm512_min_ps(vscaled12x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled13x0123456789ABCDEF = _mm512_min_ps(vscaled13x0123456789ABCDEF, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled6x0123456789ABCDEF);
    vacc7x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled7x0123456789ABCDEF);
    vacc8x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled8x0123456789ABCDEF);
    vacc9x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled9x0123456789ABCDEF);
    vacc10x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled10x0123456789ABCDEF);
    vacc11x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled11x0123456789ABCDEF);
    vacc12x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled12x0123456789ABCDEF);
    vacc13x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled13x0123456789ABCDEF);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, voutput_zero_point);
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, voutput_zero_point);
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, voutput_zero_point);
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, voutput_zero_point);
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, voutput_zero_point);
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, voutput_zero_point);
    vacc7x0123456789ABCDEF = _mm512_add_epi32(vacc7x0123456789ABCDEF, voutput_zero_point);
    vacc8x0123456789ABCDEF = _mm512_add_epi32(vacc8x0123456789ABCDEF, voutput_zero_point);
    vacc9x0123456789ABCDEF = _mm512_add_epi32(vacc9x0123456789ABCDEF, voutput_zero_point);
    vacc10x0123456789ABCDEF = _mm512_add_epi32(vacc10x0123456789ABCDEF, voutput_zero_point);
    vacc11x0123456789ABCDEF = _mm512_add_epi32(vacc11x0123456789ABCDEF, voutput_zero_point);
    vacc12x0123456789ABCDEF = _mm512_add_epi32(vacc12x0123456789ABCDEF, voutput_zero_point);
    vacc13x0123456789ABCDEF = _mm512_add_epi32(vacc13x0123456789ABCDEF, voutput_zero_point);

    __m128i vout0x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc0x0123456789ABCDEF);
    __m128i vout1x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc1x0123456789ABCDEF);
    __m128i vout2x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc2x0123456789ABCDEF);
    __m128i vout3x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc3x0123456789ABCDEF);
    __m128i vout4x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc4x0123456789ABCDEF);
    __m128i vout5x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc5x0123456789ABCDEF);
    __m128i vout6x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc6x0123456789ABCDEF);
    __m128i vout7x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc7x0123456789ABCDEF);
    __m128i vout8x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc8x0123456789ABCDEF);
    __m128i vout9x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc9x0123456789ABCDEF);
    __m128i vout10x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc10x0123456789ABCDEF);
    __m128i vout11x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc11x0123456789ABCDEF);
    __m128i vout12x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc12x0123456789ABCDEF);
    __m128i vout13x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc13x0123456789ABCDEF);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epi8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = _mm_max_epi8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = _mm_max_epi8(vout3x0123456789ABCDEF, voutput_min);
    vout4x0123456789ABCDEF = _mm_max_epi8(vout4x0123456789ABCDEF, voutput_min);
    vout5x0123456789ABCDEF = _mm_max_epi8(vout5x0123456789ABCDEF, voutput_min);
    vout6x0123456789ABCDEF = _mm_max_epi8(vout6x0123456789ABCDEF, voutput_min);
    vout7x0123456789ABCDEF = _mm_max_epi8(vout7x0123456789ABCDEF, voutput_min);
    vout8x0123456789ABCDEF = _mm_max_epi8(vout8x0123456789ABCDEF, voutput_min);
    vout9x0123456789ABCDEF = _mm_max_epi8(vout9x0123456789ABCDEF, voutput_min);
    vout10x0123456789ABCDEF = _mm_max_epi8(vout10x0123456789ABCDEF, voutput_min);
    vout11x0123456789ABCDEF = _mm_max_epi8(vout11x0123456789ABCDEF, voutput_min);
    vout12x0123456789ABCDEF = _mm_max_epi8(vout12x0123456789ABCDEF, voutput_min);
    vout13x0123456789ABCDEF = _mm_max_epi8(vout13x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      _mm_storeu_si128((__m128i*) c1, vout1x0123456789ABCDEF);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      _mm_storeu_si128((__m128i*) c2, vout2x0123456789ABCDEF);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      _mm_storeu_si128((__m128i*) c3, vout3x0123456789ABCDEF);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      _mm_storeu_si128((__m128i*) c4, vout4x0123456789ABCDEF);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      _mm_storeu_si128((__m128i*) c5, vout5x0123456789ABCDEF);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      _mm_storeu_si128((__m128i*) c6, vout6x0123456789ABCDEF);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      _mm_storeu_si128((__m128i*) c7, vout7x0123456789ABCDEF);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);
      _mm_storeu_si128((__m128i*) c8, vout8x0123456789ABCDEF);
      c8 = (int8_t*) ((uintptr_t) c8 + cn_stride);
      a8 = (const int8_t*) ((uintptr_t) a8 - kc);
      _mm_storeu_si128((__m128i*) c9, vout9x0123456789ABCDEF);
      c9 = (int8_t*) ((uintptr_t) c9 + cn_stride);
      a9 = (const int8_t*) ((uintptr_t) a9 - kc);
      _mm_storeu_si128((__m128i*) c10, vout10x0123456789ABCDEF);
      c10 = (int8_t*) ((uintptr_t) c10 + cn_stride);
      a10 = (const int8_t*) ((uintptr_t) a10 - kc);
      _mm_storeu_si128((__m128i*) c11, vout11x0123456789ABCDEF);
      c11 = (int8_t*) ((uintptr_t) c11 + cn_stride);
      a11 = (const int8_t*) ((uintptr_t) a11 - kc);
      _mm_storeu_si128((__m128i*) c12, vout12x0123456789ABCDEF);
      c12 = (int8_t*) ((uintptr_t) c12 + cn_stride);
      a12 = (const int8_t*) ((uintptr_t) a12 - kc);
      _mm_storeu_si128((__m128i*) c13, vout13x0123456789ABCDEF);
      c13 = (int8_t*) ((uintptr_t) c13 + cn_stride);
      a13 = (const int8_t*) ((uintptr_t) a13 - kc);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - UINT32_C(1));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1, vmask, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2, vmask, vout2x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c3, vmask, vout3x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c4, vmask, vout4x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c5, vmask, vout5x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c6, vmask, vout6x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c7, vmask, vout7x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c8, vmask, vout8x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c9, vmask, vout9x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c10, vmask, vout10x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c11, vmask, vout11x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c12, vmask, vout12x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c13, vmask, vout13x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
