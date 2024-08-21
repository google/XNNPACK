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
#include "xnnpack/prefetch.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c8__avxvnni_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  const __m256i vsign_mask = _mm256_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m256 voutput_max_less_zero_point = _mm256_set1_ps(params->fp32_avxvnni.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_set1_epi32(params->fp32_avxvnni.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avxvnni.output_min);
  do {
    __m256i vacc0x0123 = _mm256_cvtepu32_epi64(_mm_load_si128((const __m128i*) w));
    __m256i vacc0x4567 = _mm256_cvtepu32_epi64(_mm_load_si128((const __m128i*) ((const int32_t*) w + 4)));
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
    w = (const int32_t*) w + 8;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

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

        const __m256i vb01234567x0123 = _mm256_load_si256(w);
        const __m256i vb89ABCDEFx0123 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
        const __m256i vb01234567x4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 64));
        const __m256i vb89ABCDEFx4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 96));
        xnn_prefetch_to_l1((const int8_t*) w + 896);

        vacc0x0123 = _mm256_dpbusd_avx_epi32(vacc0x0123, va0x01234567, vb01234567x0123);
        vacc0x4567 = _mm256_dpbusd_avx_epi32(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
        vacc1x0123 = _mm256_dpbusd_avx_epi32(vacc1x0123, va1x01234567, vb01234567x0123);
        vacc1x4567 = _mm256_dpbusd_avx_epi32(vacc1x4567, va1x01234567, vb89ABCDEFx0123);
        vacc2x0123 = _mm256_dpbusd_avx_epi32(vacc2x0123, va2x01234567, vb01234567x0123);
        vacc2x4567 = _mm256_dpbusd_avx_epi32(vacc2x4567, va2x01234567, vb89ABCDEFx0123);
        vacc3x0123 = _mm256_dpbusd_avx_epi32(vacc3x0123, va3x01234567, vb01234567x0123);
        vacc3x4567 = _mm256_dpbusd_avx_epi32(vacc3x4567, va3x01234567, vb89ABCDEFx0123);
        vacc4x0123 = _mm256_dpbusd_avx_epi32(vacc4x0123, va4x01234567, vb01234567x0123);
        vacc4x4567 = _mm256_dpbusd_avx_epi32(vacc4x4567, va4x01234567, vb89ABCDEFx0123);
        vacc5x0123 = _mm256_dpbusd_avx_epi32(vacc5x0123, va5x01234567, vb01234567x0123);
        vacc5x4567 = _mm256_dpbusd_avx_epi32(vacc5x4567, va5x01234567, vb89ABCDEFx0123);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        vacc0x0123 = _mm256_dpbusd_avx_epi32(vacc0x0123, va0x89ABCDEF, vb01234567x4567);
        vacc0x4567 = _mm256_dpbusd_avx_epi32(vacc0x4567, va0x89ABCDEF, vb89ABCDEFx4567);
        vacc1x0123 = _mm256_dpbusd_avx_epi32(vacc1x0123, va1x89ABCDEF, vb01234567x4567);
        vacc1x4567 = _mm256_dpbusd_avx_epi32(vacc1x4567, va1x89ABCDEF, vb89ABCDEFx4567);
        vacc2x0123 = _mm256_dpbusd_avx_epi32(vacc2x0123, va2x89ABCDEF, vb01234567x4567);
        vacc2x4567 = _mm256_dpbusd_avx_epi32(vacc2x4567, va2x89ABCDEF, vb89ABCDEFx4567);
        vacc3x0123 = _mm256_dpbusd_avx_epi32(vacc3x0123, va3x89ABCDEF, vb01234567x4567);
        vacc3x4567 = _mm256_dpbusd_avx_epi32(vacc3x4567, va3x89ABCDEF, vb89ABCDEFx4567);
        vacc4x0123 = _mm256_dpbusd_avx_epi32(vacc4x0123, va4x89ABCDEF, vb01234567x4567);
        vacc4x4567 = _mm256_dpbusd_avx_epi32(vacc4x4567, va4x89ABCDEF, vb89ABCDEFx4567);
        vacc5x0123 = _mm256_dpbusd_avx_epi32(vacc5x0123, va5x89ABCDEF, vb01234567x4567);
        vacc5x4567 = _mm256_dpbusd_avx_epi32(vacc5x4567, va5x89ABCDEF, vb89ABCDEFx4567);

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

        const __m256i vb01234567x0123 = _mm256_load_si256(w);
        const __m256i vb89ABCDEFx0123 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));

        vacc0x0123 = _mm256_dpbusd_avx_epi32(vacc0x0123, va0x01234567, vb01234567x0123);
        vacc0x4567 = _mm256_dpbusd_avx_epi32(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
        vacc1x0123 = _mm256_dpbusd_avx_epi32(vacc1x0123, va1x01234567, vb01234567x0123);
        vacc1x4567 = _mm256_dpbusd_avx_epi32(vacc1x4567, va1x01234567, vb89ABCDEFx0123);
        vacc2x0123 = _mm256_dpbusd_avx_epi32(vacc2x0123, va2x01234567, vb01234567x0123);
        vacc2x4567 = _mm256_dpbusd_avx_epi32(vacc2x4567, va2x01234567, vb89ABCDEFx0123);
        vacc3x0123 = _mm256_dpbusd_avx_epi32(vacc3x0123, va3x01234567, vb01234567x0123);
        vacc3x4567 = _mm256_dpbusd_avx_epi32(vacc3x4567, va3x01234567, vb89ABCDEFx0123);
        vacc4x0123 = _mm256_dpbusd_avx_epi32(vacc4x0123, va4x01234567, vb01234567x0123);
        vacc4x4567 = _mm256_dpbusd_avx_epi32(vacc4x4567, va4x01234567, vb89ABCDEFx0123);
        vacc5x0123 = _mm256_dpbusd_avx_epi32(vacc5x0123, va5x01234567, vb01234567x0123);
        vacc5x4567 = _mm256_dpbusd_avx_epi32(vacc5x4567, va5x01234567, vb89ABCDEFx0123);
        xnn_prefetch_to_l1((const int8_t*) w + 960);

        w = (const int8_t*) w + 64;
        k -= 8 * sizeof(int8_t);
      }

      p -= 6 * sizeof(void*);
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

    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vout1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vout2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);
    __m256 vout3x01234567 = _mm256_cvtepi32_ps(vacc3x01234567);
    __m256 vout4x01234567 = _mm256_cvtepi32_ps(vacc4x01234567);
    __m256 vout5x01234567 = _mm256_cvtepi32_ps(vacc5x01234567);

    const __m256 vscale01234567 = _mm256_load_ps(w);
    w = (const float*) w + 8;
    vout0x01234567 = _mm256_mul_ps(vout0x01234567, vscale01234567);
    vout1x01234567 = _mm256_mul_ps(vout1x01234567, vscale01234567);
    vout2x01234567 = _mm256_mul_ps(vout2x01234567, vscale01234567);
    vout3x01234567 = _mm256_mul_ps(vout3x01234567, vscale01234567);
    vout4x01234567 = _mm256_mul_ps(vout4x01234567, vscale01234567);
    vout5x01234567 = _mm256_mul_ps(vout5x01234567, vscale01234567);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, voutput_max_less_zero_point);
    vout1x01234567 = _mm256_min_ps(vout1x01234567, voutput_max_less_zero_point);
    vout2x01234567 = _mm256_min_ps(vout2x01234567, voutput_max_less_zero_point);
    vout3x01234567 = _mm256_min_ps(vout3x01234567, voutput_max_less_zero_point);
    vout4x01234567 = _mm256_min_ps(vout4x01234567, voutput_max_less_zero_point);
    vout5x01234567 = _mm256_min_ps(vout5x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vout0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vout1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vout2x01234567);
    vacc3x01234567 = _mm256_cvtps_epi32(vout3x01234567);
    vacc4x01234567 = _mm256_cvtps_epi32(vout4x01234567);
    vacc5x01234567 = _mm256_cvtps_epi32(vout5x01234567);

    vacc0x01234567 = _mm256_add_epi32(vacc0x01234567, voutput_zero_point);
    vacc1x01234567 = _mm256_add_epi32(vacc1x01234567, voutput_zero_point);
    vacc2x01234567 = _mm256_add_epi32(vacc2x01234567, voutput_zero_point);
    vacc3x01234567 = _mm256_add_epi32(vacc3x01234567, voutput_zero_point);
    vacc4x01234567 = _mm256_add_epi32(vacc4x01234567, voutput_zero_point);
    vacc5x01234567 = _mm256_add_epi32(vacc5x01234567, voutput_zero_point);

    vacc0x01234567 = _mm256_packs_epi32(vacc0x01234567, _mm256_castsi128_si256(_mm256_extracti128_si256(vacc0x01234567, 1)));
    __m128i voutb0x01234567 = _mm256_castsi256_si128(_mm256_packs_epi16(vacc0x01234567, vacc0x01234567));
    vacc1x01234567 = _mm256_packs_epi32(vacc1x01234567, _mm256_castsi128_si256(_mm256_extracti128_si256(vacc1x01234567, 1)));
    __m128i voutb1x01234567 = _mm256_castsi256_si128(_mm256_packs_epi16(vacc1x01234567, vacc1x01234567));
    vacc2x01234567 = _mm256_packs_epi32(vacc2x01234567, _mm256_castsi128_si256(_mm256_extracti128_si256(vacc2x01234567, 1)));
    __m128i voutb2x01234567 = _mm256_castsi256_si128(_mm256_packs_epi16(vacc2x01234567, vacc2x01234567));
    vacc3x01234567 = _mm256_packs_epi32(vacc3x01234567, _mm256_castsi128_si256(_mm256_extracti128_si256(vacc3x01234567, 1)));
    __m128i voutb3x01234567 = _mm256_castsi256_si128(_mm256_packs_epi16(vacc3x01234567, vacc3x01234567));
    vacc4x01234567 = _mm256_packs_epi32(vacc4x01234567, _mm256_castsi128_si256(_mm256_extracti128_si256(vacc4x01234567, 1)));
    __m128i voutb4x01234567 = _mm256_castsi256_si128(_mm256_packs_epi16(vacc4x01234567, vacc4x01234567));
    vacc5x01234567 = _mm256_packs_epi32(vacc5x01234567, _mm256_castsi128_si256(_mm256_extracti128_si256(vacc5x01234567, 1)));
    __m128i voutb5x01234567 = _mm256_castsi256_si128(_mm256_packs_epi16(vacc5x01234567, vacc5x01234567));

    voutb0x01234567 = _mm_max_epi8(voutb0x01234567, voutput_min);
    voutb1x01234567 = _mm_max_epi8(voutb1x01234567, voutput_min);
    voutb2x01234567 = _mm_max_epi8(voutb2x01234567, voutput_min);
    voutb3x01234567 = _mm_max_epi8(voutb3x01234567, voutput_min);
    voutb4x01234567 = _mm_max_epi8(voutb4x01234567, voutput_min);
    voutb5x01234567 = _mm_max_epi8(voutb5x01234567, voutput_min);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c5, voutb5x01234567);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storel_epi64((__m128i*) c4, voutb4x01234567);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storel_epi64((__m128i*) c3, voutb3x01234567);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storel_epi64((__m128i*) c2, voutb2x01234567);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storel_epi64((__m128i*) c1, voutb1x01234567);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storel_epi64((__m128i*) c0, voutb0x01234567);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c5, voutb5x01234567);
        c5 += 4;
        _mm_storeu_si32(c4, voutb4x01234567);
        c4 += 4;
        _mm_storeu_si32(c3, voutb3x01234567);
        c3 += 4;
        _mm_storeu_si32(c2, voutb2x01234567);
        c2 += 4;
        _mm_storeu_si32(c1, voutb1x01234567);
        c1 += 4;
        _mm_storeu_si32(c0, voutb0x01234567);
        c0 += 4;
        voutb5x01234567 = _mm_srli_epi64(voutb5x01234567, 32);
        voutb4x01234567 = _mm_srli_epi64(voutb4x01234567, 32);
        voutb3x01234567 = _mm_srli_epi64(voutb3x01234567, 32);
        voutb2x01234567 = _mm_srli_epi64(voutb2x01234567, 32);
        voutb1x01234567 = _mm_srli_epi64(voutb1x01234567, 32);
        voutb0x01234567 = _mm_srli_epi64(voutb0x01234567, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c5, (uint16_t) _mm_extract_epi16(voutb5x01234567, 0));
        c5 += 2;
        unaligned_store_u16(c4, (uint16_t) _mm_extract_epi16(voutb4x01234567, 0));
        c4 += 2;
        unaligned_store_u16(c3, (uint16_t) _mm_extract_epi16(voutb3x01234567, 0));
        c3 += 2;
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(voutb2x01234567, 0));
        c2 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(voutb1x01234567, 0));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(voutb0x01234567, 0));
        c0 += 2;
        voutb5x01234567 = _mm_srli_epi32(voutb5x01234567, 16);
        voutb4x01234567 = _mm_srli_epi32(voutb4x01234567, 16);
        voutb3x01234567 = _mm_srli_epi32(voutb3x01234567, 16);
        voutb2x01234567 = _mm_srli_epi32(voutb2x01234567, 16);
        voutb1x01234567 = _mm_srli_epi32(voutb1x01234567, 16);
        voutb0x01234567 = _mm_srli_epi32(voutb0x01234567, 16);
      }
      if (nc & 1) {
        *c5 = (int8_t) _mm_extract_epi8(voutb5x01234567, 0);
        *c4 = (int8_t) _mm_extract_epi8(voutb4x01234567, 0);
        *c3 = (int8_t) _mm_extract_epi8(voutb3x01234567, 0);
        *c2 = (int8_t) _mm_extract_epi8(voutb2x01234567, 0);
        *c1 = (int8_t) _mm_extract_epi8(voutb1x01234567, 0);
        *c0 = (int8_t) _mm_extract_epi8(voutb0x01234567, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
