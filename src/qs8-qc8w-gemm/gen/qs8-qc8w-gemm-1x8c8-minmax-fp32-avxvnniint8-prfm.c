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
#include "xnnpack/prefetch.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const __m256 voutput_max_less_zero_point = _mm256_set1_ps((int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point);
  const __m256i voutput_zero_point = _mm256_set1_epi32(params->fp32_scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->fp32_scalar.output_min);
  // XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  // XNN_FORCE_REALIZATION(voutput_zero_point);
  // XNN_FORCE_REALIZATION(voutput_min);
  do {
    __m256i vacc0x0123 = _mm256_cvtepu32_epi64(_mm_load_si128((const __m128i*) w));
    __m256i vacc0x4567 = _mm256_cvtepu32_epi64(_mm_load_si128((const __m128i*) ((const int32_t*) w + 4)));
    __m256i vacc1x0x0123 = _mm256_setzero_si256();
    __m256i vacc1x0x4567 = _mm256_setzero_si256();
    w = (const int32_t*) w + 8;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m256i va0x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      const __m256i va0x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
      a0 += 16;

      const __m256i vb01234567x0123 = _mm256_load_si256(w);
      const __m256i vb89ABCDEFx0123 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vb01234567x4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 64));
      const __m256i vb89ABCDEFx4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 96));
      xnn_prefetch_to_l1((const int8_t*) w + 896);

      vacc0x0123 = _mm256_dpbssd_epi32(vacc0x0123, va0x01234567, vb01234567x0123);
      vacc0x4567 = _mm256_dpbssd_epi32(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
      xnn_prefetch_to_l1((const int8_t*) w + 960);
      vacc1x0x0123 = _mm256_dpbssd_epi32(vacc1x0x0123, va0x89ABCDEF, vb01234567x4567);
      vacc1x0x4567 = _mm256_dpbssd_epi32(vacc1x0x4567, va0x89ABCDEF, vb89ABCDEFx4567);

      w = (const int8_t*) w + 128;
      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const __m256i va0x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      a0 += 8;

      const __m256i vb01234567x0123 = _mm256_load_si256(w);
      const __m256i vb89ABCDEFx0123 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));

      vacc0x0123 = _mm256_dpbssd_epi32(vacc0x0123, va0x01234567, vb01234567x0123);
      vacc0x4567 = _mm256_dpbssd_epi32(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      w = (const int8_t*) w + 64;
      k -= 8 * sizeof(int8_t);
    }
    vacc0x0123 = _mm256_add_epi32(vacc0x0123, vacc1x0x0123);
    vacc0x4567 = _mm256_add_epi32(vacc0x4567, vacc1x0x4567);

    // Add adjacent pairs
    const __m256i vsum0x02134657 = _mm256_hadd_epi32(vacc0x0123, vacc0x4567);
    __m256i vacc0x01234567 = _mm256_permute4x64_epi64(vsum0x02134657, _MM_SHUFFLE(3, 1, 2, 0));

    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 vscale01234567 = _mm256_load_ps(w);
    w = (const float*) w + 8;
    vout0x01234567 = _mm256_mul_ps(vout0x01234567, vscale01234567);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vout0x01234567);

    vacc0x01234567 = _mm256_add_epi32(vacc0x01234567, voutput_zero_point);

    vacc0x01234567 = _mm256_packs_epi32(vacc0x01234567, _mm256_castsi128_si256(_mm256_extracti128_si256(vacc0x01234567, 1)));
    __m128i voutb0x01234567 = _mm256_castsi256_si128(_mm256_packs_epi16(vacc0x01234567, vacc0x01234567));

    voutb0x01234567 = _mm_max_epi8(voutb0x01234567, voutput_min);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, voutb0x01234567);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, voutb0x01234567);
        c0 += 4;
        voutb0x01234567 = _mm_srli_epi64(voutb0x01234567, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(voutb0x01234567, 0));
        c0 += 2;
        voutb0x01234567 = _mm_srli_epi32(voutb0x01234567, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(voutb0x01234567, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
