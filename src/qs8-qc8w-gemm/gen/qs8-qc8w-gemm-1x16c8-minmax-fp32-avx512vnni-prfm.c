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


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm(
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

  const __m512i vsign_mask = _mm512_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m512 voutput_max_less_zero_point = _mm512_set1_ps(params->fp32_avx512vnni.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_avx512vnni.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512vnni.output_min);
  do {
    __m512i vacc0x01234567 = _mm512_cvtepu32_epi64(_mm256_load_si256((const __m256i*) w));
    __m512i vacc0x89ABCDEF = _mm512_cvtepu32_epi64(_mm256_load_si256((const __m256i*) ((const int32_t*) w + 8)));
    __m512i vacc1x0x01234567 = _mm512_setzero_epi32();
    __m512i vacc1x0x89ABCDEF = _mm512_setzero_epi32();
    w = (const int32_t*) w + 16;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0)), vsign_mask);
      const __m512i va0x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8)), vsign_mask);
      a0 += 16;

      const __m512i vb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vb01234567x89ABCDEF = _mm512_load_si512((const int8_t*) w + 128);
      const __m512i vb89ABCDEFx89ABCDEF = _mm512_load_si512((const int8_t*) w + 192);
      xnn_prefetch_to_l1((const int8_t*) w + 768);
      xnn_prefetch_to_l1((const int8_t*) w + 832);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);
      vacc1x0x01234567 = _mm512_dpbusd_epi32(vacc1x0x01234567, va0x89ABCDEF, vb01234567x89ABCDEF);
      vacc1x0x89ABCDEF = _mm512_dpbusd_epi32(vacc1x0x89ABCDEF, va0x89ABCDEF, vb89ABCDEFx89ABCDEF);

      w = (const int8_t*) w + 256;
      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0)), vsign_mask);
      a0 += 8;

      const __m512i vb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      w = (const int8_t*) w + 128;
      k -= 8 * sizeof(int8_t);
    }
    vacc0x01234567 = _mm512_add_epi32(vacc0x01234567, vacc1x0x01234567);
    vacc0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, vacc1x0x89ABCDEF);

    // Add adjacent pairs
    const __m512i vidx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i vsum0x01234567 = _mm512_add_epi32(vacc0x01234567, _mm512_srli_epi64(vacc0x01234567, 32));
    const __m512i vsum0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, _mm512_srli_epi64(vacc0x89ABCDEF, 32));
    __m512i vacc0x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum0x01234567, vidx, vsum0x89ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const float*) w + 16;
    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale012345678ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);

    __m128i vout0x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc0x0123456789ABCDEF);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - UINT32_C(1));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
