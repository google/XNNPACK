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
#include "src/xnnpack/prefetch.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f16_qc2w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm(
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
  assert(mr <= 1);
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

  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
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
    __m512i vacc1x0x01234567 = _mm512_setzero_epi32();
    __m512i vacc1x0x89ABCDEF = _mm512_setzero_epi32();
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

      const __m512i vb01234567x1 = _mm512_and_si512(_mm512_srli_epi32(vbb01234567x0123456789ABCDEF, 2), vmask);
      const __m512i vb89ABCDEFx1 = _mm512_and_si512(_mm512_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 2), vmask);
      const __m512i va0x1 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8));
      vacc1x0x01234567 = _mm512_dpbusd_epi32(vacc1x0x01234567, vb01234567x1, va0x1);
      vacc1x0x89ABCDEF = _mm512_dpbusd_epi32(vacc1x0x89ABCDEF, vb89ABCDEFx1, va0x1);

      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      const __m512i vb01234567x2 = _mm512_and_si512(_mm512_srli_epi32(vbb01234567x0123456789ABCDEF, 4), vmask);
      const __m512i vb89ABCDEFx2 = _mm512_and_si512(_mm512_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 4), vmask);
      const __m512i va0x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 16));
      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x2, va0x2);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx2, va0x2);

      const __m512i vb01234567x3 = _mm512_and_si512(_mm512_srli_epi32(vbb01234567x0123456789ABCDEF, 6), vmask);
      const __m512i vb89ABCDEFx3 = _mm512_and_si512(_mm512_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 6), vmask);
      const __m512i va0x3 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 24));
      a0 += 32;
      vacc1x0x01234567 = _mm512_dpbusd_epi32(vacc1x0x01234567, vb01234567x3, va0x3);
      vacc1x0x89ABCDEF = _mm512_dpbusd_epi32(vacc1x0x89ABCDEF, vb89ABCDEFx3, va0x3);

      w = (const int8_t*) w + 128;
      k -= 32 * sizeof(int8_t);
    }
    while (k >= 16 * sizeof(int8_t)) {
      const __m512i va0x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
      const __m512i va0x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8));
      a0 += 16;

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
      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);
      vacc1x0x01234567 = _mm512_dpbusd_epi32(vacc1x0x01234567, vb01234567x89ABCDEF, va0x89ABCDEF);
      vacc1x0x89ABCDEF = _mm512_dpbusd_epi32(vacc1x0x89ABCDEF, vb89ABCDEFx89ABCDEF, va0x89ABCDEF);

      w = (const int8_t*) w + 128;
      k -= 16 * sizeof(int8_t);
      if (k != 0) {
        const __m512i vb01234567x2 = _mm512_and_si512(_mm512_srli_epi32(vbb01234567x01234567, 4), vmask);
        const __m512i vb89ABCDEFx2 = _mm512_and_si512(_mm512_srli_epi32(vbb89ABCDEFx01234567, 4), vmask);
        const __m512i va0x2 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
        a0 += 8;
        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x2, va0x2);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx2, va0x2);
        k -= 8 * sizeof(int8_t);
      }
    }

    if (k != 0) {
      const __m512i va0x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
      a0 += 8;

      const __m512i vbb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vbb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vb01234567x01234567 = _mm512_and_si512(vbb01234567x01234567, vmask);
      const __m512i vb89ABCDEFx01234567 = _mm512_and_si512(vbb89ABCDEFx01234567, vmask);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, vb01234567x01234567, va0x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, vb89ABCDEFx01234567, va0x01234567);
      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      w = (const int8_t*) w + 128;
      k -= 8 * sizeof(int8_t);
    }
    assert(k == 0);
    vacc0x01234567 = _mm512_add_epi32(vacc0x01234567, vacc1x0x01234567);
    vacc0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, vacc1x0x89ABCDEF);

    // Add adjacent pairs
    const __m512i vidx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i vsum0x01234567 = _mm512_add_epi32(vacc0x01234567, _mm512_srli_epi64(vacc0x01234567, 32));
    const __m512i vsum0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, _mm512_srli_epi64(vacc0x89ABCDEF, 32));
    __m512i vacc0x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum0x01234567, vidx, vsum0x89ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);

    const __m512 rh_zero_points_0123456789ABCDEF = _mm512_load_ps((const float*) kzp);
    const __m512 biased_rh_zero_points_0123456789ABCDEF = _mm512_add_ps(rh_zero_points_0123456789ABCDEF, _mm512_set1_ps(2.0f));

    // Subtract out the scaled left-hand row sums.
    const __m512 lh_row_sum_0 = _mm512_set1_ps(row_sum[0]);
    vscaled0x0123456789ABCDEF = _mm512_fnmadd_ps(biased_rh_zero_points_0123456789ABCDEF, lh_row_sum_0, vscaled0x0123456789ABCDEF);
    // Add the product of left/right-hand zero points and `kc`.
    const __m512 vscaled_lh_zero_point_0 = _mm512_set1_ps((float) original_kc * quantization_params[0].zero_point);
    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(rh_zero_points_0123456789ABCDEF, vscaled_lh_zero_point_0, vscaled0x0123456789ABCDEF);
    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_si256((__m256i*) c0, _mm512_cvtps_ph(vscaled0x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 16-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm256_mask_storeu_epi16(c0, vmask, _mm512_cvtps_ph(vscaled0x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      nc = 0;
    }
  } while (nc != 0);
}
