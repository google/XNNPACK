// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx8c8-avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
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


void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x8c8__avx2_madd(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
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
  float* c0 = c;

  const __m256i vinput_zero_point0 = _mm256_set1_epi32((int) quantization_params[0].zero_point);
  const __m256 voutput_min = _mm256_set1_ps(params->scalar.min);
  const __m256 voutput_max = _mm256_set1_ps(params->scalar.max);
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  const __m256i vmask = _mm256_set1_epi8(0x03);
  do {
    const __m256i vksum01234567 = _mm256_load_si256(w);
    __m256i vsum0x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point0);
    __m256i vacc0x0123 = _mm256_setzero_si256();
    __m256i vacc0x4567 = _mm256_setzero_si256();
    __m256i vacc1x0x0123 = _mm256_setzero_si256();
    __m256i vacc1x0x4567 = _mm256_setzero_si256();
    w = (const int32_t*) w + 8;
    // TODO: move kernel zero point after weights
    const void* kzp = w;
    w = (const float*)w + 8;

    size_t k = kc;
    while (k >= 32 * sizeof(int8_t)) {
      const __m256i va0x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      const __m256i va0x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
      const __m256i va0x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 16));
      const __m256i va0x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 24));
      a0 += 32;

      const __m256i vbb01234567x0123456789ABCDEF = _mm256_load_si256(w);
      const __m256i vbb89ABCDEFx0123456789ABCDEF = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vb01234567x0123 = _mm256_and_si256(vbb01234567x0123456789ABCDEF, vmask);
      const __m256i vb89ABCDEFx0123 = _mm256_and_si256(vbb89ABCDEFx0123456789ABCDEF, vmask);
      const __m256i vbs01234567x4567 = _mm256_srli_epi32(vbb01234567x0123456789ABCDEF, 2);
      const __m256i vbs89ABCDEFx4567 = _mm256_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 2);
      const __m256i vb01234567x4567 = _mm256_and_si256(vbs01234567x4567, vmask);
      const __m256i vb89ABCDEFx4567 = _mm256_and_si256(vbs89ABCDEFx4567, vmask);
      const __m256i vbs01234567x89AB = _mm256_srli_epi32(vbb01234567x0123456789ABCDEF, 4);
      const __m256i vbs89ABCDEFx89AB = _mm256_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 4);
      const __m256i vb01234567x89AB = _mm256_and_si256(vbs01234567x89AB, vmask);
      const __m256i vb89ABCDEFx89AB = _mm256_and_si256(vbs89ABCDEFx89AB, vmask);
      const __m256i vbs01234567xCDEF = _mm256_srli_epi32(vbb01234567x0123456789ABCDEF, 6);
      const __m256i vbs89ABCDEFxCDEF = _mm256_srli_epi32(vbb89ABCDEFx0123456789ABCDEF, 6);
      const __m256i vb01234567xCDEF = _mm256_and_si256(vbs01234567xCDEF, vmask);
      const __m256i vb89ABCDEFxCDEF = _mm256_and_si256(vbs89ABCDEFxCDEF, vmask);

      vacc0x0123 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x0123, va0x0, vb01234567x0123);
      vacc0x4567 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x4567, va0x0, vb89ABCDEFx0123);
      vacc1x0x0123 = _mm256_dpbusd_epi32_madd_kzp2(vacc1x0x0123, va0x1, vb01234567x4567);
      vacc1x0x4567 = _mm256_dpbusd_epi32_madd_kzp2(vacc1x0x4567, va0x1, vb89ABCDEFx4567);
      vacc0x0123 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x0123, va0x2, vb01234567x89AB);
      vacc0x4567 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x4567, va0x2, vb89ABCDEFx89AB);
      vacc1x0x0123 = _mm256_dpbusd_epi32_madd_kzp2(vacc1x0x0123, va0x3, vb01234567xCDEF);
      vacc1x0x4567 = _mm256_dpbusd_epi32_madd_kzp2(vacc1x0x4567, va0x3, vb89ABCDEFxCDEF);
      w = (const int8_t*) w + 64;
      k -= 32 * sizeof(int8_t);
    }
    while (k >= 16 * sizeof(int8_t)) {
      const __m256i va0x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      const __m256i va0x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
      a0 += 16;

      // 2 planes of 2 bit.  potentially 3rd plane handled later
      const __m256i vbb01234567x01234567 = _mm256_load_si256(w);
      const __m256i vbb89ABCDEFx01234567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vbs01234567x4567 = _mm256_srli_epi32(vbb01234567x01234567, 2);
      const __m256i vbs89ABCDEFx4567 = _mm256_srli_epi32(vbb89ABCDEFx01234567, 2);
      const __m256i vb01234567x0123 = _mm256_and_si256(vbb01234567x01234567, vmask);
      const __m256i vb89ABCDEFx0123 = _mm256_and_si256(vbb89ABCDEFx01234567, vmask);
      const __m256i vb01234567x4567 = _mm256_and_si256(vbs01234567x4567, vmask);
      const __m256i vb89ABCDEFx4567 = _mm256_and_si256(vbs89ABCDEFx4567, vmask);

      vacc0x0123 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x0123, va0x01234567, vb01234567x0123);
      vacc0x4567 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
      vacc1x0x0123 = _mm256_dpbusd_epi32_madd_kzp2(vacc1x0x0123, va0x89ABCDEF, vb01234567x4567);
      vacc1x0x4567 = _mm256_dpbusd_epi32_madd_kzp2(vacc1x0x4567, va0x89ABCDEF, vb89ABCDEFx4567);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
      // 3rd plane for 2 bit
      if (k != 0) {
        const __m256i va0x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
        a0 += 8;

        // mask 3rd plane of 2 bit.
        const __m256i vbs01234567x89AB = _mm256_srli_epi32(vbb01234567x01234567, 4);
        const __m256i vbs89ABCDEFx89AB = _mm256_srli_epi32(vbb89ABCDEFx01234567, 4);
        const __m256i vb01234567x89AB = _mm256_and_si256(vbs01234567x89AB, vmask);
        const __m256i vb89ABCDEFx89AB = _mm256_and_si256(vbs89ABCDEFx89AB, vmask);

        vacc0x0123 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x0123, va0x3, vb01234567x89AB);
        vacc0x4567 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x4567, va0x3, vb89ABCDEFx89AB);
        k -= 8 * sizeof(int8_t);
      }
    }

    if (k != 0) {
      const __m256i va0x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      a0 += 8;

      // 1 plane of 2 bit.
      const __m256i vbb01234567x01234567 = _mm256_load_si256(w);
      const __m256i vbb89ABCDEFx01234567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vb01234567x0123 = _mm256_and_si256(vbb01234567x01234567, vmask);
      const __m256i vb89ABCDEFx0123 = _mm256_and_si256(vbb89ABCDEFx01234567, vmask);

      vacc0x0123 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x0123, va0x01234567, vb01234567x0123);
      vacc0x4567 = _mm256_dpbusd_epi32_madd_kzp2(vacc0x4567, va0x01234567, vb89ABCDEFx0123);

      w = (const int8_t*) w + 64;
      k -= 8 * sizeof(int8_t);
    }
    // Make sure there were no leftovers.
    assert(k == 0);
    vacc0x0123 = _mm256_add_epi32(vacc0x0123, vacc1x0x0123);
    vacc0x4567 = _mm256_add_epi32(vacc0x4567, vacc1x0x4567);

    // Add adjacent pairs
    const __m256i vsum0x02134657 = _mm256_hadd_epi32(vacc0x0123, vacc0x4567);
    __m256i vacc0x01234567 = _mm256_permute4x64_epi64(vsum0x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    vacc0x01234567 = _mm256_add_epi32(vacc0x01234567, vsum0x01234567);

    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 rh_zero_points_01234567 = _mm256_load_ps((const float*) kzp);
    kzp = (const float*)kzp + 8;

    // Subtract out the scaled left-hand row sums.
    const __m256 lh_row_sum_0 = _mm256_set1_ps(row_sum[0]);
    vout0x01234567 = _mm256_fnmadd_ps(rh_zero_points_01234567, lh_row_sum_0, vout0x01234567);
    // Add the product of left/right-hand zero points and `kc`.
    // TODO: use kc
    const __m256 vscaled_lh_zero_point_0 = _mm256_set1_ps((float)original_kc * quantization_params[0].zero_point);
    vout0x01234567 = _mm256_fmadd_ps(rh_zero_points_01234567, vscaled_lh_zero_point_0, vout0x01234567);
    vout0x01234567 = _mm256_mul_ps(vout0x01234567, _mm256_set1_ps(quantization_params[0].inv_scale));

    const __m256 vfilter_output_scale01234567 = _mm256_load_ps((const float*) w);
    const __m256 vbias01234567 = _mm256_load_ps((const float*) w + 8);
    w = (const float*) w + 16;

    vout0x01234567 = _mm256_fmadd_ps(vout0x01234567, vfilter_output_scale01234567, vbias01234567);

    vout0x01234567 = _mm256_max_ps(vout0x01234567, voutput_min);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      _mm256_storeu_ps(c0, vout0x01234567);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      nc -= 8;
    } else {
      __m128 vout0x0123 = _mm256_castps256_ps128(vout0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vout0x0123);
        c0 += 4;
        vout0x0123 = _mm256_extractf128_ps(vout0x01234567, 1);
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        c0 += 2;
        vout0x0123 = _mm_movehl_ps(vout0x0123, vout0x0123);
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
