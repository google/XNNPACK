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

  const __m256 voutput_min = _mm256_set1_ps(params->scalar.min);
  const __m256 voutput_max = _mm256_set1_ps(params->scalar.max);
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  const __m256i vmask = _mm256_set1_epi8(0x03);
  const __m256i vmask4 = _mm256_set1_epi8(0x0C);
  const __m256i vminus_one = _mm256_set1_epi16(-1);
  // -(kzp + 2) * row_sum + kzp * kc * zero_point is evaluated as
  // -2 * row_sum + kzp * (kc * zero_point - row_sum).
  const float neg_two_row_sum0 = -2.0f * row_sum[0];
  const float kzp_scale0 = (float) original_kc * quantization_params[0].zero_point - row_sum[0];
  do {
    const int32_t* ksum = (const int32_t*) w;
    __m256i vacc0x0123 = _mm256_setzero_si256();
    __m256i vacc0x4567 = _mm256_setzero_si256();
    w = (const int32_t*) w + 8;
    // TODO: move kernel zero point after weights
    w = (const float*) w + 8;

    size_t k = kc;
    // Planes 0 and 2 use the weights in bits [1:0] (scale 1) and planes 1 and
    // 3 use the weights in bits [3:2] (scale 4), so one shift by 4 unpacks all
    // 4 planes. The scale 4 sums take at most 8 products of magnitude 3072
    // per 128 bytes of k, so they stay within 24576 and cannot overflow int16.
    do {
      __m256i vsum0x0123 = _mm256_setzero_si256();
      __m256i vsum0x4567 = _mm256_setzero_si256();
      __m256i vsum4x0x0123 = _mm256_setzero_si256();
      __m256i vsum4x0x4567 = _mm256_setzero_si256();
      size_t kb = min(k, 128 * sizeof(int8_t));
      k -= kb;
      for (; kb >= 32 * sizeof(int8_t); kb -= 32 * sizeof(int8_t)) {
        const __m256i vbb0123 = _mm256_load_si256(w);
        const __m256i vbb4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
        w = (const int8_t*) w + 64;

        const __m256i vbs0123 = _mm256_srli_epi32(vbb0123, 4);
        const __m256i vbs4567 = _mm256_srli_epi32(vbb4567, 4);
        const __m256i vb0123x0 = _mm256_and_si256(vbb0123, vmask);
        const __m256i vb4567x0 = _mm256_and_si256(vbb4567, vmask);
        const __m256i vb0123x1 = _mm256_and_si256(vbb0123, vmask4);
        const __m256i vb4567x1 = _mm256_and_si256(vbb4567, vmask4);
        const __m256i vb0123x2 = _mm256_and_si256(vbs0123, vmask);
        const __m256i vb4567x2 = _mm256_and_si256(vbs4567, vmask);
        const __m256i vb0123x3 = _mm256_and_si256(vbs0123, vmask4);
        const __m256i vb4567x3 = _mm256_and_si256(vbs4567, vmask4);

        const __m256i va0x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
        const __m256i va0x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
        const __m256i va0x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 16));
        const __m256i va0x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 24));
        a0 += 32;

        vsum0x0123 = _mm256_add_epi16(vsum0x0123, _mm256_maddubs_epi16(vb0123x0, va0x0));
        vsum0x4567 = _mm256_add_epi16(vsum0x4567, _mm256_maddubs_epi16(vb4567x0, va0x0));
        vsum4x0x0123 = _mm256_add_epi16(vsum4x0x0123, _mm256_maddubs_epi16(vb0123x1, va0x1));
        vsum4x0x4567 = _mm256_add_epi16(vsum4x0x4567, _mm256_maddubs_epi16(vb4567x1, va0x1));
        vsum0x0123 = _mm256_add_epi16(vsum0x0123, _mm256_maddubs_epi16(vb0123x2, va0x2));
        vsum0x4567 = _mm256_add_epi16(vsum0x4567, _mm256_maddubs_epi16(vb4567x2, va0x2));
        vsum4x0x0123 = _mm256_add_epi16(vsum4x0x0123, _mm256_maddubs_epi16(vb0123x3, va0x3));
        vsum4x0x4567 = _mm256_add_epi16(vsum4x0x4567, _mm256_maddubs_epi16(vb4567x3, va0x3));
      }
      if (kb != 0) {
        const __m256i vbb0123 = _mm256_load_si256(w);
        const __m256i vbb4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
        w = (const int8_t*) w + 64;

        const __m256i va0x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
        vsum0x0123 = _mm256_add_epi16(vsum0x0123, _mm256_maddubs_epi16(_mm256_and_si256(vbb0123, vmask), va0x0));
        vsum0x4567 = _mm256_add_epi16(vsum0x4567, _mm256_maddubs_epi16(_mm256_and_si256(vbb4567, vmask), va0x0));
        if (kb > 8 * sizeof(int8_t)) {
          const __m256i va0x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
          vsum4x0x0123 = _mm256_add_epi16(vsum4x0x0123, _mm256_maddubs_epi16(_mm256_and_si256(vbb0123, vmask4), va0x1));
          vsum4x0x4567 = _mm256_add_epi16(vsum4x0x4567, _mm256_maddubs_epi16(_mm256_and_si256(vbb4567, vmask4), va0x1));
          if (kb > 16 * sizeof(int8_t)) {
            const __m256i va0x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 16));
            vsum0x0123 = _mm256_add_epi16(vsum0x0123, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi32(vbb0123, 4), vmask), va0x2));
            vsum0x4567 = _mm256_add_epi16(vsum0x4567, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi32(vbb4567, 4), vmask), va0x2));
          }
        }
        a0 += kb;
      }
      vsum0x0123 = _mm256_add_epi16(vsum0x0123, _mm256_srai_epi16(vsum4x0x0123, 2));
      vsum0x4567 = _mm256_add_epi16(vsum0x4567, _mm256_srai_epi16(vsum4x0x4567, 2));
      vacc0x0123 = _mm256_sub_epi32(vacc0x0123, _mm256_madd_epi16(vsum0x0123, vminus_one));
      vacc0x4567 = _mm256_sub_epi32(vacc0x4567, _mm256_madd_epi16(vsum0x4567, vminus_one));
    } while (k != 0);

    // Add adjacent pairs
    const __m256i vsum0x02134657 = _mm256_hadd_epi32(vacc0x0123, vacc0x4567);
    __m256i vacc0x01234567 = _mm256_permute4x64_epi64(vsum0x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vksum01234567 = _mm256_load_si256((const __m256i*) ksum);
    vacc0x01234567 = _mm256_add_epi32(vacc0x01234567, _mm256_mullo_epi32(vksum01234567, _mm256_set1_epi32((int) quantization_params[0].zero_point)));

    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);

    const __m256 rh_zero_points_01234567 = _mm256_load_ps((const float*) (ksum + 8));

    // Subtract out the left-hand row sums scaled by the weight bias of 2.
    vout0x01234567 = _mm256_add_ps(vout0x01234567, _mm256_set1_ps(neg_two_row_sum0));
    // Add the right-hand zero points scaled by the left-hand zero points
    // and `original_kc` less the left-hand row sums.
    vout0x01234567 = _mm256_fmadd_ps(rh_zero_points_01234567, _mm256_set1_ps(kzp_scale0), vout0x01234567);
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
