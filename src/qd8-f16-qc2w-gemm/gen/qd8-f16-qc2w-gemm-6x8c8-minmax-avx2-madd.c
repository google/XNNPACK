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


void xnn_qd8_f16_qc2w_gemm_minmax_ukernel_6x8c8__avx2_madd(
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
  assert(mr <= 6);
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
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const __m256 voutput_min = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m256 voutput_max = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  const __m256i vmask = _mm256_set1_epi8(0x03);
  const __m256i vminus_one = _mm256_set1_epi16(-1);
  // -(kzp + 2) * row_sum + kzp * kc * zero_point is evaluated as
  // -2 * row_sum + kzp * (kc * zero_point - row_sum).
  const float neg_two_row_sum0 = -2.0f * row_sum[0];
  const float kzp_scale0 = (float) original_kc * quantization_params[0].zero_point - row_sum[0];
  const float neg_two_row_sum1 = -2.0f * row_sum[1];
  const float kzp_scale1 = (float) original_kc * quantization_params[1].zero_point - row_sum[1];
  const float neg_two_row_sum2 = -2.0f * row_sum[2];
  const float kzp_scale2 = (float) original_kc * quantization_params[2].zero_point - row_sum[2];
  const float neg_two_row_sum3 = -2.0f * row_sum[3];
  const float kzp_scale3 = (float) original_kc * quantization_params[3].zero_point - row_sum[3];
  const float neg_two_row_sum4 = -2.0f * row_sum[4];
  const float kzp_scale4 = (float) original_kc * quantization_params[4].zero_point - row_sum[4];
  const float neg_two_row_sum5 = -2.0f * row_sum[5];
  const float kzp_scale5 = (float) original_kc * quantization_params[5].zero_point - row_sum[5];
  do {
    const int32_t* ksum = (const int32_t*) w;
    __m256i vacc0x0123 = _mm256_setzero_si256();
    __m256i vacc0x4567 = _mm256_setzero_si256();
    __m256i vacc1x0123 = _mm256_setzero_si256();
    __m256i vacc1x4567 = _mm256_setzero_si256();
    __m256i vacc2x0123 = _mm256_setzero_si256();
    __m256i vacc2x4567 = _mm256_setzero_si256();
    __m256i vacc3x0123 = _mm256_setzero_si256();
    __m256i vacc3x4567 = _mm256_setzero_si256();
    __m256i vacc4x0123 = _mm256_setzero_si256();
    __m256i vacc4x4567 = _mm256_setzero_si256();
    __m256i vacc5x0123 = _mm256_setzero_si256();
    __m256i vacc5x4567 = _mm256_setzero_si256();
    w = (const int32_t*) w + 8;
    // TODO: move kernel zero point after weights
    w = (const float*) w + 8;

    size_t k = kc;
    // The 4 planes of a pairwise u2 * s8 sum are bounded by 4 * 768 = 3072, so
    // they are summed in int16 and widened to int32 once.
    while (k >= 32 * sizeof(int8_t)) {
      const __m256i vbb0123 = _mm256_load_si256(w);
      const __m256i vbb4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      w = (const int8_t*) w + 64;

      const __m256i vb0123x0 = _mm256_and_si256(vbb0123, vmask);
      const __m256i vb4567x0 = _mm256_and_si256(vbb4567, vmask);
      const __m256i vb0123x1 = _mm256_and_si256(_mm256_srli_epi32(vbb0123, 2), vmask);
      const __m256i vb4567x1 = _mm256_and_si256(_mm256_srli_epi32(vbb4567, 2), vmask);
      const __m256i vb0123x2 = _mm256_and_si256(_mm256_srli_epi32(vbb0123, 4), vmask);
      const __m256i vb4567x2 = _mm256_and_si256(_mm256_srli_epi32(vbb4567, 4), vmask);
      const __m256i vb0123x3 = _mm256_and_si256(_mm256_srli_epi32(vbb0123, 6), vmask);
      const __m256i vb4567x3 = _mm256_and_si256(_mm256_srli_epi32(vbb4567, 6), vmask);

      const __m256i va0x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      const __m256i va0x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
      const __m256i va0x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 16));
      const __m256i va0x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 24));
      a0 += 32;

      const __m256i vsum0x0123 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x0, va0x0),
              _mm256_maddubs_epi16(vb0123x1, va0x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x2, va0x2),
              _mm256_maddubs_epi16(vb0123x3, va0x3)));
      const __m256i vsum0x4567 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x0, va0x0),
              _mm256_maddubs_epi16(vb4567x1, va0x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x2, va0x2),
              _mm256_maddubs_epi16(vb4567x3, va0x3)));
      vacc0x0123 = _mm256_sub_epi32(vacc0x0123, _mm256_madd_epi16(vsum0x0123, vminus_one));
      vacc0x4567 = _mm256_sub_epi32(vacc0x4567, _mm256_madd_epi16(vsum0x4567, vminus_one));
      const __m256i va1x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1));
      const __m256i va1x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8));
      const __m256i va1x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1 + 16));
      const __m256i va1x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1 + 24));
      a1 += 32;

      const __m256i vsum1x0123 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x0, va1x0),
              _mm256_maddubs_epi16(vb0123x1, va1x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x2, va1x2),
              _mm256_maddubs_epi16(vb0123x3, va1x3)));
      const __m256i vsum1x4567 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x0, va1x0),
              _mm256_maddubs_epi16(vb4567x1, va1x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x2, va1x2),
              _mm256_maddubs_epi16(vb4567x3, va1x3)));
      vacc1x0123 = _mm256_sub_epi32(vacc1x0123, _mm256_madd_epi16(vsum1x0123, vminus_one));
      vacc1x4567 = _mm256_sub_epi32(vacc1x4567, _mm256_madd_epi16(vsum1x4567, vminus_one));
      const __m256i va2x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2));
      const __m256i va2x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8));
      const __m256i va2x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2 + 16));
      const __m256i va2x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2 + 24));
      a2 += 32;

      const __m256i vsum2x0123 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x0, va2x0),
              _mm256_maddubs_epi16(vb0123x1, va2x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x2, va2x2),
              _mm256_maddubs_epi16(vb0123x3, va2x3)));
      const __m256i vsum2x4567 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x0, va2x0),
              _mm256_maddubs_epi16(vb4567x1, va2x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x2, va2x2),
              _mm256_maddubs_epi16(vb4567x3, va2x3)));
      vacc2x0123 = _mm256_sub_epi32(vacc2x0123, _mm256_madd_epi16(vsum2x0123, vminus_one));
      vacc2x4567 = _mm256_sub_epi32(vacc2x4567, _mm256_madd_epi16(vsum2x4567, vminus_one));
      const __m256i va3x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3));
      const __m256i va3x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3 + 8));
      const __m256i va3x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3 + 16));
      const __m256i va3x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3 + 24));
      a3 += 32;

      const __m256i vsum3x0123 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x0, va3x0),
              _mm256_maddubs_epi16(vb0123x1, va3x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x2, va3x2),
              _mm256_maddubs_epi16(vb0123x3, va3x3)));
      const __m256i vsum3x4567 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x0, va3x0),
              _mm256_maddubs_epi16(vb4567x1, va3x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x2, va3x2),
              _mm256_maddubs_epi16(vb4567x3, va3x3)));
      vacc3x0123 = _mm256_sub_epi32(vacc3x0123, _mm256_madd_epi16(vsum3x0123, vminus_one));
      vacc3x4567 = _mm256_sub_epi32(vacc3x4567, _mm256_madd_epi16(vsum3x4567, vminus_one));
      const __m256i va4x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4));
      const __m256i va4x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4 + 8));
      const __m256i va4x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4 + 16));
      const __m256i va4x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4 + 24));
      a4 += 32;

      const __m256i vsum4x0123 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x0, va4x0),
              _mm256_maddubs_epi16(vb0123x1, va4x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x2, va4x2),
              _mm256_maddubs_epi16(vb0123x3, va4x3)));
      const __m256i vsum4x4567 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x0, va4x0),
              _mm256_maddubs_epi16(vb4567x1, va4x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x2, va4x2),
              _mm256_maddubs_epi16(vb4567x3, va4x3)));
      vacc4x0123 = _mm256_sub_epi32(vacc4x0123, _mm256_madd_epi16(vsum4x0123, vminus_one));
      vacc4x4567 = _mm256_sub_epi32(vacc4x4567, _mm256_madd_epi16(vsum4x4567, vminus_one));
      const __m256i va5x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5));
      const __m256i va5x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5 + 8));
      const __m256i va5x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5 + 16));
      const __m256i va5x3 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5 + 24));
      a5 += 32;

      const __m256i vsum5x0123 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x0, va5x0),
              _mm256_maddubs_epi16(vb0123x1, va5x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb0123x2, va5x2),
              _mm256_maddubs_epi16(vb0123x3, va5x3)));
      const __m256i vsum5x4567 = _mm256_add_epi16(
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x0, va5x0),
              _mm256_maddubs_epi16(vb4567x1, va5x1)),
          _mm256_add_epi16(
              _mm256_maddubs_epi16(vb4567x2, va5x2),
              _mm256_maddubs_epi16(vb4567x3, va5x3)));
      vacc5x0123 = _mm256_sub_epi32(vacc5x0123, _mm256_madd_epi16(vsum5x0123, vminus_one));
      vacc5x4567 = _mm256_sub_epi32(vacc5x4567, _mm256_madd_epi16(vsum5x4567, vminus_one));

      k -= 32 * sizeof(int8_t);
    }
    if (k != 0) {
      const __m256i vbb0123 = _mm256_load_si256(w);
      const __m256i vbb4567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      w = (const int8_t*) w + 64;

      const __m256i vb0123x0 = _mm256_and_si256(vbb0123, vmask);
      const __m256i vb4567x0 = _mm256_and_si256(vbb4567, vmask);
      const __m256i va0x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      __m256i vsum0x0123 = _mm256_maddubs_epi16(vb0123x0, va0x0);
      __m256i vsum0x4567 = _mm256_maddubs_epi16(vb4567x0, va0x0);
      const __m256i va1x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1));
      __m256i vsum1x0123 = _mm256_maddubs_epi16(vb0123x0, va1x0);
      __m256i vsum1x4567 = _mm256_maddubs_epi16(vb4567x0, va1x0);
      const __m256i va2x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2));
      __m256i vsum2x0123 = _mm256_maddubs_epi16(vb0123x0, va2x0);
      __m256i vsum2x4567 = _mm256_maddubs_epi16(vb4567x0, va2x0);
      const __m256i va3x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3));
      __m256i vsum3x0123 = _mm256_maddubs_epi16(vb0123x0, va3x0);
      __m256i vsum3x4567 = _mm256_maddubs_epi16(vb4567x0, va3x0);
      const __m256i va4x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4));
      __m256i vsum4x0123 = _mm256_maddubs_epi16(vb0123x0, va4x0);
      __m256i vsum4x4567 = _mm256_maddubs_epi16(vb4567x0, va4x0);
      const __m256i va5x0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5));
      __m256i vsum5x0123 = _mm256_maddubs_epi16(vb0123x0, va5x0);
      __m256i vsum5x4567 = _mm256_maddubs_epi16(vb4567x0, va5x0);
      if (k > 8 * sizeof(int8_t)) {
        const __m256i vb0123x1 = _mm256_and_si256(_mm256_srli_epi32(vbb0123, 2), vmask);
        const __m256i vb4567x1 = _mm256_and_si256(_mm256_srli_epi32(vbb4567, 2), vmask);
        const __m256i va0x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
        vsum0x0123 = _mm256_add_epi16(vsum0x0123, _mm256_maddubs_epi16(vb0123x1, va0x1));
        vsum0x4567 = _mm256_add_epi16(vsum0x4567, _mm256_maddubs_epi16(vb4567x1, va0x1));
        const __m256i va1x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8));
        vsum1x0123 = _mm256_add_epi16(vsum1x0123, _mm256_maddubs_epi16(vb0123x1, va1x1));
        vsum1x4567 = _mm256_add_epi16(vsum1x4567, _mm256_maddubs_epi16(vb4567x1, va1x1));
        const __m256i va2x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8));
        vsum2x0123 = _mm256_add_epi16(vsum2x0123, _mm256_maddubs_epi16(vb0123x1, va2x1));
        vsum2x4567 = _mm256_add_epi16(vsum2x4567, _mm256_maddubs_epi16(vb4567x1, va2x1));
        const __m256i va3x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3 + 8));
        vsum3x0123 = _mm256_add_epi16(vsum3x0123, _mm256_maddubs_epi16(vb0123x1, va3x1));
        vsum3x4567 = _mm256_add_epi16(vsum3x4567, _mm256_maddubs_epi16(vb4567x1, va3x1));
        const __m256i va4x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4 + 8));
        vsum4x0123 = _mm256_add_epi16(vsum4x0123, _mm256_maddubs_epi16(vb0123x1, va4x1));
        vsum4x4567 = _mm256_add_epi16(vsum4x4567, _mm256_maddubs_epi16(vb4567x1, va4x1));
        const __m256i va5x1 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5 + 8));
        vsum5x0123 = _mm256_add_epi16(vsum5x0123, _mm256_maddubs_epi16(vb0123x1, va5x1));
        vsum5x4567 = _mm256_add_epi16(vsum5x4567, _mm256_maddubs_epi16(vb4567x1, va5x1));
        if (k > 16 * sizeof(int8_t)) {
          const __m256i vb0123x2 = _mm256_and_si256(_mm256_srli_epi32(vbb0123, 4), vmask);
          const __m256i vb4567x2 = _mm256_and_si256(_mm256_srli_epi32(vbb4567, 4), vmask);
          const __m256i va0x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 16));
          vsum0x0123 = _mm256_add_epi16(vsum0x0123, _mm256_maddubs_epi16(vb0123x2, va0x2));
          vsum0x4567 = _mm256_add_epi16(vsum0x4567, _mm256_maddubs_epi16(vb4567x2, va0x2));
          const __m256i va1x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1 + 16));
          vsum1x0123 = _mm256_add_epi16(vsum1x0123, _mm256_maddubs_epi16(vb0123x2, va1x2));
          vsum1x4567 = _mm256_add_epi16(vsum1x4567, _mm256_maddubs_epi16(vb4567x2, va1x2));
          const __m256i va2x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2 + 16));
          vsum2x0123 = _mm256_add_epi16(vsum2x0123, _mm256_maddubs_epi16(vb0123x2, va2x2));
          vsum2x4567 = _mm256_add_epi16(vsum2x4567, _mm256_maddubs_epi16(vb4567x2, va2x2));
          const __m256i va3x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3 + 16));
          vsum3x0123 = _mm256_add_epi16(vsum3x0123, _mm256_maddubs_epi16(vb0123x2, va3x2));
          vsum3x4567 = _mm256_add_epi16(vsum3x4567, _mm256_maddubs_epi16(vb4567x2, va3x2));
          const __m256i va4x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4 + 16));
          vsum4x0123 = _mm256_add_epi16(vsum4x0123, _mm256_maddubs_epi16(vb0123x2, va4x2));
          vsum4x4567 = _mm256_add_epi16(vsum4x4567, _mm256_maddubs_epi16(vb4567x2, va4x2));
          const __m256i va5x2 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5 + 16));
          vsum5x0123 = _mm256_add_epi16(vsum5x0123, _mm256_maddubs_epi16(vb0123x2, va5x2));
          vsum5x4567 = _mm256_add_epi16(vsum5x4567, _mm256_maddubs_epi16(vb4567x2, va5x2));
        }
      }
      a0 += k;
      vacc0x0123 = _mm256_sub_epi32(vacc0x0123, _mm256_madd_epi16(vsum0x0123, vminus_one));
      vacc0x4567 = _mm256_sub_epi32(vacc0x4567, _mm256_madd_epi16(vsum0x4567, vminus_one));
      a1 += k;
      vacc1x0123 = _mm256_sub_epi32(vacc1x0123, _mm256_madd_epi16(vsum1x0123, vminus_one));
      vacc1x4567 = _mm256_sub_epi32(vacc1x4567, _mm256_madd_epi16(vsum1x4567, vminus_one));
      a2 += k;
      vacc2x0123 = _mm256_sub_epi32(vacc2x0123, _mm256_madd_epi16(vsum2x0123, vminus_one));
      vacc2x4567 = _mm256_sub_epi32(vacc2x4567, _mm256_madd_epi16(vsum2x4567, vminus_one));
      a3 += k;
      vacc3x0123 = _mm256_sub_epi32(vacc3x0123, _mm256_madd_epi16(vsum3x0123, vminus_one));
      vacc3x4567 = _mm256_sub_epi32(vacc3x4567, _mm256_madd_epi16(vsum3x4567, vminus_one));
      a4 += k;
      vacc4x0123 = _mm256_sub_epi32(vacc4x0123, _mm256_madd_epi16(vsum4x0123, vminus_one));
      vacc4x4567 = _mm256_sub_epi32(vacc4x4567, _mm256_madd_epi16(vsum4x4567, vminus_one));
      a5 += k;
      vacc5x0123 = _mm256_sub_epi32(vacc5x0123, _mm256_madd_epi16(vsum5x0123, vminus_one));
      vacc5x4567 = _mm256_sub_epi32(vacc5x4567, _mm256_madd_epi16(vsum5x4567, vminus_one));
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
    const __m256i vksum01234567 = _mm256_load_si256((const __m256i*) ksum);
    vacc0x01234567 = _mm256_add_epi32(vacc0x01234567, _mm256_mullo_epi32(vksum01234567, _mm256_set1_epi32((int) quantization_params[0].zero_point)));
    vacc1x01234567 = _mm256_add_epi32(vacc1x01234567, _mm256_mullo_epi32(vksum01234567, _mm256_set1_epi32((int) quantization_params[1].zero_point)));
    vacc2x01234567 = _mm256_add_epi32(vacc2x01234567, _mm256_mullo_epi32(vksum01234567, _mm256_set1_epi32((int) quantization_params[2].zero_point)));
    vacc3x01234567 = _mm256_add_epi32(vacc3x01234567, _mm256_mullo_epi32(vksum01234567, _mm256_set1_epi32((int) quantization_params[3].zero_point)));
    vacc4x01234567 = _mm256_add_epi32(vacc4x01234567, _mm256_mullo_epi32(vksum01234567, _mm256_set1_epi32((int) quantization_params[4].zero_point)));
    vacc5x01234567 = _mm256_add_epi32(vacc5x01234567, _mm256_mullo_epi32(vksum01234567, _mm256_set1_epi32((int) quantization_params[5].zero_point)));

    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vout1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vout2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);
    __m256 vout3x01234567 = _mm256_cvtepi32_ps(vacc3x01234567);
    __m256 vout4x01234567 = _mm256_cvtepi32_ps(vacc4x01234567);
    __m256 vout5x01234567 = _mm256_cvtepi32_ps(vacc5x01234567);

    const __m256 rh_zero_points_01234567 = _mm256_load_ps((const float*) (ksum + 8));

    // Subtract out the left-hand row sums scaled by the weight bias of 2.
    vout0x01234567 = _mm256_add_ps(vout0x01234567, _mm256_set1_ps(neg_two_row_sum0));
    vout1x01234567 = _mm256_add_ps(vout1x01234567, _mm256_set1_ps(neg_two_row_sum1));
    vout2x01234567 = _mm256_add_ps(vout2x01234567, _mm256_set1_ps(neg_two_row_sum2));
    vout3x01234567 = _mm256_add_ps(vout3x01234567, _mm256_set1_ps(neg_two_row_sum3));
    vout4x01234567 = _mm256_add_ps(vout4x01234567, _mm256_set1_ps(neg_two_row_sum4));
    vout5x01234567 = _mm256_add_ps(vout5x01234567, _mm256_set1_ps(neg_two_row_sum5));
    // Add the right-hand zero points scaled by the left-hand zero points
    // and `original_kc` less the left-hand row sums.
    vout0x01234567 = _mm256_fmadd_ps(rh_zero_points_01234567, _mm256_set1_ps(kzp_scale0), vout0x01234567);
    vout1x01234567 = _mm256_fmadd_ps(rh_zero_points_01234567, _mm256_set1_ps(kzp_scale1), vout1x01234567);
    vout2x01234567 = _mm256_fmadd_ps(rh_zero_points_01234567, _mm256_set1_ps(kzp_scale2), vout2x01234567);
    vout3x01234567 = _mm256_fmadd_ps(rh_zero_points_01234567, _mm256_set1_ps(kzp_scale3), vout3x01234567);
    vout4x01234567 = _mm256_fmadd_ps(rh_zero_points_01234567, _mm256_set1_ps(kzp_scale4), vout4x01234567);
    vout5x01234567 = _mm256_fmadd_ps(rh_zero_points_01234567, _mm256_set1_ps(kzp_scale5), vout5x01234567);
    vout0x01234567 = _mm256_mul_ps(vout0x01234567, _mm256_set1_ps(quantization_params[0].inv_scale));
    vout1x01234567 = _mm256_mul_ps(vout1x01234567, _mm256_set1_ps(quantization_params[1].inv_scale));
    vout2x01234567 = _mm256_mul_ps(vout2x01234567, _mm256_set1_ps(quantization_params[2].inv_scale));
    vout3x01234567 = _mm256_mul_ps(vout3x01234567, _mm256_set1_ps(quantization_params[3].inv_scale));
    vout4x01234567 = _mm256_mul_ps(vout4x01234567, _mm256_set1_ps(quantization_params[4].inv_scale));
    vout5x01234567 = _mm256_mul_ps(vout5x01234567, _mm256_set1_ps(quantization_params[5].inv_scale));

    const __m256 vfilter_output_scale01234567 = _mm256_load_ps((const float*) w);
    const __m256 vbias01234567 = _mm256_load_ps((const float*) w + 8);
    w = (const float*) w + 16;

    vout0x01234567 = _mm256_fmadd_ps(vout0x01234567, vfilter_output_scale01234567, vbias01234567);
    vout1x01234567 = _mm256_fmadd_ps(vout1x01234567, vfilter_output_scale01234567, vbias01234567);
    vout2x01234567 = _mm256_fmadd_ps(vout2x01234567, vfilter_output_scale01234567, vbias01234567);
    vout3x01234567 = _mm256_fmadd_ps(vout3x01234567, vfilter_output_scale01234567, vbias01234567);
    vout4x01234567 = _mm256_fmadd_ps(vout4x01234567, vfilter_output_scale01234567, vbias01234567);
    vout5x01234567 = _mm256_fmadd_ps(vout5x01234567, vfilter_output_scale01234567, vbias01234567);

    vout0x01234567 = _mm256_max_ps(vout0x01234567, voutput_min);
    vout1x01234567 = _mm256_max_ps(vout1x01234567, voutput_min);
    vout2x01234567 = _mm256_max_ps(vout2x01234567, voutput_min);
    vout3x01234567 = _mm256_max_ps(vout3x01234567, voutput_min);
    vout4x01234567 = _mm256_max_ps(vout4x01234567, voutput_min);
    vout5x01234567 = _mm256_max_ps(vout5x01234567, voutput_min);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, voutput_max);
    vout1x01234567 = _mm256_min_ps(vout1x01234567, voutput_max);
    vout2x01234567 = _mm256_min_ps(vout2x01234567, voutput_max);
    vout3x01234567 = _mm256_min_ps(vout3x01234567, voutput_max);
    vout4x01234567 = _mm256_min_ps(vout4x01234567, voutput_max);
    vout5x01234567 = _mm256_min_ps(vout5x01234567, voutput_max);

    __m128i vfp16out0x01234567 = _mm256_cvtps_ph(vout0x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out1x01234567 = _mm256_cvtps_ph(vout1x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out2x01234567 = _mm256_cvtps_ph(vout2x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out3x01234567 = _mm256_cvtps_ph(vout3x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out4x01234567 = _mm256_cvtps_ph(vout4x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out5x01234567 = _mm256_cvtps_ph(vout5x01234567, _MM_FROUND_TO_NEAREST_INT);
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
      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c0, vfp16out0x01234567);
        c0 += 4;
        vfp16out0x01234567 = _mm_unpackhi_epi64(vfp16out0x01234567, vfp16out0x01234567);
        _mm_storel_epi64((__m128i*) c1, vfp16out1x01234567);
        c1 += 4;
        vfp16out1x01234567 = _mm_unpackhi_epi64(vfp16out1x01234567, vfp16out1x01234567);
        _mm_storel_epi64((__m128i*) c2, vfp16out2x01234567);
        c2 += 4;
        vfp16out2x01234567 = _mm_unpackhi_epi64(vfp16out2x01234567, vfp16out2x01234567);
        _mm_storel_epi64((__m128i*) c3, vfp16out3x01234567);
        c3 += 4;
        vfp16out3x01234567 = _mm_unpackhi_epi64(vfp16out3x01234567, vfp16out3x01234567);
        _mm_storel_epi64((__m128i*) c4, vfp16out4x01234567);
        c4 += 4;
        vfp16out4x01234567 = _mm_unpackhi_epi64(vfp16out4x01234567, vfp16out4x01234567);
        _mm_storel_epi64((__m128i*) c5, vfp16out5x01234567);
        c5 += 4;
        vfp16out5x01234567 = _mm_unpackhi_epi64(vfp16out5x01234567, vfp16out5x01234567);
      }
      if (nc & 2) {
        _mm_storeu_si32(c0, vfp16out0x01234567);
        c0 += 2;
        vfp16out0x01234567 = _mm_srli_epi64(vfp16out0x01234567, 32);
        _mm_storeu_si32(c1, vfp16out1x01234567);
        c1 += 2;
        vfp16out1x01234567 = _mm_srli_epi64(vfp16out1x01234567, 32);
        _mm_storeu_si32(c2, vfp16out2x01234567);
        c2 += 2;
        vfp16out2x01234567 = _mm_srli_epi64(vfp16out2x01234567, 32);
        _mm_storeu_si32(c3, vfp16out3x01234567);
        c3 += 2;
        vfp16out3x01234567 = _mm_srli_epi64(vfp16out3x01234567, 32);
        _mm_storeu_si32(c4, vfp16out4x01234567);
        c4 += 2;
        vfp16out4x01234567 = _mm_srli_epi64(vfp16out4x01234567, 32);
        _mm_storeu_si32(c5, vfp16out5x01234567);
        c5 += 2;
        vfp16out5x01234567 = _mm_srli_epi64(vfp16out5x01234567, 32);
      }
      if (nc & 1) {
        *c0 = (uint16_t) _mm_extract_epi16(vfp16out0x01234567, 0);
        *c1 = (uint16_t) _mm_extract_epi16(vfp16out1x01234567, 0);
        *c2 = (uint16_t) _mm_extract_epi16(vfp16out2x01234567, 0);
        *c3 = (uint16_t) _mm_extract_epi16(vfp16out3x01234567, 0);
        *c4 = (uint16_t) _mm_extract_epi16(vfp16out4x01234567, 0);
        *c5 = (uint16_t) _mm_extract_epi16(vfp16out5x01234567, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
