// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_qs8_vcvt_ukernel__avx2_u64(
    size_t batch,
    const float* input,
    int8_t* output,
    const struct xnn_f32_qs8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  // *cvtps_epi32 maps all floats out of bounds of int to INT_MIN, so we need to clamp at the max to avoid overflow.
  // INT16_MAX is exactly representable as a float, and is plenty large (this clamp is applied after scaling).
  const __m256 voverflow_max = _mm256_set1_ps((float) INT16_MAX);
  XNN_FORCE_REALIZATION(voverflow_max);

  const __m256 vscale = _mm256_set1_ps(params->scalar.scale);
  const __m256i voutput_zero_point = _mm256_set1_epi16(params->scalar.output_zero_point);
  XNN_ALIGN(32) static const uint32_t shuffle_mask[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  const __m256i vshuffle_mask = _mm256_load_si256((const __m256i*) shuffle_mask);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    __m256 vx01 = _mm256_loadu_ps(input);
    __m256 vx23 = _mm256_loadu_ps(input + 8);
    __m256 vx45 = _mm256_loadu_ps(input + 16);
    __m256 vx67 = _mm256_loadu_ps(input + 24);
    __m256 vx89 = _mm256_loadu_ps(input + 32);
    __m256 vxAB = _mm256_loadu_ps(input + 40);
    __m256 vxCD = _mm256_loadu_ps(input + 48);
    __m256 vxEF = _mm256_loadu_ps(input + 56);
    input += 64;

    vx01 = _mm256_mul_ps(vx01, vscale);
    vx23 = _mm256_mul_ps(vx23, vscale);
    vx45 = _mm256_mul_ps(vx45, vscale);
    vx67 = _mm256_mul_ps(vx67, vscale);
    vx89 = _mm256_mul_ps(vx89, vscale);
    vxAB = _mm256_mul_ps(vxAB, vscale);
    vxCD = _mm256_mul_ps(vxCD, vscale);
    vxEF = _mm256_mul_ps(vxEF, vscale);

    vx01 = _mm256_min_ps(vx01, voverflow_max);
    vx23 = _mm256_min_ps(vx23, voverflow_max);
    vx45 = _mm256_min_ps(vx45, voverflow_max);
    vx67 = _mm256_min_ps(vx67, voverflow_max);
    vx89 = _mm256_min_ps(vx89, voverflow_max);
    vxAB = _mm256_min_ps(vxAB, voverflow_max);
    vxCD = _mm256_min_ps(vxCD, voverflow_max);
    vxEF = _mm256_min_ps(vxEF, voverflow_max);

    const __m256i vacc01 = _mm256_cvtps_epi32(vx01);
    const __m256i vacc23 = _mm256_cvtps_epi32(vx23);
    const __m256i vacc45 = _mm256_cvtps_epi32(vx45);
    const __m256i vacc67 = _mm256_cvtps_epi32(vx67);
    const __m256i vacc89 = _mm256_cvtps_epi32(vx89);
    const __m256i vaccAB = _mm256_cvtps_epi32(vxAB);
    const __m256i vaccCD = _mm256_cvtps_epi32(vxCD);
    const __m256i vaccEF = _mm256_cvtps_epi32(vxEF);

    __m256i vacc0213 = _mm256_packs_epi32(vacc01, vacc23);
    __m256i vacc4657 = _mm256_packs_epi32(vacc45, vacc67);
    __m256i vacc8A9B = _mm256_packs_epi32(vacc89, vaccAB);
    __m256i vaccCEDF = _mm256_packs_epi32(vaccCD, vaccEF);

    vacc0213 = _mm256_adds_epi16(vacc0213, voutput_zero_point);
    vacc4657 = _mm256_adds_epi16(vacc4657, voutput_zero_point);
    vacc8A9B = _mm256_adds_epi16(vacc8A9B, voutput_zero_point);
    vaccCEDF = _mm256_adds_epi16(vaccCEDF, voutput_zero_point);

    const __m256i vy02461357 = _mm256_packs_epi16(vacc0213, vacc4657);
    const __m256i vy8ACE9BDF = _mm256_packs_epi16(vacc8A9B, vaccCEDF);

    __m256i vy01234567 = _mm256_permutevar8x32_epi32(vy02461357, vshuffle_mask);
    __m256i vy89ABCDEF = _mm256_permutevar8x32_epi32(vy8ACE9BDF, vshuffle_mask);

    _mm256_storeu_si256((__m256i*) output, vy01234567);
    _mm256_storeu_si256((__m256i*) (output + 32), vy89ABCDEF);
    output += 64;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voverflow_max);
    input += 8;

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, _mm256_castsi256_si128(voutput_zero_point));
    vy = _mm_packs_epi16(vy, vy);

    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    __m256 vx = _mm256_maskload_ps(input, vmask);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voverflow_max);

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, _mm256_castsi256_si128(voutput_zero_point));
    vy = _mm_packs_epi16(vy, vy);

    if (batch & (4 * sizeof(float))) {
      _mm_storeu_si32(output, vy);
      output += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storeu_si16(output, vy);
      output += 2;
      vy = _mm_srli_epi32(vy, 16);
    }
    if (batch & (1 * sizeof(float))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
