// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/avx.c.in
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


void xnn_f32_qu8_vcvt_ukernel__avx_u16(
    size_t batch,
    const float* input,
    uint8_t* output,
    const struct xnn_f32_qu8_cvt_params* restrict params)
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
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m256 vx01234567 = _mm256_loadu_ps(input);
    __m256 vx89ABCDEF = _mm256_loadu_ps(input + 8);
    input += 16;

    vx01234567 = _mm256_mul_ps(vx01234567, vscale);
    vx89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vscale);

    vx01234567 = _mm256_min_ps(vx01234567, voverflow_max);
    vx89ABCDEF = _mm256_min_ps(vx89ABCDEF, voverflow_max);

    const __m256i vacc01234567 = _mm256_cvtps_epi32(vx01234567);
    const __m256i vacc89ABCDEF = _mm256_cvtps_epi32(vx89ABCDEF);

    __m128i vy01234567 = _mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extractf128_si256(vacc01234567, 1));
    __m128i vy89ABCDEF = _mm_packs_epi32(_mm256_castsi256_si128(vacc89ABCDEF), _mm256_extractf128_si256(vacc89ABCDEF, 1));

    vy01234567 = _mm_adds_epi16(vy01234567, voutput_zero_point);
    vy89ABCDEF = _mm_adds_epi16(vy89ABCDEF, voutput_zero_point);

    __m128i vy0123456789ABCDEF = _mm_packus_epi16(vy01234567, vy89ABCDEF);

    _mm_storeu_si128((__m128i*) output, vy0123456789ABCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voverflow_max);
    input += 8;

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extractf128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packus_epi16(vy, vy);

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

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extractf128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packus_epi16(vy, vy);

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
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
