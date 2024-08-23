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

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_f32_qs8_vcvt_ukernel__avx_u24(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vscale = _mm256_set1_ps(params->scalar.scale);
  const __m256 voutput_max_less_zero_point = _mm256_set1_ps((float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point));
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    __m256 vx01234567 = _mm256_loadu_ps(input);
    __m256 vx89ABCDEF = _mm256_loadu_ps(input + 8);
    __m256 vxGHIJKLMN = _mm256_loadu_ps(input + 16);
    input += 24;

    vx01234567 = _mm256_mul_ps(vx01234567, vscale);
    vx89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vscale);
    vxGHIJKLMN = _mm256_mul_ps(vxGHIJKLMN, vscale);

    vx01234567 = _mm256_min_ps(vx01234567, voutput_max_less_zero_point);
    vx89ABCDEF = _mm256_min_ps(vx89ABCDEF, voutput_max_less_zero_point);
    vxGHIJKLMN = _mm256_min_ps(vxGHIJKLMN, voutput_max_less_zero_point);

    const __m256i vacc01234567 = _mm256_cvtps_epi32(vx01234567);
    const __m256i vacc89ABCDEF = _mm256_cvtps_epi32(vx89ABCDEF);
    const __m256i vaccGHIJKLMN = _mm256_cvtps_epi32(vxGHIJKLMN);

    __m128i vy01234567 = _mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extractf128_si256(vacc01234567, 1));
    __m128i vy89ABCDEF = _mm_packs_epi32(_mm256_castsi256_si128(vacc89ABCDEF), _mm256_extractf128_si256(vacc89ABCDEF, 1));
    __m128i vyGHIJKLMN = _mm_packs_epi32(_mm256_castsi256_si128(vaccGHIJKLMN), _mm256_extractf128_si256(vaccGHIJKLMN, 1));

    vy01234567 = _mm_adds_epi16(vy01234567, voutput_zero_point);
    vy89ABCDEF = _mm_adds_epi16(vy89ABCDEF, voutput_zero_point);
    vyGHIJKLMN = _mm_adds_epi16(vyGHIJKLMN, voutput_zero_point);

    __m128i vy0123456789ABCDEF = _mm_packs_epi16(vy01234567, vy89ABCDEF);
    vyGHIJKLMN = _mm_packs_epi16(vyGHIJKLMN, vyGHIJKLMN);

    vy0123456789ABCDEF = _mm_max_epi8(vy0123456789ABCDEF, voutput_min);
    vyGHIJKLMN = _mm_max_epi8(vyGHIJKLMN, voutput_min);

    _mm_storeu_si128((__m128i*) output, vy0123456789ABCDEF);
    _mm_storel_epi64((__m128i*) (output + 16), vyGHIJKLMN);
    output += 24;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);
    input += 8;

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extractf128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, voutput_min);

    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    __m256 vx = _mm256_maskload_ps(input, vmask);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extractf128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, voutput_min);

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
