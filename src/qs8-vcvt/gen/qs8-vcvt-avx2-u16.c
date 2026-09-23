// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_vcvt_ukernel__avx2_u16(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vbias = _mm256_set1_epi32(
      (int32_t) (((uint32_t) (int32_t) params->scalar.output_zero_point) << 16) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point +
      INT32_C(0x8000));
  const __m256i vmultiplier = _mm256_set1_epi32(params->scalar.multiplier);
  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(vmultiplier);
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    __m256i vacc32_lo = _mm256_cvtepi8_epi32(vx);
    __m256i vacc32_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(vx, 8));
    vacc32_lo = _mm256_add_epi32(_mm256_mullo_epi32(vacc32_lo, vmultiplier), vbias);
    vacc32_hi = _mm256_add_epi32(_mm256_mullo_epi32(vacc32_hi, vmultiplier), vbias);
    vacc32_lo = _mm256_srai_epi32(vacc32_lo, 16);
    vacc32_hi = _mm256_srai_epi32(vacc32_hi, 16);
    __m256i vacc = _mm256_packs_epi32(vacc32_lo, vacc32_hi);
    vacc = _mm256_permute4x64_epi64(vacc, _MM_SHUFFLE(3, 1, 2, 0));
    input += 16;

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    const __m128i vy = _mm_packs_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    __m256i vacc32_lo = _mm256_cvtepi8_epi32(vx);
    __m256i vacc32_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(vx, 8));
    vacc32_lo = _mm256_add_epi32(_mm256_mullo_epi32(vacc32_lo, vmultiplier), vbias);
    vacc32_hi = _mm256_add_epi32(_mm256_mullo_epi32(vacc32_hi, vmultiplier), vbias);
    vacc32_lo = _mm256_srai_epi32(vacc32_lo, 16);
    vacc32_hi = _mm256_srai_epi32(vacc32_hi, 16);
    __m256i vacc = _mm256_packs_epi32(vacc32_lo, vacc32_hi);
    vacc = _mm256_permute4x64_epi64(vacc, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    __m128i vy = _mm_packs_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    if (batch & (8 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
