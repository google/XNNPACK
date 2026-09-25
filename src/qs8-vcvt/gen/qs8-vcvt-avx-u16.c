// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/sse4.c.in
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


void xnn_qs8_vcvt_ukernel__avx_u16(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(
      (int32_t) (((uint32_t) (int32_t) params->scalar.output_zero_point) << 16) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point +
      INT32_C(0x8000));
  const __m128i vmultiplier = _mm_set1_epi32(params->scalar.multiplier);
  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(vmultiplier);
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m128i vacc0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacc1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 8)));
    input += 16;

    __m128i vacc_lo0 = _mm_cvtepi16_epi32(vacc0);
    __m128i vacc_hi0 = _mm_cvtepi16_epi32(_mm_srli_si128(vacc0, 8));
    __m128i vacc_lo1 = _mm_cvtepi16_epi32(vacc1);
    __m128i vacc_hi1 = _mm_cvtepi16_epi32(_mm_srli_si128(vacc1, 8));

    vacc_lo0 = _mm_add_epi32(_mm_mullo_epi32(vacc_lo0, vmultiplier), vbias);
    vacc_hi0 = _mm_add_epi32(_mm_mullo_epi32(vacc_hi0, vmultiplier), vbias);
    vacc_lo1 = _mm_add_epi32(_mm_mullo_epi32(vacc_lo1, vmultiplier), vbias);
    vacc_hi1 = _mm_add_epi32(_mm_mullo_epi32(vacc_hi1, vmultiplier), vbias);

    vacc_lo0 = _mm_srai_epi32(vacc_lo0, 16);
    vacc_hi0 = _mm_srai_epi32(vacc_hi0, 16);
    vacc_lo1 = _mm_srai_epi32(vacc_lo1, 16);
    vacc_hi1 = _mm_srai_epi32(vacc_hi1, 16);

    vacc0 = _mm_packs_epi32(vacc_lo0, vacc_hi0);
    vacc1 = _mm_packs_epi32(vacc_lo1, vacc_hi1);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);

    _mm_storeu_si128((__m128i*) output, vy0);
    output += 16;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i vacc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacc_lo = _mm_cvtepi16_epi32(vacc);
    __m128i vacc_hi = _mm_cvtepi16_epi32(_mm_srli_si128(vacc, 8));
    vacc_lo = _mm_add_epi32(_mm_mullo_epi32(vacc_lo, vmultiplier), vbias);
    vacc_hi = _mm_add_epi32(_mm_mullo_epi32(vacc_hi, vmultiplier), vbias);
    vacc_lo = _mm_srai_epi32(vacc_lo, 16);
    vacc_hi = _mm_srai_epi32(vacc_hi, 16);
    vacc = _mm_packs_epi32(vacc_lo, vacc_hi);
    input += 8;

    const __m128i vy = _mm_packs_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m128i vacc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacc_lo = _mm_cvtepi16_epi32(vacc);
    __m128i vacc_hi = _mm_cvtepi16_epi32(_mm_srli_si128(vacc, 8));
    vacc_lo = _mm_add_epi32(_mm_mullo_epi32(vacc_lo, vmultiplier), vbias);
    vacc_hi = _mm_add_epi32(_mm_mullo_epi32(vacc_hi, vmultiplier), vbias);
    vacc_lo = _mm_srai_epi32(vacc_lo, 16);
    vacc_hi = _mm_srai_epi32(vacc_hi, 16);
    vacc = _mm_packs_epi32(vacc_lo, vacc_hi);

    __m128i vy = _mm_packs_epi16(vacc, vacc);
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
