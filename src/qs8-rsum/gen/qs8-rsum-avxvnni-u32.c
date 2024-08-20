// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_qs8_rsum_ukernel__avxvnni_u32(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const union xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_ALIGN(32) static const int8_t onemask_table[64] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m256i vone = _mm256_set1_epi8(1);
  __m256i vacc0 = _mm256_setzero_si256();
  if (XNN_UNLIKELY(batch != 0)) {
    for (; batch >= 32; batch -= 32) {
      vacc0 = _mm256_dpbusd_epi32(vacc0, vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
    }

    // Remainder is between 17 and 31 bytes, so process 32 bytes (overread of up to 15)
    if (XNN_UNLIKELY(batch >= 17)) {
      assert(batch >= 17 && batch <= 31);
      const __m256i vonemask = _mm256_loadu_si256((const __m256i*) &onemask_table[32 - batch]);
      vacc0 = _mm256_dpbusd_epi32(vacc0, vonemask, _mm256_loadu_si256((const __m256i*) input));
    // Remainder is between 1 and 16 bytes, so process 16 bytes (overread of up to 15)
    } else if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 16);
      const __m256i vonemask = _mm256_loadu_si256((const __m256i*) &onemask_table[32 - batch]);
      vacc0 = _mm256_dpbusd_epi32(vacc0, vonemask, _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) input)));
    }
  }

  __m128i vacc_lo = _mm_add_epi32(_mm256_castsi256_si128(vacc0), _mm256_extractf128_si256(vacc0, 1));
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);

  *output += _mm_cvtsi128_si32(vacc_lo);
}
