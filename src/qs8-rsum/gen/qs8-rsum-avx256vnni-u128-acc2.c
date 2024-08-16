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


void xnn_qs8_rsum_ukernel__avx256vnni_u128_acc2(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const union xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);


  const __m256i vone = _mm256_set1_epi8(1);
  __m256i vacc0 = _mm256_setzero_si256();
  __m256i vacc1 = _mm256_setzero_si256();
  for (; batch >= 128; batch -= 128) {
    vacc0 = _mm256_dpbusd_epi32(vacc0, vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
    vacc1 = _mm256_dpbusd_epi32(vacc1, vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
    vacc0 = _mm256_dpbusd_epi32(vacc0, vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
    vacc1 = _mm256_dpbusd_epi32(vacc1, vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
  }
  if (XNN_UNLIKELY(batch != 0)) {
    for (; batch >= 32; batch -= 32) {
      vacc0 = _mm256_dpbusd_epi32(vacc0, vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
    }

    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 31);
      // Prepare mask for valid 8-bit elements (depends on batch).
      const __mmask32 vmask = _cvtu32_mask32((UINT32_C(1) << batch) - 1);
      vacc0 = _mm256_dpbusd_epi32(vacc0, vone, _mm256_maskz_loadu_epi8(vmask, (const __m256i*) input)); input += 32;
    }
  }
  vacc0 = _mm256_add_epi32(vacc0, vacc1);

  __m128i vacc_lo = _mm_add_epi32(_mm256_castsi256_si128(vacc0), _mm256_extractf128_si256(vacc0, 1));
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);

  *output += _mm_cvtsi128_si32(vacc_lo);
}
