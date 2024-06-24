// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/sse41.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>

void xnn_qs8_rsum_ukernel__sse41_u32_acc2(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const union xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  __m128i vacc = _mm_setzero_si128();
  // 256 int8s may be summed into an int16 before overflowing.
  // There are 2 registers and each register has 8 lanes so batch size is 4096
  const __m128i vone = _mm_loadu_si128((const __m128i*) &params->ssse3.onemask_table[0]);
  while (batch >= 32) {
    __m128i vacc16_0 = _mm_setzero_si128();
    __m128i vacc16_1 = _mm_setzero_si128();
    for (int current_batch = min(batch, 4096); current_batch >= 32; current_batch -= 32) {
      const __m128i vt0 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt1 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;

      vacc16_0 = _mm_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt1);
    }
    vacc = _mm_add_epi32(vacc, _mm_add_epi32(_mm_cvtepi16_epi32(vacc16_0), _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_0, 8))));
    vacc = _mm_add_epi32(vacc, _mm_add_epi32(_mm_cvtepi16_epi32(vacc16_1), _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_1, 8))));
    batch = (batch >= 4096 ? (batch - 4096) : batch & 31);
  }
  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch < 32);

    __m128i vacc16 = _mm_setzero_si128();
    for (; batch >= 16; batch -= 16) {
      const __m128i vt = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16 = _mm_add_epi16(vacc16, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      const __m128i vonemask = _mm_loadu_si128((const __m128i*) &params->ssse3.onemask_table[16 - batch]);
      const __m128i vt = _mm_maddubs_epi16(vonemask, _mm_loadu_si128((const __m128i*) input));
      vacc16 = _mm_add_epi16(vacc16, vt);
    }
    vacc = _mm_add_epi32(vacc, _mm_add_epi32(_mm_cvtepi16_epi32(vacc16), _mm_cvtepi16_epi32(_mm_srli_si128(vacc16, 8))));
  }

  __m128i vacc_lo = _mm_hadd_epi32(vacc, vacc);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);

  *output += _mm_cvtsi128_si32(vacc_lo);
}
