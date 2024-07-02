// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx512skx.c.in
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

void xnn_qs8_rsum_ukernel__avx512skx_u128_acc2(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const union xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const __m512i vone = _mm512_set1_epi8(1);
  const __m512i vone_16 = _mm512_set1_epi16(1);
  __m512i vacc0 = _mm512_setzero_si512();
  __m512i vacc1 = _mm512_setzero_si512();

  // 256 int8s may be summed into an int16 before overflowing.
  // Each register has 32 lanes and there are 2 accumulators so batch size is 16384
  for (; batch >= 16384; batch -= 16384) {
    __m512i vacc16_0 = _mm512_setzero_si512();
    __m512i vacc16_1 = _mm512_setzero_si512();
    for (size_t current_batch = 16384; current_batch > 0; current_batch -= 128) {
      const __m512i vt0 = _mm512_maddubs_epi16(vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
      const __m512i vt1 = _mm512_maddubs_epi16(vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
      vacc16_0 = _mm512_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm512_add_epi16(vacc16_1, vt1);
    }
    vacc0 = _mm512_add_epi32(vacc0, _mm512_madd_epi16(vone_16, vacc16_0));
    vacc1 = _mm512_add_epi32(vacc1, _mm512_madd_epi16(vone_16, vacc16_1));
  }

  if (XNN_LIKELY(batch >= 128)) {
    assert(batch >= 1 && batch < 16384);
    __m512i vacc16_0 = _mm512_setzero_si512();
    __m512i vacc16_1 = _mm512_setzero_si512();
    for (; batch >= 128; batch -= 128) {
      const __m512i vt0 = _mm512_maddubs_epi16(vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
      const __m512i vt1 = _mm512_maddubs_epi16(vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
      vacc16_0 = _mm512_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm512_add_epi16(vacc16_1, vt1);
    }
    vacc0 = _mm512_add_epi32(vacc0, _mm512_madd_epi16(vone_16, vacc16_0));
    vacc1 = _mm512_add_epi32(vacc1, _mm512_madd_epi16(vone_16, vacc16_1));
  }
  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 8192);
    __m512i vacc16 = _mm512_setzero_si512();
    for (; batch >= 64; batch -= 64) {
      const __m512i vt = _mm512_maddubs_epi16(vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
      vacc16 = _mm512_add_epi16(vacc16, vt);
    }

    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 63);
      // Prepare mask for valid 8-bit elements (depends on batch).
      const __mmask64 vmask = _cvtu64_mask64((UINT64_C(1) << batch) - 1);
      const __m512i vt = _mm512_maddubs_epi16(vone, _mm512_maskz_loadu_epi8(vmask, input));
      vacc16 = _mm512_add_epi16(vacc16, vt);
    }
    vacc0 = _mm512_add_epi32(vacc0, _mm512_madd_epi16(vone_16, vacc16));
  }
  vacc0 = _mm512_add_epi32(vacc0, vacc1);

  int32_t res = _mm512_reduce_add_epi32(vacc0);

  *output += res;
}
