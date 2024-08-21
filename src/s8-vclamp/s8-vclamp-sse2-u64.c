// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vunary.h"


void xnn_s8_vclamp_ukernel__sse2_u64(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi8(UINT8_C(0x80));
  const __m128i voutput_max_with_bias = _mm_set1_epi8(UINT8_C(0x80) ^ params->scalar.max);
  const __m128i voutput_min_with_bias = _mm_set1_epi8(UINT8_C(0x80) ^ params->scalar.min);
  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(voutput_max_with_bias);
  XNN_FORCE_REALIZATION(voutput_min_with_bias);

  for (; batch >= 64; batch -= 64) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) input);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) input + 1);
    __m128i vacc2 = _mm_loadu_si128((const __m128i*) input + 2);
    __m128i vacc3 = _mm_loadu_si128((const __m128i*) input + 3);
    input += 64;

    vacc0 = _mm_xor_si128(vacc0, vbias);
    vacc1 = _mm_xor_si128(vacc1, vbias);
    vacc2 = _mm_xor_si128(vacc2, vbias);
    vacc3 = _mm_xor_si128(vacc3, vbias);

    vacc0 = _mm_max_epu8(vacc0, voutput_min_with_bias);
    vacc1 = _mm_max_epu8(vacc1, voutput_min_with_bias);
    vacc2 = _mm_max_epu8(vacc2, voutput_min_with_bias);
    vacc3 = _mm_max_epu8(vacc3, voutput_min_with_bias);

    vacc0 = _mm_min_epu8(vacc0, voutput_max_with_bias);
    vacc1 = _mm_min_epu8(vacc1, voutput_max_with_bias);
    vacc2 = _mm_min_epu8(vacc2, voutput_max_with_bias);
    vacc3 = _mm_min_epu8(vacc3, voutput_max_with_bias);

    vacc0 = _mm_xor_si128(vacc0, vbias);
    vacc1 = _mm_xor_si128(vacc1, vbias);
    vacc2 = _mm_xor_si128(vacc2, vbias);
    vacc3 = _mm_xor_si128(vacc3, vbias);

    _mm_storeu_si128((__m128i*) output, vacc0);
    _mm_storeu_si128((__m128i*) output + 1, vacc1);
    _mm_storeu_si128((__m128i*) output + 2, vacc2);
    _mm_storeu_si128((__m128i*) output + 3, vacc3);
    output += 64;
  }
  for (; batch >= 16; batch -= 16) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    vacc = _mm_xor_si128(vacc, vbias);
    vacc = _mm_min_epu8(vacc, voutput_max_with_bias);
    vacc = _mm_max_epu8(vacc, voutput_min_with_bias);
    vacc = _mm_xor_si128(vacc, vbias);

    _mm_storeu_si128((__m128i*) output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);

    vacc = _mm_xor_si128(vacc, vbias);
    vacc = _mm_min_epu8(vacc, voutput_max_with_bias);
    vacc = _mm_max_epu8(vacc, voutput_min_with_bias);
    vacc = _mm_xor_si128(vacc, vbias);

    if (batch & 8) {
      _mm_storel_epi64((__m128i*) output, vacc);
      output += 8;
      vacc = _mm_unpackhi_epi64(vacc, vacc);
    }
    if (batch & 4) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vacc));
      output += 4;
      vacc = _mm_srli_epi64(vacc, 32);
    }
    if (batch & 2) {
      unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vacc));
      output += 2;
      vacc = _mm_srli_epi32(vacc, 16);
    }
    if (batch & 1) {
      *output = (int8_t) _mm_cvtsi128_si32(vacc);
    }
  }
}
