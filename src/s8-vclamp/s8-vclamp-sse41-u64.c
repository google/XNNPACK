// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/unaligned.h>
#include <xnnpack/vunary.h>


void xnn_s8_vclamp_ukernel__sse41_u64(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4.max);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4.min);
  for (; batch >= 64; batch -= 64) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) input);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) input + 1);
    __m128i vacc2 = _mm_loadu_si128((const __m128i*) input + 2);
    __m128i vacc3 = _mm_loadu_si128((const __m128i*) input + 3);
    input += 64;

    vacc0 = _mm_max_epi8(vacc0, voutput_min);
    vacc1 = _mm_max_epi8(vacc1, voutput_min);
    vacc2 = _mm_max_epi8(vacc2, voutput_min);
    vacc3 = _mm_max_epi8(vacc3, voutput_min);

    vacc0 = _mm_min_epi8(vacc0, voutput_max);
    vacc1 = _mm_min_epi8(vacc1, voutput_max);
    vacc2 = _mm_min_epi8(vacc2, voutput_max);
    vacc3 = _mm_min_epi8(vacc3, voutput_max);

    _mm_storeu_si128((__m128i*) output, vacc0);
    _mm_storeu_si128((__m128i*) output + 1, vacc1);
    _mm_storeu_si128((__m128i*) output + 2, vacc2);
    _mm_storeu_si128((__m128i*) output + 3, vacc3);
    output += 64;
  }
  for (; batch >= 16; batch -= 16) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    vacc = _mm_min_epi8(vacc, voutput_max);
    vacc = _mm_max_epi8(vacc, voutput_min);

    _mm_storeu_si128((__m128i*) output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);

    vacc = _mm_min_epi8(vacc, voutput_max);
    vacc = _mm_max_epi8(vacc, voutput_min);

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
