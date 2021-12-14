// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/vunary.h>


void xnn_s8_vclamp_ukernel__sse2_x64(
    size_t n,
    const int8_t* x,
    int8_t* y,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);

  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse2.bias);
  const __m128i voutput_max_with_bias = _mm_load_si128((const __m128i*) params->sse2.max_with_bias);
  const __m128i voutput_min_with_bias = _mm_load_si128((const __m128i*) params->sse2.min_with_bias);
  for (; n >= 64; n -= 64) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) x);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) x + 1);
    __m128i vacc2 = _mm_loadu_si128((const __m128i*) x + 2);
    __m128i vacc3 = _mm_loadu_si128((const __m128i*) x + 3);
    x += 64;

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

    _mm_storeu_si128((__m128i*) y, vacc0);
    _mm_storeu_si128((__m128i*) y + 1, vacc1);
    _mm_storeu_si128((__m128i*) y + 2, vacc2);
    _mm_storeu_si128((__m128i*) y + 3, vacc3);
    y += 64;
  }
  for (; n >= 16; n -= 16) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) x);
    x += 16;

    vacc = _mm_xor_si128(vacc, vbias);
    vacc = _mm_min_epu8(vacc, voutput_max_with_bias);
    vacc = _mm_max_epu8(vacc, voutput_min_with_bias);
    vacc = _mm_xor_si128(vacc, vbias);

    _mm_storeu_si128((__m128i*) y, vacc);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) x);

    vacc = _mm_xor_si128(vacc, vbias);
    vacc = _mm_min_epu8(vacc, voutput_max_with_bias);
    vacc = _mm_max_epu8(vacc, voutput_min_with_bias);
    vacc = _mm_xor_si128(vacc, vbias);

    if (n & 8) {
      _mm_storel_epi64((__m128i*) y, vacc);
      y += 8;
      vacc = _mm_unpackhi_epi64(vacc, vacc);
    }
    if (n & 4) {
      *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vacc);
      y += 4;
      vacc = _mm_srli_epi64(vacc, 32);
    }
    if (n & 2) {
      *((uint16_t*) y) = (uint16_t) _mm_cvtsi128_si32(vacc);
      y += 2;
      vacc = _mm_srli_epi32(vacc, 16);
    }
    if (n & 1) {
      *y = (int8_t) _mm_cvtsi128_si32(vacc);
    }
  }
}
