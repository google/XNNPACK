// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/vunary.h>


void xnn_u8_vclamp_ukernel__sse2_x64(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);

  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.max);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.min);
  for (; n >= 64; n -= 64) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) x);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) x + 1);
    __m128i vacc2 = _mm_loadu_si128((const __m128i*) x + 2);
    __m128i vacc3 = _mm_loadu_si128((const __m128i*) x + 3);
    x += 64;

    vacc0 = _mm_max_epu8(vacc0, voutput_min);
    vacc1 = _mm_max_epu8(vacc1, voutput_min);
    vacc2 = _mm_max_epu8(vacc2, voutput_min);
    vacc3 = _mm_max_epu8(vacc3, voutput_min);

    vacc0 = _mm_min_epu8(vacc0, voutput_max);
    vacc1 = _mm_min_epu8(vacc1, voutput_max);
    vacc2 = _mm_min_epu8(vacc2, voutput_max);
    vacc3 = _mm_min_epu8(vacc3, voutput_max);

    _mm_storeu_si128((__m128i*) y, vacc0);
    _mm_storeu_si128((__m128i*) y + 1, vacc1);
    _mm_storeu_si128((__m128i*) y + 2, vacc2);
    _mm_storeu_si128((__m128i*) y + 3, vacc3);
    y += 64;
  }
  for (; n >= 16; n -= 16) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) x);
    x += 16;

    vacc = _mm_min_epu8(vacc, voutput_max);
    vacc = _mm_max_epu8(vacc, voutput_min);

    _mm_storeu_si128((__m128i*) y, vacc);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) x);

    vacc = _mm_min_epu8(vacc, voutput_max);
    vacc = _mm_max_epu8(vacc, voutput_min);

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
      *y = (uint8_t) _mm_cvtsi128_si32(vacc);
    }
  }
}
