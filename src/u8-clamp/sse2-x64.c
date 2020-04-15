// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/clamp.h>


void xnn_u8_clamp_ukernel__sse2_x64(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);

  const __m128i voutput_max = _mm_load_si128((const __m128i*) &params->sse2.max);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) &params->sse2.min);
  for (; n >= 64; n -= 64) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) x);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) x + 1);
    const __m128i vx2 = _mm_loadu_si128((const __m128i*) x + 2);
    const __m128i vx3 = _mm_loadu_si128((const __m128i*) x + 3);
    x += 64;

    const __m128i vy0 = _mm_min_epu8(_mm_max_epu8(vx0, voutput_min), voutput_max);
    const __m128i vy1 = _mm_min_epu8(_mm_max_epu8(vx1, voutput_min), voutput_max);
    const __m128i vy2 = _mm_min_epu8(_mm_max_epu8(vx2, voutput_min), voutput_max);
    const __m128i vy3 = _mm_min_epu8(_mm_max_epu8(vx3, voutput_min), voutput_max);

    _mm_storeu_si128((__m128i*) y, vy0);
    _mm_storeu_si128((__m128i*) y + 1, vy1);
    _mm_storeu_si128((__m128i*) y + 2, vy2);
    _mm_storeu_si128((__m128i*) y + 3, vy3);
    y += 64;
  }
  for (; n >= 8; n -= 8) {
    __m128i vout = _mm_loadl_epi64((const __m128i*) x);
    x += 8;
    vout = _mm_min_epu8(vout, voutput_max);
    vout = _mm_max_epu8(vout, voutput_min);
    _mm_storel_epi64((__m128i*) y, vout);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    __m128i vout = _mm_loadl_epi64((const __m128i*) x);
    vout = _mm_min_epu8(vout, voutput_max);
    vout = _mm_max_epu8(vout, voutput_min);
    if (n & 4) {
      *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vout);
      y += 4;
      vout = _mm_srli_epi64(vout, 32);
    }
    if (n & 2) {
      *((uint16_t*) y) = (uint16_t) _mm_extract_epi16(vout, 0);
      y += 2;
      vout = _mm_srli_epi32(vout, 16);
    }
    if (n & 1) {
      *((uint8_t*) y) = (uint8_t) _mm_cvtsi128_si32(vout);
    }
  }
}
