// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_x2_ukernel__sse2(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  uint32_t* o = output;

  while (n >= 16) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) x);
    x += 4;
    const __m128i vy = _mm_loadu_si128((const __m128i*) y);
    y += 4;
    const __m128i vxy_lo = _mm_unpacklo_epi32(vx, vy);
    const __m128i vxy_hi = _mm_unpackhi_epi32(vx, vy);
    _mm_storeu_si128((__m128i*) o, vxy_lo);
    _mm_storeu_si128((__m128i*) (o + 4), vxy_hi);
    o += 8;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      const __m128i vx = _mm_loadl_epi64((const __m128i*) x);
      x += 2;
      const __m128i vy = _mm_loadl_epi64((const __m128i*) y);
      y += 2;
      const __m128i vxy = _mm_unpacklo_epi32(vx, vy);
      _mm_storeu_si128((__m128i*) o, vxy);
      o += 4;
    }
    if (n & 4) {
      const uint32_t vx = *x;
      const uint32_t vy = *y;
      o[0] = vx;
      o[1] = vy;
    }
  }
}
