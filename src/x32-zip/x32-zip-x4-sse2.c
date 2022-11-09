// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/zip.h>


void xnn_x32_zip_x4_ukernel__sse2(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  const uint32_t* z = (const uint32_t*) ((uintptr_t) y + n);
  const uint32_t* w = (const uint32_t*) ((uintptr_t) z + n);
  uint32_t* o = output;

  while (n >= 16) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) x);
    x += 4;
    const __m128i vy = _mm_loadu_si128((const __m128i*) y);
    y += 4;
    const __m128i vz = _mm_loadu_si128((const __m128i*) z);
    z += 4;
    const __m128i vw = _mm_loadu_si128((const __m128i*) w);
    w += 4;

    const __m128i vxy_lo = _mm_unpacklo_epi32(vx, vy);
    const __m128i vxy_hi = _mm_unpackhi_epi32(vx, vy);
    const __m128i vzw_lo = _mm_unpacklo_epi32(vz, vw);
    const __m128i vzw_hi = _mm_unpackhi_epi32(vz, vw);

    const __m128i vxyzw0 = _mm_unpacklo_epi64(vxy_lo, vzw_lo);
    const __m128i vxyzw1 = _mm_unpackhi_epi64(vxy_lo, vzw_lo);
    const __m128i vxyzw2 = _mm_unpacklo_epi64(vxy_hi, vzw_hi);
    const __m128i vxyzw3 = _mm_unpackhi_epi64(vxy_hi, vzw_hi);

    _mm_storeu_si128((__m128i*) o, vxyzw0);
    _mm_storeu_si128((__m128i*) (o + 4), vxyzw1);
    _mm_storeu_si128((__m128i*) (o + 8), vxyzw2);
    _mm_storeu_si128((__m128i*) (o + 12), vxyzw3);
    o += 16;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      const __m128i vx = _mm_loadl_epi64((const __m128i*) x);
      x += 2;
      const __m128i vy = _mm_loadl_epi64((const __m128i*) y);
      y += 2;
      const __m128i vz = _mm_loadl_epi64((const __m128i*) z);
      z += 2;
      const __m128i vw = _mm_loadl_epi64((const __m128i*) w);
      w += 2;

      const __m128i vxy = _mm_unpacklo_epi32(vx, vy);
      const __m128i vzw = _mm_unpacklo_epi32(vz, vw);

      const __m128i vxyzw_lo = _mm_unpacklo_epi64(vxy, vzw);
      const __m128i vxyzw_hi = _mm_unpackhi_epi64(vxy, vzw);

      _mm_storeu_si128((__m128i*) o, vxyzw_lo);
      _mm_storeu_si128((__m128i*) (o + 4), vxyzw_hi);
      o += 8;
    }
    if (n & 4) {
      const uint32_t vx = *x;
      const uint32_t vy = *y;
      const uint32_t vz = *z;
      const uint32_t vw = *w;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o[3] = vw;
    }
  }
}
