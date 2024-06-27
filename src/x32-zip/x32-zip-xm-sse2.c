// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_xm_ukernel__sse2(
    size_t n,
    size_t m,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);
  assert(m >= 4);

  const uint32_t* w = input;
  const size_t group_increment = m * 4;
  const size_t input_increment = n * 3;
  const size_t output_increment = 16 - m * n;
  const uint32_t* last_input = (const uint32_t*) ((uintptr_t) input + n * (m - 1));
  uint32_t* last_output = (uint32_t*) ((uintptr_t) output + (m * 4 - 16));

  for (size_t i = 0; i < m; i += 4) {
    w = (const uint32_t*) ((uintptr_t) w + input_increment);
    if (w >= last_input) {
      w = last_input;
    }
    const uint32_t* z = (const uint32_t*) ((uintptr_t) w - n);
    const uint32_t* y = (const uint32_t*) ((uintptr_t) z - n);
    const uint32_t* x = (const uint32_t*) ((uintptr_t) y - n);

    size_t k = n;
    while (k >= 16) {
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

      _mm_storeu_si128((__m128i*) output, vxyzw0);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      _mm_storeu_si128((__m128i*) output, vxyzw1);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      _mm_storeu_si128((__m128i*) output, vxyzw2);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      _mm_storeu_si128((__m128i*) output, vxyzw3);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      k -= 16;
    }
    if XNN_UNLIKELY(k != 0) {
      if (k & 8) {
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

        _mm_storeu_si128((__m128i*) output, vxyzw_lo);
        output = (uint32_t*) ((uintptr_t) output + group_increment);

        _mm_storeu_si128((__m128i*) output, vxyzw_hi);
        output = (uint32_t*) ((uintptr_t) output + group_increment);
      }
      if (k & 4) {
        const uint32_t vx = *x;
        const uint32_t vy = *y;
        const uint32_t vz = *z;
        const uint32_t vw = *w++;

        output[0] = vx;
        output[1] = vy;
        output[2] = vz;
        output[3] = vw;
        output = (uint32_t*) ((uintptr_t) output + group_increment);
      }
    }
    output = (uint32_t*) ((uintptr_t) output + output_increment);
    if (output > last_output) {
      output = last_output;
    }
  }
}
