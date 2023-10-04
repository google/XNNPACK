// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/fill.h>
#include <xnnpack/unaligned.h>


void xnn_xx_fill_ukernel__sse2_u64(
    size_t rows,
    size_t channels,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern)
{
  assert(rows != 0);
  assert(channels != 0);

  const size_t output_increment = output_stride - channels;

  const __m128i vfill = _mm_shuffle_epi32(_mm_cvtsi32_si128(fill_pattern), _MM_SHUFFLE(0, 0, 0, 0));
  do {
    size_t c = channels;
    for (; c >= 64 * sizeof(uint8_t); c -= 64 * sizeof(uint8_t)) {
      _mm_storeu_si128((__m128i*) output, vfill);
      _mm_storeu_si128((__m128i*) output + 1, vfill);
      _mm_storeu_si128((__m128i*) output + 2, vfill);
      _mm_storeu_si128((__m128i*) output + 3, vfill);
      output = ((uint8_t*) output + 64);
    }
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      _mm_storeu_si128((__m128i*) output, vfill);
      output = ((uint8_t*) output + 16);
    }
    if XNN_UNLIKELY(c != 0) {
      if XNN_LIKELY(c & (8 * sizeof(uint8_t))) {
        _mm_storel_epi64(output, vfill);
        output = ((uint8_t*) output + 8);
      }
      if XNN_LIKELY(c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, fill_pattern);
        output = ((uint8_t*) output + 4);
      }
      uint32_t vfill_subpattern = fill_pattern;
      if XNN_LIKELY(c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_subpattern);
        vfill_subpattern >>= 16;
        output = ((uint8_t*) output + 2);
      }
      if XNN_LIKELY(c & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = ((uint8_t*) output + 1);
      }
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
