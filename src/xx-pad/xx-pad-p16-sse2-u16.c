// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/pad.h"
#include "xnnpack/unaligned.h"


void xnn_xx_pad_ukernel_p16__sse2_u16(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern) XNN_OOB_READS
{
  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const __m128i vfill_pattern = _mm_shuffle_epi32(_mm_cvtsi32_si128((int) fill_pattern), _MM_SHUFFLE(0, 0, 0, 0));
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      for (; l >= 16 * sizeof(uint8_t); l -= 16 * sizeof(uint8_t)) {
        _mm_storeu_si128((__m128i*) output, vfill_pattern);
        output = (uint8_t*) output + 16;
      }
      if (l & (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vfill_pattern);
        output = (uint8_t*) output + 8;
      }
      uint32_t vfill_subpattern = fill_pattern;
      if (l & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, vfill_subpattern);
        output = (uint8_t*) output + 4;
      }
      if (l & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, vfill_subpattern);
        vfill_subpattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if (l & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = (uint8_t*) output + 1;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      const __m128i vdata = _mm_loadu_si128((const __m128i*) input);
      input = (const uint8_t*) input + 16;

      _mm_storeu_si128((__m128i*) output, vdata);
      output = (uint8_t*) output + 16;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128i vdata = _mm_loadu_si128((const __m128i*) input);
      input = (const void*) ((uintptr_t) input + c);
      if (c & (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vdata);
        vdata = _mm_unpackhi_epi64(vdata, vdata);
        output = (uint8_t*) output + 8;
      }
      if (c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vdata));
        vdata = _mm_srli_epi64(vdata, 32);
        output = (uint8_t*) output + 4;
      }
      uint32_t vsubdata = (uint32_t) _mm_cvtsi128_si32(vdata);
      if (c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vsubdata);
        vsubdata >>= 16;
        output = (uint8_t*) output + 2;
      }
      if (c & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vsubdata;
        output = (uint8_t*) output + 1;
      }
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      for (; r >= 16 * sizeof(uint8_t); r -= 16 * sizeof(uint8_t)) {
        _mm_storeu_si128((__m128i*) output, vfill_pattern);
        output = (uint8_t*) output + 16;
      }
      if (r & (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vfill_pattern);
        output = (uint8_t*) output + 8;
      }
      uint32_t vfill_subpattern = fill_pattern;
      if (r & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, vfill_subpattern);
        output = (uint8_t*) output + 4;
      }
      if (r & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_subpattern);
        vfill_subpattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if (r & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = (uint8_t*) output + 1;
      }
    }

    input = (const void*) ((uintptr_t) input + input_increment);
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
