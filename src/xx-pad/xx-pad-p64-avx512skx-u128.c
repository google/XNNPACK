// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "src/xnnpack/pad.h"
#include "src/xnnpack/unaligned.h"


// Naming convention:
//   p64  - pre/post padding loop unrolled to 64 bytes (1x 512-bit vector).
//   u128 - main channel copy loop unrolled to 128 bytes (2x 512-bit vectors).
void xnn_xx_pad_ukernel_p64__avx512skx_u128(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern)
{
  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const __m512i vfill_pattern = _mm512_set1_epi32((int) fill_pattern);
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      for (; l >= 64 * sizeof(uint8_t); l -= 64 * sizeof(uint8_t)) {
        _mm512_storeu_si512((__m512i*) output, vfill_pattern);
        output = (uint8_t*) output + 64;
      }
      if XNN_UNLIKELY(l != 0) {
        const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << l) - UINT64_C(1)));
        _mm512_mask_storeu_epi8(output, vmask, vfill_pattern);
        output = (uint8_t*) output + l;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 128 * sizeof(uint8_t); c -= 128 * sizeof(uint8_t)) {
      const __m512i vdata0 = _mm512_loadu_si512((const __m512i*) input);
      const __m512i vdata1 = _mm512_loadu_si512((const __m512i*) input + 1);
      input = (const uint8_t*) input + 128;

      _mm512_storeu_si512((__m512i*) output, vdata0);
      _mm512_storeu_si512((__m512i*) output + 1, vdata1);
      output = (uint8_t*) output + 128;
    }
    for (; c >= 64 * sizeof(uint8_t); c -= 64 * sizeof(uint8_t)) {
      const __m512i vdata = _mm512_loadu_si512((const __m512i*) input);
      input = (const uint8_t*) input + 64;

      _mm512_storeu_si512((__m512i*) output, vdata);
      output = (uint8_t*) output + 64;
    }
    if XNN_UNLIKELY(c != 0) {
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << c) - UINT64_C(1)));
      const __m512i vdata = _mm512_maskz_loadu_epi8(vmask, input);
      input = (const void*) ((uintptr_t) input + c);

      _mm512_mask_storeu_epi8(output, vmask, vdata);
      output = (uint8_t*) output + c;
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      for (; r >= 64 * sizeof(uint8_t); r -= 64 * sizeof(uint8_t)) {
        _mm512_storeu_si512((__m512i*) output, vfill_pattern);
        output = (uint8_t*) output + 64;
      }
      if XNN_UNLIKELY(r != 0) {
        const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << r) - UINT64_C(1)));
        _mm512_mask_storeu_epi8(output, vmask, vfill_pattern);
        output = (uint8_t*) output + r;
      }
    }

    input = (const void*) ((uintptr_t) input + input_increment);
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
