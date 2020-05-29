// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/pad.h>


void xnn_x32_pad_ukernel__sse(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const uint32_t* fill_value,
    const uint32_t* input,
    size_t input_stride,
    uint32_t* output,
    size_t output_stride) XNN_DISABLE_TSAN
{
  assert(channels % sizeof(uint32_t) == 0);
  assert(pre_padding % sizeof(uint32_t) == 0);
  assert(post_padding % sizeof(uint32_t) == 0);
  assert(fill_value != NULL);

  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const __m128 vfill = _mm_load1_ps((const float*) fill_value);
  const float* i = (const float*) input;
  float* o = (float*) output;
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      for (; l >= 4 * sizeof(uint32_t); l -= 4 * sizeof(uint32_t)) {
        _mm_storeu_ps(o, vfill);
        o += 4;
      }
      if (l & (2 * sizeof(uint32_t))) {
        _mm_storel_pi((__m64*) o, vfill);
        o += 2;
      }
      if (l & sizeof(uint32_t)) {
        _mm_store_ss(o, vfill);
        o += 1;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      const __m128 vtmp = _mm_loadu_ps(i);
      i += 4;

      _mm_storeu_ps(o, vtmp);
      o += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128 vtmp = _mm_loadu_ps(i);
      i = (const void*) ((uintptr_t) i + c);
      if (c & (2 * sizeof(uint32_t))) {
        _mm_storel_pi((__m64*) o, vtmp);
        o += 2;

        vtmp = _mm_movehl_ps(vtmp, vtmp);
      }
      if (c & sizeof(uint32_t)) {
        _mm_store_ss(o, vtmp);
        o += 1;
      }
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      for (; r >= 4 * sizeof(uint32_t); r -= 4 * sizeof(uint32_t)) {
        _mm_storeu_ps(o, vfill);
        o += 4;
      }
      if (r & (2 * sizeof(uint32_t))) {
        _mm_storel_pi((__m64*) o, vfill);
        o += 2;
      }
      if (r & sizeof(uint32_t)) {
        _mm_store_ss(o, vfill);
        o += 1;
      }
    }

    i = (const float*) ((uintptr_t) i + input_increment);
    o = (float*) ((uintptr_t) o + output_increment);
  } while (--rows != 0);
}
