// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/fill.h>


void xnn_x32_fill_ukernel__sse(
    size_t rows,
    size_t channels,
    uint32_t* output,
    size_t output_stride,
    const uint32_t* fill_value)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint32_t) == 0);
  assert(fill_value != NULL);

  const size_t output_increment = output_stride - channels;

  const __m128 vfill = _mm_load1_ps((const float*) fill_value);
  float* o = (float*) output;
  do {
    size_t c = channels;
    for (; c >= 16 * sizeof(uint32_t); c -= 16 * sizeof(uint32_t)) {
      _mm_storeu_ps(o, vfill);
      _mm_storeu_ps(o + 4, vfill);
      _mm_storeu_ps(o + 8, vfill);
      _mm_storeu_ps(o + 12, vfill);
      o += 16;
    }
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      _mm_storeu_ps(o, vfill);
      o += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      if XNN_LIKELY(c & (2 * sizeof(uint32_t))) {
        _mm_storel_pi((__m64*) o, vfill);
        o += 2;
      }
      if XNN_LIKELY(c & (1 * sizeof(uint32_t))) {
        _mm_store_ss(o, vfill);
        o += 1;
      }
    }
    o = (void*) ((uintptr_t) o + output_increment);
  } while (--rows != 0);
}
