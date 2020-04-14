// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/pad.h>


void xnn_x32_unpool_ukernel__sse2(
    size_t p,
    size_t c,
    uint32_t f,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output)
{
  // Pre-initialize outputs with constant.
  const __m128i vf = _mm_set1_epi32((int) f);
  uint32_t** os = output;
  do {
    uint32_t* o = *os++;
    size_t k = c;
    for (; k >= 4; k -= 4) {
      _mm_storeu_si128((__m128i*) o, vf);
      o += 4;
    }
    if (k != 0) {
      if (k & 2) {
        _mm_storel_epi64((__m128i*) o, vf);
        o += 2;
      }
      if (k & 1) {
        *((int*) o) = _mm_cvtsi128_si32(vf);
      }
    }
  } while (--p != 0);

  // Copy indexed elements to output.
  size_t offset = 0;
  do {
    const uint32_t i = *index++;
    *((uint32_t*) ((uintptr_t) output[i] + offset)) = *input++;
    offset += sizeof(uint32_t);
  } while (--c != 0);
}
