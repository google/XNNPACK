// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_f16_cvt__f16c(
    size_t n,
    const float* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    const __m128i vy = _mm256_cvtps_ph(vx, _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, vy);
    o += 8;
  }
}
