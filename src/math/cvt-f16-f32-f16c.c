// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_f32_cvt__f16c(
    size_t n,
    const void* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const uint16_t* i = (const uint16_t*) input;
  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m256 vy = _mm256_cvtph_ps(vx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
}
