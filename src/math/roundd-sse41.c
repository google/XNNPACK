// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <smmintrin.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundd__sse41(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_load_ps(input);
    input += 4;

    const __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    _mm_store_ps(output, vy);
    output += 4;
  }
}
