// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/sse41.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vrndu_ukernel__sse41_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(x);
    x += 4;

    const __m128 vy0123 = _mm_round_ps(vx0123, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(y, vy0123);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m128 vx = _mm_loadu_ps(x);
    __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy);
      vy = _mm_movehl_ps(vy, vy);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy);
    }
  }
}
