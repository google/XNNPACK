// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vrsubc_minmax_ukernel__sse_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 vy_min = _mm_load_ps(params->sse.min);
  const __m128 vy_max = _mm_load_ps(params->sse.max);

  const __m128 vb = _mm_load1_ps(input_b);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0123 = _mm_loadu_ps(input_a);
    const __m128 va4567 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vy0123 = _mm_sub_ps(vb, va0123);
    __m128 vy4567 = _mm_sub_ps(vb, va4567);


    vy0123 = _mm_max_ps(vy0123, vy_min);
    vy4567 = _mm_max_ps(vy4567, vy_min);

    vy0123 = _mm_min_ps(vy0123, vy_max);
    vy4567 = _mm_min_ps(vy4567, vy_max);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va0123 = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vy0123 = _mm_sub_ps(vb, va0123);
    vy0123 = _mm_max_ps(vy0123, vy_min);
    vy0123 = _mm_min_ps(vy0123, vy_max);
    _mm_storeu_ps(output, vy0123);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va0123 = _mm_loadu_ps(input_a);

    __m128 vy0123 = _mm_sub_ps(vb, va0123);
    vy0123 = _mm_max_ps(vy0123, vy_min);
    vy0123 = _mm_min_ps(vy0123, vy_max);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy0123);
      vy0123 = _mm_movehl_ps(vy0123, vy0123);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy0123);
    }
  }
}
