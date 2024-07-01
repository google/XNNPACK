// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"


void xnn_f32_vrelu_ukernel__sse_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vzero = _mm_setzero_ps();

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m128 vacc0123 = _mm_loadu_ps(input);
    __m128 vacc4567 = _mm_loadu_ps(input + 4);
    input += 8;

    vacc0123 = _mm_max_ps(vacc0123, vzero);
    vacc4567 = _mm_max_ps(vacc4567, vzero);

    _mm_storeu_ps(output, vacc0123);
    _mm_storeu_ps(output + 4, vacc4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128 vacc = _mm_loadu_ps(input);
    input += 4;

    vacc = _mm_max_ps(vacc, vzero);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vacc = _mm_loadu_ps(input);
    vacc = _mm_max_ps(vacc, vzero);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}
