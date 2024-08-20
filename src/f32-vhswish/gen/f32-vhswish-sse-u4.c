// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vhswish_ukernel__sse_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsixth = _mm_set1_ps(0x1.555556p-3f);
  const __m128 vhalf = _mm_set1_ps(0.5f);
  const __m128 vone = _mm_set1_ps(1.0f);
  const __m128 vzero = _mm_setzero_ps();

  XNN_FORCE_REALIZATION(vsixth);
  XNN_FORCE_REALIZATION(vhalf);
  XNN_FORCE_REALIZATION(vone);
  // XNN_FORCE_REALIZATION(vzero);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    input += 4;

    __m128 vacc0123 = _mm_mul_ps(vx0123, vsixth);

    vacc0123 = _mm_add_ps(vacc0123, vhalf);

    vacc0123 = _mm_max_ps(vacc0123, vzero);

    vacc0123 = _mm_min_ps(vacc0123, vone);

    vacc0123 = _mm_mul_ps(vacc0123, vx0123);

    _mm_storeu_ps(output, vacc0123);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vacc0123 = _mm_mul_ps(vx0123, vsixth);
    vacc0123 = _mm_add_ps(vacc0123, vhalf);
    vacc0123 = _mm_max_ps(vacc0123, vzero);
    vacc0123 = _mm_min_ps(vacc0123, vone);
    vacc0123 = _mm_mul_ps(vacc0123, vx0123);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc0123);
      vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc0123);
    }
  }
}
