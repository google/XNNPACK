// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f32_rsum_ukernel__sse_u16_acc2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m128 vacc0 = _mm_setzero_ps();
  __m128 vacc1 = _mm_setzero_ps();
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m128 vt0 = _mm_loadu_ps(input);
    const __m128 vt1 = _mm_loadu_ps(input + 4);
    const __m128 vt2 = _mm_loadu_ps(input + 8);
    const __m128 vt3 = _mm_loadu_ps(input + 12);
    input += 16;

    vacc0 = _mm_add_ps(vacc0, vt0);
    vacc1 = _mm_add_ps(vacc1, vt1);
    vacc0 = _mm_add_ps(vacc0, vt2);
    vacc1 = _mm_add_ps(vacc1, vt3);
  }
  vacc0 = _mm_add_ps(vacc0, vacc1);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vt = _mm_loadu_ps(input);
    input += 4;

    vacc0 = _mm_add_ps(vacc0, vt);
  }
  vacc0 = _mm_add_ps(vacc0, _mm_movehl_ps(vacc0, vacc0));
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128 vt = _mm_load_ss(input);
      input += 1;
      vacc0 = _mm_add_ss(vacc0, vt);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  vacc0 = _mm_add_ss(vacc0, _mm_shuffle_ps(vacc0, vacc0, _MM_SHUFFLE(1, 1, 1, 1)));
  vacc0 = _mm_mul_ss(vacc0, _mm_load_ss(&params->scalar.scale));
  vacc0 = _mm_max_ss(vacc0, _mm_load_ss(&params->scalar.min));
  vacc0 = _mm_min_ss(vacc0, _mm_load_ss(&params->scalar.max));
  *output += _mm_cvtss_f32(vacc0);
}
