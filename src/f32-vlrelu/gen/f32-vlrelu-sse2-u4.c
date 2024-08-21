// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vlrelu_ukernel__sse2_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vslope = _mm_set1_ps(params->scalar.slope);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    input += 4;

    __m128 vacc0123 = _mm_mul_ps(vx0123, vslope);
    const __m128 vmask0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx0123)));

    vacc0123 = _mm_or_ps(_mm_and_ps(vacc0123, vmask0123), _mm_andnot_ps(vmask0123, vx0123));

    _mm_storeu_ps(output, vacc0123);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    __m128 vacc = _mm_mul_ps(vx, vslope);
    const __m128 vmask = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vacc = _mm_or_ps(_mm_and_ps(vacc, vmask), _mm_andnot_ps(vmask, vx));

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
