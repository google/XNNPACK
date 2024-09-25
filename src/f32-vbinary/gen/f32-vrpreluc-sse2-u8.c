// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vrpreluc_ukernel__sse2_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 vb = _mm_load1_ps(input_b);

  const __m128 vmask = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vb)));
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_mul_ps(va0, vb);
    __m128 vacc1 = _mm_mul_ps(va1, vb);

    vacc0 = _mm_or_ps(_mm_and_ps(vacc0, vmask), _mm_andnot_ps(vmask, vb));
    vacc1 = _mm_or_ps(_mm_and_ps(vacc1, vmask), _mm_andnot_ps(vmask, vb));

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_mul_ps(va, vb);
    vacc = _mm_or_ps(_mm_and_ps(vacc, vmask), _mm_andnot_ps(vmask, vb));

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_mul_ps(va, vb);
    vacc = _mm_or_ps(_mm_and_ps(vacc, vmask), _mm_andnot_ps(vmask, vb));
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
