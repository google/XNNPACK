// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vrsubc_minmax_ukernel__avx512f_u32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vb, vacc0);
    vacc1 = _mm512_sub_ps(vb, vacc1);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vb, vacc);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}
