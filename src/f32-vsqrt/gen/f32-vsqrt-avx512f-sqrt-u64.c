// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/avx512f-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vsqrt_ukernel__avx512f_sqrt_u64(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(input + 0);
    const __m512 vx1 = _mm512_loadu_ps(input + 16);
    const __m512 vx2 = _mm512_loadu_ps(input + 32);
    const __m512 vx3 = _mm512_loadu_ps(input + 48);
    input += 64;

    const __m512 vy0 = _mm512_sqrt_ps(vx0);
    const __m512 vy1 = _mm512_sqrt_ps(vx1);
    const __m512 vy2 = _mm512_sqrt_ps(vx2);
    const __m512 vy3 = _mm512_sqrt_ps(vx3);

    _mm512_storeu_ps(output + 0, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    _mm512_storeu_ps(output + 32, vy2);
    _mm512_storeu_ps(output + 48, vy3);
    output += 64;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;
    const __m512 vy = _mm512_sqrt_ps(vx);
    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_sqrt_ps(vmask, vx);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
