// Auto-generated file. Do not edit!
//   Template: src/f32-clamp/avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/clamp.h>
#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f32_clamp_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m512 vy_min = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.min));
  const __m512 vy_max = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.max));

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(x);
    x += 16;

    vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEF, vy_min);

    vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vy_max);

    _mm512_storeu_ps(y, vacc0123456789ABCDEF);
    y += 16;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(x);
    x += 16;

    vacc = _mm512_max_ps(vacc, vy_min);
    vacc = _mm512_min_ps(vacc, vy_max);

    _mm512_storeu_ps(y, vacc);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, x);
    vacc = _mm512_max_ps(vacc, vy_min);
    vacc = _mm512_min_ps(vacc, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vacc);
  }
}
