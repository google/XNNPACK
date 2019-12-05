// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/clamp.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f32_clamp_ukernel__avx512f(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m512 voutput_max = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.max));
  const __m512 voutput_min = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.min));

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vy = _mm512_min_ps(_mm512_max_ps(vx, voutput_min), voutput_max);

    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if (n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);

    const __m512 vy = _mm512_min_ps(_mm512_max_ps(vx, voutput_min), voutput_max);

    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}
