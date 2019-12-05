// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vscale.h>


void xnn_f32_vscale_ukernel__avx512f_unroll64(
    size_t n,
    const float* x,
    float* y,
    float c)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  __m512 vc = _mm512_set1_ps(c);
  for (; n >= 64 * sizeof(float); n -= 64 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(x);
    const __m512 vx1 = _mm512_loadu_ps(x + 16);
    const __m512 vx2 = _mm512_loadu_ps(x + 32);
    const __m512 vx3 = _mm512_loadu_ps(x + 48);
    x += 64;

    const __m512 vy0 = _mm512_mul_ps(vx0, vc);
    const __m512 vy1 = _mm512_mul_ps(vx1, vc);
    const __m512 vy2 = _mm512_mul_ps(vx2, vc);
    const __m512 vy3 = _mm512_mul_ps(vx3, vc);

    _mm512_storeu_ps(y, vy0);
    _mm512_storeu_ps(y + 16, vy1);
    _mm512_storeu_ps(y + 32, vy2);
    _mm512_storeu_ps(y + 48, vy3);
    y += 64;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vy = _mm512_mul_ps(vx, vc);

    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    const __m512 vy = _mm512_mul_ps(vx, vc);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}
