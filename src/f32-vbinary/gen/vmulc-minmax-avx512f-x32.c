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

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vmulc_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m512 vy_min = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.min));
  const __m512 vy_max = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.max));

  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_mul_ps(va0123456789ABCDEF, vb);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vaGHIJKLMNOPQRSTUV, vb);

    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_mul_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_mul_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}
