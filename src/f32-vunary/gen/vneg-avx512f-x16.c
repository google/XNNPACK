// Auto-generated file. Do not edit!
//   Template: src/f32-vunary/avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vneg_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_neg_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512.sign_mask);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512i vx0123456789ABCDEF = _mm512_loadu_si512(x);
    x += 16;

    const __m512i vy0123456789ABCDEF = _mm512_xor_epi32(vx0123456789ABCDEF, vsign_mask);

    _mm512_storeu_si512(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512i vx = _mm512_maskz_loadu_epi32(vmask, x);
    const __m512i vy = _mm512_xor_epi32(vx, vsign_mask);
    _mm512_mask_storeu_epi32(y, vmask, vy);
  }
}
