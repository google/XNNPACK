// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-avx512f.c.in
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


void xnn_f32_vmin_ukernel__avx512f_x16(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_MSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);


  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    a += 16;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(b);
    b += 16;

    __m512 vy0123456789ABCDEF = _mm512_min_ps(va0123456789ABCDEF, vb0123456789ABCDEF);



    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, b);

    __m512 vy = _mm512_min_ps(va, vb);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}
