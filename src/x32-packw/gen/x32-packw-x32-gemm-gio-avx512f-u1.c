// Auto-generated file. Do not edit!
//   Template: src/x32-packw/gio-avx512.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/packw.h"


void xnn_x32_packw_gemm_gio_ukernel_x32__avx512f_u1(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint32_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const __m512 vzero = _mm512_setzero_ps();
  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 32
    const float* w = (const float*) weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const __m512 vb0 = _mm512_loadu_ps(b + 0);
        const __m512 vb1 = _mm512_loadu_ps(b + 16);
        _mm512_store_ps(packed_w + 0, vb0);
        _mm512_store_ps(packed_w + 16, vb1);
        b += 32;
      } else {
        _mm512_store_ps(packed_w + 0, vzero);
        _mm512_store_ps(packed_w + 16, vzero);
      }
      packed_w += 32;

      // KC main loop 1x32
      size_t k = kc;

      // KC remainder loop
      for (; k > 0; --k) {
        const __m512 v0 = _mm512_loadu_ps(w + 0);
        const __m512 v1 = _mm512_loadu_ps(w + 16);
        _mm512_store_ps(packed_w + 0, v0);
        _mm512_store_ps(packed_w + 16, v1);
        w += k_stride;
        packed_w += 32;
      }
      w = w - kc * k_stride + 32;  // Advance to next column of 32 floats
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);

      // Prepare mask for valid 32-bit elements (depends on n).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << n) - 1) >> 0));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << n) - 1) >> 16));

      if XNN_LIKELY(b != NULL) {
        const __m512 vb0 = _mm512_maskz_loadu_ps(vmask0, b + 0);
        const __m512 vb1 = _mm512_maskz_loadu_ps(vmask1, b + 16);
        _mm512_store_ps(packed_w + 0, vb0);
        _mm512_store_ps(packed_w + 16, vb1);
        b += n;
      } else {
        _mm512_store_ps(packed_w + 0, vzero);
        _mm512_store_ps(packed_w + 16, vzero);
      }
      packed_w += 32;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const __m512 v0 = _mm512_maskz_loadu_ps(vmask0, w + 0);
        const __m512 v1 = _mm512_maskz_loadu_ps(vmask1, w + 16);
        _mm512_store_ps(packed_w + 0, v0);
        _mm512_store_ps(packed_w + 16, v1);
        w += k_stride;
        packed_w += 32;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
