// clang-format off
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

#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"


// Pack pre-transposed weights (GIO) for use by f32-gemm
void xnn_x32_packw_gemm_gio_ukernel_x32__avx512f_u8(
  size_t g,                  // Batch size (outer loop).  usually 1
  size_t nc,                 // Number of columns and typically large
  size_t kc,                 // Number of rows and typically small
  size_t nr,                 // Matches gemm and is a multiple of vector sizes
  size_t kr,                 // unused - must be 1
  size_t sr,                 // unused - must be 1
  size_t k_stride,           // Elements per row (typically same as nc)
  const uint32_t* weights,   // Weights to pack. unaligned, unpadded
  const uint32_t* bias,      // Bias to pack. unaligned, unpadded, can be NULL
  const void* scale,         // unused
  uint32_t* packed_weights,  // packed weights output buffer - aligned, padded
  size_t extra_bytes,        // number of extra bytes between weights. aligned
  const void* params)        // unused
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

      size_t k = kc;
      // KC main loop 8x32
      for (; k >= 8; k -= 8) {
        const __m512 v0_0 = _mm512_loadu_ps(w + 0 + 0 * k_stride);
        const __m512 v1_0 = _mm512_loadu_ps(w + 16 + 0 * k_stride);
        const __m512 v0_1 = _mm512_loadu_ps(w + 0 + 1 * k_stride);
        const __m512 v1_1 = _mm512_loadu_ps(w + 16 + 1 * k_stride);
        const __m512 v0_2 = _mm512_loadu_ps(w + 0 + 2 * k_stride);
        const __m512 v1_2 = _mm512_loadu_ps(w + 16 + 2 * k_stride);
        const __m512 v0_3 = _mm512_loadu_ps(w + 0 + 3 * k_stride);
        const __m512 v1_3 = _mm512_loadu_ps(w + 16 + 3 * k_stride);
        const __m512 v0_4 = _mm512_loadu_ps(w + 0 + 4 * k_stride);
        const __m512 v1_4 = _mm512_loadu_ps(w + 16 + 4 * k_stride);
        const __m512 v0_5 = _mm512_loadu_ps(w + 0 + 5 * k_stride);
        const __m512 v1_5 = _mm512_loadu_ps(w + 16 + 5 * k_stride);
        const __m512 v0_6 = _mm512_loadu_ps(w + 0 + 6 * k_stride);
        const __m512 v1_6 = _mm512_loadu_ps(w + 16 + 6 * k_stride);
        const __m512 v0_7 = _mm512_loadu_ps(w + 0 + 7 * k_stride);
        const __m512 v1_7 = _mm512_loadu_ps(w + 16 + 7 * k_stride);
        _mm512_store_ps(packed_w + 0, v0_0);
        _mm512_store_ps(packed_w + 16, v1_0);
        _mm512_store_ps(packed_w + 32, v0_1);
        _mm512_store_ps(packed_w + 48, v1_1);
        _mm512_store_ps(packed_w + 64, v0_2);
        _mm512_store_ps(packed_w + 80, v1_2);
        _mm512_store_ps(packed_w + 96, v0_3);
        _mm512_store_ps(packed_w + 112, v1_3);
        _mm512_store_ps(packed_w + 128, v0_4);
        _mm512_store_ps(packed_w + 144, v1_4);
        _mm512_store_ps(packed_w + 160, v0_5);
        _mm512_store_ps(packed_w + 176, v1_5);
        _mm512_store_ps(packed_w + 192, v0_6);
        _mm512_store_ps(packed_w + 208, v1_6);
        _mm512_store_ps(packed_w + 224, v0_7);
        _mm512_store_ps(packed_w + 240, v1_7);
        w += k_stride * 8;
        packed_w += 256;
      }

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
