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
#include "src/xnnpack/prefetch.h"


// Pack pre-transposed weights (GIO) for use by f32-gemm
void xnn_x32_packw_gemm_gio_ukernel_x16__avx512f_u8_prfm(
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
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const __m512 vzero = _mm512_setzero_ps();
  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 16
    const float* w = (const float*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m512 vb0 = _mm512_loadu_ps(b + 0);
        _mm512_store_ps(packed_w + 0, vb0);
        b += 16;
      } else {
        _mm512_store_ps(packed_w + 0, vzero);
      }
      packed_w += 16;

      size_t k = kc;
      // KC main loop 8x16
      for (; k >= 8; k -= 8) {
        const __m512 v0_0 = _mm512_loadu_ps(w + 0 + 0 * k_stride);
        const __m512 v0_1 = _mm512_loadu_ps(w + 0 + 1 * k_stride);
        const __m512 v0_2 = _mm512_loadu_ps(w + 0 + 2 * k_stride);
        const __m512 v0_3 = _mm512_loadu_ps(w + 0 + 3 * k_stride);
        const __m512 v0_4 = _mm512_loadu_ps(w + 0 + 4 * k_stride);
        const __m512 v0_5 = _mm512_loadu_ps(w + 0 + 5 * k_stride);
        const __m512 v0_6 = _mm512_loadu_ps(w + 0 + 6 * k_stride);
        const __m512 v0_7 = _mm512_loadu_ps(w + 0 + 7 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 0 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 1 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 2 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 3 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 4 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 5 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 6 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 7 * k_stride);
        _mm512_store_ps(packed_w + 0, v0_0);
        _mm512_store_ps(packed_w + 16, v0_1);
        _mm512_store_ps(packed_w + 32, v0_2);
        _mm512_store_ps(packed_w + 48, v0_3);
        _mm512_store_ps(packed_w + 64, v0_4);
        _mm512_store_ps(packed_w + 80, v0_5);
        _mm512_store_ps(packed_w + 96, v0_6);
        _mm512_store_ps(packed_w + 112, v0_7);
        w += k_stride * 8;
        packed_w += 128;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const __m512 v0 = _mm512_loadu_ps(w + 0);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        _mm512_store_ps(packed_w + 0, v0);
        w += k_stride;
        packed_w += 16;
      }
      w = w - kc * k_stride + 16;  // Advance to next column of 16 floats
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);

      // Prepare mask for valid 32-bit elements (depends on n).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << n) - 1) >> 0));

      if XNN_LIKELY(b != NULL) {
        const __m512 vb0 = _mm512_maskz_loadu_ps(vmask0, b + 0);
        _mm512_store_ps(packed_w + 0, vb0);
        b += n;
      } else {
        _mm512_store_ps(packed_w + 0, vzero);
      }
      packed_w += 16;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const __m512 v0 = _mm512_maskz_loadu_ps(vmask0, w + 0);
        _mm512_store_ps(packed_w + 0, v0);
        w += k_stride;
        packed_w += 16;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
