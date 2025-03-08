// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/gio-avx.c.in
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


void xnn_x32_packw_gemm_gio_ukernel_x8__avx_u8(
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
  assert(nr == 8);   // This kernel is for NR=8
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  static const int32_t mask_table[16] = {
    -1, -1, -1, -1, -1, -1, -1, -1,
    0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m256 vzero = _mm256_setzero_ps();
  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 8
    const float* w = (const float*) weights;
    size_t n = nc;

    for (; n >= 8; n -= 8) {
      if XNN_LIKELY(b != NULL) {
        const __m256 vb0 = _mm256_loadu_ps(b + 0);
        _mm256_store_ps(packed_w + 0, vb0);
        b += 8;
      } else {
        _mm256_store_ps(packed_w + 0, vzero);
      }
      packed_w += 8;

      size_t k = kc;
      // KC main loop 8x8
      for (; k >= 8; k -= 8) {
        const __m256 v0_0 = _mm256_loadu_ps(w + 0 + 0 * k_stride);
        const __m256 v0_1 = _mm256_loadu_ps(w + 0 + 1 * k_stride);
        const __m256 v0_2 = _mm256_loadu_ps(w + 0 + 2 * k_stride);
        const __m256 v0_3 = _mm256_loadu_ps(w + 0 + 3 * k_stride);
        const __m256 v0_4 = _mm256_loadu_ps(w + 0 + 4 * k_stride);
        const __m256 v0_5 = _mm256_loadu_ps(w + 0 + 5 * k_stride);
        const __m256 v0_6 = _mm256_loadu_ps(w + 0 + 6 * k_stride);
        const __m256 v0_7 = _mm256_loadu_ps(w + 0 + 7 * k_stride);
        _mm256_store_ps(packed_w + 0, v0_0);
        _mm256_store_ps(packed_w + 8, v0_1);
        _mm256_store_ps(packed_w + 16, v0_2);
        _mm256_store_ps(packed_w + 24, v0_3);
        _mm256_store_ps(packed_w + 32, v0_4);
        _mm256_store_ps(packed_w + 40, v0_5);
        _mm256_store_ps(packed_w + 48, v0_6);
        _mm256_store_ps(packed_w + 56, v0_7);
        w += k_stride * 8;
        packed_w += 64;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const __m256 v0 = _mm256_loadu_ps(w + 0);
        _mm256_store_ps(packed_w + 0, v0);
        w += k_stride;
        packed_w += 8;
      }
      w = w - kc * k_stride + 8;  // Advance to next column of 8 floats
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      const __m256i vmask0 = _mm256_loadu_si256((const __m256i*) &mask_table[8 - n]);

      if XNN_LIKELY(b != NULL) {
        const __m256 vb0 = _mm256_maskload_ps(b + 0, vmask0);
        _mm256_store_ps(packed_w + 0, vb0);
        b += n;
      } else {
        _mm256_store_ps(packed_w + 0, vzero);
      }
      packed_w += 8;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const __m256 v0 = _mm256_maskload_ps(w + 0, vmask0);
        _mm256_store_ps(packed_w + 0, v0);
        w += k_stride;
        packed_w += 8;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
