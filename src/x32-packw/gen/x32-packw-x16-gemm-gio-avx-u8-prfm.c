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
#include "src/xnnpack/prefetch.h"


void xnn_x32_packw_gemm_gio_ukernel_x16__avx_u8_prfm(
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
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  static const int32_t mask_table[32] = {
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m256 vzero = _mm256_setzero_ps();
  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 16
    const float* w = (const float*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m256 vb0 = _mm256_loadu_ps(b + 0);
        const __m256 vb1 = _mm256_loadu_ps(b + 8);
        _mm256_store_ps(packed_w + 0, vb0);
        _mm256_store_ps(packed_w + 8, vb1);
        b += 16;
      } else {
        _mm256_store_ps(packed_w + 0, vzero);
        _mm256_store_ps(packed_w + 8, vzero);
      }
      packed_w += 16;

      size_t k = kc;
      // KC main loop 8x16
      for (; k >= 8; k -= 8) {
        const __m256 v0_0 = _mm256_loadu_ps(w + 0 + 0 * k_stride);
        const __m256 v1_0 = _mm256_loadu_ps(w + 8 + 0 * k_stride);
        const __m256 v0_1 = _mm256_loadu_ps(w + 0 + 1 * k_stride);
        const __m256 v1_1 = _mm256_loadu_ps(w + 8 + 1 * k_stride);
        const __m256 v0_2 = _mm256_loadu_ps(w + 0 + 2 * k_stride);
        const __m256 v1_2 = _mm256_loadu_ps(w + 8 + 2 * k_stride);
        const __m256 v0_3 = _mm256_loadu_ps(w + 0 + 3 * k_stride);
        const __m256 v1_3 = _mm256_loadu_ps(w + 8 + 3 * k_stride);
        const __m256 v0_4 = _mm256_loadu_ps(w + 0 + 4 * k_stride);
        const __m256 v1_4 = _mm256_loadu_ps(w + 8 + 4 * k_stride);
        const __m256 v0_5 = _mm256_loadu_ps(w + 0 + 5 * k_stride);
        const __m256 v1_5 = _mm256_loadu_ps(w + 8 + 5 * k_stride);
        const __m256 v0_6 = _mm256_loadu_ps(w + 0 + 6 * k_stride);
        const __m256 v1_6 = _mm256_loadu_ps(w + 8 + 6 * k_stride);
        const __m256 v0_7 = _mm256_loadu_ps(w + 0 + 7 * k_stride);
        const __m256 v1_7 = _mm256_loadu_ps(w + 8 + 7 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 0 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 1 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 2 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 3 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 4 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 5 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 6 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960 + 7 * k_stride);
        _mm256_store_ps(packed_w + 0, v0_0);
        _mm256_store_ps(packed_w + 8, v1_0);
        _mm256_store_ps(packed_w + 16, v0_1);
        _mm256_store_ps(packed_w + 24, v1_1);
        _mm256_store_ps(packed_w + 32, v0_2);
        _mm256_store_ps(packed_w + 40, v1_2);
        _mm256_store_ps(packed_w + 48, v0_3);
        _mm256_store_ps(packed_w + 56, v1_3);
        _mm256_store_ps(packed_w + 64, v0_4);
        _mm256_store_ps(packed_w + 72, v1_4);
        _mm256_store_ps(packed_w + 80, v0_5);
        _mm256_store_ps(packed_w + 88, v1_5);
        _mm256_store_ps(packed_w + 96, v0_6);
        _mm256_store_ps(packed_w + 104, v1_6);
        _mm256_store_ps(packed_w + 112, v0_7);
        _mm256_store_ps(packed_w + 120, v1_7);
        w += k_stride * 8;
        packed_w += 128;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const __m256 v0 = _mm256_loadu_ps(w + 0);
        const __m256 v1 = _mm256_loadu_ps(w + 8);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        _mm256_store_ps(packed_w + 0, v0);
        _mm256_store_ps(packed_w + 8, v1);
        w += k_stride;
        packed_w += 16;
      }
      w = w - kc * k_stride + 16;  // Advance to next column of 16 floats
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      const __m256i vmask0 = _mm256_loadu_si256((const __m256i*) &mask_table[16 - n]);
      const __m256i vmask1 = _mm256_loadu_si256((const __m256i*) &mask_table[24 - n]);

      if XNN_LIKELY(b != NULL) {
        const __m256 vb0 = _mm256_maskload_ps(b + 0, vmask0);
        const __m256 vb1 = _mm256_maskload_ps(b + 8, vmask1);
        _mm256_store_ps(packed_w + 0, vb0);
        _mm256_store_ps(packed_w + 8, vb1);
        b += n;
      } else {
        _mm256_store_ps(packed_w + 0, vzero);
        _mm256_store_ps(packed_w + 8, vzero);
      }
      packed_w += 16;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const __m256 v0 = _mm256_maskload_ps(w + 0, vmask0);
        const __m256 v1 = _mm256_maskload_ps(w + 8, vmask1);
        _mm256_store_ps(packed_w + 0, v0);
        _mm256_store_ps(packed_w + 8, v1);
        w += k_stride;
        packed_w += 16;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
