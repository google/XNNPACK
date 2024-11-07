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

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/packw.h"
#include "xnnpack/prefetch.h"


void xnn_x32_packw_gemm_gio_ukernel_x32__avx_u8_prfm(
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
  static const int32_t mask_table[64] = {
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m256 vzero = _mm256_setzero_ps();
  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 32
    const float* w = (const float*) weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const __m256 vb0 = _mm256_loadu_ps(b + 0);
        const __m256 vb8 = _mm256_loadu_ps(b + 8);
        const __m256 vb16 = _mm256_loadu_ps(b + 16);
        const __m256 vb24 = _mm256_loadu_ps(b + 24);
        _mm256_store_ps(packed_w + 0, vb0);
        _mm256_store_ps(packed_w + 8, vb8);
        _mm256_store_ps(packed_w + 16, vb16);
        _mm256_store_ps(packed_w + 24, vb24);
        b += 32;
      } else {
        _mm256_store_ps(packed_w + 0, vzero);
        _mm256_store_ps(packed_w + 8, vzero);
        _mm256_store_ps(packed_w + 16, vzero);
        _mm256_store_ps(packed_w + 24, vzero);
      }
      packed_w += 32;

      size_t k = kc;
      // KC main loop 8x32
      for (; k >= 8; k -= 8) {
        const __m256 v0_0 = _mm256_loadu_ps(w + 0 + 0 * k_stride);
        const __m256 v8_0 = _mm256_loadu_ps(w + 8 + 0 * k_stride);
        const __m256 v16_0 = _mm256_loadu_ps(w + 16 + 0 * k_stride);
        const __m256 v24_0 = _mm256_loadu_ps(w + 24 + 0 * k_stride);
        const __m256 v0_1 = _mm256_loadu_ps(w + 0 + 1 * k_stride);
        const __m256 v8_1 = _mm256_loadu_ps(w + 8 + 1 * k_stride);
        const __m256 v16_1 = _mm256_loadu_ps(w + 16 + 1 * k_stride);
        const __m256 v24_1 = _mm256_loadu_ps(w + 24 + 1 * k_stride);
        const __m256 v0_2 = _mm256_loadu_ps(w + 0 + 2 * k_stride);
        const __m256 v8_2 = _mm256_loadu_ps(w + 8 + 2 * k_stride);
        const __m256 v16_2 = _mm256_loadu_ps(w + 16 + 2 * k_stride);
        const __m256 v24_2 = _mm256_loadu_ps(w + 24 + 2 * k_stride);
        const __m256 v0_3 = _mm256_loadu_ps(w + 0 + 3 * k_stride);
        const __m256 v8_3 = _mm256_loadu_ps(w + 8 + 3 * k_stride);
        const __m256 v16_3 = _mm256_loadu_ps(w + 16 + 3 * k_stride);
        const __m256 v24_3 = _mm256_loadu_ps(w + 24 + 3 * k_stride);
        const __m256 v0_4 = _mm256_loadu_ps(w + 0 + 4 * k_stride);
        const __m256 v8_4 = _mm256_loadu_ps(w + 8 + 4 * k_stride);
        const __m256 v16_4 = _mm256_loadu_ps(w + 16 + 4 * k_stride);
        const __m256 v24_4 = _mm256_loadu_ps(w + 24 + 4 * k_stride);
        const __m256 v0_5 = _mm256_loadu_ps(w + 0 + 5 * k_stride);
        const __m256 v8_5 = _mm256_loadu_ps(w + 8 + 5 * k_stride);
        const __m256 v16_5 = _mm256_loadu_ps(w + 16 + 5 * k_stride);
        const __m256 v24_5 = _mm256_loadu_ps(w + 24 + 5 * k_stride);
        const __m256 v0_6 = _mm256_loadu_ps(w + 0 + 6 * k_stride);
        const __m256 v8_6 = _mm256_loadu_ps(w + 8 + 6 * k_stride);
        const __m256 v16_6 = _mm256_loadu_ps(w + 16 + 6 * k_stride);
        const __m256 v24_6 = _mm256_loadu_ps(w + 24 + 6 * k_stride);
        const __m256 v0_7 = _mm256_loadu_ps(w + 0 + 7 * k_stride);
        const __m256 v8_7 = _mm256_loadu_ps(w + 8 + 7 * k_stride);
        const __m256 v16_7 = _mm256_loadu_ps(w + 16 + 7 * k_stride);
        const __m256 v24_7 = _mm256_loadu_ps(w + 24 + 7 * k_stride);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        _mm256_store_ps(packed_w + 0, v0_0);
        _mm256_store_ps(packed_w + 8, v8_0);
        _mm256_store_ps(packed_w + 16, v16_0);
        _mm256_store_ps(packed_w + 24, v24_0);
        _mm256_store_ps(packed_w + 32, v0_1);
        _mm256_store_ps(packed_w + 40, v8_1);
        _mm256_store_ps(packed_w + 48, v16_1);
        _mm256_store_ps(packed_w + 56, v24_1);
        _mm256_store_ps(packed_w + 64, v0_2);
        _mm256_store_ps(packed_w + 72, v8_2);
        _mm256_store_ps(packed_w + 80, v16_2);
        _mm256_store_ps(packed_w + 88, v24_2);
        _mm256_store_ps(packed_w + 96, v0_3);
        _mm256_store_ps(packed_w + 104, v8_3);
        _mm256_store_ps(packed_w + 112, v16_3);
        _mm256_store_ps(packed_w + 120, v24_3);
        _mm256_store_ps(packed_w + 128, v0_4);
        _mm256_store_ps(packed_w + 136, v8_4);
        _mm256_store_ps(packed_w + 144, v16_4);
        _mm256_store_ps(packed_w + 152, v24_4);
        _mm256_store_ps(packed_w + 160, v0_5);
        _mm256_store_ps(packed_w + 168, v8_5);
        _mm256_store_ps(packed_w + 176, v16_5);
        _mm256_store_ps(packed_w + 184, v24_5);
        _mm256_store_ps(packed_w + 192, v0_6);
        _mm256_store_ps(packed_w + 200, v8_6);
        _mm256_store_ps(packed_w + 208, v16_6);
        _mm256_store_ps(packed_w + 216, v24_6);
        _mm256_store_ps(packed_w + 224, v0_7);
        _mm256_store_ps(packed_w + 232, v8_7);
        _mm256_store_ps(packed_w + 240, v16_7);
        _mm256_store_ps(packed_w + 248, v24_7);
        w += k_stride * 8;
        packed_w += 256;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const __m256 v0 = _mm256_loadu_ps(w + 0);
        const __m256 v8 = _mm256_loadu_ps(w + 8);
        const __m256 v16 = _mm256_loadu_ps(w + 16);
        const __m256 v24 = _mm256_loadu_ps(w + 24);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        xnn_prefetch_to_l1((const int8_t*) w + 1024);
        _mm256_store_ps(packed_w + 0, v0);
        _mm256_store_ps(packed_w + 8, v8);
        _mm256_store_ps(packed_w + 16, v16);
        _mm256_store_ps(packed_w + 24, v24);
        w += k_stride;
        packed_w += 32;
      }
      w = w - kc * k_stride + 32;  // Advance to next column of 32 floats
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);
      const __m256i vmask0 = _mm256_loadu_si256((const __m256i*) &mask_table[32 - n]);
      const __m256i vmask8 = _mm256_loadu_si256((const __m256i*) &mask_table[40 - n]);
      const __m256i vmask16 = _mm256_loadu_si256((const __m256i*) &mask_table[48 - n]);
      const __m256i vmask24 = _mm256_loadu_si256((const __m256i*) &mask_table[56 - n]);

      if XNN_LIKELY(b != NULL) {
        const __m256 vb0 = _mm256_maskload_ps(b + 0, vmask0);
        const __m256 vb8 = _mm256_maskload_ps(b + 8, vmask8);
        const __m256 vb16 = _mm256_maskload_ps(b + 16, vmask16);
        const __m256 vb24 = _mm256_maskload_ps(b + 24, vmask24);
        _mm256_store_ps(packed_w + 0, vb0);
        _mm256_store_ps(packed_w + 8, vb8);
        _mm256_store_ps(packed_w + 16, vb16);
        _mm256_store_ps(packed_w + 24, vb24);
        b += n;
      } else {
        _mm256_store_ps(packed_w + 0, vzero);
        _mm256_store_ps(packed_w + 8, vzero);
        _mm256_store_ps(packed_w + 16, vzero);
        _mm256_store_ps(packed_w + 24, vzero);
      }
      packed_w += 32;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const __m256 v0 = _mm256_maskload_ps(w + 0, vmask0);
        const __m256 v8 = _mm256_maskload_ps(w + 8, vmask8);
        const __m256 v16 = _mm256_maskload_ps(w + 16, vmask16);
        const __m256 v24 = _mm256_maskload_ps(w + 24, vmask24);
        _mm256_store_ps(packed_w + 0, v0);
        _mm256_store_ps(packed_w + 8, v8);
        _mm256_store_ps(packed_w + 16, v16);
        _mm256_store_ps(packed_w + 24, v24);
        w += k_stride;
        packed_w += 32;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
