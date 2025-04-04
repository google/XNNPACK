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


void xnn_x32_packw_gemm_gio_ukernel_x32__avx_u1(
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
        const __m256 vb1 = _mm256_loadu_ps(b + 8);
        const __m256 vb2 = _mm256_loadu_ps(b + 16);
        const __m256 vb3 = _mm256_loadu_ps(b + 24);
        _mm256_store_ps(packed_w + 0, vb0);
        _mm256_store_ps(packed_w + 8, vb1);
        _mm256_store_ps(packed_w + 16, vb2);
        _mm256_store_ps(packed_w + 24, vb3);
        b += 32;
      } else {
        _mm256_store_ps(packed_w + 0, vzero);
        _mm256_store_ps(packed_w + 8, vzero);
        _mm256_store_ps(packed_w + 16, vzero);
        _mm256_store_ps(packed_w + 24, vzero);
      }
      packed_w += 32;

      size_t k = kc;

      // KC remainder loop
      for (; k > 0; --k) {
        const __m256 v0 = _mm256_loadu_ps(w + 0);
        const __m256 v1 = _mm256_loadu_ps(w + 8);
        const __m256 v2 = _mm256_loadu_ps(w + 16);
        const __m256 v3 = _mm256_loadu_ps(w + 24);
        _mm256_store_ps(packed_w + 0, v0);
        _mm256_store_ps(packed_w + 8, v1);
        _mm256_store_ps(packed_w + 16, v2);
        _mm256_store_ps(packed_w + 24, v3);
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
      const __m256i vmask1 = _mm256_loadu_si256((const __m256i*) &mask_table[40 - n]);
      const __m256i vmask2 = _mm256_loadu_si256((const __m256i*) &mask_table[48 - n]);
      const __m256i vmask3 = _mm256_loadu_si256((const __m256i*) &mask_table[56 - n]);

      if XNN_LIKELY(b != NULL) {
        const __m256 vb0 = _mm256_maskload_ps(b + 0, vmask0);
        const __m256 vb1 = _mm256_maskload_ps(b + 8, vmask1);
        const __m256 vb2 = _mm256_maskload_ps(b + 16, vmask2);
        const __m256 vb3 = _mm256_maskload_ps(b + 24, vmask3);
        _mm256_store_ps(packed_w + 0, vb0);
        _mm256_store_ps(packed_w + 8, vb1);
        _mm256_store_ps(packed_w + 16, vb2);
        _mm256_store_ps(packed_w + 24, vb3);
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
        const __m256 v1 = _mm256_maskload_ps(w + 8, vmask1);
        const __m256 v2 = _mm256_maskload_ps(w + 16, vmask2);
        const __m256 v3 = _mm256_maskload_ps(w + 24, vmask3);
        _mm256_store_ps(packed_w + 0, v0);
        _mm256_store_ps(packed_w + 8, v1);
        _mm256_store_ps(packed_w + 16, v2);
        _mm256_store_ps(packed_w + 24, v3);
        w += k_stride;
        packed_w += 32;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
