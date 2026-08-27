// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-packw/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"
#include "src/xnnpack/prefetch.h"


void xnn_x16_packw_gemm_goi_ukernel_x64__avx512skx_u16_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 64);   // This kernel is for NR=64
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint16_t* b = bias;
  uint16_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 64
    const uint16_t* w_base = weights;
    size_t n = nc;

    for (; n >= 64; n -= 64) {
      if XNN_LIKELY(b != NULL) {
        const __m512i vb0 = _mm512_loadu_si512((const __m512i*) (b + 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vb0);
        const __m512i vb32 = _mm512_loadu_si512((const __m512i*) (b + 32));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), vb32);
        b += 64;
      } else {
        const __m512i vzero = _mm512_setzero_si512();
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vzero);
        _mm512_storeu_si512((__m512i*) (packed_w + 32), vzero);
      }
      packed_w += 64;

      const ptrdiff_t stride1 = (ptrdiff_t) kc;
      const ptrdiff_t stride2 = stride1 * 2;
      const ptrdiff_t stride3 = stride1 * 3;
      const ptrdiff_t stride4 = stride1 * 4;
      const ptrdiff_t stride5 = stride1 * 5;
      const ptrdiff_t stride6 = stride1 * 6;
      const ptrdiff_t stride7 = stride1 * 7;

      const uint16_t* w0 = w_base;
      const uint16_t* w8 = w0 + 8 * stride1;
      const uint16_t* w16 = w0 + 16 * stride1;
      const uint16_t* w24 = w0 + 24 * stride1;
      const uint16_t* w32 = w0 + 32 * stride1;
      const uint16_t* w40 = w0 + 40 * stride1;
      const uint16_t* w48 = w0 + 48 * stride1;
      const uint16_t* w56 = w0 + 56 * stride1;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w8);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w16);
      xnn_prefetch_to_l1((const int8_t*) w16 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w24);
      xnn_prefetch_to_l1((const int8_t*) w24 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w32);
      xnn_prefetch_to_l1((const int8_t*) w32 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w32 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w40);
      xnn_prefetch_to_l1((const int8_t*) w40 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w40 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w48);
      xnn_prefetch_to_l1((const int8_t*) w48 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w48 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w56);
      xnn_prefetch_to_l1((const int8_t*) w56 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w56 + stride7) + 64);

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        const __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        const __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) (w0 + stride1));
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) (w8 + stride1));
        const __m256i v17 = _mm256_loadu_si256((const __m256i*) (w16 + stride1));
        const __m256i v25 = _mm256_loadu_si256((const __m256i*) (w24 + stride1));
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) (w0 + stride2));
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) (w8 + stride2));
        const __m256i v18 = _mm256_loadu_si256((const __m256i*) (w16 + stride2));
        const __m256i v26 = _mm256_loadu_si256((const __m256i*) (w24 + stride2));
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) (w0 + stride3));
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) (w8 + stride3));
        const __m256i v19 = _mm256_loadu_si256((const __m256i*) (w16 + stride3));
        const __m256i v27 = _mm256_loadu_si256((const __m256i*) (w24 + stride3));
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) (w0 + stride4));
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) (w8 + stride4));
        const __m256i v20 = _mm256_loadu_si256((const __m256i*) (w16 + stride4));
        const __m256i v28 = _mm256_loadu_si256((const __m256i*) (w24 + stride4));
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) (w0 + stride5));
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) (w8 + stride5));
        const __m256i v21 = _mm256_loadu_si256((const __m256i*) (w16 + stride5));
        const __m256i v29 = _mm256_loadu_si256((const __m256i*) (w24 + stride5));
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) (w0 + stride6));
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) (w8 + stride6));
        const __m256i v22 = _mm256_loadu_si256((const __m256i*) (w16 + stride6));
        const __m256i v30 = _mm256_loadu_si256((const __m256i*) (w24 + stride6));
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) (w0 + stride7));
        const __m256i v15 = _mm256_loadu_si256((const __m256i*) (w8 + stride7));
        const __m256i v23 = _mm256_loadu_si256((const __m256i*) (w16 + stride7));
        const __m256i v31 = _mm256_loadu_si256((const __m256i*) (w24 + stride7));
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);
        w0 += 16;
        w8 += 16;
        w16 += 16;
        w24 += 16;
        const __m256i v32 = _mm256_loadu_si256((const __m256i*) w32);
        const __m256i v40 = _mm256_loadu_si256((const __m256i*) w40);
        const __m256i v48 = _mm256_loadu_si256((const __m256i*) w48);
        const __m256i v56 = _mm256_loadu_si256((const __m256i*) w56);
        __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v40, 1);
        __m512i z40 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v56, 1);
        const __m256i v33 = _mm256_loadu_si256((const __m256i*) (w32 + stride1));
        const __m256i v41 = _mm256_loadu_si256((const __m256i*) (w40 + stride1));
        const __m256i v49 = _mm256_loadu_si256((const __m256i*) (w48 + stride1));
        const __m256i v57 = _mm256_loadu_si256((const __m256i*) (w56 + stride1));
        __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v41, 1);
        __m512i z41 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v57, 1);
        const __m256i v34 = _mm256_loadu_si256((const __m256i*) (w32 + stride2));
        const __m256i v42 = _mm256_loadu_si256((const __m256i*) (w40 + stride2));
        const __m256i v50 = _mm256_loadu_si256((const __m256i*) (w48 + stride2));
        const __m256i v58 = _mm256_loadu_si256((const __m256i*) (w56 + stride2));
        __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v42, 1);
        __m512i z42 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v58, 1);
        const __m256i v35 = _mm256_loadu_si256((const __m256i*) (w32 + stride3));
        const __m256i v43 = _mm256_loadu_si256((const __m256i*) (w40 + stride3));
        const __m256i v51 = _mm256_loadu_si256((const __m256i*) (w48 + stride3));
        const __m256i v59 = _mm256_loadu_si256((const __m256i*) (w56 + stride3));
        __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v43, 1);
        __m512i z43 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v59, 1);
        const __m256i v36 = _mm256_loadu_si256((const __m256i*) (w32 + stride4));
        const __m256i v44 = _mm256_loadu_si256((const __m256i*) (w40 + stride4));
        const __m256i v52 = _mm256_loadu_si256((const __m256i*) (w48 + stride4));
        const __m256i v60 = _mm256_loadu_si256((const __m256i*) (w56 + stride4));
        __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v36), v44, 1);
        __m512i z44 = _mm512_inserti64x4(_mm512_castsi256_si512(v52), v60, 1);
        const __m256i v37 = _mm256_loadu_si256((const __m256i*) (w32 + stride5));
        const __m256i v45 = _mm256_loadu_si256((const __m256i*) (w40 + stride5));
        const __m256i v53 = _mm256_loadu_si256((const __m256i*) (w48 + stride5));
        const __m256i v61 = _mm256_loadu_si256((const __m256i*) (w56 + stride5));
        __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v37), v45, 1);
        __m512i z45 = _mm512_inserti64x4(_mm512_castsi256_si512(v53), v61, 1);
        const __m256i v38 = _mm256_loadu_si256((const __m256i*) (w32 + stride6));
        const __m256i v46 = _mm256_loadu_si256((const __m256i*) (w40 + stride6));
        const __m256i v54 = _mm256_loadu_si256((const __m256i*) (w48 + stride6));
        const __m256i v62 = _mm256_loadu_si256((const __m256i*) (w56 + stride6));
        __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v38), v46, 1);
        __m512i z46 = _mm512_inserti64x4(_mm512_castsi256_si512(v54), v62, 1);
        const __m256i v39 = _mm256_loadu_si256((const __m256i*) (w32 + stride7));
        const __m256i v47 = _mm256_loadu_si256((const __m256i*) (w40 + stride7));
        const __m256i v55 = _mm256_loadu_si256((const __m256i*) (w48 + stride7));
        const __m256i v63 = _mm256_loadu_si256((const __m256i*) (w56 + stride7));
        __m512i z39 = _mm512_inserti64x4(_mm512_castsi256_si512(v39), v47, 1);
        __m512i z47 = _mm512_inserti64x4(_mm512_castsi256_si512(v55), v63, 1);
        w32 += 16;
        w40 += 16;
        w48 += 16;
        w56 += 16;

        const __m512i t0 = _mm512_unpacklo_epi16(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi16(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi16(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi16(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi16(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi16(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi16(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi16(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi32(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi32(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi32(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi32(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi32(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi32(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi32(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi32(t5, t7);

        const __m512i w0 = _mm512_unpacklo_epi64(u0, u4);
        const __m512i w1 = _mm512_unpackhi_epi64(u0, u4);
        const __m512i w2 = _mm512_unpacklo_epi64(u1, u5);
        const __m512i w3 = _mm512_unpackhi_epi64(u1, u5);
        const __m512i w4 = _mm512_unpacklo_epi64(u2, u6);
        const __m512i w5 = _mm512_unpackhi_epi64(u2, u6);
        const __m512i w6 = _mm512_unpacklo_epi64(u3, u7);
        const __m512i w7 = _mm512_unpackhi_epi64(u3, u7);
        const __m512i t8 = _mm512_unpacklo_epi16(z8, z9);
        const __m512i t9 = _mm512_unpackhi_epi16(z8, z9);
        const __m512i t10 = _mm512_unpacklo_epi16(z10, z11);
        const __m512i t11 = _mm512_unpackhi_epi16(z10, z11);
        const __m512i t12 = _mm512_unpacklo_epi16(z12, z13);
        const __m512i t13 = _mm512_unpackhi_epi16(z12, z13);
        const __m512i t14 = _mm512_unpacklo_epi16(z14, z15);
        const __m512i t15 = _mm512_unpackhi_epi16(z14, z15);

        const __m512i u8 = _mm512_unpacklo_epi32(t8, t10);
        const __m512i u9 = _mm512_unpackhi_epi32(t8, t10);
        const __m512i u10 = _mm512_unpacklo_epi32(t9, t11);
        const __m512i u11 = _mm512_unpackhi_epi32(t9, t11);
        const __m512i u12 = _mm512_unpacklo_epi32(t12, t14);
        const __m512i u13 = _mm512_unpackhi_epi32(t12, t14);
        const __m512i u14 = _mm512_unpacklo_epi32(t13, t15);
        const __m512i u15 = _mm512_unpackhi_epi32(t13, t15);

        const __m512i w8 = _mm512_unpacklo_epi64(u8, u12);
        const __m512i w9 = _mm512_unpackhi_epi64(u8, u12);
        const __m512i w10 = _mm512_unpacklo_epi64(u9, u13);
        const __m512i w11 = _mm512_unpackhi_epi64(u9, u13);
        const __m512i w12 = _mm512_unpacklo_epi64(u10, u14);
        const __m512i w13 = _mm512_unpackhi_epi64(u10, u14);
        const __m512i w14 = _mm512_unpacklo_epi64(u11, u15);
        const __m512i w15 = _mm512_unpackhi_epi64(u11, u15);

        const __m512i out0_0 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_8 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 512), out0_8);
        const __m512i out0_1 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_9 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 576), out0_9);
        const __m512i out0_2 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_10 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 640), out0_10);
        const __m512i out0_3 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_11 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 704), out0_11);
        const __m512i out0_4 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_12 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_4);
        _mm512_storeu_si512((__m512i*) (packed_w + 768), out0_12);
        const __m512i out0_5 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_13 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 320), out0_5);
        _mm512_storeu_si512((__m512i*) (packed_w + 832), out0_13);
        const __m512i out0_6 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_14 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_6);
        _mm512_storeu_si512((__m512i*) (packed_w + 896), out0_14);
        const __m512i out0_7 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_15 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 448), out0_7);
        _mm512_storeu_si512((__m512i*) (packed_w + 960), out0_15);
        const __m512i t32 = _mm512_unpacklo_epi16(z32, z33);
        const __m512i t33 = _mm512_unpackhi_epi16(z32, z33);
        const __m512i t34 = _mm512_unpacklo_epi16(z34, z35);
        const __m512i t35 = _mm512_unpackhi_epi16(z34, z35);
        const __m512i t36 = _mm512_unpacklo_epi16(z36, z37);
        const __m512i t37 = _mm512_unpackhi_epi16(z36, z37);
        const __m512i t38 = _mm512_unpacklo_epi16(z38, z39);
        const __m512i t39 = _mm512_unpackhi_epi16(z38, z39);

        const __m512i u32 = _mm512_unpacklo_epi32(t32, t34);
        const __m512i u33 = _mm512_unpackhi_epi32(t32, t34);
        const __m512i u34 = _mm512_unpacklo_epi32(t33, t35);
        const __m512i u35 = _mm512_unpackhi_epi32(t33, t35);
        const __m512i u36 = _mm512_unpacklo_epi32(t36, t38);
        const __m512i u37 = _mm512_unpackhi_epi32(t36, t38);
        const __m512i u38 = _mm512_unpacklo_epi32(t37, t39);
        const __m512i u39 = _mm512_unpackhi_epi32(t37, t39);

        const __m512i w32 = _mm512_unpacklo_epi64(u32, u36);
        const __m512i w33 = _mm512_unpackhi_epi64(u32, u36);
        const __m512i w34 = _mm512_unpacklo_epi64(u33, u37);
        const __m512i w35 = _mm512_unpackhi_epi64(u33, u37);
        const __m512i w36 = _mm512_unpacklo_epi64(u34, u38);
        const __m512i w37 = _mm512_unpackhi_epi64(u34, u38);
        const __m512i w38 = _mm512_unpacklo_epi64(u35, u39);
        const __m512i w39 = _mm512_unpackhi_epi64(u35, u39);
        const __m512i t40 = _mm512_unpacklo_epi16(z40, z41);
        const __m512i t41 = _mm512_unpackhi_epi16(z40, z41);
        const __m512i t42 = _mm512_unpacklo_epi16(z42, z43);
        const __m512i t43 = _mm512_unpackhi_epi16(z42, z43);
        const __m512i t44 = _mm512_unpacklo_epi16(z44, z45);
        const __m512i t45 = _mm512_unpackhi_epi16(z44, z45);
        const __m512i t46 = _mm512_unpacklo_epi16(z46, z47);
        const __m512i t47 = _mm512_unpackhi_epi16(z46, z47);

        const __m512i u40 = _mm512_unpacklo_epi32(t40, t42);
        const __m512i u41 = _mm512_unpackhi_epi32(t40, t42);
        const __m512i u42 = _mm512_unpacklo_epi32(t41, t43);
        const __m512i u43 = _mm512_unpackhi_epi32(t41, t43);
        const __m512i u44 = _mm512_unpacklo_epi32(t44, t46);
        const __m512i u45 = _mm512_unpackhi_epi32(t44, t46);
        const __m512i u46 = _mm512_unpacklo_epi32(t45, t47);
        const __m512i u47 = _mm512_unpackhi_epi32(t45, t47);

        const __m512i w40 = _mm512_unpacklo_epi64(u40, u44);
        const __m512i w41 = _mm512_unpackhi_epi64(u40, u44);
        const __m512i w42 = _mm512_unpacklo_epi64(u41, u45);
        const __m512i w43 = _mm512_unpackhi_epi64(u41, u45);
        const __m512i w44 = _mm512_unpacklo_epi64(u42, u46);
        const __m512i w45 = _mm512_unpackhi_epi64(u42, u46);
        const __m512i w46 = _mm512_unpacklo_epi64(u43, u47);
        const __m512i w47 = _mm512_unpackhi_epi64(u43, u47);

        const __m512i out32_0 = _mm512_shuffle_i32x4(w32, w40, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_8 = _mm512_shuffle_i32x4(w32, w40, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out32_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 544), out32_8);
        const __m512i out32_1 = _mm512_shuffle_i32x4(w33, w41, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_9 = _mm512_shuffle_i32x4(w33, w41, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out32_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 608), out32_9);
        const __m512i out32_2 = _mm512_shuffle_i32x4(w34, w42, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_10 = _mm512_shuffle_i32x4(w34, w42, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 160), out32_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 672), out32_10);
        const __m512i out32_3 = _mm512_shuffle_i32x4(w35, w43, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_11 = _mm512_shuffle_i32x4(w35, w43, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 224), out32_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 736), out32_11);
        const __m512i out32_4 = _mm512_shuffle_i32x4(w36, w44, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_12 = _mm512_shuffle_i32x4(w36, w44, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 288), out32_4);
        _mm512_storeu_si512((__m512i*) (packed_w + 800), out32_12);
        const __m512i out32_5 = _mm512_shuffle_i32x4(w37, w45, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_13 = _mm512_shuffle_i32x4(w37, w45, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 352), out32_5);
        _mm512_storeu_si512((__m512i*) (packed_w + 864), out32_13);
        const __m512i out32_6 = _mm512_shuffle_i32x4(w38, w46, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_14 = _mm512_shuffle_i32x4(w38, w46, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 416), out32_6);
        _mm512_storeu_si512((__m512i*) (packed_w + 928), out32_14);
        const __m512i out32_7 = _mm512_shuffle_i32x4(w39, w47, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_15 = _mm512_shuffle_i32x4(w39, w47, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 480), out32_7);
        _mm512_storeu_si512((__m512i*) (packed_w + 992), out32_15);
        packed_w += 1024;
      }

      // KC remainder (1..15) using AVX-512BW masking
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);
        const __mmask16 kmask = _cvtu32_mask16((1U << k) - 1U);

        const __m256i v0 = _mm256_maskz_loadu_epi16(kmask, w0);
        const __m256i v8 = _mm256_maskz_loadu_epi16(kmask, w8);
        const __m256i v16 = _mm256_maskz_loadu_epi16(kmask, w16);
        const __m256i v24 = _mm256_maskz_loadu_epi16(kmask, w24);
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_maskz_loadu_epi16(kmask, w0 + stride1);
        const __m256i v9 = _mm256_maskz_loadu_epi16(kmask, w8 + stride1);
        const __m256i v17 = _mm256_maskz_loadu_epi16(kmask, w16 + stride1);
        const __m256i v25 = _mm256_maskz_loadu_epi16(kmask, w24 + stride1);
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_maskz_loadu_epi16(kmask, w0 + stride2);
        const __m256i v10 = _mm256_maskz_loadu_epi16(kmask, w8 + stride2);
        const __m256i v18 = _mm256_maskz_loadu_epi16(kmask, w16 + stride2);
        const __m256i v26 = _mm256_maskz_loadu_epi16(kmask, w24 + stride2);
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_maskz_loadu_epi16(kmask, w0 + stride3);
        const __m256i v11 = _mm256_maskz_loadu_epi16(kmask, w8 + stride3);
        const __m256i v19 = _mm256_maskz_loadu_epi16(kmask, w16 + stride3);
        const __m256i v27 = _mm256_maskz_loadu_epi16(kmask, w24 + stride3);
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_maskz_loadu_epi16(kmask, w0 + stride4);
        const __m256i v12 = _mm256_maskz_loadu_epi16(kmask, w8 + stride4);
        const __m256i v20 = _mm256_maskz_loadu_epi16(kmask, w16 + stride4);
        const __m256i v28 = _mm256_maskz_loadu_epi16(kmask, w24 + stride4);
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_maskz_loadu_epi16(kmask, w0 + stride5);
        const __m256i v13 = _mm256_maskz_loadu_epi16(kmask, w8 + stride5);
        const __m256i v21 = _mm256_maskz_loadu_epi16(kmask, w16 + stride5);
        const __m256i v29 = _mm256_maskz_loadu_epi16(kmask, w24 + stride5);
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_maskz_loadu_epi16(kmask, w0 + stride6);
        const __m256i v14 = _mm256_maskz_loadu_epi16(kmask, w8 + stride6);
        const __m256i v22 = _mm256_maskz_loadu_epi16(kmask, w16 + stride6);
        const __m256i v30 = _mm256_maskz_loadu_epi16(kmask, w24 + stride6);
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_maskz_loadu_epi16(kmask, w0 + stride7);
        const __m256i v15 = _mm256_maskz_loadu_epi16(kmask, w8 + stride7);
        const __m256i v23 = _mm256_maskz_loadu_epi16(kmask, w16 + stride7);
        const __m256i v31 = _mm256_maskz_loadu_epi16(kmask, w24 + stride7);
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);
        w0 += k;
        w8 += k;
        w16 += k;
        w24 += k;
        const __m256i v32 = _mm256_maskz_loadu_epi16(kmask, w32);
        const __m256i v40 = _mm256_maskz_loadu_epi16(kmask, w40);
        const __m256i v48 = _mm256_maskz_loadu_epi16(kmask, w48);
        const __m256i v56 = _mm256_maskz_loadu_epi16(kmask, w56);
        __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v40, 1);
        __m512i z40 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v56, 1);
        const __m256i v33 = _mm256_maskz_loadu_epi16(kmask, w32 + stride1);
        const __m256i v41 = _mm256_maskz_loadu_epi16(kmask, w40 + stride1);
        const __m256i v49 = _mm256_maskz_loadu_epi16(kmask, w48 + stride1);
        const __m256i v57 = _mm256_maskz_loadu_epi16(kmask, w56 + stride1);
        __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v41, 1);
        __m512i z41 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v57, 1);
        const __m256i v34 = _mm256_maskz_loadu_epi16(kmask, w32 + stride2);
        const __m256i v42 = _mm256_maskz_loadu_epi16(kmask, w40 + stride2);
        const __m256i v50 = _mm256_maskz_loadu_epi16(kmask, w48 + stride2);
        const __m256i v58 = _mm256_maskz_loadu_epi16(kmask, w56 + stride2);
        __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v42, 1);
        __m512i z42 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v58, 1);
        const __m256i v35 = _mm256_maskz_loadu_epi16(kmask, w32 + stride3);
        const __m256i v43 = _mm256_maskz_loadu_epi16(kmask, w40 + stride3);
        const __m256i v51 = _mm256_maskz_loadu_epi16(kmask, w48 + stride3);
        const __m256i v59 = _mm256_maskz_loadu_epi16(kmask, w56 + stride3);
        __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v43, 1);
        __m512i z43 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v59, 1);
        const __m256i v36 = _mm256_maskz_loadu_epi16(kmask, w32 + stride4);
        const __m256i v44 = _mm256_maskz_loadu_epi16(kmask, w40 + stride4);
        const __m256i v52 = _mm256_maskz_loadu_epi16(kmask, w48 + stride4);
        const __m256i v60 = _mm256_maskz_loadu_epi16(kmask, w56 + stride4);
        __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v36), v44, 1);
        __m512i z44 = _mm512_inserti64x4(_mm512_castsi256_si512(v52), v60, 1);
        const __m256i v37 = _mm256_maskz_loadu_epi16(kmask, w32 + stride5);
        const __m256i v45 = _mm256_maskz_loadu_epi16(kmask, w40 + stride5);
        const __m256i v53 = _mm256_maskz_loadu_epi16(kmask, w48 + stride5);
        const __m256i v61 = _mm256_maskz_loadu_epi16(kmask, w56 + stride5);
        __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v37), v45, 1);
        __m512i z45 = _mm512_inserti64x4(_mm512_castsi256_si512(v53), v61, 1);
        const __m256i v38 = _mm256_maskz_loadu_epi16(kmask, w32 + stride6);
        const __m256i v46 = _mm256_maskz_loadu_epi16(kmask, w40 + stride6);
        const __m256i v54 = _mm256_maskz_loadu_epi16(kmask, w48 + stride6);
        const __m256i v62 = _mm256_maskz_loadu_epi16(kmask, w56 + stride6);
        __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v38), v46, 1);
        __m512i z46 = _mm512_inserti64x4(_mm512_castsi256_si512(v54), v62, 1);
        const __m256i v39 = _mm256_maskz_loadu_epi16(kmask, w32 + stride7);
        const __m256i v47 = _mm256_maskz_loadu_epi16(kmask, w40 + stride7);
        const __m256i v55 = _mm256_maskz_loadu_epi16(kmask, w48 + stride7);
        const __m256i v63 = _mm256_maskz_loadu_epi16(kmask, w56 + stride7);
        __m512i z39 = _mm512_inserti64x4(_mm512_castsi256_si512(v39), v47, 1);
        __m512i z47 = _mm512_inserti64x4(_mm512_castsi256_si512(v55), v63, 1);
        w32 += k;
        w40 += k;
        w48 += k;
        w56 += k;

        const __m512i t0 = _mm512_unpacklo_epi16(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi16(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi16(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi16(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi16(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi16(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi16(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi16(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi32(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi32(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi32(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi32(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi32(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi32(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi32(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi32(t5, t7);

        const __m512i w0 = _mm512_unpacklo_epi64(u0, u4);
        const __m512i w1 = _mm512_unpackhi_epi64(u0, u4);
        const __m512i w2 = _mm512_unpacklo_epi64(u1, u5);
        const __m512i w3 = _mm512_unpackhi_epi64(u1, u5);
        const __m512i w4 = _mm512_unpacklo_epi64(u2, u6);
        const __m512i w5 = _mm512_unpackhi_epi64(u2, u6);
        const __m512i w6 = _mm512_unpacklo_epi64(u3, u7);
        const __m512i w7 = _mm512_unpackhi_epi64(u3, u7);
        const __m512i t8 = _mm512_unpacklo_epi16(z8, z9);
        const __m512i t9 = _mm512_unpackhi_epi16(z8, z9);
        const __m512i t10 = _mm512_unpacklo_epi16(z10, z11);
        const __m512i t11 = _mm512_unpackhi_epi16(z10, z11);
        const __m512i t12 = _mm512_unpacklo_epi16(z12, z13);
        const __m512i t13 = _mm512_unpackhi_epi16(z12, z13);
        const __m512i t14 = _mm512_unpacklo_epi16(z14, z15);
        const __m512i t15 = _mm512_unpackhi_epi16(z14, z15);

        const __m512i u8 = _mm512_unpacklo_epi32(t8, t10);
        const __m512i u9 = _mm512_unpackhi_epi32(t8, t10);
        const __m512i u10 = _mm512_unpacklo_epi32(t9, t11);
        const __m512i u11 = _mm512_unpackhi_epi32(t9, t11);
        const __m512i u12 = _mm512_unpacklo_epi32(t12, t14);
        const __m512i u13 = _mm512_unpackhi_epi32(t12, t14);
        const __m512i u14 = _mm512_unpacklo_epi32(t13, t15);
        const __m512i u15 = _mm512_unpackhi_epi32(t13, t15);

        const __m512i w8 = _mm512_unpacklo_epi64(u8, u12);
        const __m512i w9 = _mm512_unpackhi_epi64(u8, u12);
        const __m512i w10 = _mm512_unpacklo_epi64(u9, u13);
        const __m512i w11 = _mm512_unpackhi_epi64(u9, u13);
        const __m512i w12 = _mm512_unpacklo_epi64(u10, u14);
        const __m512i w13 = _mm512_unpackhi_epi64(u10, u14);
        const __m512i w14 = _mm512_unpacklo_epi64(u11, u15);
        const __m512i w15 = _mm512_unpackhi_epi64(u11, u15);

        const __m512i out0_0 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 0) {
          _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        }
        const __m512i out0_8 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 512), out0_8);
        }
        const __m512i out0_1 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 1) {
          _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_1);
        }
        const __m512i out0_9 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 9) {
          _mm512_storeu_si512((__m512i*) (packed_w + 576), out0_9);
        }
        const __m512i out0_2 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_2);
        }
        const __m512i out0_10 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 640), out0_10);
        }
        const __m512i out0_3 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 3) {
          _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_3);
        }
        const __m512i out0_11 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 11) {
          _mm512_storeu_si512((__m512i*) (packed_w + 704), out0_11);
        }
        const __m512i out0_4 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_4);
        }
        const __m512i out0_12 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 768), out0_12);
        }
        const __m512i out0_5 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 5) {
          _mm512_storeu_si512((__m512i*) (packed_w + 320), out0_5);
        }
        const __m512i out0_13 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 13) {
          _mm512_storeu_si512((__m512i*) (packed_w + 832), out0_13);
        }
        const __m512i out0_6 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_6);
        }
        const __m512i out0_14 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 896), out0_14);
        }
        const __m512i out0_7 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 7) {
          _mm512_storeu_si512((__m512i*) (packed_w + 448), out0_7);
        }
        const __m512i out0_15 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 15) {
          _mm512_storeu_si512((__m512i*) (packed_w + 960), out0_15);
        }
        const __m512i t32 = _mm512_unpacklo_epi16(z32, z33);
        const __m512i t33 = _mm512_unpackhi_epi16(z32, z33);
        const __m512i t34 = _mm512_unpacklo_epi16(z34, z35);
        const __m512i t35 = _mm512_unpackhi_epi16(z34, z35);
        const __m512i t36 = _mm512_unpacklo_epi16(z36, z37);
        const __m512i t37 = _mm512_unpackhi_epi16(z36, z37);
        const __m512i t38 = _mm512_unpacklo_epi16(z38, z39);
        const __m512i t39 = _mm512_unpackhi_epi16(z38, z39);

        const __m512i u32 = _mm512_unpacklo_epi32(t32, t34);
        const __m512i u33 = _mm512_unpackhi_epi32(t32, t34);
        const __m512i u34 = _mm512_unpacklo_epi32(t33, t35);
        const __m512i u35 = _mm512_unpackhi_epi32(t33, t35);
        const __m512i u36 = _mm512_unpacklo_epi32(t36, t38);
        const __m512i u37 = _mm512_unpackhi_epi32(t36, t38);
        const __m512i u38 = _mm512_unpacklo_epi32(t37, t39);
        const __m512i u39 = _mm512_unpackhi_epi32(t37, t39);

        const __m512i w32 = _mm512_unpacklo_epi64(u32, u36);
        const __m512i w33 = _mm512_unpackhi_epi64(u32, u36);
        const __m512i w34 = _mm512_unpacklo_epi64(u33, u37);
        const __m512i w35 = _mm512_unpackhi_epi64(u33, u37);
        const __m512i w36 = _mm512_unpacklo_epi64(u34, u38);
        const __m512i w37 = _mm512_unpackhi_epi64(u34, u38);
        const __m512i w38 = _mm512_unpacklo_epi64(u35, u39);
        const __m512i w39 = _mm512_unpackhi_epi64(u35, u39);
        const __m512i t40 = _mm512_unpacklo_epi16(z40, z41);
        const __m512i t41 = _mm512_unpackhi_epi16(z40, z41);
        const __m512i t42 = _mm512_unpacklo_epi16(z42, z43);
        const __m512i t43 = _mm512_unpackhi_epi16(z42, z43);
        const __m512i t44 = _mm512_unpacklo_epi16(z44, z45);
        const __m512i t45 = _mm512_unpackhi_epi16(z44, z45);
        const __m512i t46 = _mm512_unpacklo_epi16(z46, z47);
        const __m512i t47 = _mm512_unpackhi_epi16(z46, z47);

        const __m512i u40 = _mm512_unpacklo_epi32(t40, t42);
        const __m512i u41 = _mm512_unpackhi_epi32(t40, t42);
        const __m512i u42 = _mm512_unpacklo_epi32(t41, t43);
        const __m512i u43 = _mm512_unpackhi_epi32(t41, t43);
        const __m512i u44 = _mm512_unpacklo_epi32(t44, t46);
        const __m512i u45 = _mm512_unpackhi_epi32(t44, t46);
        const __m512i u46 = _mm512_unpacklo_epi32(t45, t47);
        const __m512i u47 = _mm512_unpackhi_epi32(t45, t47);

        const __m512i w40 = _mm512_unpacklo_epi64(u40, u44);
        const __m512i w41 = _mm512_unpackhi_epi64(u40, u44);
        const __m512i w42 = _mm512_unpacklo_epi64(u41, u45);
        const __m512i w43 = _mm512_unpackhi_epi64(u41, u45);
        const __m512i w44 = _mm512_unpacklo_epi64(u42, u46);
        const __m512i w45 = _mm512_unpackhi_epi64(u42, u46);
        const __m512i w46 = _mm512_unpacklo_epi64(u43, u47);
        const __m512i w47 = _mm512_unpackhi_epi64(u43, u47);

        const __m512i out32_0 = _mm512_shuffle_i32x4(w32, w40, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 0) {
          _mm512_storeu_si512((__m512i*) (packed_w + 32), out32_0);
        }
        const __m512i out32_8 = _mm512_shuffle_i32x4(w32, w40, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 544), out32_8);
        }
        const __m512i out32_1 = _mm512_shuffle_i32x4(w33, w41, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 1) {
          _mm512_storeu_si512((__m512i*) (packed_w + 96), out32_1);
        }
        const __m512i out32_9 = _mm512_shuffle_i32x4(w33, w41, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 9) {
          _mm512_storeu_si512((__m512i*) (packed_w + 608), out32_9);
        }
        const __m512i out32_2 = _mm512_shuffle_i32x4(w34, w42, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 160), out32_2);
        }
        const __m512i out32_10 = _mm512_shuffle_i32x4(w34, w42, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 672), out32_10);
        }
        const __m512i out32_3 = _mm512_shuffle_i32x4(w35, w43, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 3) {
          _mm512_storeu_si512((__m512i*) (packed_w + 224), out32_3);
        }
        const __m512i out32_11 = _mm512_shuffle_i32x4(w35, w43, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 11) {
          _mm512_storeu_si512((__m512i*) (packed_w + 736), out32_11);
        }
        const __m512i out32_4 = _mm512_shuffle_i32x4(w36, w44, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 288), out32_4);
        }
        const __m512i out32_12 = _mm512_shuffle_i32x4(w36, w44, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 800), out32_12);
        }
        const __m512i out32_5 = _mm512_shuffle_i32x4(w37, w45, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 5) {
          _mm512_storeu_si512((__m512i*) (packed_w + 352), out32_5);
        }
        const __m512i out32_13 = _mm512_shuffle_i32x4(w37, w45, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 13) {
          _mm512_storeu_si512((__m512i*) (packed_w + 864), out32_13);
        }
        const __m512i out32_6 = _mm512_shuffle_i32x4(w38, w46, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 416), out32_6);
        }
        const __m512i out32_14 = _mm512_shuffle_i32x4(w38, w46, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 928), out32_14);
        }
        const __m512i out32_7 = _mm512_shuffle_i32x4(w39, w47, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 7) {
          _mm512_storeu_si512((__m512i*) (packed_w + 480), out32_7);
        }
        const __m512i out32_15 = _mm512_shuffle_i32x4(w39, w47, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 15) {
          _mm512_storeu_si512((__m512i*) (packed_w + 992), out32_15);
        }
        packed_w += 64 * k;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w_base += 64 * stride1;
    }

    // NC remainder (1..63)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 63);
      const uint16_t* w0 = w_base;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++ = *b++;
        } while (--nb != 0);
        packed_w += (64 - n);
      } else {
        const __m512i vzero = _mm512_setzero_si512();
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vzero);
        _mm512_storeu_si512((__m512i*) (packed_w + 32), vzero);
        packed_w += 64;
      }

      // NR remainder has less than 64 rows so last row is not loaded
      const uint16_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint16_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint16_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint16_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint16_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint16_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint16_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint16_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint16_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint16_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint16_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint16_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint16_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint16_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const uint16_t* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const uint16_t* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const uint16_t* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const uint16_t* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const uint16_t* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const uint16_t* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const uint16_t* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const uint16_t* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const uint16_t* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const uint16_t* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const uint16_t* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const uint16_t* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const uint16_t* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const uint16_t* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const uint16_t* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const uint16_t* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }
      const uint16_t* w31 = w30 + kc;
      if XNN_UNPREDICTABLE(n < 32) {
        w31 = w30;
      }
      const uint16_t* w32 = w31 + kc;
      if XNN_UNPREDICTABLE(n <= 32) {
        w32 = w31;
      }
      const uint16_t* w33 = w32 + kc;
      if XNN_UNPREDICTABLE(n < 34) {
        w33 = w32;
      }
      const uint16_t* w34 = w33 + kc;
      if XNN_UNPREDICTABLE(n <= 34) {
        w34 = w33;
      }
      const uint16_t* w35 = w34 + kc;
      if XNN_UNPREDICTABLE(n < 36) {
        w35 = w34;
      }
      const uint16_t* w36 = w35 + kc;
      if XNN_UNPREDICTABLE(n <= 36) {
        w36 = w35;
      }
      const uint16_t* w37 = w36 + kc;
      if XNN_UNPREDICTABLE(n < 38) {
        w37 = w36;
      }
      const uint16_t* w38 = w37 + kc;
      if XNN_UNPREDICTABLE(n <= 38) {
        w38 = w37;
      }
      const uint16_t* w39 = w38 + kc;
      if XNN_UNPREDICTABLE(n < 40) {
        w39 = w38;
      }
      const uint16_t* w40 = w39 + kc;
      if XNN_UNPREDICTABLE(n <= 40) {
        w40 = w39;
      }
      const uint16_t* w41 = w40 + kc;
      if XNN_UNPREDICTABLE(n < 42) {
        w41 = w40;
      }
      const uint16_t* w42 = w41 + kc;
      if XNN_UNPREDICTABLE(n <= 42) {
        w42 = w41;
      }
      const uint16_t* w43 = w42 + kc;
      if XNN_UNPREDICTABLE(n < 44) {
        w43 = w42;
      }
      const uint16_t* w44 = w43 + kc;
      if XNN_UNPREDICTABLE(n <= 44) {
        w44 = w43;
      }
      const uint16_t* w45 = w44 + kc;
      if XNN_UNPREDICTABLE(n < 46) {
        w45 = w44;
      }
      const uint16_t* w46 = w45 + kc;
      if XNN_UNPREDICTABLE(n <= 46) {
        w46 = w45;
      }
      const uint16_t* w47 = w46 + kc;
      if XNN_UNPREDICTABLE(n < 48) {
        w47 = w46;
      }
      const uint16_t* w48 = w47 + kc;
      if XNN_UNPREDICTABLE(n <= 48) {
        w48 = w47;
      }
      const uint16_t* w49 = w48 + kc;
      if XNN_UNPREDICTABLE(n < 50) {
        w49 = w48;
      }
      const uint16_t* w50 = w49 + kc;
      if XNN_UNPREDICTABLE(n <= 50) {
        w50 = w49;
      }
      const uint16_t* w51 = w50 + kc;
      if XNN_UNPREDICTABLE(n < 52) {
        w51 = w50;
      }
      const uint16_t* w52 = w51 + kc;
      if XNN_UNPREDICTABLE(n <= 52) {
        w52 = w51;
      }
      const uint16_t* w53 = w52 + kc;
      if XNN_UNPREDICTABLE(n < 54) {
        w53 = w52;
      }
      const uint16_t* w54 = w53 + kc;
      if XNN_UNPREDICTABLE(n <= 54) {
        w54 = w53;
      }
      const uint16_t* w55 = w54 + kc;
      if XNN_UNPREDICTABLE(n < 56) {
        w55 = w54;
      }
      const uint16_t* w56 = w55 + kc;
      if XNN_UNPREDICTABLE(n <= 56) {
        w56 = w55;
      }
      const uint16_t* w57 = w56 + kc;
      if XNN_UNPREDICTABLE(n < 58) {
        w57 = w56;
      }
      const uint16_t* w58 = w57 + kc;
      if XNN_UNPREDICTABLE(n <= 58) {
        w58 = w57;
      }
      const uint16_t* w59 = w58 + kc;
      if XNN_UNPREDICTABLE(n < 60) {
        w59 = w58;
      }
      const uint16_t* w60 = w59 + kc;
      if XNN_UNPREDICTABLE(n <= 60) {
        w60 = w59;
      }
      const uint16_t* w61 = w60 + kc;
      if XNN_UNPREDICTABLE(n < 62) {
        w61 = w60;
      }
      const uint16_t* w62 = w61 + kc;
      if XNN_UNPREDICTABLE(n <= 62) {
        w62 = w61;
      }

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        w0 += 16;
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        w8 += 16;
        const __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        w16 += 16;
        const __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        w24 += 16;
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        w1 += 16;
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        w9 += 16;
        const __m256i v17 = _mm256_loadu_si256((const __m256i*) w17);
        w17 += 16;
        const __m256i v25 = _mm256_loadu_si256((const __m256i*) w25);
        w25 += 16;
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        w2 += 16;
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        w10 += 16;
        const __m256i v18 = _mm256_loadu_si256((const __m256i*) w18);
        w18 += 16;
        const __m256i v26 = _mm256_loadu_si256((const __m256i*) w26);
        w26 += 16;
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        w3 += 16;
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        w11 += 16;
        const __m256i v19 = _mm256_loadu_si256((const __m256i*) w19);
        w19 += 16;
        const __m256i v27 = _mm256_loadu_si256((const __m256i*) w27);
        w27 += 16;
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        w4 += 16;
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        w12 += 16;
        const __m256i v20 = _mm256_loadu_si256((const __m256i*) w20);
        w20 += 16;
        const __m256i v28 = _mm256_loadu_si256((const __m256i*) w28);
        w28 += 16;
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        w5 += 16;
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        w13 += 16;
        const __m256i v21 = _mm256_loadu_si256((const __m256i*) w21);
        w21 += 16;
        const __m256i v29 = _mm256_loadu_si256((const __m256i*) w29);
        w29 += 16;
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        w6 += 16;
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        w14 += 16;
        const __m256i v22 = _mm256_loadu_si256((const __m256i*) w22);
        w22 += 16;
        const __m256i v30 = _mm256_loadu_si256((const __m256i*) w30);
        w30 += 16;
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        w7 += 16;
        const __m256i v15 = _mm256_loadu_si256((const __m256i*) w15);
        w15 += 16;
        const __m256i v23 = _mm256_loadu_si256((const __m256i*) w23);
        w23 += 16;
        const __m256i v31 = _mm256_loadu_si256((const __m256i*) w31);
        w31 += 16;
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);
        const __m256i v32 = _mm256_loadu_si256((const __m256i*) w32);
        w32 += 16;
        const __m256i v40 = _mm256_loadu_si256((const __m256i*) w40);
        w40 += 16;
        const __m256i v48 = _mm256_loadu_si256((const __m256i*) w48);
        w48 += 16;
        const __m256i v56 = _mm256_loadu_si256((const __m256i*) w56);
        w56 += 16;
        __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v40, 1);
        __m512i z40 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v56, 1);
        const __m256i v33 = _mm256_loadu_si256((const __m256i*) w33);
        w33 += 16;
        const __m256i v41 = _mm256_loadu_si256((const __m256i*) w41);
        w41 += 16;
        const __m256i v49 = _mm256_loadu_si256((const __m256i*) w49);
        w49 += 16;
        const __m256i v57 = _mm256_loadu_si256((const __m256i*) w57);
        w57 += 16;
        __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v41, 1);
        __m512i z41 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v57, 1);
        const __m256i v34 = _mm256_loadu_si256((const __m256i*) w34);
        w34 += 16;
        const __m256i v42 = _mm256_loadu_si256((const __m256i*) w42);
        w42 += 16;
        const __m256i v50 = _mm256_loadu_si256((const __m256i*) w50);
        w50 += 16;
        const __m256i v58 = _mm256_loadu_si256((const __m256i*) w58);
        w58 += 16;
        __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v42, 1);
        __m512i z42 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v58, 1);
        const __m256i v35 = _mm256_loadu_si256((const __m256i*) w35);
        w35 += 16;
        const __m256i v43 = _mm256_loadu_si256((const __m256i*) w43);
        w43 += 16;
        const __m256i v51 = _mm256_loadu_si256((const __m256i*) w51);
        w51 += 16;
        const __m256i v59 = _mm256_loadu_si256((const __m256i*) w59);
        w59 += 16;
        __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v43, 1);
        __m512i z43 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v59, 1);
        const __m256i v36 = _mm256_loadu_si256((const __m256i*) w36);
        w36 += 16;
        const __m256i v44 = _mm256_loadu_si256((const __m256i*) w44);
        w44 += 16;
        const __m256i v52 = _mm256_loadu_si256((const __m256i*) w52);
        w52 += 16;
        const __m256i v60 = _mm256_loadu_si256((const __m256i*) w60);
        w60 += 16;
        __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v36), v44, 1);
        __m512i z44 = _mm512_inserti64x4(_mm512_castsi256_si512(v52), v60, 1);
        const __m256i v37 = _mm256_loadu_si256((const __m256i*) w37);
        w37 += 16;
        const __m256i v45 = _mm256_loadu_si256((const __m256i*) w45);
        w45 += 16;
        const __m256i v53 = _mm256_loadu_si256((const __m256i*) w53);
        w53 += 16;
        const __m256i v61 = _mm256_loadu_si256((const __m256i*) w61);
        w61 += 16;
        __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v37), v45, 1);
        __m512i z45 = _mm512_inserti64x4(_mm512_castsi256_si512(v53), v61, 1);
        const __m256i v38 = _mm256_loadu_si256((const __m256i*) w38);
        w38 += 16;
        const __m256i v46 = _mm256_loadu_si256((const __m256i*) w46);
        w46 += 16;
        const __m256i v54 = _mm256_loadu_si256((const __m256i*) w54);
        w54 += 16;
        const __m256i v62 = _mm256_loadu_si256((const __m256i*) w62);
        w62 += 16;
        __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v38), v46, 1);
        __m512i z46 = _mm512_inserti64x4(_mm512_castsi256_si512(v54), v62, 1);
        const __m256i v39 = _mm256_loadu_si256((const __m256i*) w39);
        w39 += 16;
        const __m256i v47 = _mm256_loadu_si256((const __m256i*) w47);
        w47 += 16;
        const __m256i v55 = _mm256_loadu_si256((const __m256i*) w55);
        w55 += 16;
        const __m256i v63 = _mm256_setzero_si256();
        __m512i z39 = _mm512_inserti64x4(_mm512_castsi256_si512(v39), v47, 1);
        __m512i z47 = _mm512_inserti64x4(_mm512_castsi256_si512(v55), v63, 1);

        const __m512i t0 = _mm512_unpacklo_epi16(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi16(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi16(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi16(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi16(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi16(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi16(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi16(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi32(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi32(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi32(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi32(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi32(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi32(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi32(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi32(t5, t7);

        const __m512i w0 = _mm512_unpacklo_epi64(u0, u4);
        const __m512i w1 = _mm512_unpackhi_epi64(u0, u4);
        const __m512i w2 = _mm512_unpacklo_epi64(u1, u5);
        const __m512i w3 = _mm512_unpackhi_epi64(u1, u5);
        const __m512i w4 = _mm512_unpacklo_epi64(u2, u6);
        const __m512i w5 = _mm512_unpackhi_epi64(u2, u6);
        const __m512i w6 = _mm512_unpacklo_epi64(u3, u7);
        const __m512i w7 = _mm512_unpackhi_epi64(u3, u7);
        const __m512i t8 = _mm512_unpacklo_epi16(z8, z9);
        const __m512i t9 = _mm512_unpackhi_epi16(z8, z9);
        const __m512i t10 = _mm512_unpacklo_epi16(z10, z11);
        const __m512i t11 = _mm512_unpackhi_epi16(z10, z11);
        const __m512i t12 = _mm512_unpacklo_epi16(z12, z13);
        const __m512i t13 = _mm512_unpackhi_epi16(z12, z13);
        const __m512i t14 = _mm512_unpacklo_epi16(z14, z15);
        const __m512i t15 = _mm512_unpackhi_epi16(z14, z15);

        const __m512i u8 = _mm512_unpacklo_epi32(t8, t10);
        const __m512i u9 = _mm512_unpackhi_epi32(t8, t10);
        const __m512i u10 = _mm512_unpacklo_epi32(t9, t11);
        const __m512i u11 = _mm512_unpackhi_epi32(t9, t11);
        const __m512i u12 = _mm512_unpacklo_epi32(t12, t14);
        const __m512i u13 = _mm512_unpackhi_epi32(t12, t14);
        const __m512i u14 = _mm512_unpacklo_epi32(t13, t15);
        const __m512i u15 = _mm512_unpackhi_epi32(t13, t15);

        const __m512i w8 = _mm512_unpacklo_epi64(u8, u12);
        const __m512i w9 = _mm512_unpackhi_epi64(u8, u12);
        const __m512i w10 = _mm512_unpacklo_epi64(u9, u13);
        const __m512i w11 = _mm512_unpackhi_epi64(u9, u13);
        const __m512i w12 = _mm512_unpacklo_epi64(u10, u14);
        const __m512i w13 = _mm512_unpackhi_epi64(u10, u14);
        const __m512i w14 = _mm512_unpacklo_epi64(u11, u15);
        const __m512i w15 = _mm512_unpackhi_epi64(u11, u15);

        const __m512i out0_0 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_8 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 512), out0_8);
        const __m512i out0_1 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_9 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 576), out0_9);
        const __m512i out0_2 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_10 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 640), out0_10);
        const __m512i out0_3 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_11 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 704), out0_11);
        const __m512i out0_4 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_12 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_4);
        _mm512_storeu_si512((__m512i*) (packed_w + 768), out0_12);
        const __m512i out0_5 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_13 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 320), out0_5);
        _mm512_storeu_si512((__m512i*) (packed_w + 832), out0_13);
        const __m512i out0_6 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_14 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_6);
        _mm512_storeu_si512((__m512i*) (packed_w + 896), out0_14);
        const __m512i out0_7 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_15 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 448), out0_7);
        _mm512_storeu_si512((__m512i*) (packed_w + 960), out0_15);
        const __m512i t32 = _mm512_unpacklo_epi16(z32, z33);
        const __m512i t33 = _mm512_unpackhi_epi16(z32, z33);
        const __m512i t34 = _mm512_unpacklo_epi16(z34, z35);
        const __m512i t35 = _mm512_unpackhi_epi16(z34, z35);
        const __m512i t36 = _mm512_unpacklo_epi16(z36, z37);
        const __m512i t37 = _mm512_unpackhi_epi16(z36, z37);
        const __m512i t38 = _mm512_unpacklo_epi16(z38, z39);
        const __m512i t39 = _mm512_unpackhi_epi16(z38, z39);

        const __m512i u32 = _mm512_unpacklo_epi32(t32, t34);
        const __m512i u33 = _mm512_unpackhi_epi32(t32, t34);
        const __m512i u34 = _mm512_unpacklo_epi32(t33, t35);
        const __m512i u35 = _mm512_unpackhi_epi32(t33, t35);
        const __m512i u36 = _mm512_unpacklo_epi32(t36, t38);
        const __m512i u37 = _mm512_unpackhi_epi32(t36, t38);
        const __m512i u38 = _mm512_unpacklo_epi32(t37, t39);
        const __m512i u39 = _mm512_unpackhi_epi32(t37, t39);

        const __m512i w32 = _mm512_unpacklo_epi64(u32, u36);
        const __m512i w33 = _mm512_unpackhi_epi64(u32, u36);
        const __m512i w34 = _mm512_unpacklo_epi64(u33, u37);
        const __m512i w35 = _mm512_unpackhi_epi64(u33, u37);
        const __m512i w36 = _mm512_unpacklo_epi64(u34, u38);
        const __m512i w37 = _mm512_unpackhi_epi64(u34, u38);
        const __m512i w38 = _mm512_unpacklo_epi64(u35, u39);
        const __m512i w39 = _mm512_unpackhi_epi64(u35, u39);
        const __m512i t40 = _mm512_unpacklo_epi16(z40, z41);
        const __m512i t41 = _mm512_unpackhi_epi16(z40, z41);
        const __m512i t42 = _mm512_unpacklo_epi16(z42, z43);
        const __m512i t43 = _mm512_unpackhi_epi16(z42, z43);
        const __m512i t44 = _mm512_unpacklo_epi16(z44, z45);
        const __m512i t45 = _mm512_unpackhi_epi16(z44, z45);
        const __m512i t46 = _mm512_unpacklo_epi16(z46, z47);
        const __m512i t47 = _mm512_unpackhi_epi16(z46, z47);

        const __m512i u40 = _mm512_unpacklo_epi32(t40, t42);
        const __m512i u41 = _mm512_unpackhi_epi32(t40, t42);
        const __m512i u42 = _mm512_unpacklo_epi32(t41, t43);
        const __m512i u43 = _mm512_unpackhi_epi32(t41, t43);
        const __m512i u44 = _mm512_unpacklo_epi32(t44, t46);
        const __m512i u45 = _mm512_unpackhi_epi32(t44, t46);
        const __m512i u46 = _mm512_unpacklo_epi32(t45, t47);
        const __m512i u47 = _mm512_unpackhi_epi32(t45, t47);

        const __m512i w40 = _mm512_unpacklo_epi64(u40, u44);
        const __m512i w41 = _mm512_unpackhi_epi64(u40, u44);
        const __m512i w42 = _mm512_unpacklo_epi64(u41, u45);
        const __m512i w43 = _mm512_unpackhi_epi64(u41, u45);
        const __m512i w44 = _mm512_unpacklo_epi64(u42, u46);
        const __m512i w45 = _mm512_unpackhi_epi64(u42, u46);
        const __m512i w46 = _mm512_unpacklo_epi64(u43, u47);
        const __m512i w47 = _mm512_unpackhi_epi64(u43, u47);

        const __m512i out32_0 = _mm512_shuffle_i32x4(w32, w40, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_8 = _mm512_shuffle_i32x4(w32, w40, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out32_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 544), out32_8);
        const __m512i out32_1 = _mm512_shuffle_i32x4(w33, w41, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_9 = _mm512_shuffle_i32x4(w33, w41, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out32_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 608), out32_9);
        const __m512i out32_2 = _mm512_shuffle_i32x4(w34, w42, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_10 = _mm512_shuffle_i32x4(w34, w42, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 160), out32_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 672), out32_10);
        const __m512i out32_3 = _mm512_shuffle_i32x4(w35, w43, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_11 = _mm512_shuffle_i32x4(w35, w43, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 224), out32_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 736), out32_11);
        const __m512i out32_4 = _mm512_shuffle_i32x4(w36, w44, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_12 = _mm512_shuffle_i32x4(w36, w44, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 288), out32_4);
        _mm512_storeu_si512((__m512i*) (packed_w + 800), out32_12);
        const __m512i out32_5 = _mm512_shuffle_i32x4(w37, w45, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_13 = _mm512_shuffle_i32x4(w37, w45, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 352), out32_5);
        _mm512_storeu_si512((__m512i*) (packed_w + 864), out32_13);
        const __m512i out32_6 = _mm512_shuffle_i32x4(w38, w46, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_14 = _mm512_shuffle_i32x4(w38, w46, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 416), out32_6);
        _mm512_storeu_si512((__m512i*) (packed_w + 928), out32_14);
        const __m512i out32_7 = _mm512_shuffle_i32x4(w39, w47, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_15 = _mm512_shuffle_i32x4(w39, w47, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 480), out32_7);
        _mm512_storeu_si512((__m512i*) (packed_w + 992), out32_15);
        packed_w += 1024;
      }

      // KC remainder (1..15) using AVX-512BW masking
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);
        const __mmask16 kmask = _cvtu32_mask16((1U << k) - 1U);

        const __m256i v0 = _mm256_maskz_loadu_epi16(kmask, w0);
        w0 += k;
        const __m256i v8 = _mm256_maskz_loadu_epi16(kmask, w8);
        w8 += k;
        const __m256i v16 = _mm256_maskz_loadu_epi16(kmask, w16);
        w16 += k;
        const __m256i v24 = _mm256_maskz_loadu_epi16(kmask, w24);
        w24 += k;
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_maskz_loadu_epi16(kmask, w1);
        w1 += k;
        const __m256i v9 = _mm256_maskz_loadu_epi16(kmask, w9);
        w9 += k;
        const __m256i v17 = _mm256_maskz_loadu_epi16(kmask, w17);
        w17 += k;
        const __m256i v25 = _mm256_maskz_loadu_epi16(kmask, w25);
        w25 += k;
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_maskz_loadu_epi16(kmask, w2);
        w2 += k;
        const __m256i v10 = _mm256_maskz_loadu_epi16(kmask, w10);
        w10 += k;
        const __m256i v18 = _mm256_maskz_loadu_epi16(kmask, w18);
        w18 += k;
        const __m256i v26 = _mm256_maskz_loadu_epi16(kmask, w26);
        w26 += k;
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_maskz_loadu_epi16(kmask, w3);
        w3 += k;
        const __m256i v11 = _mm256_maskz_loadu_epi16(kmask, w11);
        w11 += k;
        const __m256i v19 = _mm256_maskz_loadu_epi16(kmask, w19);
        w19 += k;
        const __m256i v27 = _mm256_maskz_loadu_epi16(kmask, w27);
        w27 += k;
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_maskz_loadu_epi16(kmask, w4);
        w4 += k;
        const __m256i v12 = _mm256_maskz_loadu_epi16(kmask, w12);
        w12 += k;
        const __m256i v20 = _mm256_maskz_loadu_epi16(kmask, w20);
        w20 += k;
        const __m256i v28 = _mm256_maskz_loadu_epi16(kmask, w28);
        w28 += k;
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_maskz_loadu_epi16(kmask, w5);
        w5 += k;
        const __m256i v13 = _mm256_maskz_loadu_epi16(kmask, w13);
        w13 += k;
        const __m256i v21 = _mm256_maskz_loadu_epi16(kmask, w21);
        w21 += k;
        const __m256i v29 = _mm256_maskz_loadu_epi16(kmask, w29);
        w29 += k;
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_maskz_loadu_epi16(kmask, w6);
        w6 += k;
        const __m256i v14 = _mm256_maskz_loadu_epi16(kmask, w14);
        w14 += k;
        const __m256i v22 = _mm256_maskz_loadu_epi16(kmask, w22);
        w22 += k;
        const __m256i v30 = _mm256_maskz_loadu_epi16(kmask, w30);
        w30 += k;
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_maskz_loadu_epi16(kmask, w7);
        w7 += k;
        const __m256i v15 = _mm256_maskz_loadu_epi16(kmask, w15);
        w15 += k;
        const __m256i v23 = _mm256_maskz_loadu_epi16(kmask, w23);
        w23 += k;
        const __m256i v31 = _mm256_maskz_loadu_epi16(kmask, w31);
        w31 += k;
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);
        const __m256i v32 = _mm256_maskz_loadu_epi16(kmask, w32);
        w32 += k;
        const __m256i v40 = _mm256_maskz_loadu_epi16(kmask, w40);
        w40 += k;
        const __m256i v48 = _mm256_maskz_loadu_epi16(kmask, w48);
        w48 += k;
        const __m256i v56 = _mm256_maskz_loadu_epi16(kmask, w56);
        w56 += k;
        __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v40, 1);
        __m512i z40 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v56, 1);
        const __m256i v33 = _mm256_maskz_loadu_epi16(kmask, w33);
        w33 += k;
        const __m256i v41 = _mm256_maskz_loadu_epi16(kmask, w41);
        w41 += k;
        const __m256i v49 = _mm256_maskz_loadu_epi16(kmask, w49);
        w49 += k;
        const __m256i v57 = _mm256_maskz_loadu_epi16(kmask, w57);
        w57 += k;
        __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v41, 1);
        __m512i z41 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v57, 1);
        const __m256i v34 = _mm256_maskz_loadu_epi16(kmask, w34);
        w34 += k;
        const __m256i v42 = _mm256_maskz_loadu_epi16(kmask, w42);
        w42 += k;
        const __m256i v50 = _mm256_maskz_loadu_epi16(kmask, w50);
        w50 += k;
        const __m256i v58 = _mm256_maskz_loadu_epi16(kmask, w58);
        w58 += k;
        __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v42, 1);
        __m512i z42 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v58, 1);
        const __m256i v35 = _mm256_maskz_loadu_epi16(kmask, w35);
        w35 += k;
        const __m256i v43 = _mm256_maskz_loadu_epi16(kmask, w43);
        w43 += k;
        const __m256i v51 = _mm256_maskz_loadu_epi16(kmask, w51);
        w51 += k;
        const __m256i v59 = _mm256_maskz_loadu_epi16(kmask, w59);
        w59 += k;
        __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v43, 1);
        __m512i z43 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v59, 1);
        const __m256i v36 = _mm256_maskz_loadu_epi16(kmask, w36);
        w36 += k;
        const __m256i v44 = _mm256_maskz_loadu_epi16(kmask, w44);
        w44 += k;
        const __m256i v52 = _mm256_maskz_loadu_epi16(kmask, w52);
        w52 += k;
        const __m256i v60 = _mm256_maskz_loadu_epi16(kmask, w60);
        w60 += k;
        __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v36), v44, 1);
        __m512i z44 = _mm512_inserti64x4(_mm512_castsi256_si512(v52), v60, 1);
        const __m256i v37 = _mm256_maskz_loadu_epi16(kmask, w37);
        w37 += k;
        const __m256i v45 = _mm256_maskz_loadu_epi16(kmask, w45);
        w45 += k;
        const __m256i v53 = _mm256_maskz_loadu_epi16(kmask, w53);
        w53 += k;
        const __m256i v61 = _mm256_maskz_loadu_epi16(kmask, w61);
        w61 += k;
        __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v37), v45, 1);
        __m512i z45 = _mm512_inserti64x4(_mm512_castsi256_si512(v53), v61, 1);
        const __m256i v38 = _mm256_maskz_loadu_epi16(kmask, w38);
        w38 += k;
        const __m256i v46 = _mm256_maskz_loadu_epi16(kmask, w46);
        w46 += k;
        const __m256i v54 = _mm256_maskz_loadu_epi16(kmask, w54);
        w54 += k;
        const __m256i v62 = _mm256_maskz_loadu_epi16(kmask, w62);
        w62 += k;
        __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v38), v46, 1);
        __m512i z46 = _mm512_inserti64x4(_mm512_castsi256_si512(v54), v62, 1);
        const __m256i v39 = _mm256_maskz_loadu_epi16(kmask, w39);
        w39 += k;
        const __m256i v47 = _mm256_maskz_loadu_epi16(kmask, w47);
        w47 += k;
        const __m256i v55 = _mm256_maskz_loadu_epi16(kmask, w55);
        w55 += k;
        const __m256i v63 = _mm256_setzero_si256();
        __m512i z39 = _mm512_inserti64x4(_mm512_castsi256_si512(v39), v47, 1);
        __m512i z47 = _mm512_inserti64x4(_mm512_castsi256_si512(v55), v63, 1);

        const __m512i t0 = _mm512_unpacklo_epi16(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi16(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi16(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi16(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi16(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi16(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi16(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi16(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi32(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi32(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi32(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi32(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi32(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi32(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi32(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi32(t5, t7);

        const __m512i w0 = _mm512_unpacklo_epi64(u0, u4);
        const __m512i w1 = _mm512_unpackhi_epi64(u0, u4);
        const __m512i w2 = _mm512_unpacklo_epi64(u1, u5);
        const __m512i w3 = _mm512_unpackhi_epi64(u1, u5);
        const __m512i w4 = _mm512_unpacklo_epi64(u2, u6);
        const __m512i w5 = _mm512_unpackhi_epi64(u2, u6);
        const __m512i w6 = _mm512_unpacklo_epi64(u3, u7);
        const __m512i w7 = _mm512_unpackhi_epi64(u3, u7);
        const __m512i t8 = _mm512_unpacklo_epi16(z8, z9);
        const __m512i t9 = _mm512_unpackhi_epi16(z8, z9);
        const __m512i t10 = _mm512_unpacklo_epi16(z10, z11);
        const __m512i t11 = _mm512_unpackhi_epi16(z10, z11);
        const __m512i t12 = _mm512_unpacklo_epi16(z12, z13);
        const __m512i t13 = _mm512_unpackhi_epi16(z12, z13);
        const __m512i t14 = _mm512_unpacklo_epi16(z14, z15);
        const __m512i t15 = _mm512_unpackhi_epi16(z14, z15);

        const __m512i u8 = _mm512_unpacklo_epi32(t8, t10);
        const __m512i u9 = _mm512_unpackhi_epi32(t8, t10);
        const __m512i u10 = _mm512_unpacklo_epi32(t9, t11);
        const __m512i u11 = _mm512_unpackhi_epi32(t9, t11);
        const __m512i u12 = _mm512_unpacklo_epi32(t12, t14);
        const __m512i u13 = _mm512_unpackhi_epi32(t12, t14);
        const __m512i u14 = _mm512_unpacklo_epi32(t13, t15);
        const __m512i u15 = _mm512_unpackhi_epi32(t13, t15);

        const __m512i w8 = _mm512_unpacklo_epi64(u8, u12);
        const __m512i w9 = _mm512_unpackhi_epi64(u8, u12);
        const __m512i w10 = _mm512_unpacklo_epi64(u9, u13);
        const __m512i w11 = _mm512_unpackhi_epi64(u9, u13);
        const __m512i w12 = _mm512_unpacklo_epi64(u10, u14);
        const __m512i w13 = _mm512_unpackhi_epi64(u10, u14);
        const __m512i w14 = _mm512_unpacklo_epi64(u11, u15);
        const __m512i w15 = _mm512_unpackhi_epi64(u11, u15);

        const __m512i out0_0 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 0) {
          _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        }
        const __m512i out0_8 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 512), out0_8);
        }
        const __m512i out0_1 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 1) {
          _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_1);
        }
        const __m512i out0_9 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 9) {
          _mm512_storeu_si512((__m512i*) (packed_w + 576), out0_9);
        }
        const __m512i out0_2 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_2);
        }
        const __m512i out0_10 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 640), out0_10);
        }
        const __m512i out0_3 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 3) {
          _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_3);
        }
        const __m512i out0_11 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 11) {
          _mm512_storeu_si512((__m512i*) (packed_w + 704), out0_11);
        }
        const __m512i out0_4 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_4);
        }
        const __m512i out0_12 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 768), out0_12);
        }
        const __m512i out0_5 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 5) {
          _mm512_storeu_si512((__m512i*) (packed_w + 320), out0_5);
        }
        const __m512i out0_13 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 13) {
          _mm512_storeu_si512((__m512i*) (packed_w + 832), out0_13);
        }
        const __m512i out0_6 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_6);
        }
        const __m512i out0_14 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 896), out0_14);
        }
        const __m512i out0_7 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 7) {
          _mm512_storeu_si512((__m512i*) (packed_w + 448), out0_7);
        }
        const __m512i out0_15 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 15) {
          _mm512_storeu_si512((__m512i*) (packed_w + 960), out0_15);
        }
        const __m512i t32 = _mm512_unpacklo_epi16(z32, z33);
        const __m512i t33 = _mm512_unpackhi_epi16(z32, z33);
        const __m512i t34 = _mm512_unpacklo_epi16(z34, z35);
        const __m512i t35 = _mm512_unpackhi_epi16(z34, z35);
        const __m512i t36 = _mm512_unpacklo_epi16(z36, z37);
        const __m512i t37 = _mm512_unpackhi_epi16(z36, z37);
        const __m512i t38 = _mm512_unpacklo_epi16(z38, z39);
        const __m512i t39 = _mm512_unpackhi_epi16(z38, z39);

        const __m512i u32 = _mm512_unpacklo_epi32(t32, t34);
        const __m512i u33 = _mm512_unpackhi_epi32(t32, t34);
        const __m512i u34 = _mm512_unpacklo_epi32(t33, t35);
        const __m512i u35 = _mm512_unpackhi_epi32(t33, t35);
        const __m512i u36 = _mm512_unpacklo_epi32(t36, t38);
        const __m512i u37 = _mm512_unpackhi_epi32(t36, t38);
        const __m512i u38 = _mm512_unpacklo_epi32(t37, t39);
        const __m512i u39 = _mm512_unpackhi_epi32(t37, t39);

        const __m512i w32 = _mm512_unpacklo_epi64(u32, u36);
        const __m512i w33 = _mm512_unpackhi_epi64(u32, u36);
        const __m512i w34 = _mm512_unpacklo_epi64(u33, u37);
        const __m512i w35 = _mm512_unpackhi_epi64(u33, u37);
        const __m512i w36 = _mm512_unpacklo_epi64(u34, u38);
        const __m512i w37 = _mm512_unpackhi_epi64(u34, u38);
        const __m512i w38 = _mm512_unpacklo_epi64(u35, u39);
        const __m512i w39 = _mm512_unpackhi_epi64(u35, u39);
        const __m512i t40 = _mm512_unpacklo_epi16(z40, z41);
        const __m512i t41 = _mm512_unpackhi_epi16(z40, z41);
        const __m512i t42 = _mm512_unpacklo_epi16(z42, z43);
        const __m512i t43 = _mm512_unpackhi_epi16(z42, z43);
        const __m512i t44 = _mm512_unpacklo_epi16(z44, z45);
        const __m512i t45 = _mm512_unpackhi_epi16(z44, z45);
        const __m512i t46 = _mm512_unpacklo_epi16(z46, z47);
        const __m512i t47 = _mm512_unpackhi_epi16(z46, z47);

        const __m512i u40 = _mm512_unpacklo_epi32(t40, t42);
        const __m512i u41 = _mm512_unpackhi_epi32(t40, t42);
        const __m512i u42 = _mm512_unpacklo_epi32(t41, t43);
        const __m512i u43 = _mm512_unpackhi_epi32(t41, t43);
        const __m512i u44 = _mm512_unpacklo_epi32(t44, t46);
        const __m512i u45 = _mm512_unpackhi_epi32(t44, t46);
        const __m512i u46 = _mm512_unpacklo_epi32(t45, t47);
        const __m512i u47 = _mm512_unpackhi_epi32(t45, t47);

        const __m512i w40 = _mm512_unpacklo_epi64(u40, u44);
        const __m512i w41 = _mm512_unpackhi_epi64(u40, u44);
        const __m512i w42 = _mm512_unpacklo_epi64(u41, u45);
        const __m512i w43 = _mm512_unpackhi_epi64(u41, u45);
        const __m512i w44 = _mm512_unpacklo_epi64(u42, u46);
        const __m512i w45 = _mm512_unpackhi_epi64(u42, u46);
        const __m512i w46 = _mm512_unpacklo_epi64(u43, u47);
        const __m512i w47 = _mm512_unpackhi_epi64(u43, u47);

        const __m512i out32_0 = _mm512_shuffle_i32x4(w32, w40, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 0) {
          _mm512_storeu_si512((__m512i*) (packed_w + 32), out32_0);
        }
        const __m512i out32_8 = _mm512_shuffle_i32x4(w32, w40, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 544), out32_8);
        }
        const __m512i out32_1 = _mm512_shuffle_i32x4(w33, w41, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 1) {
          _mm512_storeu_si512((__m512i*) (packed_w + 96), out32_1);
        }
        const __m512i out32_9 = _mm512_shuffle_i32x4(w33, w41, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 9) {
          _mm512_storeu_si512((__m512i*) (packed_w + 608), out32_9);
        }
        const __m512i out32_2 = _mm512_shuffle_i32x4(w34, w42, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 160), out32_2);
        }
        const __m512i out32_10 = _mm512_shuffle_i32x4(w34, w42, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 672), out32_10);
        }
        const __m512i out32_3 = _mm512_shuffle_i32x4(w35, w43, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 3) {
          _mm512_storeu_si512((__m512i*) (packed_w + 224), out32_3);
        }
        const __m512i out32_11 = _mm512_shuffle_i32x4(w35, w43, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 11) {
          _mm512_storeu_si512((__m512i*) (packed_w + 736), out32_11);
        }
        const __m512i out32_4 = _mm512_shuffle_i32x4(w36, w44, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 288), out32_4);
        }
        const __m512i out32_12 = _mm512_shuffle_i32x4(w36, w44, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 800), out32_12);
        }
        const __m512i out32_5 = _mm512_shuffle_i32x4(w37, w45, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 5) {
          _mm512_storeu_si512((__m512i*) (packed_w + 352), out32_5);
        }
        const __m512i out32_13 = _mm512_shuffle_i32x4(w37, w45, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 13) {
          _mm512_storeu_si512((__m512i*) (packed_w + 864), out32_13);
        }
        const __m512i out32_6 = _mm512_shuffle_i32x4(w38, w46, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 416), out32_6);
        }
        const __m512i out32_14 = _mm512_shuffle_i32x4(w38, w46, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 928), out32_14);
        }
        const __m512i out32_7 = _mm512_shuffle_i32x4(w39, w47, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 7) {
          _mm512_storeu_si512((__m512i*) (packed_w + 480), out32_7);
        }
        const __m512i out32_15 = _mm512_shuffle_i32x4(w39, w47, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 15) {
          _mm512_storeu_si512((__m512i*) (packed_w + 992), out32_15);
        }
        packed_w += 64 * k;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
