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
    const uint16_t* w0 = weights;
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

      const uint16_t* w1 = w0 + kc;
      const uint16_t* w2 = w1 + kc;
      const uint16_t* w3 = w2 + kc;
      const uint16_t* w4 = w3 + kc;
      const uint16_t* w5 = w4 + kc;
      const uint16_t* w6 = w5 + kc;
      const uint16_t* w7 = w6 + kc;
      const uint16_t* w8 = w7 + kc;
      const uint16_t* w9 = w8 + kc;
      const uint16_t* w10 = w9 + kc;
      const uint16_t* w11 = w10 + kc;
      const uint16_t* w12 = w11 + kc;
      const uint16_t* w13 = w12 + kc;
      const uint16_t* w14 = w13 + kc;
      const uint16_t* w15 = w14 + kc;
      const uint16_t* w16 = w15 + kc;
      const uint16_t* w17 = w16 + kc;
      const uint16_t* w18 = w17 + kc;
      const uint16_t* w19 = w18 + kc;
      const uint16_t* w20 = w19 + kc;
      const uint16_t* w21 = w20 + kc;
      const uint16_t* w22 = w21 + kc;
      const uint16_t* w23 = w22 + kc;
      const uint16_t* w24 = w23 + kc;
      const uint16_t* w25 = w24 + kc;
      const uint16_t* w26 = w25 + kc;
      const uint16_t* w27 = w26 + kc;
      const uint16_t* w28 = w27 + kc;
      const uint16_t* w29 = w28 + kc;
      const uint16_t* w30 = w29 + kc;
      const uint16_t* w31 = w30 + kc;
      const uint16_t* w32 = w31 + kc;
      const uint16_t* w33 = w32 + kc;
      const uint16_t* w34 = w33 + kc;
      const uint16_t* w35 = w34 + kc;
      const uint16_t* w36 = w35 + kc;
      const uint16_t* w37 = w36 + kc;
      const uint16_t* w38 = w37 + kc;
      const uint16_t* w39 = w38 + kc;
      const uint16_t* w40 = w39 + kc;
      const uint16_t* w41 = w40 + kc;
      const uint16_t* w42 = w41 + kc;
      const uint16_t* w43 = w42 + kc;
      const uint16_t* w44 = w43 + kc;
      const uint16_t* w45 = w44 + kc;
      const uint16_t* w46 = w45 + kc;
      const uint16_t* w47 = w46 + kc;
      const uint16_t* w48 = w47 + kc;
      const uint16_t* w49 = w48 + kc;
      const uint16_t* w50 = w49 + kc;
      const uint16_t* w51 = w50 + kc;
      const uint16_t* w52 = w51 + kc;
      const uint16_t* w53 = w52 + kc;
      const uint16_t* w54 = w53 + kc;
      const uint16_t* w55 = w54 + kc;
      const uint16_t* w56 = w55 + kc;
      const uint16_t* w57 = w56 + kc;
      const uint16_t* w58 = w57 + kc;
      const uint16_t* w59 = w58 + kc;
      const uint16_t* w60 = w59 + kc;
      const uint16_t* w61 = w60 + kc;
      const uint16_t* w62 = w61 + kc;
      const uint16_t* w63 = w62 + kc;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);
      xnn_prefetch_to_l1((const int8_t*) w16);
      xnn_prefetch_to_l1((const int8_t*) w16 + 64);
      xnn_prefetch_to_l1((const int8_t*) w17);
      xnn_prefetch_to_l1((const int8_t*) w17 + 64);
      xnn_prefetch_to_l1((const int8_t*) w18);
      xnn_prefetch_to_l1((const int8_t*) w18 + 64);
      xnn_prefetch_to_l1((const int8_t*) w19);
      xnn_prefetch_to_l1((const int8_t*) w19 + 64);
      xnn_prefetch_to_l1((const int8_t*) w20);
      xnn_prefetch_to_l1((const int8_t*) w20 + 64);
      xnn_prefetch_to_l1((const int8_t*) w21);
      xnn_prefetch_to_l1((const int8_t*) w21 + 64);
      xnn_prefetch_to_l1((const int8_t*) w22);
      xnn_prefetch_to_l1((const int8_t*) w22 + 64);
      xnn_prefetch_to_l1((const int8_t*) w23);
      xnn_prefetch_to_l1((const int8_t*) w23 + 64);
      xnn_prefetch_to_l1((const int8_t*) w24);
      xnn_prefetch_to_l1((const int8_t*) w24 + 64);
      xnn_prefetch_to_l1((const int8_t*) w25);
      xnn_prefetch_to_l1((const int8_t*) w25 + 64);
      xnn_prefetch_to_l1((const int8_t*) w26);
      xnn_prefetch_to_l1((const int8_t*) w26 + 64);
      xnn_prefetch_to_l1((const int8_t*) w27);
      xnn_prefetch_to_l1((const int8_t*) w27 + 64);
      xnn_prefetch_to_l1((const int8_t*) w28);
      xnn_prefetch_to_l1((const int8_t*) w28 + 64);
      xnn_prefetch_to_l1((const int8_t*) w29);
      xnn_prefetch_to_l1((const int8_t*) w29 + 64);
      xnn_prefetch_to_l1((const int8_t*) w30);
      xnn_prefetch_to_l1((const int8_t*) w30 + 64);
      xnn_prefetch_to_l1((const int8_t*) w31);
      xnn_prefetch_to_l1((const int8_t*) w31 + 64);
      xnn_prefetch_to_l1((const int8_t*) w32);
      xnn_prefetch_to_l1((const int8_t*) w32 + 64);
      xnn_prefetch_to_l1((const int8_t*) w33);
      xnn_prefetch_to_l1((const int8_t*) w33 + 64);
      xnn_prefetch_to_l1((const int8_t*) w34);
      xnn_prefetch_to_l1((const int8_t*) w34 + 64);
      xnn_prefetch_to_l1((const int8_t*) w35);
      xnn_prefetch_to_l1((const int8_t*) w35 + 64);
      xnn_prefetch_to_l1((const int8_t*) w36);
      xnn_prefetch_to_l1((const int8_t*) w36 + 64);
      xnn_prefetch_to_l1((const int8_t*) w37);
      xnn_prefetch_to_l1((const int8_t*) w37 + 64);
      xnn_prefetch_to_l1((const int8_t*) w38);
      xnn_prefetch_to_l1((const int8_t*) w38 + 64);
      xnn_prefetch_to_l1((const int8_t*) w39);
      xnn_prefetch_to_l1((const int8_t*) w39 + 64);
      xnn_prefetch_to_l1((const int8_t*) w40);
      xnn_prefetch_to_l1((const int8_t*) w40 + 64);
      xnn_prefetch_to_l1((const int8_t*) w41);
      xnn_prefetch_to_l1((const int8_t*) w41 + 64);
      xnn_prefetch_to_l1((const int8_t*) w42);
      xnn_prefetch_to_l1((const int8_t*) w42 + 64);
      xnn_prefetch_to_l1((const int8_t*) w43);
      xnn_prefetch_to_l1((const int8_t*) w43 + 64);
      xnn_prefetch_to_l1((const int8_t*) w44);
      xnn_prefetch_to_l1((const int8_t*) w44 + 64);
      xnn_prefetch_to_l1((const int8_t*) w45);
      xnn_prefetch_to_l1((const int8_t*) w45 + 64);
      xnn_prefetch_to_l1((const int8_t*) w46);
      xnn_prefetch_to_l1((const int8_t*) w46 + 64);
      xnn_prefetch_to_l1((const int8_t*) w47);
      xnn_prefetch_to_l1((const int8_t*) w47 + 64);
      xnn_prefetch_to_l1((const int8_t*) w48);
      xnn_prefetch_to_l1((const int8_t*) w48 + 64);
      xnn_prefetch_to_l1((const int8_t*) w49);
      xnn_prefetch_to_l1((const int8_t*) w49 + 64);
      xnn_prefetch_to_l1((const int8_t*) w50);
      xnn_prefetch_to_l1((const int8_t*) w50 + 64);
      xnn_prefetch_to_l1((const int8_t*) w51);
      xnn_prefetch_to_l1((const int8_t*) w51 + 64);
      xnn_prefetch_to_l1((const int8_t*) w52);
      xnn_prefetch_to_l1((const int8_t*) w52 + 64);
      xnn_prefetch_to_l1((const int8_t*) w53);
      xnn_prefetch_to_l1((const int8_t*) w53 + 64);
      xnn_prefetch_to_l1((const int8_t*) w54);
      xnn_prefetch_to_l1((const int8_t*) w54 + 64);
      xnn_prefetch_to_l1((const int8_t*) w55);
      xnn_prefetch_to_l1((const int8_t*) w55 + 64);
      xnn_prefetch_to_l1((const int8_t*) w56);
      xnn_prefetch_to_l1((const int8_t*) w56 + 64);
      xnn_prefetch_to_l1((const int8_t*) w57);
      xnn_prefetch_to_l1((const int8_t*) w57 + 64);
      xnn_prefetch_to_l1((const int8_t*) w58);
      xnn_prefetch_to_l1((const int8_t*) w58 + 64);
      xnn_prefetch_to_l1((const int8_t*) w59);
      xnn_prefetch_to_l1((const int8_t*) w59 + 64);
      xnn_prefetch_to_l1((const int8_t*) w60);
      xnn_prefetch_to_l1((const int8_t*) w60 + 64);
      xnn_prefetch_to_l1((const int8_t*) w61);
      xnn_prefetch_to_l1((const int8_t*) w61 + 64);
      xnn_prefetch_to_l1((const int8_t*) w62);
      xnn_prefetch_to_l1((const int8_t*) w62 + 64);
      xnn_prefetch_to_l1((const int8_t*) w63);
      xnn_prefetch_to_l1((const int8_t*) w63 + 64);

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        const __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        const __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        w0 += 16;
        w8 += 16;
        w16 += 16;
        w24 += 16;
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        const __m256i v17 = _mm256_loadu_si256((const __m256i*) w17);
        const __m256i v25 = _mm256_loadu_si256((const __m256i*) w25);
        w1 += 16;
        w9 += 16;
        w17 += 16;
        w25 += 16;
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        const __m256i v18 = _mm256_loadu_si256((const __m256i*) w18);
        const __m256i v26 = _mm256_loadu_si256((const __m256i*) w26);
        w2 += 16;
        w10 += 16;
        w18 += 16;
        w26 += 16;
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        const __m256i v19 = _mm256_loadu_si256((const __m256i*) w19);
        const __m256i v27 = _mm256_loadu_si256((const __m256i*) w27);
        w3 += 16;
        w11 += 16;
        w19 += 16;
        w27 += 16;
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        const __m256i v20 = _mm256_loadu_si256((const __m256i*) w20);
        const __m256i v28 = _mm256_loadu_si256((const __m256i*) w28);
        w4 += 16;
        w12 += 16;
        w20 += 16;
        w28 += 16;
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        const __m256i v21 = _mm256_loadu_si256((const __m256i*) w21);
        const __m256i v29 = _mm256_loadu_si256((const __m256i*) w29);
        w5 += 16;
        w13 += 16;
        w21 += 16;
        w29 += 16;
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        const __m256i v22 = _mm256_loadu_si256((const __m256i*) w22);
        const __m256i v30 = _mm256_loadu_si256((const __m256i*) w30);
        w6 += 16;
        w14 += 16;
        w22 += 16;
        w30 += 16;
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        const __m256i v15 = _mm256_loadu_si256((const __m256i*) w15);
        const __m256i v23 = _mm256_loadu_si256((const __m256i*) w23);
        const __m256i v31 = _mm256_loadu_si256((const __m256i*) w31);
        w7 += 16;
        w15 += 16;
        w23 += 16;
        w31 += 16;
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);
        const __m256i v32 = _mm256_loadu_si256((const __m256i*) w32);
        const __m256i v40 = _mm256_loadu_si256((const __m256i*) w40);
        const __m256i v48 = _mm256_loadu_si256((const __m256i*) w48);
        const __m256i v56 = _mm256_loadu_si256((const __m256i*) w56);
        w32 += 16;
        w40 += 16;
        w48 += 16;
        w56 += 16;
        __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v40, 1);
        __m512i z40 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v56, 1);
        const __m256i v33 = _mm256_loadu_si256((const __m256i*) w33);
        const __m256i v41 = _mm256_loadu_si256((const __m256i*) w41);
        const __m256i v49 = _mm256_loadu_si256((const __m256i*) w49);
        const __m256i v57 = _mm256_loadu_si256((const __m256i*) w57);
        w33 += 16;
        w41 += 16;
        w49 += 16;
        w57 += 16;
        __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v41, 1);
        __m512i z41 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v57, 1);
        const __m256i v34 = _mm256_loadu_si256((const __m256i*) w34);
        const __m256i v42 = _mm256_loadu_si256((const __m256i*) w42);
        const __m256i v50 = _mm256_loadu_si256((const __m256i*) w50);
        const __m256i v58 = _mm256_loadu_si256((const __m256i*) w58);
        w34 += 16;
        w42 += 16;
        w50 += 16;
        w58 += 16;
        __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v42, 1);
        __m512i z42 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v58, 1);
        const __m256i v35 = _mm256_loadu_si256((const __m256i*) w35);
        const __m256i v43 = _mm256_loadu_si256((const __m256i*) w43);
        const __m256i v51 = _mm256_loadu_si256((const __m256i*) w51);
        const __m256i v59 = _mm256_loadu_si256((const __m256i*) w59);
        w35 += 16;
        w43 += 16;
        w51 += 16;
        w59 += 16;
        __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v43, 1);
        __m512i z43 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v59, 1);
        const __m256i v36 = _mm256_loadu_si256((const __m256i*) w36);
        const __m256i v44 = _mm256_loadu_si256((const __m256i*) w44);
        const __m256i v52 = _mm256_loadu_si256((const __m256i*) w52);
        const __m256i v60 = _mm256_loadu_si256((const __m256i*) w60);
        w36 += 16;
        w44 += 16;
        w52 += 16;
        w60 += 16;
        __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v36), v44, 1);
        __m512i z44 = _mm512_inserti64x4(_mm512_castsi256_si512(v52), v60, 1);
        const __m256i v37 = _mm256_loadu_si256((const __m256i*) w37);
        const __m256i v45 = _mm256_loadu_si256((const __m256i*) w45);
        const __m256i v53 = _mm256_loadu_si256((const __m256i*) w53);
        const __m256i v61 = _mm256_loadu_si256((const __m256i*) w61);
        w37 += 16;
        w45 += 16;
        w53 += 16;
        w61 += 16;
        __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v37), v45, 1);
        __m512i z45 = _mm512_inserti64x4(_mm512_castsi256_si512(v53), v61, 1);
        const __m256i v38 = _mm256_loadu_si256((const __m256i*) w38);
        const __m256i v46 = _mm256_loadu_si256((const __m256i*) w46);
        const __m256i v54 = _mm256_loadu_si256((const __m256i*) w54);
        const __m256i v62 = _mm256_loadu_si256((const __m256i*) w62);
        w38 += 16;
        w46 += 16;
        w54 += 16;
        w62 += 16;
        __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v38), v46, 1);
        __m512i z46 = _mm512_inserti64x4(_mm512_castsi256_si512(v54), v62, 1);
        const __m256i v39 = _mm256_loadu_si256((const __m256i*) w39);
        const __m256i v47 = _mm256_loadu_si256((const __m256i*) w47);
        const __m256i v55 = _mm256_loadu_si256((const __m256i*) w55);
        const __m256i v63 = _mm256_loadu_si256((const __m256i*) w63);
        w39 += 16;
        w47 += 16;
        w55 += 16;
        w63 += 16;
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

      // KC remainder (1..15)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);

        if (k & 8) {
          __m256i v0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w0));
          w0 += 8;
          __m256i v1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w1));
          w1 += 8;
          __m256i v2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w2));
          w2 += 8;
          __m256i v3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w3));
          w3 += 8;
          __m256i v4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w4));
          w4 += 8;
          __m256i v5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w5));
          w5 += 8;
          __m256i v6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w6));
          w6 += 8;
          __m256i v7 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w7));
          w7 += 8;
          __m256i v8 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w8));
          w8 += 8;
          __m256i v9 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w9));
          w9 += 8;
          __m256i v10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w10));
          w10 += 8;
          __m256i v11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w11));
          w11 += 8;
          __m256i v12 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w12));
          w12 += 8;
          __m256i v13 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w13));
          w13 += 8;
          __m256i v14 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w14));
          w14 += 8;
          __m256i v15 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w15));
          w15 += 8;
          __m256i v16 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w16));
          w16 += 8;
          __m256i v17 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w17));
          w17 += 8;
          __m256i v18 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w18));
          w18 += 8;
          __m256i v19 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w19));
          w19 += 8;
          __m256i v20 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w20));
          w20 += 8;
          __m256i v21 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w21));
          w21 += 8;
          __m256i v22 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w22));
          w22 += 8;
          __m256i v23 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w23));
          w23 += 8;
          __m256i v24 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w24));
          w24 += 8;
          __m256i v25 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w25));
          w25 += 8;
          __m256i v26 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w26));
          w26 += 8;
          __m256i v27 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w27));
          w27 += 8;
          __m256i v28 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w28));
          w28 += 8;
          __m256i v29 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w29));
          w29 += 8;
          __m256i v30 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w30));
          w30 += 8;
          __m256i v31 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w31));
          w31 += 8;
          __m256i v32 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w32));
          w32 += 8;
          __m256i v33 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w33));
          w33 += 8;
          __m256i v34 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w34));
          w34 += 8;
          __m256i v35 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w35));
          w35 += 8;
          __m256i v36 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w36));
          w36 += 8;
          __m256i v37 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w37));
          w37 += 8;
          __m256i v38 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w38));
          w38 += 8;
          __m256i v39 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w39));
          w39 += 8;
          __m256i v40 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w40));
          w40 += 8;
          __m256i v41 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w41));
          w41 += 8;
          __m256i v42 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w42));
          w42 += 8;
          __m256i v43 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w43));
          w43 += 8;
          __m256i v44 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w44));
          w44 += 8;
          __m256i v45 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w45));
          w45 += 8;
          __m256i v46 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w46));
          w46 += 8;
          __m256i v47 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w47));
          w47 += 8;
          __m256i v48 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w48));
          w48 += 8;
          __m256i v49 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w49));
          w49 += 8;
          __m256i v50 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w50));
          w50 += 8;
          __m256i v51 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w51));
          w51 += 8;
          __m256i v52 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w52));
          w52 += 8;
          __m256i v53 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w53));
          w53 += 8;
          __m256i v54 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w54));
          w54 += 8;
          __m256i v55 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w55));
          w55 += 8;
          __m256i v56 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w56));
          w56 += 8;
          __m256i v57 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w57));
          w57 += 8;
          __m256i v58 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w58));
          w58 += 8;
          __m256i v59 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w59));
          w59 += 8;
          __m256i v60 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w60));
          w60 += 8;
          __m256i v61 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w61));
          w61 += 8;
          __m256i v62 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w62));
          w62 += 8;
          __m256i v63 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w63));
          w63 += 8;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t1 = _mm256_unpackhi_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t3 = _mm256_unpackhi_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t5 = _mm256_unpackhi_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);
          const __m256i t7 = _mm256_unpackhi_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
          const __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
          const __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
          const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
          const __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
          const __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
          const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
          const __m256i w4 = _mm256_unpacklo_epi64(u2, u6);
          const __m256i w5 = _mm256_unpackhi_epi64(u2, u6);
          const __m256i w6 = _mm256_unpacklo_epi64(u3, u7);
          const __m256i w7 = _mm256_unpackhi_epi64(u3, u7);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t9 = _mm256_unpackhi_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t11 = _mm256_unpackhi_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t13 = _mm256_unpackhi_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);
          const __m256i t15 = _mm256_unpackhi_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
          const __m256i u10 = _mm256_unpacklo_epi32(t9, t11);
          const __m256i u11 = _mm256_unpackhi_epi32(t9, t11);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
          const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);
          const __m256i u14 = _mm256_unpacklo_epi32(t13, t15);
          const __m256i u15 = _mm256_unpackhi_epi32(t13, t15);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
          const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);
          const __m256i w12 = _mm256_unpacklo_epi64(u10, u14);
          const __m256i w13 = _mm256_unpackhi_epi64(u10, u14);
          const __m256i w14 = _mm256_unpacklo_epi64(u11, u15);
          const __m256i w15 = _mm256_unpackhi_epi64(u11, u15);
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t17 = _mm256_unpackhi_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t19 = _mm256_unpackhi_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t21 = _mm256_unpackhi_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);
          const __m256i t23 = _mm256_unpackhi_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
          const __m256i u18 = _mm256_unpacklo_epi32(t17, t19);
          const __m256i u19 = _mm256_unpackhi_epi32(t17, t19);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
          const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);
          const __m256i u22 = _mm256_unpacklo_epi32(t21, t23);
          const __m256i u23 = _mm256_unpackhi_epi32(t21, t23);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
          const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
          const __m256i w20 = _mm256_unpacklo_epi64(u18, u22);
          const __m256i w21 = _mm256_unpackhi_epi64(u18, u22);
          const __m256i w22 = _mm256_unpacklo_epi64(u19, u23);
          const __m256i w23 = _mm256_unpackhi_epi64(u19, u23);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t25 = _mm256_unpackhi_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t27 = _mm256_unpackhi_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t29 = _mm256_unpackhi_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);
          const __m256i t31 = _mm256_unpackhi_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
          const __m256i u26 = _mm256_unpacklo_epi32(t25, t27);
          const __m256i u27 = _mm256_unpackhi_epi32(t25, t27);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
          const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);
          const __m256i u30 = _mm256_unpacklo_epi32(t29, t31);
          const __m256i u31 = _mm256_unpackhi_epi32(t29, t31);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
          const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);
          const __m256i w28 = _mm256_unpacklo_epi64(u26, u30);
          const __m256i w29 = _mm256_unpackhi_epi64(u26, u30);
          const __m256i w30 = _mm256_unpacklo_epi64(u27, u31);
          const __m256i w31 = _mm256_unpackhi_epi64(u27, u31);
          const __m256i t32 = _mm256_unpacklo_epi16(v32, v33);
          const __m256i t33 = _mm256_unpackhi_epi16(v32, v33);
          const __m256i t34 = _mm256_unpacklo_epi16(v34, v35);
          const __m256i t35 = _mm256_unpackhi_epi16(v34, v35);
          const __m256i t36 = _mm256_unpacklo_epi16(v36, v37);
          const __m256i t37 = _mm256_unpackhi_epi16(v36, v37);
          const __m256i t38 = _mm256_unpacklo_epi16(v38, v39);
          const __m256i t39 = _mm256_unpackhi_epi16(v38, v39);

          const __m256i u32 = _mm256_unpacklo_epi32(t32, t34);
          const __m256i u33 = _mm256_unpackhi_epi32(t32, t34);
          const __m256i u34 = _mm256_unpacklo_epi32(t33, t35);
          const __m256i u35 = _mm256_unpackhi_epi32(t33, t35);
          const __m256i u36 = _mm256_unpacklo_epi32(t36, t38);
          const __m256i u37 = _mm256_unpackhi_epi32(t36, t38);
          const __m256i u38 = _mm256_unpacklo_epi32(t37, t39);
          const __m256i u39 = _mm256_unpackhi_epi32(t37, t39);

          const __m256i w32 = _mm256_unpacklo_epi64(u32, u36);
          const __m256i w33 = _mm256_unpackhi_epi64(u32, u36);
          const __m256i w34 = _mm256_unpacklo_epi64(u33, u37);
          const __m256i w35 = _mm256_unpackhi_epi64(u33, u37);
          const __m256i w36 = _mm256_unpacklo_epi64(u34, u38);
          const __m256i w37 = _mm256_unpackhi_epi64(u34, u38);
          const __m256i w38 = _mm256_unpacklo_epi64(u35, u39);
          const __m256i w39 = _mm256_unpackhi_epi64(u35, u39);
          const __m256i t40 = _mm256_unpacklo_epi16(v40, v41);
          const __m256i t41 = _mm256_unpackhi_epi16(v40, v41);
          const __m256i t42 = _mm256_unpacklo_epi16(v42, v43);
          const __m256i t43 = _mm256_unpackhi_epi16(v42, v43);
          const __m256i t44 = _mm256_unpacklo_epi16(v44, v45);
          const __m256i t45 = _mm256_unpackhi_epi16(v44, v45);
          const __m256i t46 = _mm256_unpacklo_epi16(v46, v47);
          const __m256i t47 = _mm256_unpackhi_epi16(v46, v47);

          const __m256i u40 = _mm256_unpacklo_epi32(t40, t42);
          const __m256i u41 = _mm256_unpackhi_epi32(t40, t42);
          const __m256i u42 = _mm256_unpacklo_epi32(t41, t43);
          const __m256i u43 = _mm256_unpackhi_epi32(t41, t43);
          const __m256i u44 = _mm256_unpacklo_epi32(t44, t46);
          const __m256i u45 = _mm256_unpackhi_epi32(t44, t46);
          const __m256i u46 = _mm256_unpacklo_epi32(t45, t47);
          const __m256i u47 = _mm256_unpackhi_epi32(t45, t47);

          const __m256i w40 = _mm256_unpacklo_epi64(u40, u44);
          const __m256i w41 = _mm256_unpackhi_epi64(u40, u44);
          const __m256i w42 = _mm256_unpacklo_epi64(u41, u45);
          const __m256i w43 = _mm256_unpackhi_epi64(u41, u45);
          const __m256i w44 = _mm256_unpacklo_epi64(u42, u46);
          const __m256i w45 = _mm256_unpackhi_epi64(u42, u46);
          const __m256i w46 = _mm256_unpacklo_epi64(u43, u47);
          const __m256i w47 = _mm256_unpackhi_epi64(u43, u47);
          const __m256i t48 = _mm256_unpacklo_epi16(v48, v49);
          const __m256i t49 = _mm256_unpackhi_epi16(v48, v49);
          const __m256i t50 = _mm256_unpacklo_epi16(v50, v51);
          const __m256i t51 = _mm256_unpackhi_epi16(v50, v51);
          const __m256i t52 = _mm256_unpacklo_epi16(v52, v53);
          const __m256i t53 = _mm256_unpackhi_epi16(v52, v53);
          const __m256i t54 = _mm256_unpacklo_epi16(v54, v55);
          const __m256i t55 = _mm256_unpackhi_epi16(v54, v55);

          const __m256i u48 = _mm256_unpacklo_epi32(t48, t50);
          const __m256i u49 = _mm256_unpackhi_epi32(t48, t50);
          const __m256i u50 = _mm256_unpacklo_epi32(t49, t51);
          const __m256i u51 = _mm256_unpackhi_epi32(t49, t51);
          const __m256i u52 = _mm256_unpacklo_epi32(t52, t54);
          const __m256i u53 = _mm256_unpackhi_epi32(t52, t54);
          const __m256i u54 = _mm256_unpacklo_epi32(t53, t55);
          const __m256i u55 = _mm256_unpackhi_epi32(t53, t55);

          const __m256i w48 = _mm256_unpacklo_epi64(u48, u52);
          const __m256i w49 = _mm256_unpackhi_epi64(u48, u52);
          const __m256i w50 = _mm256_unpacklo_epi64(u49, u53);
          const __m256i w51 = _mm256_unpackhi_epi64(u49, u53);
          const __m256i w52 = _mm256_unpacklo_epi64(u50, u54);
          const __m256i w53 = _mm256_unpackhi_epi64(u50, u54);
          const __m256i w54 = _mm256_unpacklo_epi64(u51, u55);
          const __m256i w55 = _mm256_unpackhi_epi64(u51, u55);
          const __m256i t56 = _mm256_unpacklo_epi16(v56, v57);
          const __m256i t57 = _mm256_unpackhi_epi16(v56, v57);
          const __m256i t58 = _mm256_unpacklo_epi16(v58, v59);
          const __m256i t59 = _mm256_unpackhi_epi16(v58, v59);
          const __m256i t60 = _mm256_unpacklo_epi16(v60, v61);
          const __m256i t61 = _mm256_unpackhi_epi16(v60, v61);
          const __m256i t62 = _mm256_unpacklo_epi16(v62, v63);
          const __m256i t63 = _mm256_unpackhi_epi16(v62, v63);

          const __m256i u56 = _mm256_unpacklo_epi32(t56, t58);
          const __m256i u57 = _mm256_unpackhi_epi32(t56, t58);
          const __m256i u58 = _mm256_unpacklo_epi32(t57, t59);
          const __m256i u59 = _mm256_unpackhi_epi32(t57, t59);
          const __m256i u60 = _mm256_unpacklo_epi32(t60, t62);
          const __m256i u61 = _mm256_unpackhi_epi32(t60, t62);
          const __m256i u62 = _mm256_unpacklo_epi32(t61, t63);
          const __m256i u63 = _mm256_unpackhi_epi32(t61, t63);

          const __m256i w56 = _mm256_unpacklo_epi64(u56, u60);
          const __m256i w57 = _mm256_unpackhi_epi64(u56, u60);
          const __m256i w58 = _mm256_unpacklo_epi64(u57, u61);
          const __m256i w59 = _mm256_unpackhi_epi64(u57, u61);
          const __m256i w60 = _mm256_unpacklo_epi64(u58, u62);
          const __m256i w61 = _mm256_unpackhi_epi64(u58, u62);
          const __m256i w62 = _mm256_unpacklo_epi64(u59, u63);
          const __m256i w63 = _mm256_unpackhi_epi64(u59, u63);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w32));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w40));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w48));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w56));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w33));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w41));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w49));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w57));
          _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_castsi256_si128(w18));
          _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_castsi256_si128(w26));
          _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_castsi256_si128(w34));
          _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_castsi256_si128(w42));
          _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_castsi256_si128(w50));
          _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_castsi256_si128(w58));
          _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_castsi256_si128(w19));
          _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_castsi256_si128(w27));
          _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_castsi256_si128(w35));
          _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_castsi256_si128(w43));
          _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_castsi256_si128(w51));
          _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_castsi256_si128(w59));
          _mm_storeu_si128((__m128i*) (packed_w + 256), _mm256_castsi256_si128(w4));
          _mm_storeu_si128((__m128i*) (packed_w + 264), _mm256_castsi256_si128(w12));
          _mm_storeu_si128((__m128i*) (packed_w + 272), _mm256_castsi256_si128(w20));
          _mm_storeu_si128((__m128i*) (packed_w + 280), _mm256_castsi256_si128(w28));
          _mm_storeu_si128((__m128i*) (packed_w + 288), _mm256_castsi256_si128(w36));
          _mm_storeu_si128((__m128i*) (packed_w + 296), _mm256_castsi256_si128(w44));
          _mm_storeu_si128((__m128i*) (packed_w + 304), _mm256_castsi256_si128(w52));
          _mm_storeu_si128((__m128i*) (packed_w + 312), _mm256_castsi256_si128(w60));
          _mm_storeu_si128((__m128i*) (packed_w + 320), _mm256_castsi256_si128(w5));
          _mm_storeu_si128((__m128i*) (packed_w + 328), _mm256_castsi256_si128(w13));
          _mm_storeu_si128((__m128i*) (packed_w + 336), _mm256_castsi256_si128(w21));
          _mm_storeu_si128((__m128i*) (packed_w + 344), _mm256_castsi256_si128(w29));
          _mm_storeu_si128((__m128i*) (packed_w + 352), _mm256_castsi256_si128(w37));
          _mm_storeu_si128((__m128i*) (packed_w + 360), _mm256_castsi256_si128(w45));
          _mm_storeu_si128((__m128i*) (packed_w + 368), _mm256_castsi256_si128(w53));
          _mm_storeu_si128((__m128i*) (packed_w + 376), _mm256_castsi256_si128(w61));
          _mm_storeu_si128((__m128i*) (packed_w + 384), _mm256_castsi256_si128(w6));
          _mm_storeu_si128((__m128i*) (packed_w + 392), _mm256_castsi256_si128(w14));
          _mm_storeu_si128((__m128i*) (packed_w + 400), _mm256_castsi256_si128(w22));
          _mm_storeu_si128((__m128i*) (packed_w + 408), _mm256_castsi256_si128(w30));
          _mm_storeu_si128((__m128i*) (packed_w + 416), _mm256_castsi256_si128(w38));
          _mm_storeu_si128((__m128i*) (packed_w + 424), _mm256_castsi256_si128(w46));
          _mm_storeu_si128((__m128i*) (packed_w + 432), _mm256_castsi256_si128(w54));
          _mm_storeu_si128((__m128i*) (packed_w + 440), _mm256_castsi256_si128(w62));
          _mm_storeu_si128((__m128i*) (packed_w + 448), _mm256_castsi256_si128(w7));
          _mm_storeu_si128((__m128i*) (packed_w + 456), _mm256_castsi256_si128(w15));
          _mm_storeu_si128((__m128i*) (packed_w + 464), _mm256_castsi256_si128(w23));
          _mm_storeu_si128((__m128i*) (packed_w + 472), _mm256_castsi256_si128(w31));
          _mm_storeu_si128((__m128i*) (packed_w + 480), _mm256_castsi256_si128(w39));
          _mm_storeu_si128((__m128i*) (packed_w + 488), _mm256_castsi256_si128(w47));
          _mm_storeu_si128((__m128i*) (packed_w + 496), _mm256_castsi256_si128(w55));
          _mm_storeu_si128((__m128i*) (packed_w + 504), _mm256_castsi256_si128(w63));
          packed_w += 512;
        }

        if (k & 4) {
          __m256i v0 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 4;
          __m256i v1 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 4;
          __m256i v2 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 4;
          __m256i v3 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 4;
          __m256i v4 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 4;
          __m256i v5 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 4;
          __m256i v6 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 4;
          __m256i v7 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 4;
          __m256i v8 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 4;
          __m256i v9 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 4;
          __m256i v10 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 4;
          __m256i v11 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 4;
          __m256i v12 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 4;
          __m256i v13 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 4;
          __m256i v14 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 4;
          __m256i v15 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w15));
          w15 += 4;
          __m256i v16 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w16));
          w16 += 4;
          __m256i v17 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w17));
          w17 += 4;
          __m256i v18 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w18));
          w18 += 4;
          __m256i v19 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w19));
          w19 += 4;
          __m256i v20 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w20));
          w20 += 4;
          __m256i v21 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w21));
          w21 += 4;
          __m256i v22 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w22));
          w22 += 4;
          __m256i v23 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w23));
          w23 += 4;
          __m256i v24 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w24));
          w24 += 4;
          __m256i v25 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w25));
          w25 += 4;
          __m256i v26 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w26));
          w26 += 4;
          __m256i v27 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w27));
          w27 += 4;
          __m256i v28 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w28));
          w28 += 4;
          __m256i v29 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w29));
          w29 += 4;
          __m256i v30 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w30));
          w30 += 4;
          __m256i v31 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w31));
          w31 += 4;
          __m256i v32 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w32));
          w32 += 4;
          __m256i v33 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w33));
          w33 += 4;
          __m256i v34 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w34));
          w34 += 4;
          __m256i v35 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w35));
          w35 += 4;
          __m256i v36 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w36));
          w36 += 4;
          __m256i v37 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w37));
          w37 += 4;
          __m256i v38 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w38));
          w38 += 4;
          __m256i v39 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w39));
          w39 += 4;
          __m256i v40 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w40));
          w40 += 4;
          __m256i v41 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w41));
          w41 += 4;
          __m256i v42 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w42));
          w42 += 4;
          __m256i v43 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w43));
          w43 += 4;
          __m256i v44 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w44));
          w44 += 4;
          __m256i v45 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w45));
          w45 += 4;
          __m256i v46 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w46));
          w46 += 4;
          __m256i v47 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w47));
          w47 += 4;
          __m256i v48 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w48));
          w48 += 4;
          __m256i v49 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w49));
          w49 += 4;
          __m256i v50 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w50));
          w50 += 4;
          __m256i v51 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w51));
          w51 += 4;
          __m256i v52 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w52));
          w52 += 4;
          __m256i v53 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w53));
          w53 += 4;
          __m256i v54 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w54));
          w54 += 4;
          __m256i v55 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w55));
          w55 += 4;
          __m256i v56 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w56));
          w56 += 4;
          __m256i v57 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w57));
          w57 += 4;
          __m256i v58 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w58));
          w58 += 4;
          __m256i v59 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w59));
          w59 += 4;
          __m256i v60 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w60));
          w60 += 4;
          __m256i v61 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w61));
          w61 += 4;
          __m256i v62 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w62));
          w62 += 4;
          __m256i v63 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w63));
          w63 += 4;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
          const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
          const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
          const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
          const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
          const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
          const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
          const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
          const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);
          const __m256i t32 = _mm256_unpacklo_epi16(v32, v33);
          const __m256i t34 = _mm256_unpacklo_epi16(v34, v35);
          const __m256i t36 = _mm256_unpacklo_epi16(v36, v37);
          const __m256i t38 = _mm256_unpacklo_epi16(v38, v39);

          const __m256i u32 = _mm256_unpacklo_epi32(t32, t34);
          const __m256i u33 = _mm256_unpackhi_epi32(t32, t34);
          const __m256i u36 = _mm256_unpacklo_epi32(t36, t38);
          const __m256i u37 = _mm256_unpackhi_epi32(t36, t38);

          const __m256i w32 = _mm256_unpacklo_epi64(u32, u36);
          const __m256i w33 = _mm256_unpackhi_epi64(u32, u36);
          const __m256i w34 = _mm256_unpacklo_epi64(u33, u37);
          const __m256i w35 = _mm256_unpackhi_epi64(u33, u37);
          const __m256i t40 = _mm256_unpacklo_epi16(v40, v41);
          const __m256i t42 = _mm256_unpacklo_epi16(v42, v43);
          const __m256i t44 = _mm256_unpacklo_epi16(v44, v45);
          const __m256i t46 = _mm256_unpacklo_epi16(v46, v47);

          const __m256i u40 = _mm256_unpacklo_epi32(t40, t42);
          const __m256i u41 = _mm256_unpackhi_epi32(t40, t42);
          const __m256i u44 = _mm256_unpacklo_epi32(t44, t46);
          const __m256i u45 = _mm256_unpackhi_epi32(t44, t46);

          const __m256i w40 = _mm256_unpacklo_epi64(u40, u44);
          const __m256i w41 = _mm256_unpackhi_epi64(u40, u44);
          const __m256i w42 = _mm256_unpacklo_epi64(u41, u45);
          const __m256i w43 = _mm256_unpackhi_epi64(u41, u45);
          const __m256i t48 = _mm256_unpacklo_epi16(v48, v49);
          const __m256i t50 = _mm256_unpacklo_epi16(v50, v51);
          const __m256i t52 = _mm256_unpacklo_epi16(v52, v53);
          const __m256i t54 = _mm256_unpacklo_epi16(v54, v55);

          const __m256i u48 = _mm256_unpacklo_epi32(t48, t50);
          const __m256i u49 = _mm256_unpackhi_epi32(t48, t50);
          const __m256i u52 = _mm256_unpacklo_epi32(t52, t54);
          const __m256i u53 = _mm256_unpackhi_epi32(t52, t54);

          const __m256i w48 = _mm256_unpacklo_epi64(u48, u52);
          const __m256i w49 = _mm256_unpackhi_epi64(u48, u52);
          const __m256i w50 = _mm256_unpacklo_epi64(u49, u53);
          const __m256i w51 = _mm256_unpackhi_epi64(u49, u53);
          const __m256i t56 = _mm256_unpacklo_epi16(v56, v57);
          const __m256i t58 = _mm256_unpacklo_epi16(v58, v59);
          const __m256i t60 = _mm256_unpacklo_epi16(v60, v61);
          const __m256i t62 = _mm256_unpacklo_epi16(v62, v63);

          const __m256i u56 = _mm256_unpacklo_epi32(t56, t58);
          const __m256i u57 = _mm256_unpackhi_epi32(t56, t58);
          const __m256i u60 = _mm256_unpacklo_epi32(t60, t62);
          const __m256i u61 = _mm256_unpackhi_epi32(t60, t62);

          const __m256i w56 = _mm256_unpacklo_epi64(u56, u60);
          const __m256i w57 = _mm256_unpackhi_epi64(u56, u60);
          const __m256i w58 = _mm256_unpacklo_epi64(u57, u61);
          const __m256i w59 = _mm256_unpackhi_epi64(u57, u61);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w32));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w40));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w48));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w56));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w33));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w41));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w49));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w57));
          _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_castsi256_si128(w18));
          _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_castsi256_si128(w26));
          _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_castsi256_si128(w34));
          _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_castsi256_si128(w42));
          _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_castsi256_si128(w50));
          _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_castsi256_si128(w58));
          _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_castsi256_si128(w19));
          _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_castsi256_si128(w27));
          _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_castsi256_si128(w35));
          _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_castsi256_si128(w43));
          _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_castsi256_si128(w51));
          _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_castsi256_si128(w59));
          packed_w += 256;
        }

        if (k & 2) {
          __m256i v0 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w0)));
          w0 += 2;
          __m256i v1 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w1)));
          w1 += 2;
          __m256i v2 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w2)));
          w2 += 2;
          __m256i v3 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w3)));
          w3 += 2;
          __m256i v4 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w4)));
          w4 += 2;
          __m256i v5 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w5)));
          w5 += 2;
          __m256i v6 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w6)));
          w6 += 2;
          __m256i v7 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w7)));
          w7 += 2;
          __m256i v8 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w8)));
          w8 += 2;
          __m256i v9 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w9)));
          w9 += 2;
          __m256i v10 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w10)));
          w10 += 2;
          __m256i v11 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w11)));
          w11 += 2;
          __m256i v12 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w12)));
          w12 += 2;
          __m256i v13 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w13)));
          w13 += 2;
          __m256i v14 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w14)));
          w14 += 2;
          __m256i v15 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w15)));
          w15 += 2;
          __m256i v16 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w16)));
          w16 += 2;
          __m256i v17 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w17)));
          w17 += 2;
          __m256i v18 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w18)));
          w18 += 2;
          __m256i v19 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w19)));
          w19 += 2;
          __m256i v20 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w20)));
          w20 += 2;
          __m256i v21 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w21)));
          w21 += 2;
          __m256i v22 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w22)));
          w22 += 2;
          __m256i v23 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w23)));
          w23 += 2;
          __m256i v24 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w24)));
          w24 += 2;
          __m256i v25 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w25)));
          w25 += 2;
          __m256i v26 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w26)));
          w26 += 2;
          __m256i v27 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w27)));
          w27 += 2;
          __m256i v28 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w28)));
          w28 += 2;
          __m256i v29 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w29)));
          w29 += 2;
          __m256i v30 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w30)));
          w30 += 2;
          __m256i v31 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w31)));
          w31 += 2;
          __m256i v32 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w32)));
          w32 += 2;
          __m256i v33 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w33)));
          w33 += 2;
          __m256i v34 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w34)));
          w34 += 2;
          __m256i v35 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w35)));
          w35 += 2;
          __m256i v36 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w36)));
          w36 += 2;
          __m256i v37 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w37)));
          w37 += 2;
          __m256i v38 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w38)));
          w38 += 2;
          __m256i v39 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w39)));
          w39 += 2;
          __m256i v40 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w40)));
          w40 += 2;
          __m256i v41 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w41)));
          w41 += 2;
          __m256i v42 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w42)));
          w42 += 2;
          __m256i v43 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w43)));
          w43 += 2;
          __m256i v44 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w44)));
          w44 += 2;
          __m256i v45 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w45)));
          w45 += 2;
          __m256i v46 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w46)));
          w46 += 2;
          __m256i v47 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w47)));
          w47 += 2;
          __m256i v48 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w48)));
          w48 += 2;
          __m256i v49 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w49)));
          w49 += 2;
          __m256i v50 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w50)));
          w50 += 2;
          __m256i v51 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w51)));
          w51 += 2;
          __m256i v52 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w52)));
          w52 += 2;
          __m256i v53 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w53)));
          w53 += 2;
          __m256i v54 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w54)));
          w54 += 2;
          __m256i v55 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w55)));
          w55 += 2;
          __m256i v56 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w56)));
          w56 += 2;
          __m256i v57 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w57)));
          w57 += 2;
          __m256i v58 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w58)));
          w58 += 2;
          __m256i v59 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w59)));
          w59 += 2;
          __m256i v60 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w60)));
          w60 += 2;
          __m256i v61 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w61)));
          w61 += 2;
          __m256i v62 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w62)));
          w62 += 2;
          __m256i v63 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w63)));
          w63 += 2;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i t32 = _mm256_unpacklo_epi16(v32, v33);
          const __m256i t34 = _mm256_unpacklo_epi16(v34, v35);
          const __m256i t36 = _mm256_unpacklo_epi16(v36, v37);
          const __m256i t38 = _mm256_unpacklo_epi16(v38, v39);

          const __m256i u32 = _mm256_unpacklo_epi32(t32, t34);
          const __m256i u36 = _mm256_unpacklo_epi32(t36, t38);

          const __m256i w32 = _mm256_unpacklo_epi64(u32, u36);
          const __m256i w33 = _mm256_unpackhi_epi64(u32, u36);
          const __m256i t40 = _mm256_unpacklo_epi16(v40, v41);
          const __m256i t42 = _mm256_unpacklo_epi16(v42, v43);
          const __m256i t44 = _mm256_unpacklo_epi16(v44, v45);
          const __m256i t46 = _mm256_unpacklo_epi16(v46, v47);

          const __m256i u40 = _mm256_unpacklo_epi32(t40, t42);
          const __m256i u44 = _mm256_unpacklo_epi32(t44, t46);

          const __m256i w40 = _mm256_unpacklo_epi64(u40, u44);
          const __m256i w41 = _mm256_unpackhi_epi64(u40, u44);
          const __m256i t48 = _mm256_unpacklo_epi16(v48, v49);
          const __m256i t50 = _mm256_unpacklo_epi16(v50, v51);
          const __m256i t52 = _mm256_unpacklo_epi16(v52, v53);
          const __m256i t54 = _mm256_unpacklo_epi16(v54, v55);

          const __m256i u48 = _mm256_unpacklo_epi32(t48, t50);
          const __m256i u52 = _mm256_unpacklo_epi32(t52, t54);

          const __m256i w48 = _mm256_unpacklo_epi64(u48, u52);
          const __m256i w49 = _mm256_unpackhi_epi64(u48, u52);
          const __m256i t56 = _mm256_unpacklo_epi16(v56, v57);
          const __m256i t58 = _mm256_unpacklo_epi16(v58, v59);
          const __m256i t60 = _mm256_unpacklo_epi16(v60, v61);
          const __m256i t62 = _mm256_unpacklo_epi16(v62, v63);

          const __m256i u56 = _mm256_unpacklo_epi32(t56, t58);
          const __m256i u60 = _mm256_unpacklo_epi32(t60, t62);

          const __m256i w56 = _mm256_unpacklo_epi64(u56, u60);
          const __m256i w57 = _mm256_unpackhi_epi64(u56, u60);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w32));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w40));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w48));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w56));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w33));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w41));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w49));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w57));
          packed_w += 128;
        }

        if (k & 1) {
          __m256i v0 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w0, 0));
          w0 += 1;
          __m256i v1 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w1, 0));
          w1 += 1;
          __m256i v2 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w2, 0));
          w2 += 1;
          __m256i v3 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w3, 0));
          w3 += 1;
          __m256i v4 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w4, 0));
          w4 += 1;
          __m256i v5 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w5, 0));
          w5 += 1;
          __m256i v6 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w6, 0));
          w6 += 1;
          __m256i v7 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w7, 0));
          w7 += 1;
          __m256i v8 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w8, 0));
          w8 += 1;
          __m256i v9 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w9, 0));
          w9 += 1;
          __m256i v10 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w10, 0));
          w10 += 1;
          __m256i v11 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w11, 0));
          w11 += 1;
          __m256i v12 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w12, 0));
          w12 += 1;
          __m256i v13 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w13, 0));
          w13 += 1;
          __m256i v14 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w14, 0));
          w14 += 1;
          __m256i v15 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w15, 0));
          w15 += 1;
          __m256i v16 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w16, 0));
          w16 += 1;
          __m256i v17 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w17, 0));
          w17 += 1;
          __m256i v18 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w18, 0));
          w18 += 1;
          __m256i v19 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w19, 0));
          w19 += 1;
          __m256i v20 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w20, 0));
          w20 += 1;
          __m256i v21 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w21, 0));
          w21 += 1;
          __m256i v22 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w22, 0));
          w22 += 1;
          __m256i v23 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w23, 0));
          w23 += 1;
          __m256i v24 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w24, 0));
          w24 += 1;
          __m256i v25 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w25, 0));
          w25 += 1;
          __m256i v26 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w26, 0));
          w26 += 1;
          __m256i v27 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w27, 0));
          w27 += 1;
          __m256i v28 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w28, 0));
          w28 += 1;
          __m256i v29 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w29, 0));
          w29 += 1;
          __m256i v30 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w30, 0));
          w30 += 1;
          __m256i v31 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w31, 0));
          w31 += 1;
          __m256i v32 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w32, 0));
          w32 += 1;
          __m256i v33 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w33, 0));
          w33 += 1;
          __m256i v34 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w34, 0));
          w34 += 1;
          __m256i v35 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w35, 0));
          w35 += 1;
          __m256i v36 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w36, 0));
          w36 += 1;
          __m256i v37 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w37, 0));
          w37 += 1;
          __m256i v38 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w38, 0));
          w38 += 1;
          __m256i v39 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w39, 0));
          w39 += 1;
          __m256i v40 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w40, 0));
          w40 += 1;
          __m256i v41 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w41, 0));
          w41 += 1;
          __m256i v42 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w42, 0));
          w42 += 1;
          __m256i v43 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w43, 0));
          w43 += 1;
          __m256i v44 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w44, 0));
          w44 += 1;
          __m256i v45 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w45, 0));
          w45 += 1;
          __m256i v46 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w46, 0));
          w46 += 1;
          __m256i v47 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w47, 0));
          w47 += 1;
          __m256i v48 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w48, 0));
          w48 += 1;
          __m256i v49 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w49, 0));
          w49 += 1;
          __m256i v50 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w50, 0));
          w50 += 1;
          __m256i v51 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w51, 0));
          w51 += 1;
          __m256i v52 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w52, 0));
          w52 += 1;
          __m256i v53 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w53, 0));
          w53 += 1;
          __m256i v54 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w54, 0));
          w54 += 1;
          __m256i v55 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w55, 0));
          w55 += 1;
          __m256i v56 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w56, 0));
          w56 += 1;
          __m256i v57 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w57, 0));
          w57 += 1;
          __m256i v58 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w58, 0));
          w58 += 1;
          __m256i v59 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w59, 0));
          w59 += 1;
          __m256i v60 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w60, 0));
          w60 += 1;
          __m256i v61 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w61, 0));
          w61 += 1;
          __m256i v62 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w62, 0));
          w62 += 1;
          __m256i v63 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w63, 0));
          w63 += 1;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i t32 = _mm256_unpacklo_epi16(v32, v33);
          const __m256i t34 = _mm256_unpacklo_epi16(v34, v35);
          const __m256i t36 = _mm256_unpacklo_epi16(v36, v37);
          const __m256i t38 = _mm256_unpacklo_epi16(v38, v39);

          const __m256i u32 = _mm256_unpacklo_epi32(t32, t34);
          const __m256i u36 = _mm256_unpacklo_epi32(t36, t38);

          const __m256i w32 = _mm256_unpacklo_epi64(u32, u36);
          const __m256i t40 = _mm256_unpacklo_epi16(v40, v41);
          const __m256i t42 = _mm256_unpacklo_epi16(v42, v43);
          const __m256i t44 = _mm256_unpacklo_epi16(v44, v45);
          const __m256i t46 = _mm256_unpacklo_epi16(v46, v47);

          const __m256i u40 = _mm256_unpacklo_epi32(t40, t42);
          const __m256i u44 = _mm256_unpacklo_epi32(t44, t46);

          const __m256i w40 = _mm256_unpacklo_epi64(u40, u44);
          const __m256i t48 = _mm256_unpacklo_epi16(v48, v49);
          const __m256i t50 = _mm256_unpacklo_epi16(v50, v51);
          const __m256i t52 = _mm256_unpacklo_epi16(v52, v53);
          const __m256i t54 = _mm256_unpacklo_epi16(v54, v55);

          const __m256i u48 = _mm256_unpacklo_epi32(t48, t50);
          const __m256i u52 = _mm256_unpacklo_epi32(t52, t54);

          const __m256i w48 = _mm256_unpacklo_epi64(u48, u52);
          const __m256i t56 = _mm256_unpacklo_epi16(v56, v57);
          const __m256i t58 = _mm256_unpacklo_epi16(v58, v59);
          const __m256i t60 = _mm256_unpacklo_epi16(v60, v61);
          const __m256i t62 = _mm256_unpacklo_epi16(v62, v63);

          const __m256i u56 = _mm256_unpacklo_epi32(t56, t58);
          const __m256i u60 = _mm256_unpacklo_epi32(t60, t62);

          const __m256i w56 = _mm256_unpacklo_epi64(u56, u60);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w32));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w40));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w48));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w56));
          packed_w += 64;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w63;
    }

    // NC remainder (1..63)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 63);
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

      // KC remainder (1..15)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);

        if (k & 8) {
          __m256i v0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w0));
          w0 += 8;
          __m256i v1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w1));
          w1 += 8;
          __m256i v2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w2));
          w2 += 8;
          __m256i v3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w3));
          w3 += 8;
          __m256i v4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w4));
          w4 += 8;
          __m256i v5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w5));
          w5 += 8;
          __m256i v6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w6));
          w6 += 8;
          __m256i v7 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w7));
          w7 += 8;
          __m256i v8 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w8));
          w8 += 8;
          __m256i v9 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w9));
          w9 += 8;
          __m256i v10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w10));
          w10 += 8;
          __m256i v11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w11));
          w11 += 8;
          __m256i v12 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w12));
          w12 += 8;
          __m256i v13 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w13));
          w13 += 8;
          __m256i v14 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w14));
          w14 += 8;
          __m256i v15 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w15));
          w15 += 8;
          __m256i v16 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w16));
          w16 += 8;
          __m256i v17 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w17));
          w17 += 8;
          __m256i v18 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w18));
          w18 += 8;
          __m256i v19 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w19));
          w19 += 8;
          __m256i v20 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w20));
          w20 += 8;
          __m256i v21 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w21));
          w21 += 8;
          __m256i v22 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w22));
          w22 += 8;
          __m256i v23 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w23));
          w23 += 8;
          __m256i v24 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w24));
          w24 += 8;
          __m256i v25 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w25));
          w25 += 8;
          __m256i v26 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w26));
          w26 += 8;
          __m256i v27 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w27));
          w27 += 8;
          __m256i v28 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w28));
          w28 += 8;
          __m256i v29 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w29));
          w29 += 8;
          __m256i v30 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w30));
          w30 += 8;
          __m256i v31 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w31));
          w31 += 8;
          __m256i v32 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w32));
          w32 += 8;
          __m256i v33 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w33));
          w33 += 8;
          __m256i v34 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w34));
          w34 += 8;
          __m256i v35 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w35));
          w35 += 8;
          __m256i v36 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w36));
          w36 += 8;
          __m256i v37 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w37));
          w37 += 8;
          __m256i v38 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w38));
          w38 += 8;
          __m256i v39 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w39));
          w39 += 8;
          __m256i v40 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w40));
          w40 += 8;
          __m256i v41 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w41));
          w41 += 8;
          __m256i v42 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w42));
          w42 += 8;
          __m256i v43 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w43));
          w43 += 8;
          __m256i v44 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w44));
          w44 += 8;
          __m256i v45 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w45));
          w45 += 8;
          __m256i v46 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w46));
          w46 += 8;
          __m256i v47 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w47));
          w47 += 8;
          __m256i v48 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w48));
          w48 += 8;
          __m256i v49 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w49));
          w49 += 8;
          __m256i v50 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w50));
          w50 += 8;
          __m256i v51 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w51));
          w51 += 8;
          __m256i v52 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w52));
          w52 += 8;
          __m256i v53 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w53));
          w53 += 8;
          __m256i v54 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w54));
          w54 += 8;
          __m256i v55 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w55));
          w55 += 8;
          __m256i v56 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w56));
          w56 += 8;
          __m256i v57 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w57));
          w57 += 8;
          __m256i v58 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w58));
          w58 += 8;
          __m256i v59 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w59));
          w59 += 8;
          __m256i v60 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w60));
          w60 += 8;
          __m256i v61 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w61));
          w61 += 8;
          __m256i v62 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w62));
          w62 += 8;
          __m256i v63 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t1 = _mm256_unpackhi_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t3 = _mm256_unpackhi_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t5 = _mm256_unpackhi_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);
          const __m256i t7 = _mm256_unpackhi_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
          const __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
          const __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
          const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
          const __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
          const __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
          const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
          const __m256i w4 = _mm256_unpacklo_epi64(u2, u6);
          const __m256i w5 = _mm256_unpackhi_epi64(u2, u6);
          const __m256i w6 = _mm256_unpacklo_epi64(u3, u7);
          const __m256i w7 = _mm256_unpackhi_epi64(u3, u7);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t9 = _mm256_unpackhi_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t11 = _mm256_unpackhi_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t13 = _mm256_unpackhi_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);
          const __m256i t15 = _mm256_unpackhi_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
          const __m256i u10 = _mm256_unpacklo_epi32(t9, t11);
          const __m256i u11 = _mm256_unpackhi_epi32(t9, t11);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
          const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);
          const __m256i u14 = _mm256_unpacklo_epi32(t13, t15);
          const __m256i u15 = _mm256_unpackhi_epi32(t13, t15);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
          const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);
          const __m256i w12 = _mm256_unpacklo_epi64(u10, u14);
          const __m256i w13 = _mm256_unpackhi_epi64(u10, u14);
          const __m256i w14 = _mm256_unpacklo_epi64(u11, u15);
          const __m256i w15 = _mm256_unpackhi_epi64(u11, u15);
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t17 = _mm256_unpackhi_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t19 = _mm256_unpackhi_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t21 = _mm256_unpackhi_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);
          const __m256i t23 = _mm256_unpackhi_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
          const __m256i u18 = _mm256_unpacklo_epi32(t17, t19);
          const __m256i u19 = _mm256_unpackhi_epi32(t17, t19);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
          const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);
          const __m256i u22 = _mm256_unpacklo_epi32(t21, t23);
          const __m256i u23 = _mm256_unpackhi_epi32(t21, t23);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
          const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
          const __m256i w20 = _mm256_unpacklo_epi64(u18, u22);
          const __m256i w21 = _mm256_unpackhi_epi64(u18, u22);
          const __m256i w22 = _mm256_unpacklo_epi64(u19, u23);
          const __m256i w23 = _mm256_unpackhi_epi64(u19, u23);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t25 = _mm256_unpackhi_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t27 = _mm256_unpackhi_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t29 = _mm256_unpackhi_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);
          const __m256i t31 = _mm256_unpackhi_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
          const __m256i u26 = _mm256_unpacklo_epi32(t25, t27);
          const __m256i u27 = _mm256_unpackhi_epi32(t25, t27);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
          const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);
          const __m256i u30 = _mm256_unpacklo_epi32(t29, t31);
          const __m256i u31 = _mm256_unpackhi_epi32(t29, t31);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
          const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);
          const __m256i w28 = _mm256_unpacklo_epi64(u26, u30);
          const __m256i w29 = _mm256_unpackhi_epi64(u26, u30);
          const __m256i w30 = _mm256_unpacklo_epi64(u27, u31);
          const __m256i w31 = _mm256_unpackhi_epi64(u27, u31);
          const __m256i t32 = _mm256_unpacklo_epi16(v32, v33);
          const __m256i t33 = _mm256_unpackhi_epi16(v32, v33);
          const __m256i t34 = _mm256_unpacklo_epi16(v34, v35);
          const __m256i t35 = _mm256_unpackhi_epi16(v34, v35);
          const __m256i t36 = _mm256_unpacklo_epi16(v36, v37);
          const __m256i t37 = _mm256_unpackhi_epi16(v36, v37);
          const __m256i t38 = _mm256_unpacklo_epi16(v38, v39);
          const __m256i t39 = _mm256_unpackhi_epi16(v38, v39);

          const __m256i u32 = _mm256_unpacklo_epi32(t32, t34);
          const __m256i u33 = _mm256_unpackhi_epi32(t32, t34);
          const __m256i u34 = _mm256_unpacklo_epi32(t33, t35);
          const __m256i u35 = _mm256_unpackhi_epi32(t33, t35);
          const __m256i u36 = _mm256_unpacklo_epi32(t36, t38);
          const __m256i u37 = _mm256_unpackhi_epi32(t36, t38);
          const __m256i u38 = _mm256_unpacklo_epi32(t37, t39);
          const __m256i u39 = _mm256_unpackhi_epi32(t37, t39);

          const __m256i w32 = _mm256_unpacklo_epi64(u32, u36);
          const __m256i w33 = _mm256_unpackhi_epi64(u32, u36);
          const __m256i w34 = _mm256_unpacklo_epi64(u33, u37);
          const __m256i w35 = _mm256_unpackhi_epi64(u33, u37);
          const __m256i w36 = _mm256_unpacklo_epi64(u34, u38);
          const __m256i w37 = _mm256_unpackhi_epi64(u34, u38);
          const __m256i w38 = _mm256_unpacklo_epi64(u35, u39);
          const __m256i w39 = _mm256_unpackhi_epi64(u35, u39);
          const __m256i t40 = _mm256_unpacklo_epi16(v40, v41);
          const __m256i t41 = _mm256_unpackhi_epi16(v40, v41);
          const __m256i t42 = _mm256_unpacklo_epi16(v42, v43);
          const __m256i t43 = _mm256_unpackhi_epi16(v42, v43);
          const __m256i t44 = _mm256_unpacklo_epi16(v44, v45);
          const __m256i t45 = _mm256_unpackhi_epi16(v44, v45);
          const __m256i t46 = _mm256_unpacklo_epi16(v46, v47);
          const __m256i t47 = _mm256_unpackhi_epi16(v46, v47);

          const __m256i u40 = _mm256_unpacklo_epi32(t40, t42);
          const __m256i u41 = _mm256_unpackhi_epi32(t40, t42);
          const __m256i u42 = _mm256_unpacklo_epi32(t41, t43);
          const __m256i u43 = _mm256_unpackhi_epi32(t41, t43);
          const __m256i u44 = _mm256_unpacklo_epi32(t44, t46);
          const __m256i u45 = _mm256_unpackhi_epi32(t44, t46);
          const __m256i u46 = _mm256_unpacklo_epi32(t45, t47);
          const __m256i u47 = _mm256_unpackhi_epi32(t45, t47);

          const __m256i w40 = _mm256_unpacklo_epi64(u40, u44);
          const __m256i w41 = _mm256_unpackhi_epi64(u40, u44);
          const __m256i w42 = _mm256_unpacklo_epi64(u41, u45);
          const __m256i w43 = _mm256_unpackhi_epi64(u41, u45);
          const __m256i w44 = _mm256_unpacklo_epi64(u42, u46);
          const __m256i w45 = _mm256_unpackhi_epi64(u42, u46);
          const __m256i w46 = _mm256_unpacklo_epi64(u43, u47);
          const __m256i w47 = _mm256_unpackhi_epi64(u43, u47);
          const __m256i t48 = _mm256_unpacklo_epi16(v48, v49);
          const __m256i t49 = _mm256_unpackhi_epi16(v48, v49);
          const __m256i t50 = _mm256_unpacklo_epi16(v50, v51);
          const __m256i t51 = _mm256_unpackhi_epi16(v50, v51);
          const __m256i t52 = _mm256_unpacklo_epi16(v52, v53);
          const __m256i t53 = _mm256_unpackhi_epi16(v52, v53);
          const __m256i t54 = _mm256_unpacklo_epi16(v54, v55);
          const __m256i t55 = _mm256_unpackhi_epi16(v54, v55);

          const __m256i u48 = _mm256_unpacklo_epi32(t48, t50);
          const __m256i u49 = _mm256_unpackhi_epi32(t48, t50);
          const __m256i u50 = _mm256_unpacklo_epi32(t49, t51);
          const __m256i u51 = _mm256_unpackhi_epi32(t49, t51);
          const __m256i u52 = _mm256_unpacklo_epi32(t52, t54);
          const __m256i u53 = _mm256_unpackhi_epi32(t52, t54);
          const __m256i u54 = _mm256_unpacklo_epi32(t53, t55);
          const __m256i u55 = _mm256_unpackhi_epi32(t53, t55);

          const __m256i w48 = _mm256_unpacklo_epi64(u48, u52);
          const __m256i w49 = _mm256_unpackhi_epi64(u48, u52);
          const __m256i w50 = _mm256_unpacklo_epi64(u49, u53);
          const __m256i w51 = _mm256_unpackhi_epi64(u49, u53);
          const __m256i w52 = _mm256_unpacklo_epi64(u50, u54);
          const __m256i w53 = _mm256_unpackhi_epi64(u50, u54);
          const __m256i w54 = _mm256_unpacklo_epi64(u51, u55);
          const __m256i w55 = _mm256_unpackhi_epi64(u51, u55);
          const __m256i t56 = _mm256_unpacklo_epi16(v56, v57);
          const __m256i t57 = _mm256_unpackhi_epi16(v56, v57);
          const __m256i t58 = _mm256_unpacklo_epi16(v58, v59);
          const __m256i t59 = _mm256_unpackhi_epi16(v58, v59);
          const __m256i t60 = _mm256_unpacklo_epi16(v60, v61);
          const __m256i t61 = _mm256_unpackhi_epi16(v60, v61);
          const __m256i t62 = _mm256_unpacklo_epi16(v62, v63);
          const __m256i t63 = _mm256_unpackhi_epi16(v62, v63);

          const __m256i u56 = _mm256_unpacklo_epi32(t56, t58);
          const __m256i u57 = _mm256_unpackhi_epi32(t56, t58);
          const __m256i u58 = _mm256_unpacklo_epi32(t57, t59);
          const __m256i u59 = _mm256_unpackhi_epi32(t57, t59);
          const __m256i u60 = _mm256_unpacklo_epi32(t60, t62);
          const __m256i u61 = _mm256_unpackhi_epi32(t60, t62);
          const __m256i u62 = _mm256_unpacklo_epi32(t61, t63);
          const __m256i u63 = _mm256_unpackhi_epi32(t61, t63);

          const __m256i w56 = _mm256_unpacklo_epi64(u56, u60);
          const __m256i w57 = _mm256_unpackhi_epi64(u56, u60);
          const __m256i w58 = _mm256_unpacklo_epi64(u57, u61);
          const __m256i w59 = _mm256_unpackhi_epi64(u57, u61);
          const __m256i w60 = _mm256_unpacklo_epi64(u58, u62);
          const __m256i w61 = _mm256_unpackhi_epi64(u58, u62);
          const __m256i w62 = _mm256_unpacklo_epi64(u59, u63);
          const __m256i w63 = _mm256_unpackhi_epi64(u59, u63);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w32));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w40));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w48));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w56));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w33));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w41));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w49));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w57));
          _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_castsi256_si128(w18));
          _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_castsi256_si128(w26));
          _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_castsi256_si128(w34));
          _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_castsi256_si128(w42));
          _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_castsi256_si128(w50));
          _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_castsi256_si128(w58));
          _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_castsi256_si128(w19));
          _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_castsi256_si128(w27));
          _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_castsi256_si128(w35));
          _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_castsi256_si128(w43));
          _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_castsi256_si128(w51));
          _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_castsi256_si128(w59));
          _mm_storeu_si128((__m128i*) (packed_w + 256), _mm256_castsi256_si128(w4));
          _mm_storeu_si128((__m128i*) (packed_w + 264), _mm256_castsi256_si128(w12));
          _mm_storeu_si128((__m128i*) (packed_w + 272), _mm256_castsi256_si128(w20));
          _mm_storeu_si128((__m128i*) (packed_w + 280), _mm256_castsi256_si128(w28));
          _mm_storeu_si128((__m128i*) (packed_w + 288), _mm256_castsi256_si128(w36));
          _mm_storeu_si128((__m128i*) (packed_w + 296), _mm256_castsi256_si128(w44));
          _mm_storeu_si128((__m128i*) (packed_w + 304), _mm256_castsi256_si128(w52));
          _mm_storeu_si128((__m128i*) (packed_w + 312), _mm256_castsi256_si128(w60));
          _mm_storeu_si128((__m128i*) (packed_w + 320), _mm256_castsi256_si128(w5));
          _mm_storeu_si128((__m128i*) (packed_w + 328), _mm256_castsi256_si128(w13));
          _mm_storeu_si128((__m128i*) (packed_w + 336), _mm256_castsi256_si128(w21));
          _mm_storeu_si128((__m128i*) (packed_w + 344), _mm256_castsi256_si128(w29));
          _mm_storeu_si128((__m128i*) (packed_w + 352), _mm256_castsi256_si128(w37));
          _mm_storeu_si128((__m128i*) (packed_w + 360), _mm256_castsi256_si128(w45));
          _mm_storeu_si128((__m128i*) (packed_w + 368), _mm256_castsi256_si128(w53));
          _mm_storeu_si128((__m128i*) (packed_w + 376), _mm256_castsi256_si128(w61));
          _mm_storeu_si128((__m128i*) (packed_w + 384), _mm256_castsi256_si128(w6));
          _mm_storeu_si128((__m128i*) (packed_w + 392), _mm256_castsi256_si128(w14));
          _mm_storeu_si128((__m128i*) (packed_w + 400), _mm256_castsi256_si128(w22));
          _mm_storeu_si128((__m128i*) (packed_w + 408), _mm256_castsi256_si128(w30));
          _mm_storeu_si128((__m128i*) (packed_w + 416), _mm256_castsi256_si128(w38));
          _mm_storeu_si128((__m128i*) (packed_w + 424), _mm256_castsi256_si128(w46));
          _mm_storeu_si128((__m128i*) (packed_w + 432), _mm256_castsi256_si128(w54));
          _mm_storeu_si128((__m128i*) (packed_w + 440), _mm256_castsi256_si128(w62));
          _mm_storeu_si128((__m128i*) (packed_w + 448), _mm256_castsi256_si128(w7));
          _mm_storeu_si128((__m128i*) (packed_w + 456), _mm256_castsi256_si128(w15));
          _mm_storeu_si128((__m128i*) (packed_w + 464), _mm256_castsi256_si128(w23));
          _mm_storeu_si128((__m128i*) (packed_w + 472), _mm256_castsi256_si128(w31));
          _mm_storeu_si128((__m128i*) (packed_w + 480), _mm256_castsi256_si128(w39));
          _mm_storeu_si128((__m128i*) (packed_w + 488), _mm256_castsi256_si128(w47));
          _mm_storeu_si128((__m128i*) (packed_w + 496), _mm256_castsi256_si128(w55));
          _mm_storeu_si128((__m128i*) (packed_w + 504), _mm256_castsi256_si128(w63));
          packed_w += 512;
        }

        if (k & 4) {
          __m256i v0 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 4;
          __m256i v1 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 4;
          __m256i v2 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 4;
          __m256i v3 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 4;
          __m256i v4 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 4;
          __m256i v5 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 4;
          __m256i v6 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 4;
          __m256i v7 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 4;
          __m256i v8 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 4;
          __m256i v9 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 4;
          __m256i v10 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 4;
          __m256i v11 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 4;
          __m256i v12 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 4;
          __m256i v13 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 4;
          __m256i v14 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 4;
          __m256i v15 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w15));
          w15 += 4;
          __m256i v16 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w16));
          w16 += 4;
          __m256i v17 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w17));
          w17 += 4;
          __m256i v18 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w18));
          w18 += 4;
          __m256i v19 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w19));
          w19 += 4;
          __m256i v20 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w20));
          w20 += 4;
          __m256i v21 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w21));
          w21 += 4;
          __m256i v22 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w22));
          w22 += 4;
          __m256i v23 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w23));
          w23 += 4;
          __m256i v24 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w24));
          w24 += 4;
          __m256i v25 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w25));
          w25 += 4;
          __m256i v26 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w26));
          w26 += 4;
          __m256i v27 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w27));
          w27 += 4;
          __m256i v28 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w28));
          w28 += 4;
          __m256i v29 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w29));
          w29 += 4;
          __m256i v30 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w30));
          w30 += 4;
          __m256i v31 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w31));
          w31 += 4;
          __m256i v32 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w32));
          w32 += 4;
          __m256i v33 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w33));
          w33 += 4;
          __m256i v34 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w34));
          w34 += 4;
          __m256i v35 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w35));
          w35 += 4;
          __m256i v36 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w36));
          w36 += 4;
          __m256i v37 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w37));
          w37 += 4;
          __m256i v38 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w38));
          w38 += 4;
          __m256i v39 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w39));
          w39 += 4;
          __m256i v40 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w40));
          w40 += 4;
          __m256i v41 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w41));
          w41 += 4;
          __m256i v42 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w42));
          w42 += 4;
          __m256i v43 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w43));
          w43 += 4;
          __m256i v44 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w44));
          w44 += 4;
          __m256i v45 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w45));
          w45 += 4;
          __m256i v46 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w46));
          w46 += 4;
          __m256i v47 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w47));
          w47 += 4;
          __m256i v48 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w48));
          w48 += 4;
          __m256i v49 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w49));
          w49 += 4;
          __m256i v50 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w50));
          w50 += 4;
          __m256i v51 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w51));
          w51 += 4;
          __m256i v52 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w52));
          w52 += 4;
          __m256i v53 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w53));
          w53 += 4;
          __m256i v54 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w54));
          w54 += 4;
          __m256i v55 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w55));
          w55 += 4;
          __m256i v56 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w56));
          w56 += 4;
          __m256i v57 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w57));
          w57 += 4;
          __m256i v58 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w58));
          w58 += 4;
          __m256i v59 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w59));
          w59 += 4;
          __m256i v60 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w60));
          w60 += 4;
          __m256i v61 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w61));
          w61 += 4;
          __m256i v62 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w62));
          w62 += 4;
          __m256i v63 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
          const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
          const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
          const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
          const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
          const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
          const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
          const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
          const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);
          const __m256i t32 = _mm256_unpacklo_epi16(v32, v33);
          const __m256i t34 = _mm256_unpacklo_epi16(v34, v35);
          const __m256i t36 = _mm256_unpacklo_epi16(v36, v37);
          const __m256i t38 = _mm256_unpacklo_epi16(v38, v39);

          const __m256i u32 = _mm256_unpacklo_epi32(t32, t34);
          const __m256i u33 = _mm256_unpackhi_epi32(t32, t34);
          const __m256i u36 = _mm256_unpacklo_epi32(t36, t38);
          const __m256i u37 = _mm256_unpackhi_epi32(t36, t38);

          const __m256i w32 = _mm256_unpacklo_epi64(u32, u36);
          const __m256i w33 = _mm256_unpackhi_epi64(u32, u36);
          const __m256i w34 = _mm256_unpacklo_epi64(u33, u37);
          const __m256i w35 = _mm256_unpackhi_epi64(u33, u37);
          const __m256i t40 = _mm256_unpacklo_epi16(v40, v41);
          const __m256i t42 = _mm256_unpacklo_epi16(v42, v43);
          const __m256i t44 = _mm256_unpacklo_epi16(v44, v45);
          const __m256i t46 = _mm256_unpacklo_epi16(v46, v47);

          const __m256i u40 = _mm256_unpacklo_epi32(t40, t42);
          const __m256i u41 = _mm256_unpackhi_epi32(t40, t42);
          const __m256i u44 = _mm256_unpacklo_epi32(t44, t46);
          const __m256i u45 = _mm256_unpackhi_epi32(t44, t46);

          const __m256i w40 = _mm256_unpacklo_epi64(u40, u44);
          const __m256i w41 = _mm256_unpackhi_epi64(u40, u44);
          const __m256i w42 = _mm256_unpacklo_epi64(u41, u45);
          const __m256i w43 = _mm256_unpackhi_epi64(u41, u45);
          const __m256i t48 = _mm256_unpacklo_epi16(v48, v49);
          const __m256i t50 = _mm256_unpacklo_epi16(v50, v51);
          const __m256i t52 = _mm256_unpacklo_epi16(v52, v53);
          const __m256i t54 = _mm256_unpacklo_epi16(v54, v55);

          const __m256i u48 = _mm256_unpacklo_epi32(t48, t50);
          const __m256i u49 = _mm256_unpackhi_epi32(t48, t50);
          const __m256i u52 = _mm256_unpacklo_epi32(t52, t54);
          const __m256i u53 = _mm256_unpackhi_epi32(t52, t54);

          const __m256i w48 = _mm256_unpacklo_epi64(u48, u52);
          const __m256i w49 = _mm256_unpackhi_epi64(u48, u52);
          const __m256i w50 = _mm256_unpacklo_epi64(u49, u53);
          const __m256i w51 = _mm256_unpackhi_epi64(u49, u53);
          const __m256i t56 = _mm256_unpacklo_epi16(v56, v57);
          const __m256i t58 = _mm256_unpacklo_epi16(v58, v59);
          const __m256i t60 = _mm256_unpacklo_epi16(v60, v61);
          const __m256i t62 = _mm256_unpacklo_epi16(v62, v63);

          const __m256i u56 = _mm256_unpacklo_epi32(t56, t58);
          const __m256i u57 = _mm256_unpackhi_epi32(t56, t58);
          const __m256i u60 = _mm256_unpacklo_epi32(t60, t62);
          const __m256i u61 = _mm256_unpackhi_epi32(t60, t62);

          const __m256i w56 = _mm256_unpacklo_epi64(u56, u60);
          const __m256i w57 = _mm256_unpackhi_epi64(u56, u60);
          const __m256i w58 = _mm256_unpacklo_epi64(u57, u61);
          const __m256i w59 = _mm256_unpackhi_epi64(u57, u61);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w32));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w40));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w48));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w56));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w33));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w41));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w49));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w57));
          _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_castsi256_si128(w18));
          _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_castsi256_si128(w26));
          _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_castsi256_si128(w34));
          _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_castsi256_si128(w42));
          _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_castsi256_si128(w50));
          _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_castsi256_si128(w58));
          _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_castsi256_si128(w19));
          _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_castsi256_si128(w27));
          _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_castsi256_si128(w35));
          _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_castsi256_si128(w43));
          _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_castsi256_si128(w51));
          _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_castsi256_si128(w59));
          packed_w += 256;
        }

        if (k & 2) {
          __m256i v0 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w0)));
          w0 += 2;
          __m256i v1 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w1)));
          w1 += 2;
          __m256i v2 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w2)));
          w2 += 2;
          __m256i v3 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w3)));
          w3 += 2;
          __m256i v4 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w4)));
          w4 += 2;
          __m256i v5 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w5)));
          w5 += 2;
          __m256i v6 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w6)));
          w6 += 2;
          __m256i v7 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w7)));
          w7 += 2;
          __m256i v8 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w8)));
          w8 += 2;
          __m256i v9 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w9)));
          w9 += 2;
          __m256i v10 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w10)));
          w10 += 2;
          __m256i v11 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w11)));
          w11 += 2;
          __m256i v12 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w12)));
          w12 += 2;
          __m256i v13 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w13)));
          w13 += 2;
          __m256i v14 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w14)));
          w14 += 2;
          __m256i v15 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w15)));
          w15 += 2;
          __m256i v16 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w16)));
          w16 += 2;
          __m256i v17 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w17)));
          w17 += 2;
          __m256i v18 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w18)));
          w18 += 2;
          __m256i v19 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w19)));
          w19 += 2;
          __m256i v20 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w20)));
          w20 += 2;
          __m256i v21 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w21)));
          w21 += 2;
          __m256i v22 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w22)));
          w22 += 2;
          __m256i v23 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w23)));
          w23 += 2;
          __m256i v24 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w24)));
          w24 += 2;
          __m256i v25 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w25)));
          w25 += 2;
          __m256i v26 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w26)));
          w26 += 2;
          __m256i v27 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w27)));
          w27 += 2;
          __m256i v28 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w28)));
          w28 += 2;
          __m256i v29 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w29)));
          w29 += 2;
          __m256i v30 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w30)));
          w30 += 2;
          __m256i v31 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w31)));
          w31 += 2;
          __m256i v32 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w32)));
          w32 += 2;
          __m256i v33 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w33)));
          w33 += 2;
          __m256i v34 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w34)));
          w34 += 2;
          __m256i v35 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w35)));
          w35 += 2;
          __m256i v36 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w36)));
          w36 += 2;
          __m256i v37 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w37)));
          w37 += 2;
          __m256i v38 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w38)));
          w38 += 2;
          __m256i v39 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w39)));
          w39 += 2;
          __m256i v40 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w40)));
          w40 += 2;
          __m256i v41 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w41)));
          w41 += 2;
          __m256i v42 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w42)));
          w42 += 2;
          __m256i v43 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w43)));
          w43 += 2;
          __m256i v44 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w44)));
          w44 += 2;
          __m256i v45 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w45)));
          w45 += 2;
          __m256i v46 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w46)));
          w46 += 2;
          __m256i v47 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w47)));
          w47 += 2;
          __m256i v48 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w48)));
          w48 += 2;
          __m256i v49 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w49)));
          w49 += 2;
          __m256i v50 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w50)));
          w50 += 2;
          __m256i v51 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w51)));
          w51 += 2;
          __m256i v52 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w52)));
          w52 += 2;
          __m256i v53 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w53)));
          w53 += 2;
          __m256i v54 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w54)));
          w54 += 2;
          __m256i v55 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w55)));
          w55 += 2;
          __m256i v56 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w56)));
          w56 += 2;
          __m256i v57 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w57)));
          w57 += 2;
          __m256i v58 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w58)));
          w58 += 2;
          __m256i v59 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w59)));
          w59 += 2;
          __m256i v60 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w60)));
          w60 += 2;
          __m256i v61 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w61)));
          w61 += 2;
          __m256i v62 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w62)));
          w62 += 2;
          __m256i v63 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i t32 = _mm256_unpacklo_epi16(v32, v33);
          const __m256i t34 = _mm256_unpacklo_epi16(v34, v35);
          const __m256i t36 = _mm256_unpacklo_epi16(v36, v37);
          const __m256i t38 = _mm256_unpacklo_epi16(v38, v39);

          const __m256i u32 = _mm256_unpacklo_epi32(t32, t34);
          const __m256i u36 = _mm256_unpacklo_epi32(t36, t38);

          const __m256i w32 = _mm256_unpacklo_epi64(u32, u36);
          const __m256i w33 = _mm256_unpackhi_epi64(u32, u36);
          const __m256i t40 = _mm256_unpacklo_epi16(v40, v41);
          const __m256i t42 = _mm256_unpacklo_epi16(v42, v43);
          const __m256i t44 = _mm256_unpacklo_epi16(v44, v45);
          const __m256i t46 = _mm256_unpacklo_epi16(v46, v47);

          const __m256i u40 = _mm256_unpacklo_epi32(t40, t42);
          const __m256i u44 = _mm256_unpacklo_epi32(t44, t46);

          const __m256i w40 = _mm256_unpacklo_epi64(u40, u44);
          const __m256i w41 = _mm256_unpackhi_epi64(u40, u44);
          const __m256i t48 = _mm256_unpacklo_epi16(v48, v49);
          const __m256i t50 = _mm256_unpacklo_epi16(v50, v51);
          const __m256i t52 = _mm256_unpacklo_epi16(v52, v53);
          const __m256i t54 = _mm256_unpacklo_epi16(v54, v55);

          const __m256i u48 = _mm256_unpacklo_epi32(t48, t50);
          const __m256i u52 = _mm256_unpacklo_epi32(t52, t54);

          const __m256i w48 = _mm256_unpacklo_epi64(u48, u52);
          const __m256i w49 = _mm256_unpackhi_epi64(u48, u52);
          const __m256i t56 = _mm256_unpacklo_epi16(v56, v57);
          const __m256i t58 = _mm256_unpacklo_epi16(v58, v59);
          const __m256i t60 = _mm256_unpacklo_epi16(v60, v61);
          const __m256i t62 = _mm256_unpacklo_epi16(v62, v63);

          const __m256i u56 = _mm256_unpacklo_epi32(t56, t58);
          const __m256i u60 = _mm256_unpacklo_epi32(t60, t62);

          const __m256i w56 = _mm256_unpacklo_epi64(u56, u60);
          const __m256i w57 = _mm256_unpackhi_epi64(u56, u60);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w32));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w40));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w48));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w56));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w33));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w41));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w49));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w57));
          packed_w += 128;
        }

        if (k & 1) {
          __m256i v0 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w0, 0));
          w0 += 1;
          __m256i v1 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w1, 0));
          w1 += 1;
          __m256i v2 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w2, 0));
          w2 += 1;
          __m256i v3 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w3, 0));
          w3 += 1;
          __m256i v4 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w4, 0));
          w4 += 1;
          __m256i v5 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w5, 0));
          w5 += 1;
          __m256i v6 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w6, 0));
          w6 += 1;
          __m256i v7 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w7, 0));
          w7 += 1;
          __m256i v8 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w8, 0));
          w8 += 1;
          __m256i v9 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w9, 0));
          w9 += 1;
          __m256i v10 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w10, 0));
          w10 += 1;
          __m256i v11 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w11, 0));
          w11 += 1;
          __m256i v12 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w12, 0));
          w12 += 1;
          __m256i v13 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w13, 0));
          w13 += 1;
          __m256i v14 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w14, 0));
          w14 += 1;
          __m256i v15 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w15, 0));
          w15 += 1;
          __m256i v16 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w16, 0));
          w16 += 1;
          __m256i v17 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w17, 0));
          w17 += 1;
          __m256i v18 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w18, 0));
          w18 += 1;
          __m256i v19 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w19, 0));
          w19 += 1;
          __m256i v20 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w20, 0));
          w20 += 1;
          __m256i v21 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w21, 0));
          w21 += 1;
          __m256i v22 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w22, 0));
          w22 += 1;
          __m256i v23 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w23, 0));
          w23 += 1;
          __m256i v24 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w24, 0));
          w24 += 1;
          __m256i v25 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w25, 0));
          w25 += 1;
          __m256i v26 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w26, 0));
          w26 += 1;
          __m256i v27 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w27, 0));
          w27 += 1;
          __m256i v28 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w28, 0));
          w28 += 1;
          __m256i v29 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w29, 0));
          w29 += 1;
          __m256i v30 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w30, 0));
          w30 += 1;
          __m256i v31 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w31, 0));
          w31 += 1;
          __m256i v32 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w32, 0));
          w32 += 1;
          __m256i v33 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w33, 0));
          w33 += 1;
          __m256i v34 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w34, 0));
          w34 += 1;
          __m256i v35 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w35, 0));
          w35 += 1;
          __m256i v36 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w36, 0));
          w36 += 1;
          __m256i v37 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w37, 0));
          w37 += 1;
          __m256i v38 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w38, 0));
          w38 += 1;
          __m256i v39 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w39, 0));
          w39 += 1;
          __m256i v40 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w40, 0));
          w40 += 1;
          __m256i v41 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w41, 0));
          w41 += 1;
          __m256i v42 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w42, 0));
          w42 += 1;
          __m256i v43 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w43, 0));
          w43 += 1;
          __m256i v44 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w44, 0));
          w44 += 1;
          __m256i v45 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w45, 0));
          w45 += 1;
          __m256i v46 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w46, 0));
          w46 += 1;
          __m256i v47 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w47, 0));
          w47 += 1;
          __m256i v48 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w48, 0));
          w48 += 1;
          __m256i v49 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w49, 0));
          w49 += 1;
          __m256i v50 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w50, 0));
          w50 += 1;
          __m256i v51 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w51, 0));
          w51 += 1;
          __m256i v52 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w52, 0));
          w52 += 1;
          __m256i v53 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w53, 0));
          w53 += 1;
          __m256i v54 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w54, 0));
          w54 += 1;
          __m256i v55 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w55, 0));
          w55 += 1;
          __m256i v56 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w56, 0));
          w56 += 1;
          __m256i v57 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w57, 0));
          w57 += 1;
          __m256i v58 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w58, 0));
          w58 += 1;
          __m256i v59 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w59, 0));
          w59 += 1;
          __m256i v60 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w60, 0));
          w60 += 1;
          __m256i v61 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w61, 0));
          w61 += 1;
          __m256i v62 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w62, 0));
          w62 += 1;
          __m256i v63 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i t32 = _mm256_unpacklo_epi16(v32, v33);
          const __m256i t34 = _mm256_unpacklo_epi16(v34, v35);
          const __m256i t36 = _mm256_unpacklo_epi16(v36, v37);
          const __m256i t38 = _mm256_unpacklo_epi16(v38, v39);

          const __m256i u32 = _mm256_unpacklo_epi32(t32, t34);
          const __m256i u36 = _mm256_unpacklo_epi32(t36, t38);

          const __m256i w32 = _mm256_unpacklo_epi64(u32, u36);
          const __m256i t40 = _mm256_unpacklo_epi16(v40, v41);
          const __m256i t42 = _mm256_unpacklo_epi16(v42, v43);
          const __m256i t44 = _mm256_unpacklo_epi16(v44, v45);
          const __m256i t46 = _mm256_unpacklo_epi16(v46, v47);

          const __m256i u40 = _mm256_unpacklo_epi32(t40, t42);
          const __m256i u44 = _mm256_unpacklo_epi32(t44, t46);

          const __m256i w40 = _mm256_unpacklo_epi64(u40, u44);
          const __m256i t48 = _mm256_unpacklo_epi16(v48, v49);
          const __m256i t50 = _mm256_unpacklo_epi16(v50, v51);
          const __m256i t52 = _mm256_unpacklo_epi16(v52, v53);
          const __m256i t54 = _mm256_unpacklo_epi16(v54, v55);

          const __m256i u48 = _mm256_unpacklo_epi32(t48, t50);
          const __m256i u52 = _mm256_unpacklo_epi32(t52, t54);

          const __m256i w48 = _mm256_unpacklo_epi64(u48, u52);
          const __m256i t56 = _mm256_unpacklo_epi16(v56, v57);
          const __m256i t58 = _mm256_unpacklo_epi16(v58, v59);
          const __m256i t60 = _mm256_unpacklo_epi16(v60, v61);
          const __m256i t62 = _mm256_unpacklo_epi16(v62, v63);

          const __m256i u56 = _mm256_unpacklo_epi32(t56, t58);
          const __m256i u60 = _mm256_unpacklo_epi32(t60, t62);

          const __m256i w56 = _mm256_unpacklo_epi64(u56, u60);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w32));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w40));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w48));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w56));
          packed_w += 64;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
