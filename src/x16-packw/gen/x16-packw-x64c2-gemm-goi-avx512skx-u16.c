// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-packw/c2-avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"


void xnn_x16_packw_gemm_goi_ukernel_x64c2__avx512skx_u16(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
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
  assert(kr == 2);
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

      const ptrdiff_t stride1 = (ptrdiff_t) n_stride;
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

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) (w0 + stride4));
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) (w8 + stride4));
        const __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v4, 1);
        const __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v8), v12, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) (w0 + stride1));
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) (w0 + stride5));
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) (w8 + stride1));
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) (w8 + stride5));
        const __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v5, 1);
        const __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v9), v13, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) (w0 + stride2));
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) (w0 + stride6));
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) (w8 + stride2));
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) (w8 + stride6));
        const __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v6, 1);
        const __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v10), v14, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) (w0 + stride3));
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) (w0 + stride7));
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) (w8 + stride3));
        const __m256i v15 = _mm256_loadu_si256((const __m256i*) (w8 + stride7));
        const __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v7, 1);
        const __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v11), v15, 1);
        w0 += 16;
        w8 += 16;

        const __m512i t0 = _mm512_unpacklo_epi32(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi32(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi32(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi32(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi32(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi32(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi32(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi32(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi64(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi64(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi64(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi64(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi64(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi64(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi64(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi64(t5, t7);

        const __m512i out0_0 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_4 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 512), out0_4);
        const __m512i out0_1 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_5 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 640), out0_5);
        const __m512i out0_2 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_6 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 768), out0_6);
        const __m512i out0_3 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_7 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 896), out0_7);
        const __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        const __m256i v20 = _mm256_loadu_si256((const __m256i*) (w16 + stride4));
        const __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        const __m256i v28 = _mm256_loadu_si256((const __m256i*) (w24 + stride4));
        const __m512i z16 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v20, 1);
        const __m512i z20 = _mm512_inserti64x4(_mm512_castsi256_si512(v24), v28, 1);
        const __m256i v17 = _mm256_loadu_si256((const __m256i*) (w16 + stride1));
        const __m256i v21 = _mm256_loadu_si256((const __m256i*) (w16 + stride5));
        const __m256i v25 = _mm256_loadu_si256((const __m256i*) (w24 + stride1));
        const __m256i v29 = _mm256_loadu_si256((const __m256i*) (w24 + stride5));
        const __m512i z17 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v21, 1);
        const __m512i z21 = _mm512_inserti64x4(_mm512_castsi256_si512(v25), v29, 1);
        const __m256i v18 = _mm256_loadu_si256((const __m256i*) (w16 + stride2));
        const __m256i v22 = _mm256_loadu_si256((const __m256i*) (w16 + stride6));
        const __m256i v26 = _mm256_loadu_si256((const __m256i*) (w24 + stride2));
        const __m256i v30 = _mm256_loadu_si256((const __m256i*) (w24 + stride6));
        const __m512i z18 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v22, 1);
        const __m512i z22 = _mm512_inserti64x4(_mm512_castsi256_si512(v26), v30, 1);
        const __m256i v19 = _mm256_loadu_si256((const __m256i*) (w16 + stride3));
        const __m256i v23 = _mm256_loadu_si256((const __m256i*) (w16 + stride7));
        const __m256i v27 = _mm256_loadu_si256((const __m256i*) (w24 + stride3));
        const __m256i v31 = _mm256_loadu_si256((const __m256i*) (w24 + stride7));
        const __m512i z19 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v23, 1);
        const __m512i z23 = _mm512_inserti64x4(_mm512_castsi256_si512(v27), v31, 1);
        w16 += 16;
        w24 += 16;

        const __m512i t16 = _mm512_unpacklo_epi32(z16, z17);
        const __m512i t17 = _mm512_unpackhi_epi32(z16, z17);
        const __m512i t18 = _mm512_unpacklo_epi32(z18, z19);
        const __m512i t19 = _mm512_unpackhi_epi32(z18, z19);
        const __m512i t20 = _mm512_unpacklo_epi32(z20, z21);
        const __m512i t21 = _mm512_unpackhi_epi32(z20, z21);
        const __m512i t22 = _mm512_unpacklo_epi32(z22, z23);
        const __m512i t23 = _mm512_unpackhi_epi32(z22, z23);

        const __m512i u16 = _mm512_unpacklo_epi64(t16, t18);
        const __m512i u17 = _mm512_unpackhi_epi64(t16, t18);
        const __m512i u18 = _mm512_unpacklo_epi64(t17, t19);
        const __m512i u19 = _mm512_unpackhi_epi64(t17, t19);
        const __m512i u20 = _mm512_unpacklo_epi64(t20, t22);
        const __m512i u21 = _mm512_unpackhi_epi64(t20, t22);
        const __m512i u22 = _mm512_unpacklo_epi64(t21, t23);
        const __m512i u23 = _mm512_unpackhi_epi64(t21, t23);

        const __m512i out16_0 = _mm512_shuffle_i32x4(u16, u20, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out16_4 = _mm512_shuffle_i32x4(u16, u20, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out16_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 544), out16_4);
        const __m512i out16_1 = _mm512_shuffle_i32x4(u17, u21, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out16_5 = _mm512_shuffle_i32x4(u17, u21, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 160), out16_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 672), out16_5);
        const __m512i out16_2 = _mm512_shuffle_i32x4(u18, u22, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out16_6 = _mm512_shuffle_i32x4(u18, u22, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 288), out16_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 800), out16_6);
        const __m512i out16_3 = _mm512_shuffle_i32x4(u19, u23, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out16_7 = _mm512_shuffle_i32x4(u19, u23, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 416), out16_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 928), out16_7);
        const __m256i v32 = _mm256_loadu_si256((const __m256i*) w32);
        const __m256i v36 = _mm256_loadu_si256((const __m256i*) (w32 + stride4));
        const __m256i v40 = _mm256_loadu_si256((const __m256i*) w40);
        const __m256i v44 = _mm256_loadu_si256((const __m256i*) (w40 + stride4));
        const __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v36, 1);
        const __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v40), v44, 1);
        const __m256i v33 = _mm256_loadu_si256((const __m256i*) (w32 + stride1));
        const __m256i v37 = _mm256_loadu_si256((const __m256i*) (w32 + stride5));
        const __m256i v41 = _mm256_loadu_si256((const __m256i*) (w40 + stride1));
        const __m256i v45 = _mm256_loadu_si256((const __m256i*) (w40 + stride5));
        const __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v37, 1);
        const __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v41), v45, 1);
        const __m256i v34 = _mm256_loadu_si256((const __m256i*) (w32 + stride2));
        const __m256i v38 = _mm256_loadu_si256((const __m256i*) (w32 + stride6));
        const __m256i v42 = _mm256_loadu_si256((const __m256i*) (w40 + stride2));
        const __m256i v46 = _mm256_loadu_si256((const __m256i*) (w40 + stride6));
        const __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v38, 1);
        const __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v42), v46, 1);
        const __m256i v35 = _mm256_loadu_si256((const __m256i*) (w32 + stride3));
        const __m256i v39 = _mm256_loadu_si256((const __m256i*) (w32 + stride7));
        const __m256i v43 = _mm256_loadu_si256((const __m256i*) (w40 + stride3));
        const __m256i v47 = _mm256_loadu_si256((const __m256i*) (w40 + stride7));
        const __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v39, 1);
        const __m512i z39 = _mm512_inserti64x4(_mm512_castsi256_si512(v43), v47, 1);
        w32 += 16;
        w40 += 16;

        const __m512i t32 = _mm512_unpacklo_epi32(z32, z33);
        const __m512i t33 = _mm512_unpackhi_epi32(z32, z33);
        const __m512i t34 = _mm512_unpacklo_epi32(z34, z35);
        const __m512i t35 = _mm512_unpackhi_epi32(z34, z35);
        const __m512i t36 = _mm512_unpacklo_epi32(z36, z37);
        const __m512i t37 = _mm512_unpackhi_epi32(z36, z37);
        const __m512i t38 = _mm512_unpacklo_epi32(z38, z39);
        const __m512i t39 = _mm512_unpackhi_epi32(z38, z39);

        const __m512i u32 = _mm512_unpacklo_epi64(t32, t34);
        const __m512i u33 = _mm512_unpackhi_epi64(t32, t34);
        const __m512i u34 = _mm512_unpacklo_epi64(t33, t35);
        const __m512i u35 = _mm512_unpackhi_epi64(t33, t35);
        const __m512i u36 = _mm512_unpacklo_epi64(t36, t38);
        const __m512i u37 = _mm512_unpackhi_epi64(t36, t38);
        const __m512i u38 = _mm512_unpacklo_epi64(t37, t39);
        const __m512i u39 = _mm512_unpackhi_epi64(t37, t39);

        const __m512i out32_0 = _mm512_shuffle_i32x4(u32, u36, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_4 = _mm512_shuffle_i32x4(u32, u36, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out32_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 576), out32_4);
        const __m512i out32_1 = _mm512_shuffle_i32x4(u33, u37, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_5 = _mm512_shuffle_i32x4(u33, u37, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 192), out32_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 704), out32_5);
        const __m512i out32_2 = _mm512_shuffle_i32x4(u34, u38, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_6 = _mm512_shuffle_i32x4(u34, u38, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 320), out32_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 832), out32_6);
        const __m512i out32_3 = _mm512_shuffle_i32x4(u35, u39, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_7 = _mm512_shuffle_i32x4(u35, u39, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 448), out32_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 960), out32_7);
        const __m256i v48 = _mm256_loadu_si256((const __m256i*) w48);
        const __m256i v52 = _mm256_loadu_si256((const __m256i*) (w48 + stride4));
        const __m256i v56 = _mm256_loadu_si256((const __m256i*) w56);
        const __m256i v60 = _mm256_loadu_si256((const __m256i*) (w56 + stride4));
        const __m512i z48 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v52, 1);
        const __m512i z52 = _mm512_inserti64x4(_mm512_castsi256_si512(v56), v60, 1);
        const __m256i v49 = _mm256_loadu_si256((const __m256i*) (w48 + stride1));
        const __m256i v53 = _mm256_loadu_si256((const __m256i*) (w48 + stride5));
        const __m256i v57 = _mm256_loadu_si256((const __m256i*) (w56 + stride1));
        const __m256i v61 = _mm256_loadu_si256((const __m256i*) (w56 + stride5));
        const __m512i z49 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v53, 1);
        const __m512i z53 = _mm512_inserti64x4(_mm512_castsi256_si512(v57), v61, 1);
        const __m256i v50 = _mm256_loadu_si256((const __m256i*) (w48 + stride2));
        const __m256i v54 = _mm256_loadu_si256((const __m256i*) (w48 + stride6));
        const __m256i v58 = _mm256_loadu_si256((const __m256i*) (w56 + stride2));
        const __m256i v62 = _mm256_loadu_si256((const __m256i*) (w56 + stride6));
        const __m512i z50 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v54, 1);
        const __m512i z54 = _mm512_inserti64x4(_mm512_castsi256_si512(v58), v62, 1);
        const __m256i v51 = _mm256_loadu_si256((const __m256i*) (w48 + stride3));
        const __m256i v55 = _mm256_loadu_si256((const __m256i*) (w48 + stride7));
        const __m256i v59 = _mm256_loadu_si256((const __m256i*) (w56 + stride3));
        const __m256i v63 = _mm256_loadu_si256((const __m256i*) (w56 + stride7));
        const __m512i z51 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v55, 1);
        const __m512i z55 = _mm512_inserti64x4(_mm512_castsi256_si512(v59), v63, 1);
        w48 += 16;
        w56 += 16;

        const __m512i t48 = _mm512_unpacklo_epi32(z48, z49);
        const __m512i t49 = _mm512_unpackhi_epi32(z48, z49);
        const __m512i t50 = _mm512_unpacklo_epi32(z50, z51);
        const __m512i t51 = _mm512_unpackhi_epi32(z50, z51);
        const __m512i t52 = _mm512_unpacklo_epi32(z52, z53);
        const __m512i t53 = _mm512_unpackhi_epi32(z52, z53);
        const __m512i t54 = _mm512_unpacklo_epi32(z54, z55);
        const __m512i t55 = _mm512_unpackhi_epi32(z54, z55);

        const __m512i u48 = _mm512_unpacklo_epi64(t48, t50);
        const __m512i u49 = _mm512_unpackhi_epi64(t48, t50);
        const __m512i u50 = _mm512_unpacklo_epi64(t49, t51);
        const __m512i u51 = _mm512_unpackhi_epi64(t49, t51);
        const __m512i u52 = _mm512_unpacklo_epi64(t52, t54);
        const __m512i u53 = _mm512_unpackhi_epi64(t52, t54);
        const __m512i u54 = _mm512_unpacklo_epi64(t53, t55);
        const __m512i u55 = _mm512_unpackhi_epi64(t53, t55);

        const __m512i out48_0 = _mm512_shuffle_i32x4(u48, u52, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out48_4 = _mm512_shuffle_i32x4(u48, u52, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out48_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 608), out48_4);
        const __m512i out48_1 = _mm512_shuffle_i32x4(u49, u53, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out48_5 = _mm512_shuffle_i32x4(u49, u53, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 224), out48_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 736), out48_5);
        const __m512i out48_2 = _mm512_shuffle_i32x4(u50, u54, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out48_6 = _mm512_shuffle_i32x4(u50, u54, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 352), out48_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 864), out48_6);
        const __m512i out48_3 = _mm512_shuffle_i32x4(u51, u55, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out48_7 = _mm512_shuffle_i32x4(u51, u55, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 480), out48_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 992), out48_7);
        packed_w += 1024;
      }

      // KC remainder (1..15) using AVX-512BW masking
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);
        const __mmask16 kmask = _cvtu32_mask16((1U << k) - 1U);

        const __m256i v0 = _mm256_maskz_loadu_epi16(kmask, w0);
        const __m256i v4 = _mm256_maskz_loadu_epi16(kmask, w0 + stride4);
        const __m256i v8 = _mm256_maskz_loadu_epi16(kmask, w8);
        const __m256i v12 = _mm256_maskz_loadu_epi16(kmask, w8 + stride4);
        const __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v4, 1);
        const __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v8), v12, 1);
        const __m256i v1 = _mm256_maskz_loadu_epi16(kmask, w0 + stride1);
        const __m256i v5 = _mm256_maskz_loadu_epi16(kmask, w0 + stride5);
        const __m256i v9 = _mm256_maskz_loadu_epi16(kmask, w8 + stride1);
        const __m256i v13 = _mm256_maskz_loadu_epi16(kmask, w8 + stride5);
        const __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v5, 1);
        const __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v9), v13, 1);
        const __m256i v2 = _mm256_maskz_loadu_epi16(kmask, w0 + stride2);
        const __m256i v6 = _mm256_maskz_loadu_epi16(kmask, w0 + stride6);
        const __m256i v10 = _mm256_maskz_loadu_epi16(kmask, w8 + stride2);
        const __m256i v14 = _mm256_maskz_loadu_epi16(kmask, w8 + stride6);
        const __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v6, 1);
        const __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v10), v14, 1);
        const __m256i v3 = _mm256_maskz_loadu_epi16(kmask, w0 + stride3);
        const __m256i v7 = _mm256_maskz_loadu_epi16(kmask, w0 + stride7);
        const __m256i v11 = _mm256_maskz_loadu_epi16(kmask, w8 + stride3);
        const __m256i v15 = _mm256_maskz_loadu_epi16(kmask, w8 + stride7);
        const __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v7, 1);
        const __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v11), v15, 1);

        const __m512i t0 = _mm512_unpacklo_epi32(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi32(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi32(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi32(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi32(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi32(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi32(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi32(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi64(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi64(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi64(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi64(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi64(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi64(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi64(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi64(t5, t7);

        const __m512i out0_0 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        const __m512i out0_4 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 512), out0_4);
        }
        const __m512i out0_1 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_1);
        }
        const __m512i out0_5 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 640), out0_5);
        }
        const __m512i out0_2 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_2);
        }
        const __m512i out0_6 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 768), out0_6);
        }
        const __m512i out0_3 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_3);
        }
        const __m512i out0_7 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 896), out0_7);
        }
        const __m256i v16 = _mm256_maskz_loadu_epi16(kmask, w16);
        const __m256i v20 = _mm256_maskz_loadu_epi16(kmask, w16 + stride4);
        const __m256i v24 = _mm256_maskz_loadu_epi16(kmask, w24);
        const __m256i v28 = _mm256_maskz_loadu_epi16(kmask, w24 + stride4);
        const __m512i z16 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v20, 1);
        const __m512i z20 = _mm512_inserti64x4(_mm512_castsi256_si512(v24), v28, 1);
        const __m256i v17 = _mm256_maskz_loadu_epi16(kmask, w16 + stride1);
        const __m256i v21 = _mm256_maskz_loadu_epi16(kmask, w16 + stride5);
        const __m256i v25 = _mm256_maskz_loadu_epi16(kmask, w24 + stride1);
        const __m256i v29 = _mm256_maskz_loadu_epi16(kmask, w24 + stride5);
        const __m512i z17 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v21, 1);
        const __m512i z21 = _mm512_inserti64x4(_mm512_castsi256_si512(v25), v29, 1);
        const __m256i v18 = _mm256_maskz_loadu_epi16(kmask, w16 + stride2);
        const __m256i v22 = _mm256_maskz_loadu_epi16(kmask, w16 + stride6);
        const __m256i v26 = _mm256_maskz_loadu_epi16(kmask, w24 + stride2);
        const __m256i v30 = _mm256_maskz_loadu_epi16(kmask, w24 + stride6);
        const __m512i z18 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v22, 1);
        const __m512i z22 = _mm512_inserti64x4(_mm512_castsi256_si512(v26), v30, 1);
        const __m256i v19 = _mm256_maskz_loadu_epi16(kmask, w16 + stride3);
        const __m256i v23 = _mm256_maskz_loadu_epi16(kmask, w16 + stride7);
        const __m256i v27 = _mm256_maskz_loadu_epi16(kmask, w24 + stride3);
        const __m256i v31 = _mm256_maskz_loadu_epi16(kmask, w24 + stride7);
        const __m512i z19 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v23, 1);
        const __m512i z23 = _mm512_inserti64x4(_mm512_castsi256_si512(v27), v31, 1);

        const __m512i t16 = _mm512_unpacklo_epi32(z16, z17);
        const __m512i t17 = _mm512_unpackhi_epi32(z16, z17);
        const __m512i t18 = _mm512_unpacklo_epi32(z18, z19);
        const __m512i t19 = _mm512_unpackhi_epi32(z18, z19);
        const __m512i t20 = _mm512_unpacklo_epi32(z20, z21);
        const __m512i t21 = _mm512_unpackhi_epi32(z20, z21);
        const __m512i t22 = _mm512_unpacklo_epi32(z22, z23);
        const __m512i t23 = _mm512_unpackhi_epi32(z22, z23);

        const __m512i u16 = _mm512_unpacklo_epi64(t16, t18);
        const __m512i u17 = _mm512_unpackhi_epi64(t16, t18);
        const __m512i u18 = _mm512_unpacklo_epi64(t17, t19);
        const __m512i u19 = _mm512_unpackhi_epi64(t17, t19);
        const __m512i u20 = _mm512_unpacklo_epi64(t20, t22);
        const __m512i u21 = _mm512_unpackhi_epi64(t20, t22);
        const __m512i u22 = _mm512_unpacklo_epi64(t21, t23);
        const __m512i u23 = _mm512_unpackhi_epi64(t21, t23);

        const __m512i out16_0 = _mm512_shuffle_i32x4(u16, u20, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out16_0);
        const __m512i out16_4 = _mm512_shuffle_i32x4(u16, u20, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 544), out16_4);
        }
        const __m512i out16_1 = _mm512_shuffle_i32x4(u17, u21, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 160), out16_1);
        }
        const __m512i out16_5 = _mm512_shuffle_i32x4(u17, u21, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 672), out16_5);
        }
        const __m512i out16_2 = _mm512_shuffle_i32x4(u18, u22, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 288), out16_2);
        }
        const __m512i out16_6 = _mm512_shuffle_i32x4(u18, u22, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 800), out16_6);
        }
        const __m512i out16_3 = _mm512_shuffle_i32x4(u19, u23, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 416), out16_3);
        }
        const __m512i out16_7 = _mm512_shuffle_i32x4(u19, u23, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 928), out16_7);
        }
        const __m256i v32 = _mm256_maskz_loadu_epi16(kmask, w32);
        const __m256i v36 = _mm256_maskz_loadu_epi16(kmask, w32 + stride4);
        const __m256i v40 = _mm256_maskz_loadu_epi16(kmask, w40);
        const __m256i v44 = _mm256_maskz_loadu_epi16(kmask, w40 + stride4);
        const __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v36, 1);
        const __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v40), v44, 1);
        const __m256i v33 = _mm256_maskz_loadu_epi16(kmask, w32 + stride1);
        const __m256i v37 = _mm256_maskz_loadu_epi16(kmask, w32 + stride5);
        const __m256i v41 = _mm256_maskz_loadu_epi16(kmask, w40 + stride1);
        const __m256i v45 = _mm256_maskz_loadu_epi16(kmask, w40 + stride5);
        const __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v37, 1);
        const __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v41), v45, 1);
        const __m256i v34 = _mm256_maskz_loadu_epi16(kmask, w32 + stride2);
        const __m256i v38 = _mm256_maskz_loadu_epi16(kmask, w32 + stride6);
        const __m256i v42 = _mm256_maskz_loadu_epi16(kmask, w40 + stride2);
        const __m256i v46 = _mm256_maskz_loadu_epi16(kmask, w40 + stride6);
        const __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v38, 1);
        const __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v42), v46, 1);
        const __m256i v35 = _mm256_maskz_loadu_epi16(kmask, w32 + stride3);
        const __m256i v39 = _mm256_maskz_loadu_epi16(kmask, w32 + stride7);
        const __m256i v43 = _mm256_maskz_loadu_epi16(kmask, w40 + stride3);
        const __m256i v47 = _mm256_maskz_loadu_epi16(kmask, w40 + stride7);
        const __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v39, 1);
        const __m512i z39 = _mm512_inserti64x4(_mm512_castsi256_si512(v43), v47, 1);

        const __m512i t32 = _mm512_unpacklo_epi32(z32, z33);
        const __m512i t33 = _mm512_unpackhi_epi32(z32, z33);
        const __m512i t34 = _mm512_unpacklo_epi32(z34, z35);
        const __m512i t35 = _mm512_unpackhi_epi32(z34, z35);
        const __m512i t36 = _mm512_unpacklo_epi32(z36, z37);
        const __m512i t37 = _mm512_unpackhi_epi32(z36, z37);
        const __m512i t38 = _mm512_unpacklo_epi32(z38, z39);
        const __m512i t39 = _mm512_unpackhi_epi32(z38, z39);

        const __m512i u32 = _mm512_unpacklo_epi64(t32, t34);
        const __m512i u33 = _mm512_unpackhi_epi64(t32, t34);
        const __m512i u34 = _mm512_unpacklo_epi64(t33, t35);
        const __m512i u35 = _mm512_unpackhi_epi64(t33, t35);
        const __m512i u36 = _mm512_unpacklo_epi64(t36, t38);
        const __m512i u37 = _mm512_unpackhi_epi64(t36, t38);
        const __m512i u38 = _mm512_unpacklo_epi64(t37, t39);
        const __m512i u39 = _mm512_unpackhi_epi64(t37, t39);

        const __m512i out32_0 = _mm512_shuffle_i32x4(u32, u36, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out32_0);
        const __m512i out32_4 = _mm512_shuffle_i32x4(u32, u36, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 576), out32_4);
        }
        const __m512i out32_1 = _mm512_shuffle_i32x4(u33, u37, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 192), out32_1);
        }
        const __m512i out32_5 = _mm512_shuffle_i32x4(u33, u37, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 704), out32_5);
        }
        const __m512i out32_2 = _mm512_shuffle_i32x4(u34, u38, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 320), out32_2);
        }
        const __m512i out32_6 = _mm512_shuffle_i32x4(u34, u38, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 832), out32_6);
        }
        const __m512i out32_3 = _mm512_shuffle_i32x4(u35, u39, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 448), out32_3);
        }
        const __m512i out32_7 = _mm512_shuffle_i32x4(u35, u39, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 960), out32_7);
        }
        const __m256i v48 = _mm256_maskz_loadu_epi16(kmask, w48);
        const __m256i v52 = _mm256_maskz_loadu_epi16(kmask, w48 + stride4);
        const __m256i v56 = _mm256_maskz_loadu_epi16(kmask, w56);
        const __m256i v60 = _mm256_maskz_loadu_epi16(kmask, w56 + stride4);
        const __m512i z48 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v52, 1);
        const __m512i z52 = _mm512_inserti64x4(_mm512_castsi256_si512(v56), v60, 1);
        const __m256i v49 = _mm256_maskz_loadu_epi16(kmask, w48 + stride1);
        const __m256i v53 = _mm256_maskz_loadu_epi16(kmask, w48 + stride5);
        const __m256i v57 = _mm256_maskz_loadu_epi16(kmask, w56 + stride1);
        const __m256i v61 = _mm256_maskz_loadu_epi16(kmask, w56 + stride5);
        const __m512i z49 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v53, 1);
        const __m512i z53 = _mm512_inserti64x4(_mm512_castsi256_si512(v57), v61, 1);
        const __m256i v50 = _mm256_maskz_loadu_epi16(kmask, w48 + stride2);
        const __m256i v54 = _mm256_maskz_loadu_epi16(kmask, w48 + stride6);
        const __m256i v58 = _mm256_maskz_loadu_epi16(kmask, w56 + stride2);
        const __m256i v62 = _mm256_maskz_loadu_epi16(kmask, w56 + stride6);
        const __m512i z50 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v54, 1);
        const __m512i z54 = _mm512_inserti64x4(_mm512_castsi256_si512(v58), v62, 1);
        const __m256i v51 = _mm256_maskz_loadu_epi16(kmask, w48 + stride3);
        const __m256i v55 = _mm256_maskz_loadu_epi16(kmask, w48 + stride7);
        const __m256i v59 = _mm256_maskz_loadu_epi16(kmask, w56 + stride3);
        const __m256i v63 = _mm256_maskz_loadu_epi16(kmask, w56 + stride7);
        const __m512i z51 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v55, 1);
        const __m512i z55 = _mm512_inserti64x4(_mm512_castsi256_si512(v59), v63, 1);

        const __m512i t48 = _mm512_unpacklo_epi32(z48, z49);
        const __m512i t49 = _mm512_unpackhi_epi32(z48, z49);
        const __m512i t50 = _mm512_unpacklo_epi32(z50, z51);
        const __m512i t51 = _mm512_unpackhi_epi32(z50, z51);
        const __m512i t52 = _mm512_unpacklo_epi32(z52, z53);
        const __m512i t53 = _mm512_unpackhi_epi32(z52, z53);
        const __m512i t54 = _mm512_unpacklo_epi32(z54, z55);
        const __m512i t55 = _mm512_unpackhi_epi32(z54, z55);

        const __m512i u48 = _mm512_unpacklo_epi64(t48, t50);
        const __m512i u49 = _mm512_unpackhi_epi64(t48, t50);
        const __m512i u50 = _mm512_unpacklo_epi64(t49, t51);
        const __m512i u51 = _mm512_unpackhi_epi64(t49, t51);
        const __m512i u52 = _mm512_unpacklo_epi64(t52, t54);
        const __m512i u53 = _mm512_unpackhi_epi64(t52, t54);
        const __m512i u54 = _mm512_unpacklo_epi64(t53, t55);
        const __m512i u55 = _mm512_unpackhi_epi64(t53, t55);

        const __m512i out48_0 = _mm512_shuffle_i32x4(u48, u52, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out48_0);
        const __m512i out48_4 = _mm512_shuffle_i32x4(u48, u52, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 608), out48_4);
        }
        const __m512i out48_1 = _mm512_shuffle_i32x4(u49, u53, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 224), out48_1);
        }
        const __m512i out48_5 = _mm512_shuffle_i32x4(u49, u53, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 736), out48_5);
        }
        const __m512i out48_2 = _mm512_shuffle_i32x4(u50, u54, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 352), out48_2);
        }
        const __m512i out48_6 = _mm512_shuffle_i32x4(u50, u54, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 864), out48_6);
        }
        const __m512i out48_3 = _mm512_shuffle_i32x4(u51, u55, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 480), out48_3);
        }
        const __m512i out48_7 = _mm512_shuffle_i32x4(u51, u55, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 992), out48_7);
        }
        packed_w += ((k + 1) >> 1) * 128;
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
      const uint16_t* w1 = w0 + n_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint16_t* w2 = w1 + n_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint16_t* w3 = w2 + n_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint16_t* w4 = w3 + n_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint16_t* w5 = w4 + n_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint16_t* w6 = w5 + n_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint16_t* w7 = w6 + n_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint16_t* w8 = w7 + n_stride;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint16_t* w9 = w8 + n_stride;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint16_t* w10 = w9 + n_stride;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint16_t* w11 = w10 + n_stride;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint16_t* w12 = w11 + n_stride;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint16_t* w13 = w12 + n_stride;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint16_t* w14 = w13 + n_stride;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const uint16_t* w15 = w14 + n_stride;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const uint16_t* w16 = w15 + n_stride;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const uint16_t* w17 = w16 + n_stride;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const uint16_t* w18 = w17 + n_stride;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const uint16_t* w19 = w18 + n_stride;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const uint16_t* w20 = w19 + n_stride;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const uint16_t* w21 = w20 + n_stride;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const uint16_t* w22 = w21 + n_stride;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const uint16_t* w23 = w22 + n_stride;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const uint16_t* w24 = w23 + n_stride;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const uint16_t* w25 = w24 + n_stride;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const uint16_t* w26 = w25 + n_stride;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const uint16_t* w27 = w26 + n_stride;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const uint16_t* w28 = w27 + n_stride;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const uint16_t* w29 = w28 + n_stride;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const uint16_t* w30 = w29 + n_stride;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }
      const uint16_t* w31 = w30 + n_stride;
      if XNN_UNPREDICTABLE(n < 32) {
        w31 = w30;
      }
      const uint16_t* w32 = w31 + n_stride;
      if XNN_UNPREDICTABLE(n <= 32) {
        w32 = w31;
      }
      const uint16_t* w33 = w32 + n_stride;
      if XNN_UNPREDICTABLE(n < 34) {
        w33 = w32;
      }
      const uint16_t* w34 = w33 + n_stride;
      if XNN_UNPREDICTABLE(n <= 34) {
        w34 = w33;
      }
      const uint16_t* w35 = w34 + n_stride;
      if XNN_UNPREDICTABLE(n < 36) {
        w35 = w34;
      }
      const uint16_t* w36 = w35 + n_stride;
      if XNN_UNPREDICTABLE(n <= 36) {
        w36 = w35;
      }
      const uint16_t* w37 = w36 + n_stride;
      if XNN_UNPREDICTABLE(n < 38) {
        w37 = w36;
      }
      const uint16_t* w38 = w37 + n_stride;
      if XNN_UNPREDICTABLE(n <= 38) {
        w38 = w37;
      }
      const uint16_t* w39 = w38 + n_stride;
      if XNN_UNPREDICTABLE(n < 40) {
        w39 = w38;
      }
      const uint16_t* w40 = w39 + n_stride;
      if XNN_UNPREDICTABLE(n <= 40) {
        w40 = w39;
      }
      const uint16_t* w41 = w40 + n_stride;
      if XNN_UNPREDICTABLE(n < 42) {
        w41 = w40;
      }
      const uint16_t* w42 = w41 + n_stride;
      if XNN_UNPREDICTABLE(n <= 42) {
        w42 = w41;
      }
      const uint16_t* w43 = w42 + n_stride;
      if XNN_UNPREDICTABLE(n < 44) {
        w43 = w42;
      }
      const uint16_t* w44 = w43 + n_stride;
      if XNN_UNPREDICTABLE(n <= 44) {
        w44 = w43;
      }
      const uint16_t* w45 = w44 + n_stride;
      if XNN_UNPREDICTABLE(n < 46) {
        w45 = w44;
      }
      const uint16_t* w46 = w45 + n_stride;
      if XNN_UNPREDICTABLE(n <= 46) {
        w46 = w45;
      }
      const uint16_t* w47 = w46 + n_stride;
      if XNN_UNPREDICTABLE(n < 48) {
        w47 = w46;
      }
      const uint16_t* w48 = w47 + n_stride;
      if XNN_UNPREDICTABLE(n <= 48) {
        w48 = w47;
      }
      const uint16_t* w49 = w48 + n_stride;
      if XNN_UNPREDICTABLE(n < 50) {
        w49 = w48;
      }
      const uint16_t* w50 = w49 + n_stride;
      if XNN_UNPREDICTABLE(n <= 50) {
        w50 = w49;
      }
      const uint16_t* w51 = w50 + n_stride;
      if XNN_UNPREDICTABLE(n < 52) {
        w51 = w50;
      }
      const uint16_t* w52 = w51 + n_stride;
      if XNN_UNPREDICTABLE(n <= 52) {
        w52 = w51;
      }
      const uint16_t* w53 = w52 + n_stride;
      if XNN_UNPREDICTABLE(n < 54) {
        w53 = w52;
      }
      const uint16_t* w54 = w53 + n_stride;
      if XNN_UNPREDICTABLE(n <= 54) {
        w54 = w53;
      }
      const uint16_t* w55 = w54 + n_stride;
      if XNN_UNPREDICTABLE(n < 56) {
        w55 = w54;
      }
      const uint16_t* w56 = w55 + n_stride;
      if XNN_UNPREDICTABLE(n <= 56) {
        w56 = w55;
      }
      const uint16_t* w57 = w56 + n_stride;
      if XNN_UNPREDICTABLE(n < 58) {
        w57 = w56;
      }
      const uint16_t* w58 = w57 + n_stride;
      if XNN_UNPREDICTABLE(n <= 58) {
        w58 = w57;
      }
      const uint16_t* w59 = w58 + n_stride;
      if XNN_UNPREDICTABLE(n < 60) {
        w59 = w58;
      }
      const uint16_t* w60 = w59 + n_stride;
      if XNN_UNPREDICTABLE(n <= 60) {
        w60 = w59;
      }
      const uint16_t* w61 = w60 + n_stride;
      if XNN_UNPREDICTABLE(n < 62) {
        w61 = w60;
      }
      const uint16_t* w62 = w61 + n_stride;
      if XNN_UNPREDICTABLE(n <= 62) {
        w62 = w61;
      }

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        w0 += 16;
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        w4 += 16;
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        w8 += 16;
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        w12 += 16;
        const __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v4, 1);
        const __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v8), v12, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        w1 += 16;
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        w5 += 16;
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        w9 += 16;
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        w13 += 16;
        const __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v5, 1);
        const __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v9), v13, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        w2 += 16;
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        w6 += 16;
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        w10 += 16;
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        w14 += 16;
        const __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v6, 1);
        const __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v10), v14, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        w3 += 16;
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        w7 += 16;
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        w11 += 16;
        const __m256i v15 = _mm256_loadu_si256((const __m256i*) w15);
        w15 += 16;
        const __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v7, 1);
        const __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v11), v15, 1);

        const __m512i t0 = _mm512_unpacklo_epi32(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi32(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi32(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi32(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi32(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi32(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi32(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi32(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi64(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi64(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi64(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi64(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi64(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi64(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi64(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi64(t5, t7);

        const __m512i out0_0 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_4 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 512), out0_4);
        const __m512i out0_1 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_5 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 640), out0_5);
        const __m512i out0_2 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_6 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 768), out0_6);
        const __m512i out0_3 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_7 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 896), out0_7);
        const __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        w16 += 16;
        const __m256i v20 = _mm256_loadu_si256((const __m256i*) w20);
        w20 += 16;
        const __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        w24 += 16;
        const __m256i v28 = _mm256_loadu_si256((const __m256i*) w28);
        w28 += 16;
        const __m512i z16 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v20, 1);
        const __m512i z20 = _mm512_inserti64x4(_mm512_castsi256_si512(v24), v28, 1);
        const __m256i v17 = _mm256_loadu_si256((const __m256i*) w17);
        w17 += 16;
        const __m256i v21 = _mm256_loadu_si256((const __m256i*) w21);
        w21 += 16;
        const __m256i v25 = _mm256_loadu_si256((const __m256i*) w25);
        w25 += 16;
        const __m256i v29 = _mm256_loadu_si256((const __m256i*) w29);
        w29 += 16;
        const __m512i z17 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v21, 1);
        const __m512i z21 = _mm512_inserti64x4(_mm512_castsi256_si512(v25), v29, 1);
        const __m256i v18 = _mm256_loadu_si256((const __m256i*) w18);
        w18 += 16;
        const __m256i v22 = _mm256_loadu_si256((const __m256i*) w22);
        w22 += 16;
        const __m256i v26 = _mm256_loadu_si256((const __m256i*) w26);
        w26 += 16;
        const __m256i v30 = _mm256_loadu_si256((const __m256i*) w30);
        w30 += 16;
        const __m512i z18 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v22, 1);
        const __m512i z22 = _mm512_inserti64x4(_mm512_castsi256_si512(v26), v30, 1);
        const __m256i v19 = _mm256_loadu_si256((const __m256i*) w19);
        w19 += 16;
        const __m256i v23 = _mm256_loadu_si256((const __m256i*) w23);
        w23 += 16;
        const __m256i v27 = _mm256_loadu_si256((const __m256i*) w27);
        w27 += 16;
        const __m256i v31 = _mm256_loadu_si256((const __m256i*) w31);
        w31 += 16;
        const __m512i z19 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v23, 1);
        const __m512i z23 = _mm512_inserti64x4(_mm512_castsi256_si512(v27), v31, 1);

        const __m512i t16 = _mm512_unpacklo_epi32(z16, z17);
        const __m512i t17 = _mm512_unpackhi_epi32(z16, z17);
        const __m512i t18 = _mm512_unpacklo_epi32(z18, z19);
        const __m512i t19 = _mm512_unpackhi_epi32(z18, z19);
        const __m512i t20 = _mm512_unpacklo_epi32(z20, z21);
        const __m512i t21 = _mm512_unpackhi_epi32(z20, z21);
        const __m512i t22 = _mm512_unpacklo_epi32(z22, z23);
        const __m512i t23 = _mm512_unpackhi_epi32(z22, z23);

        const __m512i u16 = _mm512_unpacklo_epi64(t16, t18);
        const __m512i u17 = _mm512_unpackhi_epi64(t16, t18);
        const __m512i u18 = _mm512_unpacklo_epi64(t17, t19);
        const __m512i u19 = _mm512_unpackhi_epi64(t17, t19);
        const __m512i u20 = _mm512_unpacklo_epi64(t20, t22);
        const __m512i u21 = _mm512_unpackhi_epi64(t20, t22);
        const __m512i u22 = _mm512_unpacklo_epi64(t21, t23);
        const __m512i u23 = _mm512_unpackhi_epi64(t21, t23);

        const __m512i out16_0 = _mm512_shuffle_i32x4(u16, u20, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out16_4 = _mm512_shuffle_i32x4(u16, u20, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out16_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 544), out16_4);
        const __m512i out16_1 = _mm512_shuffle_i32x4(u17, u21, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out16_5 = _mm512_shuffle_i32x4(u17, u21, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 160), out16_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 672), out16_5);
        const __m512i out16_2 = _mm512_shuffle_i32x4(u18, u22, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out16_6 = _mm512_shuffle_i32x4(u18, u22, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 288), out16_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 800), out16_6);
        const __m512i out16_3 = _mm512_shuffle_i32x4(u19, u23, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out16_7 = _mm512_shuffle_i32x4(u19, u23, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 416), out16_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 928), out16_7);
        const __m256i v32 = _mm256_loadu_si256((const __m256i*) w32);
        w32 += 16;
        const __m256i v36 = _mm256_loadu_si256((const __m256i*) w36);
        w36 += 16;
        const __m256i v40 = _mm256_loadu_si256((const __m256i*) w40);
        w40 += 16;
        const __m256i v44 = _mm256_loadu_si256((const __m256i*) w44);
        w44 += 16;
        const __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v36, 1);
        const __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v40), v44, 1);
        const __m256i v33 = _mm256_loadu_si256((const __m256i*) w33);
        w33 += 16;
        const __m256i v37 = _mm256_loadu_si256((const __m256i*) w37);
        w37 += 16;
        const __m256i v41 = _mm256_loadu_si256((const __m256i*) w41);
        w41 += 16;
        const __m256i v45 = _mm256_loadu_si256((const __m256i*) w45);
        w45 += 16;
        const __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v37, 1);
        const __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v41), v45, 1);
        const __m256i v34 = _mm256_loadu_si256((const __m256i*) w34);
        w34 += 16;
        const __m256i v38 = _mm256_loadu_si256((const __m256i*) w38);
        w38 += 16;
        const __m256i v42 = _mm256_loadu_si256((const __m256i*) w42);
        w42 += 16;
        const __m256i v46 = _mm256_loadu_si256((const __m256i*) w46);
        w46 += 16;
        const __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v38, 1);
        const __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v42), v46, 1);
        const __m256i v35 = _mm256_loadu_si256((const __m256i*) w35);
        w35 += 16;
        const __m256i v39 = _mm256_loadu_si256((const __m256i*) w39);
        w39 += 16;
        const __m256i v43 = _mm256_loadu_si256((const __m256i*) w43);
        w43 += 16;
        const __m256i v47 = _mm256_loadu_si256((const __m256i*) w47);
        w47 += 16;
        const __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v39, 1);
        const __m512i z39 = _mm512_inserti64x4(_mm512_castsi256_si512(v43), v47, 1);

        const __m512i t32 = _mm512_unpacklo_epi32(z32, z33);
        const __m512i t33 = _mm512_unpackhi_epi32(z32, z33);
        const __m512i t34 = _mm512_unpacklo_epi32(z34, z35);
        const __m512i t35 = _mm512_unpackhi_epi32(z34, z35);
        const __m512i t36 = _mm512_unpacklo_epi32(z36, z37);
        const __m512i t37 = _mm512_unpackhi_epi32(z36, z37);
        const __m512i t38 = _mm512_unpacklo_epi32(z38, z39);
        const __m512i t39 = _mm512_unpackhi_epi32(z38, z39);

        const __m512i u32 = _mm512_unpacklo_epi64(t32, t34);
        const __m512i u33 = _mm512_unpackhi_epi64(t32, t34);
        const __m512i u34 = _mm512_unpacklo_epi64(t33, t35);
        const __m512i u35 = _mm512_unpackhi_epi64(t33, t35);
        const __m512i u36 = _mm512_unpacklo_epi64(t36, t38);
        const __m512i u37 = _mm512_unpackhi_epi64(t36, t38);
        const __m512i u38 = _mm512_unpacklo_epi64(t37, t39);
        const __m512i u39 = _mm512_unpackhi_epi64(t37, t39);

        const __m512i out32_0 = _mm512_shuffle_i32x4(u32, u36, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_4 = _mm512_shuffle_i32x4(u32, u36, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out32_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 576), out32_4);
        const __m512i out32_1 = _mm512_shuffle_i32x4(u33, u37, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_5 = _mm512_shuffle_i32x4(u33, u37, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 192), out32_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 704), out32_5);
        const __m512i out32_2 = _mm512_shuffle_i32x4(u34, u38, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_6 = _mm512_shuffle_i32x4(u34, u38, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 320), out32_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 832), out32_6);
        const __m512i out32_3 = _mm512_shuffle_i32x4(u35, u39, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out32_7 = _mm512_shuffle_i32x4(u35, u39, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 448), out32_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 960), out32_7);
        const __m256i v48 = _mm256_loadu_si256((const __m256i*) w48);
        w48 += 16;
        const __m256i v52 = _mm256_loadu_si256((const __m256i*) w52);
        w52 += 16;
        const __m256i v56 = _mm256_loadu_si256((const __m256i*) w56);
        w56 += 16;
        const __m256i v60 = _mm256_loadu_si256((const __m256i*) w60);
        w60 += 16;
        const __m512i z48 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v52, 1);
        const __m512i z52 = _mm512_inserti64x4(_mm512_castsi256_si512(v56), v60, 1);
        const __m256i v49 = _mm256_loadu_si256((const __m256i*) w49);
        w49 += 16;
        const __m256i v53 = _mm256_loadu_si256((const __m256i*) w53);
        w53 += 16;
        const __m256i v57 = _mm256_loadu_si256((const __m256i*) w57);
        w57 += 16;
        const __m256i v61 = _mm256_loadu_si256((const __m256i*) w61);
        w61 += 16;
        const __m512i z49 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v53, 1);
        const __m512i z53 = _mm512_inserti64x4(_mm512_castsi256_si512(v57), v61, 1);
        const __m256i v50 = _mm256_loadu_si256((const __m256i*) w50);
        w50 += 16;
        const __m256i v54 = _mm256_loadu_si256((const __m256i*) w54);
        w54 += 16;
        const __m256i v58 = _mm256_loadu_si256((const __m256i*) w58);
        w58 += 16;
        const __m256i v62 = _mm256_loadu_si256((const __m256i*) w62);
        w62 += 16;
        const __m512i z50 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v54, 1);
        const __m512i z54 = _mm512_inserti64x4(_mm512_castsi256_si512(v58), v62, 1);
        const __m256i v51 = _mm256_loadu_si256((const __m256i*) w51);
        w51 += 16;
        const __m256i v55 = _mm256_loadu_si256((const __m256i*) w55);
        w55 += 16;
        const __m256i v59 = _mm256_loadu_si256((const __m256i*) w59);
        w59 += 16;
        const __m256i v63 = _mm256_setzero_si256();
        const __m512i z51 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v55, 1);
        const __m512i z55 = _mm512_inserti64x4(_mm512_castsi256_si512(v59), v63, 1);

        const __m512i t48 = _mm512_unpacklo_epi32(z48, z49);
        const __m512i t49 = _mm512_unpackhi_epi32(z48, z49);
        const __m512i t50 = _mm512_unpacklo_epi32(z50, z51);
        const __m512i t51 = _mm512_unpackhi_epi32(z50, z51);
        const __m512i t52 = _mm512_unpacklo_epi32(z52, z53);
        const __m512i t53 = _mm512_unpackhi_epi32(z52, z53);
        const __m512i t54 = _mm512_unpacklo_epi32(z54, z55);
        const __m512i t55 = _mm512_unpackhi_epi32(z54, z55);

        const __m512i u48 = _mm512_unpacklo_epi64(t48, t50);
        const __m512i u49 = _mm512_unpackhi_epi64(t48, t50);
        const __m512i u50 = _mm512_unpacklo_epi64(t49, t51);
        const __m512i u51 = _mm512_unpackhi_epi64(t49, t51);
        const __m512i u52 = _mm512_unpacklo_epi64(t52, t54);
        const __m512i u53 = _mm512_unpackhi_epi64(t52, t54);
        const __m512i u54 = _mm512_unpacklo_epi64(t53, t55);
        const __m512i u55 = _mm512_unpackhi_epi64(t53, t55);

        const __m512i out48_0 = _mm512_shuffle_i32x4(u48, u52, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out48_4 = _mm512_shuffle_i32x4(u48, u52, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out48_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 608), out48_4);
        const __m512i out48_1 = _mm512_shuffle_i32x4(u49, u53, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out48_5 = _mm512_shuffle_i32x4(u49, u53, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 224), out48_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 736), out48_5);
        const __m512i out48_2 = _mm512_shuffle_i32x4(u50, u54, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out48_6 = _mm512_shuffle_i32x4(u50, u54, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 352), out48_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 864), out48_6);
        const __m512i out48_3 = _mm512_shuffle_i32x4(u51, u55, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out48_7 = _mm512_shuffle_i32x4(u51, u55, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 480), out48_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 992), out48_7);
        packed_w += 1024;
      }

      // KC remainder (1..15) using AVX-512BW masking
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);
        const __mmask16 kmask = _cvtu32_mask16((1U << k) - 1U);

        const __m256i v0 = _mm256_maskz_loadu_epi16(kmask, w0);
        const __m256i v4 = _mm256_maskz_loadu_epi16(kmask, w4);
        const __m256i v8 = _mm256_maskz_loadu_epi16(kmask, w8);
        const __m256i v12 = _mm256_maskz_loadu_epi16(kmask, w12);
        const __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v4, 1);
        const __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v8), v12, 1);
        const __m256i v1 = _mm256_maskz_loadu_epi16(kmask, w1);
        const __m256i v5 = _mm256_maskz_loadu_epi16(kmask, w5);
        const __m256i v9 = _mm256_maskz_loadu_epi16(kmask, w9);
        const __m256i v13 = _mm256_maskz_loadu_epi16(kmask, w13);
        const __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v5, 1);
        const __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v9), v13, 1);
        const __m256i v2 = _mm256_maskz_loadu_epi16(kmask, w2);
        const __m256i v6 = _mm256_maskz_loadu_epi16(kmask, w6);
        const __m256i v10 = _mm256_maskz_loadu_epi16(kmask, w10);
        const __m256i v14 = _mm256_maskz_loadu_epi16(kmask, w14);
        const __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v6, 1);
        const __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v10), v14, 1);
        const __m256i v3 = _mm256_maskz_loadu_epi16(kmask, w3);
        const __m256i v7 = _mm256_maskz_loadu_epi16(kmask, w7);
        const __m256i v11 = _mm256_maskz_loadu_epi16(kmask, w11);
        const __m256i v15 = _mm256_maskz_loadu_epi16(kmask, w15);
        const __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v7, 1);
        const __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v11), v15, 1);

        const __m512i t0 = _mm512_unpacklo_epi32(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi32(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi32(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi32(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi32(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi32(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi32(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi32(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi64(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi64(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi64(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi64(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi64(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi64(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi64(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi64(t5, t7);

        const __m512i out0_0 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        const __m512i out0_4 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 512), out0_4);
        }
        const __m512i out0_1 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_1);
        }
        const __m512i out0_5 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 640), out0_5);
        }
        const __m512i out0_2 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_2);
        }
        const __m512i out0_6 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 768), out0_6);
        }
        const __m512i out0_3 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_3);
        }
        const __m512i out0_7 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 896), out0_7);
        }
        const __m256i v16 = _mm256_maskz_loadu_epi16(kmask, w16);
        const __m256i v20 = _mm256_maskz_loadu_epi16(kmask, w20);
        const __m256i v24 = _mm256_maskz_loadu_epi16(kmask, w24);
        const __m256i v28 = _mm256_maskz_loadu_epi16(kmask, w28);
        const __m512i z16 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v20, 1);
        const __m512i z20 = _mm512_inserti64x4(_mm512_castsi256_si512(v24), v28, 1);
        const __m256i v17 = _mm256_maskz_loadu_epi16(kmask, w17);
        const __m256i v21 = _mm256_maskz_loadu_epi16(kmask, w21);
        const __m256i v25 = _mm256_maskz_loadu_epi16(kmask, w25);
        const __m256i v29 = _mm256_maskz_loadu_epi16(kmask, w29);
        const __m512i z17 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v21, 1);
        const __m512i z21 = _mm512_inserti64x4(_mm512_castsi256_si512(v25), v29, 1);
        const __m256i v18 = _mm256_maskz_loadu_epi16(kmask, w18);
        const __m256i v22 = _mm256_maskz_loadu_epi16(kmask, w22);
        const __m256i v26 = _mm256_maskz_loadu_epi16(kmask, w26);
        const __m256i v30 = _mm256_maskz_loadu_epi16(kmask, w30);
        const __m512i z18 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v22, 1);
        const __m512i z22 = _mm512_inserti64x4(_mm512_castsi256_si512(v26), v30, 1);
        const __m256i v19 = _mm256_maskz_loadu_epi16(kmask, w19);
        const __m256i v23 = _mm256_maskz_loadu_epi16(kmask, w23);
        const __m256i v27 = _mm256_maskz_loadu_epi16(kmask, w27);
        const __m256i v31 = _mm256_maskz_loadu_epi16(kmask, w31);
        const __m512i z19 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v23, 1);
        const __m512i z23 = _mm512_inserti64x4(_mm512_castsi256_si512(v27), v31, 1);

        const __m512i t16 = _mm512_unpacklo_epi32(z16, z17);
        const __m512i t17 = _mm512_unpackhi_epi32(z16, z17);
        const __m512i t18 = _mm512_unpacklo_epi32(z18, z19);
        const __m512i t19 = _mm512_unpackhi_epi32(z18, z19);
        const __m512i t20 = _mm512_unpacklo_epi32(z20, z21);
        const __m512i t21 = _mm512_unpackhi_epi32(z20, z21);
        const __m512i t22 = _mm512_unpacklo_epi32(z22, z23);
        const __m512i t23 = _mm512_unpackhi_epi32(z22, z23);

        const __m512i u16 = _mm512_unpacklo_epi64(t16, t18);
        const __m512i u17 = _mm512_unpackhi_epi64(t16, t18);
        const __m512i u18 = _mm512_unpacklo_epi64(t17, t19);
        const __m512i u19 = _mm512_unpackhi_epi64(t17, t19);
        const __m512i u20 = _mm512_unpacklo_epi64(t20, t22);
        const __m512i u21 = _mm512_unpackhi_epi64(t20, t22);
        const __m512i u22 = _mm512_unpacklo_epi64(t21, t23);
        const __m512i u23 = _mm512_unpackhi_epi64(t21, t23);

        const __m512i out16_0 = _mm512_shuffle_i32x4(u16, u20, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out16_0);
        const __m512i out16_4 = _mm512_shuffle_i32x4(u16, u20, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 544), out16_4);
        }
        const __m512i out16_1 = _mm512_shuffle_i32x4(u17, u21, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 160), out16_1);
        }
        const __m512i out16_5 = _mm512_shuffle_i32x4(u17, u21, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 672), out16_5);
        }
        const __m512i out16_2 = _mm512_shuffle_i32x4(u18, u22, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 288), out16_2);
        }
        const __m512i out16_6 = _mm512_shuffle_i32x4(u18, u22, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 800), out16_6);
        }
        const __m512i out16_3 = _mm512_shuffle_i32x4(u19, u23, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 416), out16_3);
        }
        const __m512i out16_7 = _mm512_shuffle_i32x4(u19, u23, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 928), out16_7);
        }
        const __m256i v32 = _mm256_maskz_loadu_epi16(kmask, w32);
        const __m256i v36 = _mm256_maskz_loadu_epi16(kmask, w36);
        const __m256i v40 = _mm256_maskz_loadu_epi16(kmask, w40);
        const __m256i v44 = _mm256_maskz_loadu_epi16(kmask, w44);
        const __m512i z32 = _mm512_inserti64x4(_mm512_castsi256_si512(v32), v36, 1);
        const __m512i z36 = _mm512_inserti64x4(_mm512_castsi256_si512(v40), v44, 1);
        const __m256i v33 = _mm256_maskz_loadu_epi16(kmask, w33);
        const __m256i v37 = _mm256_maskz_loadu_epi16(kmask, w37);
        const __m256i v41 = _mm256_maskz_loadu_epi16(kmask, w41);
        const __m256i v45 = _mm256_maskz_loadu_epi16(kmask, w45);
        const __m512i z33 = _mm512_inserti64x4(_mm512_castsi256_si512(v33), v37, 1);
        const __m512i z37 = _mm512_inserti64x4(_mm512_castsi256_si512(v41), v45, 1);
        const __m256i v34 = _mm256_maskz_loadu_epi16(kmask, w34);
        const __m256i v38 = _mm256_maskz_loadu_epi16(kmask, w38);
        const __m256i v42 = _mm256_maskz_loadu_epi16(kmask, w42);
        const __m256i v46 = _mm256_maskz_loadu_epi16(kmask, w46);
        const __m512i z34 = _mm512_inserti64x4(_mm512_castsi256_si512(v34), v38, 1);
        const __m512i z38 = _mm512_inserti64x4(_mm512_castsi256_si512(v42), v46, 1);
        const __m256i v35 = _mm256_maskz_loadu_epi16(kmask, w35);
        const __m256i v39 = _mm256_maskz_loadu_epi16(kmask, w39);
        const __m256i v43 = _mm256_maskz_loadu_epi16(kmask, w43);
        const __m256i v47 = _mm256_maskz_loadu_epi16(kmask, w47);
        const __m512i z35 = _mm512_inserti64x4(_mm512_castsi256_si512(v35), v39, 1);
        const __m512i z39 = _mm512_inserti64x4(_mm512_castsi256_si512(v43), v47, 1);

        const __m512i t32 = _mm512_unpacklo_epi32(z32, z33);
        const __m512i t33 = _mm512_unpackhi_epi32(z32, z33);
        const __m512i t34 = _mm512_unpacklo_epi32(z34, z35);
        const __m512i t35 = _mm512_unpackhi_epi32(z34, z35);
        const __m512i t36 = _mm512_unpacklo_epi32(z36, z37);
        const __m512i t37 = _mm512_unpackhi_epi32(z36, z37);
        const __m512i t38 = _mm512_unpacklo_epi32(z38, z39);
        const __m512i t39 = _mm512_unpackhi_epi32(z38, z39);

        const __m512i u32 = _mm512_unpacklo_epi64(t32, t34);
        const __m512i u33 = _mm512_unpackhi_epi64(t32, t34);
        const __m512i u34 = _mm512_unpacklo_epi64(t33, t35);
        const __m512i u35 = _mm512_unpackhi_epi64(t33, t35);
        const __m512i u36 = _mm512_unpacklo_epi64(t36, t38);
        const __m512i u37 = _mm512_unpackhi_epi64(t36, t38);
        const __m512i u38 = _mm512_unpacklo_epi64(t37, t39);
        const __m512i u39 = _mm512_unpackhi_epi64(t37, t39);

        const __m512i out32_0 = _mm512_shuffle_i32x4(u32, u36, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out32_0);
        const __m512i out32_4 = _mm512_shuffle_i32x4(u32, u36, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 576), out32_4);
        }
        const __m512i out32_1 = _mm512_shuffle_i32x4(u33, u37, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 192), out32_1);
        }
        const __m512i out32_5 = _mm512_shuffle_i32x4(u33, u37, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 704), out32_5);
        }
        const __m512i out32_2 = _mm512_shuffle_i32x4(u34, u38, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 320), out32_2);
        }
        const __m512i out32_6 = _mm512_shuffle_i32x4(u34, u38, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 832), out32_6);
        }
        const __m512i out32_3 = _mm512_shuffle_i32x4(u35, u39, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 448), out32_3);
        }
        const __m512i out32_7 = _mm512_shuffle_i32x4(u35, u39, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 960), out32_7);
        }
        const __m256i v48 = _mm256_maskz_loadu_epi16(kmask, w48);
        const __m256i v52 = _mm256_maskz_loadu_epi16(kmask, w52);
        const __m256i v56 = _mm256_maskz_loadu_epi16(kmask, w56);
        const __m256i v60 = _mm256_maskz_loadu_epi16(kmask, w60);
        const __m512i z48 = _mm512_inserti64x4(_mm512_castsi256_si512(v48), v52, 1);
        const __m512i z52 = _mm512_inserti64x4(_mm512_castsi256_si512(v56), v60, 1);
        const __m256i v49 = _mm256_maskz_loadu_epi16(kmask, w49);
        const __m256i v53 = _mm256_maskz_loadu_epi16(kmask, w53);
        const __m256i v57 = _mm256_maskz_loadu_epi16(kmask, w57);
        const __m256i v61 = _mm256_maskz_loadu_epi16(kmask, w61);
        const __m512i z49 = _mm512_inserti64x4(_mm512_castsi256_si512(v49), v53, 1);
        const __m512i z53 = _mm512_inserti64x4(_mm512_castsi256_si512(v57), v61, 1);
        const __m256i v50 = _mm256_maskz_loadu_epi16(kmask, w50);
        const __m256i v54 = _mm256_maskz_loadu_epi16(kmask, w54);
        const __m256i v58 = _mm256_maskz_loadu_epi16(kmask, w58);
        const __m256i v62 = _mm256_maskz_loadu_epi16(kmask, w62);
        const __m512i z50 = _mm512_inserti64x4(_mm512_castsi256_si512(v50), v54, 1);
        const __m512i z54 = _mm512_inserti64x4(_mm512_castsi256_si512(v58), v62, 1);
        const __m256i v51 = _mm256_maskz_loadu_epi16(kmask, w51);
        const __m256i v55 = _mm256_maskz_loadu_epi16(kmask, w55);
        const __m256i v59 = _mm256_maskz_loadu_epi16(kmask, w59);
        const __m256i v63 = _mm256_setzero_si256();
        const __m512i z51 = _mm512_inserti64x4(_mm512_castsi256_si512(v51), v55, 1);
        const __m512i z55 = _mm512_inserti64x4(_mm512_castsi256_si512(v59), v63, 1);

        const __m512i t48 = _mm512_unpacklo_epi32(z48, z49);
        const __m512i t49 = _mm512_unpackhi_epi32(z48, z49);
        const __m512i t50 = _mm512_unpacklo_epi32(z50, z51);
        const __m512i t51 = _mm512_unpackhi_epi32(z50, z51);
        const __m512i t52 = _mm512_unpacklo_epi32(z52, z53);
        const __m512i t53 = _mm512_unpackhi_epi32(z52, z53);
        const __m512i t54 = _mm512_unpacklo_epi32(z54, z55);
        const __m512i t55 = _mm512_unpackhi_epi32(z54, z55);

        const __m512i u48 = _mm512_unpacklo_epi64(t48, t50);
        const __m512i u49 = _mm512_unpackhi_epi64(t48, t50);
        const __m512i u50 = _mm512_unpacklo_epi64(t49, t51);
        const __m512i u51 = _mm512_unpackhi_epi64(t49, t51);
        const __m512i u52 = _mm512_unpacklo_epi64(t52, t54);
        const __m512i u53 = _mm512_unpackhi_epi64(t52, t54);
        const __m512i u54 = _mm512_unpacklo_epi64(t53, t55);
        const __m512i u55 = _mm512_unpackhi_epi64(t53, t55);

        const __m512i out48_0 = _mm512_shuffle_i32x4(u48, u52, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out48_0);
        const __m512i out48_4 = _mm512_shuffle_i32x4(u48, u52, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 608), out48_4);
        }
        const __m512i out48_1 = _mm512_shuffle_i32x4(u49, u53, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 224), out48_1);
        }
        const __m512i out48_5 = _mm512_shuffle_i32x4(u49, u53, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 736), out48_5);
        }
        const __m512i out48_2 = _mm512_shuffle_i32x4(u50, u54, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 352), out48_2);
        }
        const __m512i out48_6 = _mm512_shuffle_i32x4(u50, u54, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 864), out48_6);
        }
        const __m512i out48_3 = _mm512_shuffle_i32x4(u51, u55, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 480), out48_3);
        }
        const __m512i out48_7 = _mm512_shuffle_i32x4(u51, u55, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 992), out48_7);
        }
        packed_w += ((k + 1) >> 1) * 128;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * n_stride;
  } while (--g != 0);
}
