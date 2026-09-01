// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/avx512.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
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
static XNN_INTRINSIC __m128 xnn_load_ps_k(const float* ptr, size_t k) {
  if (k == 1) {
    return _mm_load_ss(ptr);
  } else if (k == 2) {
    return _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) ptr));
  } else {
    return _mm_setr_ps(ptr[0], ptr[1], ptr[2], 0.0f);
  }
}


void xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
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
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const __m512i vperm_idx = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 32
    const float* w_base = (const float*) weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const __m512 vb0 = _mm512_loadu_ps(b);
        const __m512 vb16 = _mm512_loadu_ps(b + 16);
        _mm512_store_ps(packed_w, vb0);
        _mm512_store_ps(packed_w + 16, vb16);
        b += 32;
      } else {
        const __m512 vzero = _mm512_setzero_ps();
        _mm512_store_ps(packed_w, vzero);
        _mm512_store_ps(packed_w + 16, vzero);
      }
      packed_w += 32;

      const ptrdiff_t stride1 = (ptrdiff_t) kc;
      const ptrdiff_t stride2 = stride1 * 2;
      const ptrdiff_t stride3 = stride1 * 3;

      const float* w0 = w_base;
      const float* w4 = w0 + 4 * stride1;
      const float* w8 = w0 + 8 * stride1;
      const float* w12 = w0 + 12 * stride1;
      const float* w16 = w0 + 16 * stride1;
      const float* w20 = w0 + 20 * stride1;
      const float* w24 = w0 + 24 * stride1;
      const float* w28 = w0 + 28 * stride1;

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4 rows x 4 cols into 512-bit registers
        const __m512 v0 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w0)), _mm_loadu_ps(w0 + stride1), 1), _mm_loadu_ps(w0 + stride2), 2), _mm_loadu_ps(w0 + stride3), 3);
        const __m512 v4 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w4)), _mm_loadu_ps(w4 + stride1), 1), _mm_loadu_ps(w4 + stride2), 2), _mm_loadu_ps(w4 + stride3), 3);
        const __m512 v8 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w8)), _mm_loadu_ps(w8 + stride1), 1), _mm_loadu_ps(w8 + stride2), 2), _mm_loadu_ps(w8 + stride3), 3);
        const __m512 v12 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w12)), _mm_loadu_ps(w12 + stride1), 1), _mm_loadu_ps(w12 + stride2), 2), _mm_loadu_ps(w12 + stride3), 3);
        const __m512 v16 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w16)), _mm_loadu_ps(w16 + stride1), 1), _mm_loadu_ps(w16 + stride2), 2), _mm_loadu_ps(w16 + stride3), 3);
        const __m512 v20 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w20)), _mm_loadu_ps(w20 + stride1), 1), _mm_loadu_ps(w20 + stride2), 2), _mm_loadu_ps(w20 + stride3), 3);
        const __m512 v24 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w24)), _mm_loadu_ps(w24 + stride1), 1), _mm_loadu_ps(w24 + stride2), 2), _mm_loadu_ps(w24 + stride3), 3);
        const __m512 v28 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w28)), _mm_loadu_ps(w28 + stride1), 1), _mm_loadu_ps(w28 + stride2), 2), _mm_loadu_ps(w28 + stride3), 3);

        w0 += 4;
        w4 += 4;
        w8 += 4;
        w12 += 4;
        w16 += 4;
        w20 += 4;
        w24 += 4;
        w28 += 4;


        // Transpose 16x4 blocks: Step 1 (in-register 4x4 permute) + Step 2 (128-bit lane shuffles)
        const __m512 v0_tr = _mm512_permutexvar_ps(vperm_idx, v0);
        const __m512 v4_tr = _mm512_permutexvar_ps(vperm_idx, v4);
        const __m512 v8_tr = _mm512_permutexvar_ps(vperm_idx, v8);
        const __m512 v12_tr = _mm512_permutexvar_ps(vperm_idx, v12);

        const __m512 ab0_0 = _mm512_shuffle_f32x4(v0_tr, v4_tr, 0x44);
        const __m512 cd0_0 = _mm512_shuffle_f32x4(v8_tr, v12_tr, 0x44);
        const __m512 ab0_1 = _mm512_shuffle_f32x4(v0_tr, v4_tr, 0xEE);
        const __m512 cd0_1 = _mm512_shuffle_f32x4(v8_tr, v12_tr, 0xEE);

        const __m512 out0_0 = _mm512_shuffle_f32x4(ab0_0, cd0_0, 0x88);
        const __m512 out0_1 = _mm512_shuffle_f32x4(ab0_0, cd0_0, 0xDD);
        const __m512 out0_2 = _mm512_shuffle_f32x4(ab0_1, cd0_1, 0x88);
        const __m512 out0_3 = _mm512_shuffle_f32x4(ab0_1, cd0_1, 0xDD);

        _mm512_store_ps(packed_w + 0, out0_0);
        _mm512_store_ps(packed_w + 32, out0_1);
        _mm512_store_ps(packed_w + 64, out0_2);
        _mm512_store_ps(packed_w + 96, out0_3);
        const __m512 v16_tr = _mm512_permutexvar_ps(vperm_idx, v16);
        const __m512 v20_tr = _mm512_permutexvar_ps(vperm_idx, v20);
        const __m512 v24_tr = _mm512_permutexvar_ps(vperm_idx, v24);
        const __m512 v28_tr = _mm512_permutexvar_ps(vperm_idx, v28);

        const __m512 ab16_0 = _mm512_shuffle_f32x4(v16_tr, v20_tr, 0x44);
        const __m512 cd16_0 = _mm512_shuffle_f32x4(v24_tr, v28_tr, 0x44);
        const __m512 ab16_1 = _mm512_shuffle_f32x4(v16_tr, v20_tr, 0xEE);
        const __m512 cd16_1 = _mm512_shuffle_f32x4(v24_tr, v28_tr, 0xEE);

        const __m512 out16_0 = _mm512_shuffle_f32x4(ab16_0, cd16_0, 0x88);
        const __m512 out16_1 = _mm512_shuffle_f32x4(ab16_0, cd16_0, 0xDD);
        const __m512 out16_2 = _mm512_shuffle_f32x4(ab16_1, cd16_1, 0x88);
        const __m512 out16_3 = _mm512_shuffle_f32x4(ab16_1, cd16_1, 0xDD);

        _mm512_store_ps(packed_w + 16, out16_0);
        _mm512_store_ps(packed_w + 48, out16_1);
        _mm512_store_ps(packed_w + 80, out16_2);
        _mm512_store_ps(packed_w + 112, out16_3);

        packed_w += 128;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);

        const __m512 v0 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w0, k)), xnn_load_ps_k(w0 + stride1, k), 1), xnn_load_ps_k(w0 + stride2, k), 2), xnn_load_ps_k(w0 + stride3, k), 3);
        const __m512 v4 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w4, k)), xnn_load_ps_k(w4 + stride1, k), 1), xnn_load_ps_k(w4 + stride2, k), 2), xnn_load_ps_k(w4 + stride3, k), 3);
        const __m512 v8 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w8, k)), xnn_load_ps_k(w8 + stride1, k), 1), xnn_load_ps_k(w8 + stride2, k), 2), xnn_load_ps_k(w8 + stride3, k), 3);
        const __m512 v12 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w12, k)), xnn_load_ps_k(w12 + stride1, k), 1), xnn_load_ps_k(w12 + stride2, k), 2), xnn_load_ps_k(w12 + stride3, k), 3);

        const __m512 v0_tr = _mm512_permutexvar_ps(vperm_idx, v0);
        const __m512 v4_tr = _mm512_permutexvar_ps(vperm_idx, v4);
        const __m512 v8_tr = _mm512_permutexvar_ps(vperm_idx, v8);
        const __m512 v12_tr = _mm512_permutexvar_ps(vperm_idx, v12);

        const __m512 ab0_0 = _mm512_shuffle_f32x4(v0_tr, v4_tr, 0x44);
        const __m512 cd0_0 = _mm512_shuffle_f32x4(v8_tr, v12_tr, 0x44);
        const __m512 ab0_1 = _mm512_shuffle_f32x4(v0_tr, v4_tr, 0xEE);
        const __m512 cd0_1 = _mm512_shuffle_f32x4(v8_tr, v12_tr, 0xEE);

        const __m512 out0_0 = _mm512_shuffle_f32x4(ab0_0, cd0_0, 0x88);
        const __m512 out0_1 = _mm512_shuffle_f32x4(ab0_0, cd0_0, 0xDD);
        const __m512 out0_2 = _mm512_shuffle_f32x4(ab0_1, cd0_1, 0x88);

        _mm512_store_ps(packed_w + 0, out0_0);
        if (k > 1) {
          _mm512_store_ps(packed_w + 32, out0_1);
        }
        if (k > 2) {
          _mm512_store_ps(packed_w + 64, out0_2);
        }
        const __m512 v16 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w16, k)), xnn_load_ps_k(w16 + stride1, k), 1), xnn_load_ps_k(w16 + stride2, k), 2), xnn_load_ps_k(w16 + stride3, k), 3);
        const __m512 v20 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w20, k)), xnn_load_ps_k(w20 + stride1, k), 1), xnn_load_ps_k(w20 + stride2, k), 2), xnn_load_ps_k(w20 + stride3, k), 3);
        const __m512 v24 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w24, k)), xnn_load_ps_k(w24 + stride1, k), 1), xnn_load_ps_k(w24 + stride2, k), 2), xnn_load_ps_k(w24 + stride3, k), 3);
        const __m512 v28 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w28, k)), xnn_load_ps_k(w28 + stride1, k), 1), xnn_load_ps_k(w28 + stride2, k), 2), xnn_load_ps_k(w28 + stride3, k), 3);

        const __m512 v16_tr = _mm512_permutexvar_ps(vperm_idx, v16);
        const __m512 v20_tr = _mm512_permutexvar_ps(vperm_idx, v20);
        const __m512 v24_tr = _mm512_permutexvar_ps(vperm_idx, v24);
        const __m512 v28_tr = _mm512_permutexvar_ps(vperm_idx, v28);

        const __m512 ab16_0 = _mm512_shuffle_f32x4(v16_tr, v20_tr, 0x44);
        const __m512 cd16_0 = _mm512_shuffle_f32x4(v24_tr, v28_tr, 0x44);
        const __m512 ab16_1 = _mm512_shuffle_f32x4(v16_tr, v20_tr, 0xEE);
        const __m512 cd16_1 = _mm512_shuffle_f32x4(v24_tr, v28_tr, 0xEE);

        const __m512 out16_0 = _mm512_shuffle_f32x4(ab16_0, cd16_0, 0x88);
        const __m512 out16_1 = _mm512_shuffle_f32x4(ab16_0, cd16_0, 0xDD);
        const __m512 out16_2 = _mm512_shuffle_f32x4(ab16_1, cd16_1, 0x88);

        _mm512_store_ps(packed_w + 16, out16_0);
        if (k > 1) {
          _mm512_store_ps(packed_w + 48, out16_1);
        }
        if (k > 2) {
          _mm512_store_ps(packed_w + 80, out16_2);
        }

        packed_w += 32 * k;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w_base += 32 * stride1;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);
      const float* w0 = w_base;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (32 - n);
      } else {
        const __m512 vzero = _mm512_setzero_ps();
        _mm512_store_ps(packed_w, vzero);
        _mm512_store_ps(packed_w + 16, vzero);
        packed_w += 32;
      }

      // NR remainder has less than 32 rows so last row is not loaded
      // For SR=4 the
      const float* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const float* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const float* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const float* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const float* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const float* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const float* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const float* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const float* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const float* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const float* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const float* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const float* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const float* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const float* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const float* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const float* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const float* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const float* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const float* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const float* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const float* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const float* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const float* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const float* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const float* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const float* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const float* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const float* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const float* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4 rows x 4 cols into 512-bit registers
        const __m512 v0 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w0)), _mm_loadu_ps(w1), 1), _mm_loadu_ps(w2), 2), _mm_loadu_ps(w3), 3);
        const __m512 v4 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w4)), _mm_loadu_ps(w5), 1), _mm_loadu_ps(w6), 2), _mm_loadu_ps(w7), 3);
        const __m512 v8 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w8)), _mm_loadu_ps(w9), 1), _mm_loadu_ps(w10), 2), _mm_loadu_ps(w11), 3);
        const __m512 v12 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w12)), _mm_loadu_ps(w13), 1), _mm_loadu_ps(w14), 2), _mm_loadu_ps(w15), 3);
        const __m512 v16 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w16)), _mm_loadu_ps(w17), 1), _mm_loadu_ps(w18), 2), _mm_loadu_ps(w19), 3);
        const __m512 v20 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w20)), _mm_loadu_ps(w21), 1), _mm_loadu_ps(w22), 2), _mm_loadu_ps(w23), 3);
        const __m512 v24 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w24)), _mm_loadu_ps(w25), 1), _mm_loadu_ps(w26), 2), _mm_loadu_ps(w27), 3);
        const __m512 v28 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(w28)), _mm_loadu_ps(w29), 1), _mm_loadu_ps(w30), 2), _mm_setzero_ps(), 3);

        w0 += 4;
        w1 += 4;
        w2 += 4;
        w3 += 4;
        w4 += 4;
        w5 += 4;
        w6 += 4;
        w7 += 4;
        w8 += 4;
        w9 += 4;
        w10 += 4;
        w11 += 4;
        w12 += 4;
        w13 += 4;
        w14 += 4;
        w15 += 4;
        w16 += 4;
        w17 += 4;
        w18 += 4;
        w19 += 4;
        w20 += 4;
        w21 += 4;
        w22 += 4;
        w23 += 4;
        w24 += 4;
        w25 += 4;
        w26 += 4;
        w27 += 4;
        w28 += 4;
        w29 += 4;
        w30 += 4;


        // Transpose 16x4 blocks: Step 1 (in-register 4x4 permute) + Step 2 (128-bit lane shuffles)
        const __m512 v0_tr = _mm512_permutexvar_ps(vperm_idx, v0);
        const __m512 v4_tr = _mm512_permutexvar_ps(vperm_idx, v4);
        const __m512 v8_tr = _mm512_permutexvar_ps(vperm_idx, v8);
        const __m512 v12_tr = _mm512_permutexvar_ps(vperm_idx, v12);

        const __m512 ab0_0 = _mm512_shuffle_f32x4(v0_tr, v4_tr, 0x44);
        const __m512 cd0_0 = _mm512_shuffle_f32x4(v8_tr, v12_tr, 0x44);
        const __m512 ab0_1 = _mm512_shuffle_f32x4(v0_tr, v4_tr, 0xEE);
        const __m512 cd0_1 = _mm512_shuffle_f32x4(v8_tr, v12_tr, 0xEE);

        const __m512 out0_0 = _mm512_shuffle_f32x4(ab0_0, cd0_0, 0x88);
        const __m512 out0_1 = _mm512_shuffle_f32x4(ab0_0, cd0_0, 0xDD);
        const __m512 out0_2 = _mm512_shuffle_f32x4(ab0_1, cd0_1, 0x88);
        const __m512 out0_3 = _mm512_shuffle_f32x4(ab0_1, cd0_1, 0xDD);

        _mm512_store_ps(packed_w + 0, out0_0);
        _mm512_store_ps(packed_w + 32, out0_1);
        _mm512_store_ps(packed_w + 64, out0_2);
        _mm512_store_ps(packed_w + 96, out0_3);
        const __m512 v16_tr = _mm512_permutexvar_ps(vperm_idx, v16);
        const __m512 v20_tr = _mm512_permutexvar_ps(vperm_idx, v20);
        const __m512 v24_tr = _mm512_permutexvar_ps(vperm_idx, v24);
        const __m512 v28_tr = _mm512_permutexvar_ps(vperm_idx, v28);

        const __m512 ab16_0 = _mm512_shuffle_f32x4(v16_tr, v20_tr, 0x44);
        const __m512 cd16_0 = _mm512_shuffle_f32x4(v24_tr, v28_tr, 0x44);
        const __m512 ab16_1 = _mm512_shuffle_f32x4(v16_tr, v20_tr, 0xEE);
        const __m512 cd16_1 = _mm512_shuffle_f32x4(v24_tr, v28_tr, 0xEE);

        const __m512 out16_0 = _mm512_shuffle_f32x4(ab16_0, cd16_0, 0x88);
        const __m512 out16_1 = _mm512_shuffle_f32x4(ab16_0, cd16_0, 0xDD);
        const __m512 out16_2 = _mm512_shuffle_f32x4(ab16_1, cd16_1, 0x88);
        const __m512 out16_3 = _mm512_shuffle_f32x4(ab16_1, cd16_1, 0xDD);

        _mm512_store_ps(packed_w + 16, out16_0);
        _mm512_store_ps(packed_w + 48, out16_1);
        _mm512_store_ps(packed_w + 80, out16_2);
        _mm512_store_ps(packed_w + 112, out16_3);

        packed_w += 128;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);

        const __m512 v0 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w0, k)), xnn_load_ps_k(w1, k), 1), xnn_load_ps_k(w2, k), 2), xnn_load_ps_k(w3, k), 3);
        const __m512 v4 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w4, k)), xnn_load_ps_k(w5, k), 1), xnn_load_ps_k(w6, k), 2), xnn_load_ps_k(w7, k), 3);
        const __m512 v8 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w8, k)), xnn_load_ps_k(w9, k), 1), xnn_load_ps_k(w10, k), 2), xnn_load_ps_k(w11, k), 3);
        const __m512 v12 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w12, k)), xnn_load_ps_k(w13, k), 1), xnn_load_ps_k(w14, k), 2), xnn_load_ps_k(w15, k), 3);

        const __m512 v0_tr = _mm512_permutexvar_ps(vperm_idx, v0);
        const __m512 v4_tr = _mm512_permutexvar_ps(vperm_idx, v4);
        const __m512 v8_tr = _mm512_permutexvar_ps(vperm_idx, v8);
        const __m512 v12_tr = _mm512_permutexvar_ps(vperm_idx, v12);

        const __m512 ab0_0 = _mm512_shuffle_f32x4(v0_tr, v4_tr, 0x44);
        const __m512 cd0_0 = _mm512_shuffle_f32x4(v8_tr, v12_tr, 0x44);
        const __m512 ab0_1 = _mm512_shuffle_f32x4(v0_tr, v4_tr, 0xEE);
        const __m512 cd0_1 = _mm512_shuffle_f32x4(v8_tr, v12_tr, 0xEE);

        const __m512 out0_0 = _mm512_shuffle_f32x4(ab0_0, cd0_0, 0x88);
        const __m512 out0_1 = _mm512_shuffle_f32x4(ab0_0, cd0_0, 0xDD);
        const __m512 out0_2 = _mm512_shuffle_f32x4(ab0_1, cd0_1, 0x88);

        _mm512_store_ps(packed_w + 0, out0_0);
        if (k > 1) {
          _mm512_store_ps(packed_w + 32, out0_1);
        }
        if (k > 2) {
          _mm512_store_ps(packed_w + 64, out0_2);
        }
        const __m512 v16 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w16, k)), xnn_load_ps_k(w17, k), 1), xnn_load_ps_k(w18, k), 2), xnn_load_ps_k(w19, k), 3);
        const __m512 v20 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w20, k)), xnn_load_ps_k(w21, k), 1), xnn_load_ps_k(w22, k), 2), xnn_load_ps_k(w23, k), 3);
        const __m512 v24 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w24, k)), xnn_load_ps_k(w25, k), 1), xnn_load_ps_k(w26, k), 2), xnn_load_ps_k(w27, k), 3);
        const __m512 v28 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(xnn_load_ps_k(w28, k)), xnn_load_ps_k(w29, k), 1), xnn_load_ps_k(w30, k), 2), _mm_setzero_ps(), 3);

        const __m512 v16_tr = _mm512_permutexvar_ps(vperm_idx, v16);
        const __m512 v20_tr = _mm512_permutexvar_ps(vperm_idx, v20);
        const __m512 v24_tr = _mm512_permutexvar_ps(vperm_idx, v24);
        const __m512 v28_tr = _mm512_permutexvar_ps(vperm_idx, v28);

        const __m512 ab16_0 = _mm512_shuffle_f32x4(v16_tr, v20_tr, 0x44);
        const __m512 cd16_0 = _mm512_shuffle_f32x4(v24_tr, v28_tr, 0x44);
        const __m512 ab16_1 = _mm512_shuffle_f32x4(v16_tr, v20_tr, 0xEE);
        const __m512 cd16_1 = _mm512_shuffle_f32x4(v24_tr, v28_tr, 0xEE);

        const __m512 out16_0 = _mm512_shuffle_f32x4(ab16_0, cd16_0, 0x88);
        const __m512 out16_1 = _mm512_shuffle_f32x4(ab16_0, cd16_0, 0xDD);
        const __m512 out16_2 = _mm512_shuffle_f32x4(ab16_1, cd16_1, 0x88);

        _mm512_store_ps(packed_w + 16, out16_0);
        if (k > 1) {
          _mm512_store_ps(packed_w + 48, out16_1);
        }
        if (k > 2) {
          _mm512_store_ps(packed_w + 80, out16_2);
        }

        packed_w += 32 * k;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
