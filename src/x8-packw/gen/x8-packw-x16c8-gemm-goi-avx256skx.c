// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>  // for memcpy

#include <immintrin.h>

#include "xnnpack/packw.h"
#include "xnnpack/unaligned.h"


void xnn_x8_packw_gemm_goi_ukernel_x16c8__avx256skx(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const uint32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;


  do {
    // NC main loop multiple of 16
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m256i vb0 = _mm256_loadu_si256((const __m256i*) (b + 0));
        const __m256i vb8 = _mm256_loadu_si256((const __m256i*) (b + 8));
        _mm256_storeu_si256((__m256i*) (out + 0), vb0);
        _mm256_storeu_si256((__m256i*) (out + 32), vb8);
        b += 16;
      } else {
        _mm256_storeu_si256((__m256i*) (out + 0), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 32), _mm256_setzero_si256());
      }
      out += 16 * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;
      const int8_t* w8 = w7 + kc;
      const int8_t* w9 = w8 + kc;
      const int8_t* w10 = w9 + kc;
      const int8_t* w11 = w10 + kc;
      const int8_t* w12 = w11 + kc;
      const int8_t* w13 = w12 + kc;
      const int8_t* w14 = w13 + kc;
      const int8_t* w15 = w14 + kc;


      size_t k = kc;
      // KC main loop multiple of 16x32
      for (; k >= 32; k -= 32) {
        const __m256i v0_0123 = _mm256_loadu_si256((const __m256i*) w0);
        const __m256i v1_0123 = _mm256_loadu_si256((const __m256i*) w1);
        const __m256i v2_0123 = _mm256_loadu_si256((const __m256i*) w2);
        const __m256i v3_0123 = _mm256_loadu_si256((const __m256i*) w3);
        const __m256i v4_0123 = _mm256_loadu_si256((const __m256i*) w4);
        const __m256i v5_0123 = _mm256_loadu_si256((const __m256i*) w5);
        const __m256i v6_0123 = _mm256_loadu_si256((const __m256i*) w6);
        const __m256i v7_0123 = _mm256_loadu_si256((const __m256i*) w7);
        const __m256i v8_0123 = _mm256_loadu_si256((const __m256i*) w8);
        const __m256i v9_0123 = _mm256_loadu_si256((const __m256i*) w9);
        const __m256i v10_0123 = _mm256_loadu_si256((const __m256i*) w10);
        const __m256i v11_0123 = _mm256_loadu_si256((const __m256i*) w11);
        const __m256i v12_0123 = _mm256_loadu_si256((const __m256i*) w12);
        const __m256i v13_0123 = _mm256_loadu_si256((const __m256i*) w13);
        const __m256i v14_0123 = _mm256_loadu_si256((const __m256i*) w14);
        const __m256i v15_0123 = _mm256_loadu_si256((const __m256i*) w15);

        const __m256i v01_02 = _mm256_unpacklo_epi64(v0_0123, v1_0123);
        const __m256i v01_13 = _mm256_unpackhi_epi64(v0_0123, v1_0123);
        const __m256i v23_02 = _mm256_unpacklo_epi64(v2_0123, v3_0123);
        const __m256i v23_13 = _mm256_unpackhi_epi64(v2_0123, v3_0123);
        const __m256i v45_02 = _mm256_unpacklo_epi64(v4_0123, v5_0123);
        const __m256i v45_13 = _mm256_unpackhi_epi64(v4_0123, v5_0123);
        const __m256i v67_02 = _mm256_unpacklo_epi64(v6_0123, v7_0123);
        const __m256i v67_13 = _mm256_unpackhi_epi64(v6_0123, v7_0123);
        const __m256i v89_02 = _mm256_unpacklo_epi64(v8_0123, v9_0123);
        const __m256i v89_13 = _mm256_unpackhi_epi64(v8_0123, v9_0123);
        const __m256i v1011_02 = _mm256_unpacklo_epi64(v10_0123, v11_0123);
        const __m256i v1011_13 = _mm256_unpackhi_epi64(v10_0123, v11_0123);
        const __m256i v1213_02 = _mm256_unpacklo_epi64(v12_0123, v13_0123);
        const __m256i v1213_13 = _mm256_unpackhi_epi64(v12_0123, v13_0123);
        const __m256i v1415_02 = _mm256_unpacklo_epi64(v14_0123, v15_0123);
        const __m256i v1415_13 = _mm256_unpackhi_epi64(v14_0123, v15_0123);


        const __m256i v0_0 = _mm256_permute2f128_si256(v01_02, v23_02, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v0_1 = _mm256_permute2f128_si256(v01_13, v23_13, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v0_2 = _mm256_permute2f128_si256(v01_02, v23_02, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v0_3 = _mm256_permute2f128_si256(v01_13, v23_13, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v4_0 = _mm256_permute2f128_si256(v45_02, v67_02, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v4_1 = _mm256_permute2f128_si256(v45_13, v67_13, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v4_2 = _mm256_permute2f128_si256(v45_02, v67_02, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v4_3 = _mm256_permute2f128_si256(v45_13, v67_13, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v8_0 = _mm256_permute2f128_si256(v89_02, v1011_02, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v8_1 = _mm256_permute2f128_si256(v89_13, v1011_13, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v8_2 = _mm256_permute2f128_si256(v89_02, v1011_02, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v8_3 = _mm256_permute2f128_si256(v89_13, v1011_13, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v12_0 = _mm256_permute2f128_si256(v1213_02, v1415_02, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v12_1 = _mm256_permute2f128_si256(v1213_13, v1415_13, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v12_2 = _mm256_permute2f128_si256(v1213_02, v1415_02, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v12_3 = _mm256_permute2f128_si256(v1213_13, v1415_13, _MM_SHUFFLE(0, 3, 0, 1));


        _mm256_storeu_si256((__m256i *)&out[0],  v0_0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4_0);
        _mm256_storeu_si256((__m256i *)&out[64],  v8_0);
        _mm256_storeu_si256((__m256i *)&out[96],  v12_0);
        _mm256_storeu_si256((__m256i *)&out[128],  v0_1);
        _mm256_storeu_si256((__m256i *)&out[160],  v4_1);
        _mm256_storeu_si256((__m256i *)&out[192],  v8_1);
        _mm256_storeu_si256((__m256i *)&out[224],  v12_1);
        _mm256_storeu_si256((__m256i *)&out[256],  v0_2);
        _mm256_storeu_si256((__m256i *)&out[288],  v4_2);
        _mm256_storeu_si256((__m256i *)&out[320],  v8_2);
        _mm256_storeu_si256((__m256i *)&out[352],  v12_2);
        _mm256_storeu_si256((__m256i *)&out[384],  v0_3);
        _mm256_storeu_si256((__m256i *)&out[416],  v4_3);
        _mm256_storeu_si256((__m256i *)&out[448],  v8_3);
        _mm256_storeu_si256((__m256i *)&out[480],  v12_3);

        w0 += 32;
        w1 += 32;
        w2 += 32;
        w3 += 32;
        w4 += 32;
        w5 += 32;
        w6 += 32;
        w7 += 32;
        w8 += 32;
        w9 += 32;
        w10 += 32;
        w11 += 32;
        w12 += 32;
        w13 += 32;
        w14 += 32;
        w15 += 32;
        out += 512;
      }

      // KC main loop multiple of 16x8
      for (; k >= 8; k -= 8) {
        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);
        __m256i v8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w8));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w9)), 0x0C);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w10)), 0x30);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w11)), 0xC0);
        __m256i v12 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w12));
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w13)), 0x0C);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w14)), 0x30);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w15)), 0xC0);


        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);
        _mm256_storeu_si256((__m256i *)&out[64],  v8);
        _mm256_storeu_si256((__m256i *)&out[96],  v12);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        w8 += 8;
        w9 += 8;
        w10 += 8;
        w11 += 8;
        w12 += 8;
        w13 += 8;
        w14 += 8;
        w15 += 8;
        out += 128;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);

       // TODO: more efficient copy
        // uint64_t for KR=8 bytes
        uint64_t remainder_buffer[16] = { 0, };
        __builtin_memcpy(remainder_buffer + 0, w0, k);
        __builtin_memcpy(remainder_buffer + 1, w1, k);
        __builtin_memcpy(remainder_buffer + 2, w2, k);
        __builtin_memcpy(remainder_buffer + 3, w3, k);
        __builtin_memcpy(remainder_buffer + 4, w4, k);
        __builtin_memcpy(remainder_buffer + 5, w5, k);
        __builtin_memcpy(remainder_buffer + 6, w6, k);
        __builtin_memcpy(remainder_buffer + 7, w7, k);
        __builtin_memcpy(remainder_buffer + 8, w8, k);
        __builtin_memcpy(remainder_buffer + 9, w9, k);
        __builtin_memcpy(remainder_buffer + 10, w10, k);
        __builtin_memcpy(remainder_buffer + 11, w11, k);
        __builtin_memcpy(remainder_buffer + 12, w12, k);
        __builtin_memcpy(remainder_buffer + 13, w13, k);
        __builtin_memcpy(remainder_buffer + 14, w14, k);
        __builtin_memcpy(remainder_buffer + 15, w15, k);

        const __m256i vmask = _mm256_srli_epi64(_mm256_set1_epi32(-1), (8 - k) * sizeof(int8_t) * 8);

        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 3)), 0xC0);
        v0 = _mm256_and_si256(v0, vmask);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 7)), 0xC0);
        v4 = _mm256_and_si256(v4, vmask);
        __m256i v8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 8));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 9)), 0x0C);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 10)), 0x30);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 11)), 0xC0);
        v8 = _mm256_and_si256(v8, vmask);
        __m256i v12 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 12));
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 13)), 0x0C);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 14)), 0x30);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 15)), 0xC0);
        v12 = _mm256_and_si256(v12, vmask);

        w0 += k;
        w1 += k;
        w2 += k;
        w3 += k;
        w4 += k;
        w5 += k;
        w6 += k;
        w7 += k;
        w8 += k;
        w9 += k;
        w10 += k;
        w11 += k;
        w12 += k;
        w13 += k;
        w14 += k;
        w15 += k;


        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);
        _mm256_storeu_si256((__m256i *)&out[64],  v8);
        _mm256_storeu_si256((__m256i *)&out[96],  v12);

        out += 128;
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 15);

      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((uint32_t*) out) = *b++;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((uint32_t*) out) = 0;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      }
      out += (16 - n) * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const int8_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int8_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int8_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int8_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int8_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int8_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int8_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const int8_t* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }


      // KC main loop multiple of 16x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);
        __m256i v8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w8));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w9)), 0x0C);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w10)), 0x30);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w11)), 0xC0);
        __m256i v12 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w12));
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w13)), 0x0C);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w14)), 0x30);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w15)), 0xC0);


        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);
        _mm256_storeu_si256((__m256i *)&out[64],  v8);
        _mm256_storeu_si256((__m256i *)&out[96],  v12);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        w8 += 8;
        w9 += 8;
        w10 += 8;
        w11 += 8;
        w12 += 8;
        w13 += 8;
        w14 += 8;
        w15 += 8;
        out += 128;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);

        // TODO: more efficient copy
        // uint64_t for KR=8 bytes
        uint64_t remainder_buffer[16] = { 0, };
        __builtin_memcpy(remainder_buffer + 0, w0, k);
        __builtin_memcpy(remainder_buffer + 1, w1, k);
        __builtin_memcpy(remainder_buffer + 2, w2, k);
        __builtin_memcpy(remainder_buffer + 3, w3, k);
        __builtin_memcpy(remainder_buffer + 4, w4, k);
        __builtin_memcpy(remainder_buffer + 5, w5, k);
        __builtin_memcpy(remainder_buffer + 6, w6, k);
        __builtin_memcpy(remainder_buffer + 7, w7, k);
        __builtin_memcpy(remainder_buffer + 8, w8, k);
        __builtin_memcpy(remainder_buffer + 9, w9, k);
        __builtin_memcpy(remainder_buffer + 10, w10, k);
        __builtin_memcpy(remainder_buffer + 11, w11, k);
        __builtin_memcpy(remainder_buffer + 12, w12, k);
        __builtin_memcpy(remainder_buffer + 13, w13, k);
        __builtin_memcpy(remainder_buffer + 14, w14, k);
        __builtin_memcpy(remainder_buffer + 15, w15, k);

        const __m256i vmask = _mm256_srli_epi64(_mm256_set1_epi32(-1), (8 - k) * sizeof(int8_t) * 8);

        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 3)), 0xC0);
        v0 = _mm256_and_si256(v0, vmask);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 7)), 0xC0);
        v4 = _mm256_and_si256(v4, vmask);
        __m256i v8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 8));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 9)), 0x0C);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 10)), 0x30);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 11)), 0xC0);
        v8 = _mm256_and_si256(v8, vmask);
        __m256i v12 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 12));
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 13)), 0x0C);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 14)), 0x30);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(remainder_buffer + 15)), 0xC0);
        v12 = _mm256_and_si256(v12, vmask);

        w0 += k;
        w1 += k;
        w2 += k;
        w3 += k;
        w4 += k;
        w5 += k;
        w6 += k;
        w7 += k;
        w8 += k;
        w9 += k;
        w10 += k;
        w11 += k;
        w12 += k;
        w13 += k;
        w14 += k;
        w15 += k;


        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);
        _mm256_storeu_si256((__m256i *)&out[64],  v8);
        _mm256_storeu_si256((__m256i *)&out[96],  v12);

        out += 128;
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * kc;
  } while (--g != 0);
}
