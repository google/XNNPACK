// Auto-generated file. Do not edit!
//   Template: src/x16-packw/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "xnnpack/packw.h"


void xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16(
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
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);


  do {
    const uint16_t* w0 = weights;
    size_t n = nc;
    for (; n >= 16; n -= 16) {
      {
        __m256i vtmp;
        if XNN_LIKELY(bias != NULL) {
          vtmp = _mm256_loadu_si256((const __m256i*) bias);
          bias += 16;
        } else {
          vtmp = _mm256_setzero_si256();
        }
        _mm256_store_si256((__m256i*) packed_weights, vtmp);
        packed_weights += 16;
      }
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

      size_t k = kc;
      for (; k >= 16; k -= 16) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        w0 += 16;
        __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        w1 += 16;
        __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        w2 += 16;
        __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        w3 += 16;
        __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        w4 += 16;
        __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        w5 += 16;
        __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        w6 += 16;
        __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        w7 += 16;
        __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        w8 += 16;
        __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        w9 += 16;
        __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        w10 += 16;
        __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        w11 += 16;
        __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        w12 += 16;
        __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        w13 += 16;
        __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        w14 += 16;
        __m256i v15 = _mm256_loadu_si256((const __m256i*) w15);
        w15 += 16;
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v7);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v7);
        __m256i vt8 = _mm256_unpacklo_epi16(v8, v9);
        __m256i vt9 = _mm256_unpackhi_epi16(v8, v9);
        __m256i vt10 = _mm256_unpacklo_epi16(v10, v11);
        __m256i vt11 = _mm256_unpackhi_epi16(v10, v11);
        __m256i vt12 = _mm256_unpacklo_epi16(v12, v13);
        __m256i vt13 = _mm256_unpackhi_epi16(v12, v13);
        __m256i vt14 = _mm256_unpacklo_epi16(v14, v15);
        __m256i vt15 = _mm256_unpackhi_epi16(v14, v15);


        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);
        v8 = _mm256_unpacklo_epi32(vt8, vt10);
        v9 = _mm256_unpackhi_epi32(vt8, vt10);
        v10 = _mm256_unpacklo_epi32(vt9, vt11);
        v11 = _mm256_unpackhi_epi32(vt9, vt11);
        v12 = _mm256_unpacklo_epi32(vt12, vt14);
        v13 = _mm256_unpackhi_epi32(vt12, vt14);
        v14 = _mm256_unpacklo_epi32(vt13, vt15);
        v15 = _mm256_unpackhi_epi32(vt13, vt15);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);
        vt8 = _mm256_unpacklo_epi64(v8, v12);
        vt9 = _mm256_unpackhi_epi64(v8, v12);
        vt10 = _mm256_unpacklo_epi64(v9, v13);
        vt11 = _mm256_unpackhi_epi64(v9, v13);
        vt12 = _mm256_unpacklo_epi64(v10, v14);
        vt13 = _mm256_unpackhi_epi64(v10, v14);
        vt14 = _mm256_unpacklo_epi64(v11, v15);
        vt15 = _mm256_unpackhi_epi64(v11, v15);

        v0 = _mm256_inserti128_si256(vt0, _mm256_castsi256_si128(vt8), 1);
        v8 = _mm256_permute2x128_si256(vt0, vt8, 0x31);
        v1 = _mm256_inserti128_si256(vt1, _mm256_castsi256_si128(vt9), 1);
        v9 = _mm256_permute2x128_si256(vt1, vt9, 0x31);
        v2 = _mm256_inserti128_si256(vt2, _mm256_castsi256_si128(vt10), 1);
        v10 = _mm256_permute2x128_si256(vt2, vt10, 0x31);
        v3 = _mm256_inserti128_si256(vt3, _mm256_castsi256_si128(vt11), 1);
        v11 = _mm256_permute2x128_si256(vt3, vt11, 0x31);
        v4 = _mm256_inserti128_si256(vt4, _mm256_castsi256_si128(vt12), 1);
        v12 = _mm256_permute2x128_si256(vt4, vt12, 0x31);
        v5 = _mm256_inserti128_si256(vt5, _mm256_castsi256_si128(vt13), 1);
        v13 = _mm256_permute2x128_si256(vt5, vt13, 0x31);
        v6 = _mm256_inserti128_si256(vt6, _mm256_castsi256_si128(vt14), 1);
        v14 = _mm256_permute2x128_si256(vt6, vt14, 0x31);
        v7 = _mm256_inserti128_si256(vt7, _mm256_castsi256_si128(vt15), 1);
        v15 = _mm256_permute2x128_si256(vt7, vt15, 0x31);
        _mm256_store_si256((__m256i*) packed_weights + 0, v0);
        _mm256_store_si256((__m256i*) packed_weights + 1, v1);
        _mm256_store_si256((__m256i*) packed_weights + 2, v2);
        _mm256_store_si256((__m256i*) packed_weights + 3, v3);
        _mm256_store_si256((__m256i*) packed_weights + 4, v4);
        _mm256_store_si256((__m256i*) packed_weights + 5, v5);
        _mm256_store_si256((__m256i*) packed_weights + 6, v6);
        _mm256_store_si256((__m256i*) packed_weights + 7, v7);
        _mm256_store_si256((__m256i*) packed_weights + 8, v8);
        _mm256_store_si256((__m256i*) packed_weights + 9, v9);
        _mm256_store_si256((__m256i*) packed_weights + 10, v10);
        _mm256_store_si256((__m256i*) packed_weights + 11, v11);
        _mm256_store_si256((__m256i*) packed_weights + 12, v12);
        _mm256_store_si256((__m256i*) packed_weights + 13, v13);
        _mm256_store_si256((__m256i*) packed_weights + 14, v14);
        _mm256_store_si256((__m256i*) packed_weights + 15, v15);
        packed_weights += 256;
      }
      // KC remainder
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k < 16);
        __m256i v0;
        __m256i v1;
        __m256i v2;
        __m256i v3;
        __m256i v4;
        __m256i v5;
        __m256i v6;
        __m256i v7;
        __m256i v8;
        __m256i v9;
        __m256i v10;
        __m256i v11;
        __m256i v12;
        __m256i v13;
        __m256i v14;
        __m256i v15;
        __m256i vmask;
        switch(k) {
          case 1:
            v0 = _mm256_setzero_si256();
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[0], 0);
            v1 = _mm256_setzero_si256();
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[0], 0);
            v2 = _mm256_setzero_si256();
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[0], 0);
            v3 = _mm256_setzero_si256();
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[0], 0);
            v4 = _mm256_setzero_si256();
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[0], 0);
            v5 = _mm256_setzero_si256();
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[0], 0);
            v6 = _mm256_setzero_si256();
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[0], 0);
            v7 = _mm256_setzero_si256();
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[0], 0);
            v8 = _mm256_setzero_si256();
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[0], 0);
            v9 = _mm256_setzero_si256();
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[0], 0);
            v10 = _mm256_setzero_si256();
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[0], 0);
            v11 = _mm256_setzero_si256();
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[0], 0);
            v12 = _mm256_setzero_si256();
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[0], 0);
            v13 = _mm256_setzero_si256();
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[0], 0);
            v14 = _mm256_setzero_si256();
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[0], 0);
            v15 = _mm256_setzero_si256();
            v15 = _mm256_insert_epi16(v15, (int16_t) w15[0], 0);
            break;
          case 2:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            break;
          case 3:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[2], 2);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[2], 2);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[2], 2);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[2], 2);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[2], 2);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[2], 2);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[2], 2);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[2], 2);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[2], 2);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[2], 2);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[2], 2);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[2], 2);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[2], 2);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[2], 2);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[2], 2);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            v15 = _mm256_insert_epi16(v15, (int16_t) w15[2], 2);
            break;
          case 4:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            break;
          case 5:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[4], 4);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[4], 4);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[4], 4);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[4], 4);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[4], 4);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[4], 4);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[4], 4);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[4], 4);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[4], 4);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[4], 4);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[4], 4);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[4], 4);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[4], 4);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[4], 4);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[4], 4);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            v15 = _mm256_insert_epi16(v15, (int16_t) w15[4], 4);
            break;
          case 6:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            break;
          case 7:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[6], 6);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[6], 6);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[6], 6);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[6], 6);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[6], 6);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[6], 6);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[6], 6);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[6], 6);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[6], 6);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[6], 6);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[6], 6);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[6], 6);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[6], 6);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[6], 6);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[6], 6);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            v15 = _mm256_insert_epi16(v15, (int16_t) w15[6], 6);
            break;
          case 8:
            vmask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            break;
          case 9:
            vmask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[8], 8);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[8], 8);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[8], 8);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[8], 8);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[8], 8);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[8], 8);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[8], 8);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[8], 8);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[8], 8);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[8], 8);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[8], 8);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[8], 8);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[8], 8);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[8], 8);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[8], 8);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            v15 = _mm256_insert_epi16(v15, (int16_t) w15[8], 8);
            break;
          case 10:
            vmask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            break;
          case 11:
            vmask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[10], 10);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[10], 10);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[10], 10);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[10], 10);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[10], 10);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[10], 10);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[10], 10);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[10], 10);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[10], 10);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[10], 10);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[10], 10);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[10], 10);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[10], 10);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[10], 10);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[10], 10);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            v15 = _mm256_insert_epi16(v15, (int16_t) w15[10], 10);
            break;
          case 12:
            vmask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            break;
          case 13:
            vmask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[12], 12);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[12], 12);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[12], 12);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[12], 12);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[12], 12);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[12], 12);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[12], 12);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[12], 12);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[12], 12);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[12], 12);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[12], 12);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[12], 12);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[12], 12);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[12], 12);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[12], 12);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            v15 = _mm256_insert_epi16(v15, (int16_t) w15[12], 12);
            break;
          case 14:
            vmask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            break;
          case 15:
            vmask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[14], 14);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[14], 14);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[14], 14);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[14], 14);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[14], 14);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[14], 14);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[14], 14);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[14], 14);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[14], 14);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[14], 14);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[14], 14);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[14], 14);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[14], 14);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[14], 14);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[14], 14);
            v15 = _mm256_maskload_epi32((const int*) w15, vmask);
            v15 = _mm256_insert_epi16(v15, (int16_t) w15[14], 14);
            break;
        }
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
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v7);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v7);
        __m256i vt8 = _mm256_unpacklo_epi16(v8, v9);
        __m256i vt9 = _mm256_unpackhi_epi16(v8, v9);
        __m256i vt10 = _mm256_unpacklo_epi16(v10, v11);
        __m256i vt11 = _mm256_unpackhi_epi16(v10, v11);
        __m256i vt12 = _mm256_unpacklo_epi16(v12, v13);
        __m256i vt13 = _mm256_unpackhi_epi16(v12, v13);
        __m256i vt14 = _mm256_unpacklo_epi16(v14, v15);
        __m256i vt15 = _mm256_unpackhi_epi16(v14, v15);


        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);
        v8 = _mm256_unpacklo_epi32(vt8, vt10);
        v9 = _mm256_unpackhi_epi32(vt8, vt10);
        v10 = _mm256_unpacklo_epi32(vt9, vt11);
        v11 = _mm256_unpackhi_epi32(vt9, vt11);
        v12 = _mm256_unpacklo_epi32(vt12, vt14);
        v13 = _mm256_unpackhi_epi32(vt12, vt14);
        v14 = _mm256_unpacklo_epi32(vt13, vt15);
        v15 = _mm256_unpackhi_epi32(vt13, vt15);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);
        vt8 = _mm256_unpacklo_epi64(v8, v12);
        vt9 = _mm256_unpackhi_epi64(v8, v12);
        vt10 = _mm256_unpacklo_epi64(v9, v13);
        vt11 = _mm256_unpackhi_epi64(v9, v13);
        vt12 = _mm256_unpacklo_epi64(v10, v14);
        vt13 = _mm256_unpackhi_epi64(v10, v14);
        vt14 = _mm256_unpacklo_epi64(v11, v15);
        vt15 = _mm256_unpackhi_epi64(v11, v15);

        v0 = _mm256_inserti128_si256(vt0, _mm256_castsi256_si128(vt8), 1);
        v8 = _mm256_permute2x128_si256(vt0, vt8, 0x31);
        v1 = _mm256_inserti128_si256(vt1, _mm256_castsi256_si128(vt9), 1);
        v9 = _mm256_permute2x128_si256(vt1, vt9, 0x31);
        v2 = _mm256_inserti128_si256(vt2, _mm256_castsi256_si128(vt10), 1);
        v10 = _mm256_permute2x128_si256(vt2, vt10, 0x31);
        v3 = _mm256_inserti128_si256(vt3, _mm256_castsi256_si128(vt11), 1);
        v11 = _mm256_permute2x128_si256(vt3, vt11, 0x31);
        v4 = _mm256_inserti128_si256(vt4, _mm256_castsi256_si128(vt12), 1);
        v12 = _mm256_permute2x128_si256(vt4, vt12, 0x31);
        v5 = _mm256_inserti128_si256(vt5, _mm256_castsi256_si128(vt13), 1);
        v13 = _mm256_permute2x128_si256(vt5, vt13, 0x31);
        v6 = _mm256_inserti128_si256(vt6, _mm256_castsi256_si128(vt14), 1);
        v14 = _mm256_permute2x128_si256(vt6, vt14, 0x31);
        v7 = _mm256_inserti128_si256(vt7, _mm256_castsi256_si128(vt15), 1);
        v15 = _mm256_permute2x128_si256(vt7, vt15, 0x31);
        if (k & 8) {
          _mm256_store_si256((__m256i*) packed_weights + 0, v0);
          _mm256_store_si256((__m256i*) packed_weights + 1, v1);
          _mm256_store_si256((__m256i*) packed_weights + 2, v2);
          _mm256_store_si256((__m256i*) packed_weights + 3, v3);
          _mm256_store_si256((__m256i*) packed_weights + 4, v4);
          _mm256_store_si256((__m256i*) packed_weights + 5, v5);
          _mm256_store_si256((__m256i*) packed_weights + 6, v6);
          _mm256_store_si256((__m256i*) packed_weights + 7, v7);
          packed_weights += 128;
          v0 = v8;
          v1 = v9;
          v2 = v10;
          v3 = v11;
          v4 = v12;
          v5 = v13;
          v6 = v14;
          v7 = v15;
        }
        if (k & 4) {
          _mm256_store_si256((__m256i*) packed_weights + 0, v0);
          _mm256_store_si256((__m256i*) packed_weights + 1, v1);
          _mm256_store_si256((__m256i*) packed_weights + 2, v2);
          _mm256_store_si256((__m256i*) packed_weights + 3, v3);
          packed_weights += 64;
          v0 = v4;
          v1 = v5;
          v2 = v6;
          v3 = v7;
        }
        if (k & 2) {
          _mm256_store_si256((__m256i*) packed_weights + 0, v0);
          _mm256_store_si256((__m256i*) packed_weights + 1, v1);
          packed_weights += 32;
          v0 = v2;
          v1 = v3;
        }
        if (k & 1) {
          _mm256_store_si256((__m256i*) packed_weights + 0, v0);
          packed_weights += 16;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(bias != NULL) {
        memcpy(packed_weights, bias, n * 2);
        bias += n;
      } else {
        memset(packed_weights, 0, 32);
      }
      packed_weights += 16;
      // NR remainder has less than 16 rows so last row is not loaded
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

      size_t k = kc;
      for (; k >= 16; k -= 16) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        w0 += 16;
        __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        w1 += 16;
        __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        w2 += 16;
        __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        w3 += 16;
        __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        w4 += 16;
        __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        w5 += 16;
        __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        w6 += 16;
        __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        w7 += 16;
        __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        w8 += 16;
        __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        w9 += 16;
        __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        w10 += 16;
        __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        w11 += 16;
        __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        w12 += 16;
        __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        w13 += 16;
        __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        w14 += 16;
        __m256i v15;
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v7);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v7);
        __m256i vt8 = _mm256_unpacklo_epi16(v8, v9);
        __m256i vt9 = _mm256_unpackhi_epi16(v8, v9);
        __m256i vt10 = _mm256_unpacklo_epi16(v10, v11);
        __m256i vt11 = _mm256_unpackhi_epi16(v10, v11);
        __m256i vt12 = _mm256_unpacklo_epi16(v12, v13);
        __m256i vt13 = _mm256_unpackhi_epi16(v12, v13);
        __m256i vt14 = _mm256_unpacklo_epi16(v14, v14);
        __m256i vt15 = _mm256_unpackhi_epi16(v14, v14);


        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);
        v8 = _mm256_unpacklo_epi32(vt8, vt10);
        v9 = _mm256_unpackhi_epi32(vt8, vt10);
        v10 = _mm256_unpacklo_epi32(vt9, vt11);
        v11 = _mm256_unpackhi_epi32(vt9, vt11);
        v12 = _mm256_unpacklo_epi32(vt12, vt14);
        v13 = _mm256_unpackhi_epi32(vt12, vt14);
        v14 = _mm256_unpacklo_epi32(vt13, vt15);
        v15 = _mm256_unpackhi_epi32(vt13, vt15);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);
        vt8 = _mm256_unpacklo_epi64(v8, v12);
        vt9 = _mm256_unpackhi_epi64(v8, v12);
        vt10 = _mm256_unpacklo_epi64(v9, v13);
        vt11 = _mm256_unpackhi_epi64(v9, v13);
        vt12 = _mm256_unpacklo_epi64(v10, v14);
        vt13 = _mm256_unpackhi_epi64(v10, v14);
        vt14 = _mm256_unpacklo_epi64(v11, v15);
        vt15 = _mm256_unpackhi_epi64(v11, v15);

        v0 = _mm256_inserti128_si256(vt0, _mm256_castsi256_si128(vt8), 1);
        v8 = _mm256_permute2x128_si256(vt0, vt8, 0x31);
        v1 = _mm256_inserti128_si256(vt1, _mm256_castsi256_si128(vt9), 1);
        v9 = _mm256_permute2x128_si256(vt1, vt9, 0x31);
        v2 = _mm256_inserti128_si256(vt2, _mm256_castsi256_si128(vt10), 1);
        v10 = _mm256_permute2x128_si256(vt2, vt10, 0x31);
        v3 = _mm256_inserti128_si256(vt3, _mm256_castsi256_si128(vt11), 1);
        v11 = _mm256_permute2x128_si256(vt3, vt11, 0x31);
        v4 = _mm256_inserti128_si256(vt4, _mm256_castsi256_si128(vt12), 1);
        v12 = _mm256_permute2x128_si256(vt4, vt12, 0x31);
        v5 = _mm256_inserti128_si256(vt5, _mm256_castsi256_si128(vt13), 1);
        v13 = _mm256_permute2x128_si256(vt5, vt13, 0x31);
        v6 = _mm256_inserti128_si256(vt6, _mm256_castsi256_si128(vt14), 1);
        v14 = _mm256_permute2x128_si256(vt6, vt14, 0x31);
        v7 = _mm256_inserti128_si256(vt7, _mm256_castsi256_si128(vt15), 1);
        v15 = _mm256_permute2x128_si256(vt7, vt15, 0x31);
        _mm256_store_si256((__m256i*) packed_weights + 0, v0);
        _mm256_store_si256((__m256i*) packed_weights + 1, v1);
        _mm256_store_si256((__m256i*) packed_weights + 2, v2);
        _mm256_store_si256((__m256i*) packed_weights + 3, v3);
        _mm256_store_si256((__m256i*) packed_weights + 4, v4);
        _mm256_store_si256((__m256i*) packed_weights + 5, v5);
        _mm256_store_si256((__m256i*) packed_weights + 6, v6);
        _mm256_store_si256((__m256i*) packed_weights + 7, v7);
        _mm256_store_si256((__m256i*) packed_weights + 8, v8);
        _mm256_store_si256((__m256i*) packed_weights + 9, v9);
        _mm256_store_si256((__m256i*) packed_weights + 10, v10);
        _mm256_store_si256((__m256i*) packed_weights + 11, v11);
        _mm256_store_si256((__m256i*) packed_weights + 12, v12);
        _mm256_store_si256((__m256i*) packed_weights + 13, v13);
        _mm256_store_si256((__m256i*) packed_weights + 14, v14);
        _mm256_store_si256((__m256i*) packed_weights + 15, v15);
        packed_weights += 256;
      }

      // KC and NC remainder
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k < 16);
        __m256i v0;
        __m256i v1;
        __m256i v2;
        __m256i v3;
        __m256i v4;
        __m256i v5;
        __m256i v6;
        __m256i v7;
        __m256i v8;
        __m256i v9;
        __m256i v10;
        __m256i v11;
        __m256i v12;
        __m256i v13;
        __m256i v14;
        __m256i v15;
        __m256i vmask;
        switch(k) {
          case 1:
            v0 = _mm256_setzero_si256();
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[0], 0);
            v1 = _mm256_setzero_si256();
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[0], 0);
            v2 = _mm256_setzero_si256();
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[0], 0);
            v3 = _mm256_setzero_si256();
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[0], 0);
            v4 = _mm256_setzero_si256();
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[0], 0);
            v5 = _mm256_setzero_si256();
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[0], 0);
            v6 = _mm256_setzero_si256();
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[0], 0);
            v7 = _mm256_setzero_si256();
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[0], 0);
            v8 = _mm256_setzero_si256();
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[0], 0);
            v9 = _mm256_setzero_si256();
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[0], 0);
            v10 = _mm256_setzero_si256();
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[0], 0);
            v11 = _mm256_setzero_si256();
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[0], 0);
            v12 = _mm256_setzero_si256();
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[0], 0);
            v13 = _mm256_setzero_si256();
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[0], 0);
            v14 = _mm256_setzero_si256();
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[0], 0);
            break;
          case 2:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            break;
          case 3:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[2], 2);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[2], 2);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[2], 2);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[2], 2);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[2], 2);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[2], 2);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[2], 2);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[2], 2);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[2], 2);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[2], 2);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[2], 2);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[2], 2);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[2], 2);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[2], 2);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[2], 2);
            break;
          case 4:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            break;
          case 5:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[4], 4);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[4], 4);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[4], 4);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[4], 4);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[4], 4);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[4], 4);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[4], 4);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[4], 4);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[4], 4);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[4], 4);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[4], 4);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[4], 4);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[4], 4);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[4], 4);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[4], 4);
            break;
          case 6:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            break;
          case 7:
            vmask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[6], 6);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[6], 6);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[6], 6);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[6], 6);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[6], 6);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[6], 6);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[6], 6);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[6], 6);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[6], 6);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[6], 6);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[6], 6);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[6], 6);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[6], 6);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[6], 6);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[6], 6);
            break;
          case 8:
            vmask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            break;
          case 9:
            vmask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[8], 8);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[8], 8);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[8], 8);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[8], 8);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[8], 8);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[8], 8);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[8], 8);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[8], 8);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[8], 8);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[8], 8);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[8], 8);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[8], 8);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[8], 8);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[8], 8);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[8], 8);
            break;
          case 10:
            vmask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            break;
          case 11:
            vmask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[10], 10);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[10], 10);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[10], 10);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[10], 10);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[10], 10);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[10], 10);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[10], 10);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[10], 10);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[10], 10);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[10], 10);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[10], 10);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[10], 10);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[10], 10);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[10], 10);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[10], 10);
            break;
          case 12:
            vmask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            break;
          case 13:
            vmask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[12], 12);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[12], 12);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[12], 12);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[12], 12);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[12], 12);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[12], 12);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[12], 12);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[12], 12);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[12], 12);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[12], 12);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[12], 12);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[12], 12);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[12], 12);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[12], 12);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[12], 12);
            break;
          case 14:
            vmask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            break;
          case 15:
            vmask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
            v0 = _mm256_maskload_epi32((const int*) w0, vmask);
            v0 = _mm256_insert_epi16(v0, (int16_t) w0[14], 14);
            v1 = _mm256_maskload_epi32((const int*) w1, vmask);
            v1 = _mm256_insert_epi16(v1, (int16_t) w1[14], 14);
            v2 = _mm256_maskload_epi32((const int*) w2, vmask);
            v2 = _mm256_insert_epi16(v2, (int16_t) w2[14], 14);
            v3 = _mm256_maskload_epi32((const int*) w3, vmask);
            v3 = _mm256_insert_epi16(v3, (int16_t) w3[14], 14);
            v4 = _mm256_maskload_epi32((const int*) w4, vmask);
            v4 = _mm256_insert_epi16(v4, (int16_t) w4[14], 14);
            v5 = _mm256_maskload_epi32((const int*) w5, vmask);
            v5 = _mm256_insert_epi16(v5, (int16_t) w5[14], 14);
            v6 = _mm256_maskload_epi32((const int*) w6, vmask);
            v6 = _mm256_insert_epi16(v6, (int16_t) w6[14], 14);
            v7 = _mm256_maskload_epi32((const int*) w7, vmask);
            v7 = _mm256_insert_epi16(v7, (int16_t) w7[14], 14);
            v8 = _mm256_maskload_epi32((const int*) w8, vmask);
            v8 = _mm256_insert_epi16(v8, (int16_t) w8[14], 14);
            v9 = _mm256_maskload_epi32((const int*) w9, vmask);
            v9 = _mm256_insert_epi16(v9, (int16_t) w9[14], 14);
            v10 = _mm256_maskload_epi32((const int*) w10, vmask);
            v10 = _mm256_insert_epi16(v10, (int16_t) w10[14], 14);
            v11 = _mm256_maskload_epi32((const int*) w11, vmask);
            v11 = _mm256_insert_epi16(v11, (int16_t) w11[14], 14);
            v12 = _mm256_maskload_epi32((const int*) w12, vmask);
            v12 = _mm256_insert_epi16(v12, (int16_t) w12[14], 14);
            v13 = _mm256_maskload_epi32((const int*) w13, vmask);
            v13 = _mm256_insert_epi16(v13, (int16_t) w13[14], 14);
            v14 = _mm256_maskload_epi32((const int*) w14, vmask);
            v14 = _mm256_insert_epi16(v14, (int16_t) w14[14], 14);
            break;
        }
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
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v7);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v7);
        __m256i vt8 = _mm256_unpacklo_epi16(v8, v9);
        __m256i vt9 = _mm256_unpackhi_epi16(v8, v9);
        __m256i vt10 = _mm256_unpacklo_epi16(v10, v11);
        __m256i vt11 = _mm256_unpackhi_epi16(v10, v11);
        __m256i vt12 = _mm256_unpacklo_epi16(v12, v13);
        __m256i vt13 = _mm256_unpackhi_epi16(v12, v13);
        __m256i vt14 = _mm256_unpacklo_epi16(v14, v14);
        __m256i vt15 = _mm256_unpackhi_epi16(v14, v14);


        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);
        v8 = _mm256_unpacklo_epi32(vt8, vt10);
        v9 = _mm256_unpackhi_epi32(vt8, vt10);
        v10 = _mm256_unpacklo_epi32(vt9, vt11);
        v11 = _mm256_unpackhi_epi32(vt9, vt11);
        v12 = _mm256_unpacklo_epi32(vt12, vt14);
        v13 = _mm256_unpackhi_epi32(vt12, vt14);
        v14 = _mm256_unpacklo_epi32(vt13, vt15);
        v15 = _mm256_unpackhi_epi32(vt13, vt15);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);
        vt8 = _mm256_unpacklo_epi64(v8, v12);
        vt9 = _mm256_unpackhi_epi64(v8, v12);
        vt10 = _mm256_unpacklo_epi64(v9, v13);
        vt11 = _mm256_unpackhi_epi64(v9, v13);
        vt12 = _mm256_unpacklo_epi64(v10, v14);
        vt13 = _mm256_unpackhi_epi64(v10, v14);
        vt14 = _mm256_unpacklo_epi64(v11, v15);
        vt15 = _mm256_unpackhi_epi64(v11, v15);

        v0 = _mm256_inserti128_si256(vt0, _mm256_castsi256_si128(vt8), 1);
        v8 = _mm256_permute2x128_si256(vt0, vt8, 0x31);
        v1 = _mm256_inserti128_si256(vt1, _mm256_castsi256_si128(vt9), 1);
        v9 = _mm256_permute2x128_si256(vt1, vt9, 0x31);
        v2 = _mm256_inserti128_si256(vt2, _mm256_castsi256_si128(vt10), 1);
        v10 = _mm256_permute2x128_si256(vt2, vt10, 0x31);
        v3 = _mm256_inserti128_si256(vt3, _mm256_castsi256_si128(vt11), 1);
        v11 = _mm256_permute2x128_si256(vt3, vt11, 0x31);
        v4 = _mm256_inserti128_si256(vt4, _mm256_castsi256_si128(vt12), 1);
        v12 = _mm256_permute2x128_si256(vt4, vt12, 0x31);
        v5 = _mm256_inserti128_si256(vt5, _mm256_castsi256_si128(vt13), 1);
        v13 = _mm256_permute2x128_si256(vt5, vt13, 0x31);
        v6 = _mm256_inserti128_si256(vt6, _mm256_castsi256_si128(vt14), 1);
        v14 = _mm256_permute2x128_si256(vt6, vt14, 0x31);
        v7 = _mm256_inserti128_si256(vt7, _mm256_castsi256_si128(vt15), 1);
        v15 = _mm256_permute2x128_si256(vt7, vt15, 0x31);
        if (k & 8) {
          _mm256_store_si256((__m256i*) packed_weights + 0, v0);
          _mm256_store_si256((__m256i*) packed_weights + 1, v1);
          _mm256_store_si256((__m256i*) packed_weights + 2, v2);
          _mm256_store_si256((__m256i*) packed_weights + 3, v3);
          _mm256_store_si256((__m256i*) packed_weights + 4, v4);
          _mm256_store_si256((__m256i*) packed_weights + 5, v5);
          _mm256_store_si256((__m256i*) packed_weights + 6, v6);
          _mm256_store_si256((__m256i*) packed_weights + 7, v7);
          packed_weights += 128;
          v0 = v8;
          v1 = v9;
          v2 = v10;
          v3 = v11;
          v4 = v12;
          v5 = v13;
          v6 = v14;
          v7 = v15;
        }
        if (k & 4) {
          _mm256_store_si256((__m256i*) packed_weights + 0, v0);
          _mm256_store_si256((__m256i*) packed_weights + 1, v1);
          _mm256_store_si256((__m256i*) packed_weights + 2, v2);
          _mm256_store_si256((__m256i*) packed_weights + 3, v3);
          packed_weights += 64;
          v0 = v4;
          v1 = v5;
          v2 = v6;
          v3 = v7;
        }
        if (k & 2) {
          _mm256_store_si256((__m256i*) packed_weights + 0, v0);
          _mm256_store_si256((__m256i*) packed_weights + 1, v1);
          packed_weights += 32;
          v0 = v2;
          v1 = v3;
        }
        if (k & 1) {
          _mm256_store_si256((__m256i*) packed_weights + 0, v0);
          packed_weights += 16;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
