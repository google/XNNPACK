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


void xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16(
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
  assert(nr == 8);   // This kernel is for NR=8
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);


  do {
    const uint16_t* w0 = weights;
    size_t n = nc;
    for (; n >= 8; n -= 8) {
      {
        __m128i vtmp;
        if XNN_LIKELY(bias != NULL) {
          vtmp = _mm_loadu_si128((const __m128i*) bias);
          bias += 8;
        } else {
          vtmp = _mm_setzero_si128();
        }
        _mm_storeu_si128((__m128i*) packed_weights, vtmp);
        packed_weights += 8;
      }
      const uint16_t* w1 = w0 + kc;
      const uint16_t* w2 = w1 + kc;
      const uint16_t* w3 = w2 + kc;
      const uint16_t* w4 = w3 + kc;
      const uint16_t* w5 = w4 + kc;
      const uint16_t* w6 = w5 + kc;
      const uint16_t* w7 = w6 + kc;

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
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v7);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v7);


        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);

        v0 = _mm256_inserti128_si256(vt0, _mm256_castsi256_si128(vt1), 1);
        v1 = _mm256_permute2x128_si256(vt0, vt1, 0x31);
        v2 = _mm256_inserti128_si256(vt2, _mm256_castsi256_si128(vt3), 1);
        v3 = _mm256_permute2x128_si256(vt2, vt3, 0x31);
        v4 = _mm256_inserti128_si256(vt4, _mm256_castsi256_si128(vt5), 1);
        v5 = _mm256_permute2x128_si256(vt4, vt5, 0x31);
        v6 = _mm256_inserti128_si256(vt6, _mm256_castsi256_si128(vt7), 1);
        v7 = _mm256_permute2x128_si256(vt6, vt7, 0x31);
        _mm256_storeu_si256((__m256i*) packed_weights + 0, v0);
        _mm256_storeu_si256((__m256i*) packed_weights + 1, v2);
        _mm256_storeu_si256((__m256i*) packed_weights + 2, v4);
        _mm256_storeu_si256((__m256i*) packed_weights + 3, v6);
        _mm256_storeu_si256((__m256i*) packed_weights + 4, v1);
        _mm256_storeu_si256((__m256i*) packed_weights + 5, v3);
        _mm256_storeu_si256((__m256i*) packed_weights + 6, v5);
        _mm256_storeu_si256((__m256i*) packed_weights + 7, v7);
        packed_weights += 128;
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
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v7);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v7);


        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);

        v0 = _mm256_inserti128_si256(vt0, _mm256_castsi256_si128(vt1), 1);
        v1 = _mm256_permute2x128_si256(vt0, vt1, 0x31);
        v2 = _mm256_inserti128_si256(vt2, _mm256_castsi256_si128(vt3), 1);
        v3 = _mm256_permute2x128_si256(vt2, vt3, 0x31);
        v4 = _mm256_inserti128_si256(vt4, _mm256_castsi256_si128(vt5), 1);
        v5 = _mm256_permute2x128_si256(vt4, vt5, 0x31);
        v6 = _mm256_inserti128_si256(vt6, _mm256_castsi256_si128(vt7), 1);
        v7 = _mm256_permute2x128_si256(vt6, vt7, 0x31);
        if (k & 8) {
          _mm256_storeu_si256((__m256i*) packed_weights + 0, v0);
          _mm256_storeu_si256((__m256i*) packed_weights + 1, v2);
          _mm256_storeu_si256((__m256i*) packed_weights + 2, v4);
          _mm256_storeu_si256((__m256i*) packed_weights + 3, v6);
          packed_weights += 64;
          v0 = v1;
          v2 = v3;
          v4 = v5;
          v6 = v7;
        }
        if (k & 4) {
          _mm256_storeu_si256((__m256i*) packed_weights + 0, v0);
          _mm256_storeu_si256((__m256i*) packed_weights + 1, v2);
          packed_weights += 32;
          v0 = v4;
          v2 = v6;
        }
        if (k & 2) {
          _mm256_storeu_si256((__m256i*) packed_weights + 0, v0);
          packed_weights += 16;
          v0 = v2;
        }
        if (k & 1) {
          _mm_storeu_si128((__m128i*) packed_weights, _mm256_castsi256_si128(v0));
          packed_weights += 8;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      if XNN_LIKELY(bias != NULL) {
        memcpy(packed_weights, bias, n * 2);
        bias += n;
      } else {
        memset(packed_weights, 0, 16);
      }
      packed_weights += 8;
      // NR remainder has less than 8 rows so last row is not loaded
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
        __m256i v7;
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v6);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v6);


        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);

        v0 = _mm256_inserti128_si256(vt0, _mm256_castsi256_si128(vt1), 1);
        v1 = _mm256_permute2x128_si256(vt0, vt1, 0x31);
        v2 = _mm256_inserti128_si256(vt2, _mm256_castsi256_si128(vt3), 1);
        v3 = _mm256_permute2x128_si256(vt2, vt3, 0x31);
        v4 = _mm256_inserti128_si256(vt4, _mm256_castsi256_si128(vt5), 1);
        v5 = _mm256_permute2x128_si256(vt4, vt5, 0x31);
        v6 = _mm256_inserti128_si256(vt6, _mm256_castsi256_si128(vt7), 1);
        v7 = _mm256_permute2x128_si256(vt6, vt7, 0x31);
        _mm256_storeu_si256((__m256i*) packed_weights + 0, v0);
        _mm256_storeu_si256((__m256i*) packed_weights + 1, v2);
        _mm256_storeu_si256((__m256i*) packed_weights + 2, v4);
        _mm256_storeu_si256((__m256i*) packed_weights + 3, v6);
        _mm256_storeu_si256((__m256i*) packed_weights + 4, v1);
        _mm256_storeu_si256((__m256i*) packed_weights + 5, v3);
        _mm256_storeu_si256((__m256i*) packed_weights + 6, v5);
        _mm256_storeu_si256((__m256i*) packed_weights + 7, v7);
        packed_weights += 128;
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
            break;
        }
        w0 += k;
        w1 += k;
        w2 += k;
        w3 += k;
        w4 += k;
        w5 += k;
        w6 += k;
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v6);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v6);


        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);

        v0 = _mm256_inserti128_si256(vt0, _mm256_castsi256_si128(vt1), 1);
        v1 = _mm256_permute2x128_si256(vt0, vt1, 0x31);
        v2 = _mm256_inserti128_si256(vt2, _mm256_castsi256_si128(vt3), 1);
        v3 = _mm256_permute2x128_si256(vt2, vt3, 0x31);
        v4 = _mm256_inserti128_si256(vt4, _mm256_castsi256_si128(vt5), 1);
        v5 = _mm256_permute2x128_si256(vt4, vt5, 0x31);
        v6 = _mm256_inserti128_si256(vt6, _mm256_castsi256_si128(vt7), 1);
        v7 = _mm256_permute2x128_si256(vt6, vt7, 0x31);
        if (k & 8) {
          _mm256_storeu_si256((__m256i*) packed_weights + 0, v0);
          _mm256_storeu_si256((__m256i*) packed_weights + 1, v2);
          _mm256_storeu_si256((__m256i*) packed_weights + 2, v4);
          _mm256_storeu_si256((__m256i*) packed_weights + 3, v6);
          packed_weights += 64;
          v0 = v1;
          v2 = v3;
          v4 = v5;
          v6 = v7;
        }
        if (k & 4) {
          _mm256_storeu_si256((__m256i*) packed_weights + 0, v0);
          _mm256_storeu_si256((__m256i*) packed_weights + 1, v2);
          packed_weights += 32;
          v0 = v4;
          v2 = v6;
        }
        if (k & 2) {
          _mm256_storeu_si256((__m256i*) packed_weights + 0, v0);
          packed_weights += 16;
          v0 = v2;
        }
        if (k & 1) {
          _mm_storeu_si128((__m128i*) packed_weights, _mm256_castsi256_si128(v0));
          packed_weights += 8;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
