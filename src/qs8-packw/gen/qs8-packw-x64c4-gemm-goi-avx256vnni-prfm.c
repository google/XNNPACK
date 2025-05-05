// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/c4-avxvnni.c.in
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

#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"
#include "src/xnnpack/prefetch.h"

XNN_INLINE static uint32_t safe_load_u32(const void* src, size_t k) {
  uint32_t value = 0;
  const uint8_t* bytes = (const uint8_t*)src;
  for (size_t i = 0; i < k; ++i) {
    value |= (uint32_t) bytes[i] << (i * 8);
  }
  return value;
}


void xnn_qs8_packw_gemm_goi_ukernel_x64c4__avx256vnni_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 64);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;

  const __m256i vone = _mm256_set1_epi8(1);
  const __m256i vzeropoint = _mm256_set1_epi32((int32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 0): 0));

  do {
    // NC main loop multiple of 64
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 64; n -= 64) {
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
      const int8_t* w16 = w15 + kc;
      const int8_t* w17 = w16 + kc;
      const int8_t* w18 = w17 + kc;
      const int8_t* w19 = w18 + kc;
      const int8_t* w20 = w19 + kc;
      const int8_t* w21 = w20 + kc;
      const int8_t* w22 = w21 + kc;
      const int8_t* w23 = w22 + kc;
      const int8_t* w24 = w23 + kc;
      const int8_t* w25 = w24 + kc;
      const int8_t* w26 = w25 + kc;
      const int8_t* w27 = w26 + kc;
      const int8_t* w28 = w27 + kc;
      const int8_t* w29 = w28 + kc;
      const int8_t* w30 = w29 + kc;
      const int8_t* w31 = w30 + kc;
      const int8_t* w32 = w31 + kc;
      const int8_t* w33 = w32 + kc;
      const int8_t* w34 = w33 + kc;
      const int8_t* w35 = w34 + kc;
      const int8_t* w36 = w35 + kc;
      const int8_t* w37 = w36 + kc;
      const int8_t* w38 = w37 + kc;
      const int8_t* w39 = w38 + kc;
      const int8_t* w40 = w39 + kc;
      const int8_t* w41 = w40 + kc;
      const int8_t* w42 = w41 + kc;
      const int8_t* w43 = w42 + kc;
      const int8_t* w44 = w43 + kc;
      const int8_t* w45 = w44 + kc;
      const int8_t* w46 = w45 + kc;
      const int8_t* w47 = w46 + kc;
      const int8_t* w48 = w47 + kc;
      const int8_t* w49 = w48 + kc;
      const int8_t* w50 = w49 + kc;
      const int8_t* w51 = w50 + kc;
      const int8_t* w52 = w51 + kc;
      const int8_t* w53 = w52 + kc;
      const int8_t* w54 = w53 + kc;
      const int8_t* w55 = w54 + kc;
      const int8_t* w56 = w55 + kc;
      const int8_t* w57 = w56 + kc;
      const int8_t* w58 = w57 + kc;
      const int8_t* w59 = w58 + kc;
      const int8_t* w60 = w59 + kc;
      const int8_t* w61 = w60 + kc;
      const int8_t* w62 = w61 + kc;
      const int8_t* w63 = w62 + kc;

      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const __m256i vb0 = _mm256_loadu_si256((const __m256i*) (b + 0));
        const __m256i vb8 = _mm256_loadu_si256((const __m256i*) (b + 8));
        const __m256i vb16 = _mm256_loadu_si256((const __m256i*) (b + 16));
        const __m256i vb24 = _mm256_loadu_si256((const __m256i*) (b + 24));
        const __m256i vb32 = _mm256_loadu_si256((const __m256i*) (b + 32));
        const __m256i vb40 = _mm256_loadu_si256((const __m256i*) (b + 40));
        const __m256i vb48 = _mm256_loadu_si256((const __m256i*) (b + 48));
        const __m256i vb56 = _mm256_loadu_si256((const __m256i*) (b + 56));
        _mm256_storeu_si256((__m256i*) (out + 0), vb0);
        _mm256_storeu_si256((__m256i*) (out + 32), vb8);
        _mm256_storeu_si256((__m256i*) (out + 64), vb16);
        _mm256_storeu_si256((__m256i*) (out + 96), vb24);
        _mm256_storeu_si256((__m256i*) (out + 128), vb32);
        _mm256_storeu_si256((__m256i*) (out + 160), vb40);
        _mm256_storeu_si256((__m256i*) (out + 192), vb48);
        _mm256_storeu_si256((__m256i*) (out + 224), vb56);
        b += 64;
      } else {
        _mm256_storeu_si256((__m256i*) (out + 0), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 32), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 64), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 96), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 128), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 160), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 192), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 224), _mm256_setzero_si256());
      }
      out += 64 * sizeof(int32_t);

      xnn_prefetch_to_l1((const int8_t*) w0 + 0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w0 + 128);
      xnn_prefetch_to_l1((const int8_t*) w0 + 192);
      xnn_prefetch_to_l1((const int8_t*) w0 + 256);
      xnn_prefetch_to_l1((const int8_t*) w0 + 320);
      xnn_prefetch_to_l1((const int8_t*) w0 + 384);
      xnn_prefetch_to_l1((const int8_t*) w1 + 0);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1 + 128);
      xnn_prefetch_to_l1((const int8_t*) w1 + 192);
      xnn_prefetch_to_l1((const int8_t*) w1 + 256);
      xnn_prefetch_to_l1((const int8_t*) w1 + 320);
      xnn_prefetch_to_l1((const int8_t*) w1 + 384);
      xnn_prefetch_to_l1((const int8_t*) w2 + 0);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2 + 128);
      xnn_prefetch_to_l1((const int8_t*) w2 + 192);
      xnn_prefetch_to_l1((const int8_t*) w2 + 256);
      xnn_prefetch_to_l1((const int8_t*) w2 + 320);
      xnn_prefetch_to_l1((const int8_t*) w2 + 384);
      xnn_prefetch_to_l1((const int8_t*) w3 + 0);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3 + 128);
      xnn_prefetch_to_l1((const int8_t*) w3 + 192);
      xnn_prefetch_to_l1((const int8_t*) w3 + 256);
      xnn_prefetch_to_l1((const int8_t*) w3 + 320);
      xnn_prefetch_to_l1((const int8_t*) w3 + 384);
      xnn_prefetch_to_l1((const int8_t*) w4 + 0);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4 + 128);
      xnn_prefetch_to_l1((const int8_t*) w4 + 192);
      xnn_prefetch_to_l1((const int8_t*) w4 + 256);
      xnn_prefetch_to_l1((const int8_t*) w4 + 320);
      xnn_prefetch_to_l1((const int8_t*) w4 + 384);
      xnn_prefetch_to_l1((const int8_t*) w5 + 0);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5 + 128);
      xnn_prefetch_to_l1((const int8_t*) w5 + 192);
      xnn_prefetch_to_l1((const int8_t*) w5 + 256);
      xnn_prefetch_to_l1((const int8_t*) w5 + 320);
      xnn_prefetch_to_l1((const int8_t*) w5 + 384);
      xnn_prefetch_to_l1((const int8_t*) w6 + 0);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6 + 128);
      xnn_prefetch_to_l1((const int8_t*) w6 + 192);
      xnn_prefetch_to_l1((const int8_t*) w6 + 256);
      xnn_prefetch_to_l1((const int8_t*) w6 + 320);
      xnn_prefetch_to_l1((const int8_t*) w6 + 384);
      xnn_prefetch_to_l1((const int8_t*) w7 + 0);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7 + 128);
      xnn_prefetch_to_l1((const int8_t*) w7 + 192);
      xnn_prefetch_to_l1((const int8_t*) w7 + 256);
      xnn_prefetch_to_l1((const int8_t*) w7 + 320);
      xnn_prefetch_to_l1((const int8_t*) w7 + 384);
      xnn_prefetch_to_l1((const int8_t*) w8 + 0);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8 + 128);
      xnn_prefetch_to_l1((const int8_t*) w8 + 192);
      xnn_prefetch_to_l1((const int8_t*) w8 + 256);
      xnn_prefetch_to_l1((const int8_t*) w8 + 320);
      xnn_prefetch_to_l1((const int8_t*) w8 + 384);
      xnn_prefetch_to_l1((const int8_t*) w9 + 0);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9 + 128);
      xnn_prefetch_to_l1((const int8_t*) w9 + 192);
      xnn_prefetch_to_l1((const int8_t*) w9 + 256);
      xnn_prefetch_to_l1((const int8_t*) w9 + 320);
      xnn_prefetch_to_l1((const int8_t*) w9 + 384);
      xnn_prefetch_to_l1((const int8_t*) w10 + 0);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10 + 128);
      xnn_prefetch_to_l1((const int8_t*) w10 + 192);
      xnn_prefetch_to_l1((const int8_t*) w10 + 256);
      xnn_prefetch_to_l1((const int8_t*) w10 + 320);
      xnn_prefetch_to_l1((const int8_t*) w10 + 384);
      xnn_prefetch_to_l1((const int8_t*) w11 + 0);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11 + 128);
      xnn_prefetch_to_l1((const int8_t*) w11 + 192);
      xnn_prefetch_to_l1((const int8_t*) w11 + 256);
      xnn_prefetch_to_l1((const int8_t*) w11 + 320);
      xnn_prefetch_to_l1((const int8_t*) w11 + 384);
      xnn_prefetch_to_l1((const int8_t*) w12 + 0);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12 + 128);
      xnn_prefetch_to_l1((const int8_t*) w12 + 192);
      xnn_prefetch_to_l1((const int8_t*) w12 + 256);
      xnn_prefetch_to_l1((const int8_t*) w12 + 320);
      xnn_prefetch_to_l1((const int8_t*) w12 + 384);
      xnn_prefetch_to_l1((const int8_t*) w13 + 0);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13 + 128);
      xnn_prefetch_to_l1((const int8_t*) w13 + 192);
      xnn_prefetch_to_l1((const int8_t*) w13 + 256);
      xnn_prefetch_to_l1((const int8_t*) w13 + 320);
      xnn_prefetch_to_l1((const int8_t*) w13 + 384);
      xnn_prefetch_to_l1((const int8_t*) w14 + 0);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14 + 128);
      xnn_prefetch_to_l1((const int8_t*) w14 + 192);
      xnn_prefetch_to_l1((const int8_t*) w14 + 256);
      xnn_prefetch_to_l1((const int8_t*) w14 + 320);
      xnn_prefetch_to_l1((const int8_t*) w14 + 384);
      xnn_prefetch_to_l1((const int8_t*) w15 + 0);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15 + 128);
      xnn_prefetch_to_l1((const int8_t*) w15 + 192);
      xnn_prefetch_to_l1((const int8_t*) w15 + 256);
      xnn_prefetch_to_l1((const int8_t*) w15 + 320);
      xnn_prefetch_to_l1((const int8_t*) w15 + 384);
      xnn_prefetch_to_l1((const int8_t*) w16 + 0);
      xnn_prefetch_to_l1((const int8_t*) w16 + 64);
      xnn_prefetch_to_l1((const int8_t*) w16 + 128);
      xnn_prefetch_to_l1((const int8_t*) w16 + 192);
      xnn_prefetch_to_l1((const int8_t*) w16 + 256);
      xnn_prefetch_to_l1((const int8_t*) w16 + 320);
      xnn_prefetch_to_l1((const int8_t*) w16 + 384);
      xnn_prefetch_to_l1((const int8_t*) w17 + 0);
      xnn_prefetch_to_l1((const int8_t*) w17 + 64);
      xnn_prefetch_to_l1((const int8_t*) w17 + 128);
      xnn_prefetch_to_l1((const int8_t*) w17 + 192);
      xnn_prefetch_to_l1((const int8_t*) w17 + 256);
      xnn_prefetch_to_l1((const int8_t*) w17 + 320);
      xnn_prefetch_to_l1((const int8_t*) w17 + 384);
      xnn_prefetch_to_l1((const int8_t*) w18 + 0);
      xnn_prefetch_to_l1((const int8_t*) w18 + 64);
      xnn_prefetch_to_l1((const int8_t*) w18 + 128);
      xnn_prefetch_to_l1((const int8_t*) w18 + 192);
      xnn_prefetch_to_l1((const int8_t*) w18 + 256);
      xnn_prefetch_to_l1((const int8_t*) w18 + 320);
      xnn_prefetch_to_l1((const int8_t*) w18 + 384);
      xnn_prefetch_to_l1((const int8_t*) w19 + 0);
      xnn_prefetch_to_l1((const int8_t*) w19 + 64);
      xnn_prefetch_to_l1((const int8_t*) w19 + 128);
      xnn_prefetch_to_l1((const int8_t*) w19 + 192);
      xnn_prefetch_to_l1((const int8_t*) w19 + 256);
      xnn_prefetch_to_l1((const int8_t*) w19 + 320);
      xnn_prefetch_to_l1((const int8_t*) w19 + 384);
      xnn_prefetch_to_l1((const int8_t*) w20 + 0);
      xnn_prefetch_to_l1((const int8_t*) w20 + 64);
      xnn_prefetch_to_l1((const int8_t*) w20 + 128);
      xnn_prefetch_to_l1((const int8_t*) w20 + 192);
      xnn_prefetch_to_l1((const int8_t*) w20 + 256);
      xnn_prefetch_to_l1((const int8_t*) w20 + 320);
      xnn_prefetch_to_l1((const int8_t*) w20 + 384);
      xnn_prefetch_to_l1((const int8_t*) w21 + 0);
      xnn_prefetch_to_l1((const int8_t*) w21 + 64);
      xnn_prefetch_to_l1((const int8_t*) w21 + 128);
      xnn_prefetch_to_l1((const int8_t*) w21 + 192);
      xnn_prefetch_to_l1((const int8_t*) w21 + 256);
      xnn_prefetch_to_l1((const int8_t*) w21 + 320);
      xnn_prefetch_to_l1((const int8_t*) w21 + 384);
      xnn_prefetch_to_l1((const int8_t*) w22 + 0);
      xnn_prefetch_to_l1((const int8_t*) w22 + 64);
      xnn_prefetch_to_l1((const int8_t*) w22 + 128);
      xnn_prefetch_to_l1((const int8_t*) w22 + 192);
      xnn_prefetch_to_l1((const int8_t*) w22 + 256);
      xnn_prefetch_to_l1((const int8_t*) w22 + 320);
      xnn_prefetch_to_l1((const int8_t*) w22 + 384);
      xnn_prefetch_to_l1((const int8_t*) w23 + 0);
      xnn_prefetch_to_l1((const int8_t*) w23 + 64);
      xnn_prefetch_to_l1((const int8_t*) w23 + 128);
      xnn_prefetch_to_l1((const int8_t*) w23 + 192);
      xnn_prefetch_to_l1((const int8_t*) w23 + 256);
      xnn_prefetch_to_l1((const int8_t*) w23 + 320);
      xnn_prefetch_to_l1((const int8_t*) w23 + 384);
      xnn_prefetch_to_l1((const int8_t*) w24 + 0);
      xnn_prefetch_to_l1((const int8_t*) w24 + 64);
      xnn_prefetch_to_l1((const int8_t*) w24 + 128);
      xnn_prefetch_to_l1((const int8_t*) w24 + 192);
      xnn_prefetch_to_l1((const int8_t*) w24 + 256);
      xnn_prefetch_to_l1((const int8_t*) w24 + 320);
      xnn_prefetch_to_l1((const int8_t*) w24 + 384);
      xnn_prefetch_to_l1((const int8_t*) w25 + 0);
      xnn_prefetch_to_l1((const int8_t*) w25 + 64);
      xnn_prefetch_to_l1((const int8_t*) w25 + 128);
      xnn_prefetch_to_l1((const int8_t*) w25 + 192);
      xnn_prefetch_to_l1((const int8_t*) w25 + 256);
      xnn_prefetch_to_l1((const int8_t*) w25 + 320);
      xnn_prefetch_to_l1((const int8_t*) w25 + 384);
      xnn_prefetch_to_l1((const int8_t*) w26 + 0);
      xnn_prefetch_to_l1((const int8_t*) w26 + 64);
      xnn_prefetch_to_l1((const int8_t*) w26 + 128);
      xnn_prefetch_to_l1((const int8_t*) w26 + 192);
      xnn_prefetch_to_l1((const int8_t*) w26 + 256);
      xnn_prefetch_to_l1((const int8_t*) w26 + 320);
      xnn_prefetch_to_l1((const int8_t*) w26 + 384);
      xnn_prefetch_to_l1((const int8_t*) w27 + 0);
      xnn_prefetch_to_l1((const int8_t*) w27 + 64);
      xnn_prefetch_to_l1((const int8_t*) w27 + 128);
      xnn_prefetch_to_l1((const int8_t*) w27 + 192);
      xnn_prefetch_to_l1((const int8_t*) w27 + 256);
      xnn_prefetch_to_l1((const int8_t*) w27 + 320);
      xnn_prefetch_to_l1((const int8_t*) w27 + 384);
      xnn_prefetch_to_l1((const int8_t*) w28 + 0);
      xnn_prefetch_to_l1((const int8_t*) w28 + 64);
      xnn_prefetch_to_l1((const int8_t*) w28 + 128);
      xnn_prefetch_to_l1((const int8_t*) w28 + 192);
      xnn_prefetch_to_l1((const int8_t*) w28 + 256);
      xnn_prefetch_to_l1((const int8_t*) w28 + 320);
      xnn_prefetch_to_l1((const int8_t*) w28 + 384);
      xnn_prefetch_to_l1((const int8_t*) w29 + 0);
      xnn_prefetch_to_l1((const int8_t*) w29 + 64);
      xnn_prefetch_to_l1((const int8_t*) w29 + 128);
      xnn_prefetch_to_l1((const int8_t*) w29 + 192);
      xnn_prefetch_to_l1((const int8_t*) w29 + 256);
      xnn_prefetch_to_l1((const int8_t*) w29 + 320);
      xnn_prefetch_to_l1((const int8_t*) w29 + 384);
      xnn_prefetch_to_l1((const int8_t*) w30 + 0);
      xnn_prefetch_to_l1((const int8_t*) w30 + 64);
      xnn_prefetch_to_l1((const int8_t*) w30 + 128);
      xnn_prefetch_to_l1((const int8_t*) w30 + 192);
      xnn_prefetch_to_l1((const int8_t*) w30 + 256);
      xnn_prefetch_to_l1((const int8_t*) w30 + 320);
      xnn_prefetch_to_l1((const int8_t*) w30 + 384);
      xnn_prefetch_to_l1((const int8_t*) w31 + 0);
      xnn_prefetch_to_l1((const int8_t*) w31 + 64);
      xnn_prefetch_to_l1((const int8_t*) w31 + 128);
      xnn_prefetch_to_l1((const int8_t*) w31 + 192);
      xnn_prefetch_to_l1((const int8_t*) w31 + 256);
      xnn_prefetch_to_l1((const int8_t*) w31 + 320);
      xnn_prefetch_to_l1((const int8_t*) w31 + 384);
      xnn_prefetch_to_l1((const int8_t*) w32 + 0);
      xnn_prefetch_to_l1((const int8_t*) w32 + 64);
      xnn_prefetch_to_l1((const int8_t*) w32 + 128);
      xnn_prefetch_to_l1((const int8_t*) w32 + 192);
      xnn_prefetch_to_l1((const int8_t*) w32 + 256);
      xnn_prefetch_to_l1((const int8_t*) w32 + 320);
      xnn_prefetch_to_l1((const int8_t*) w32 + 384);
      xnn_prefetch_to_l1((const int8_t*) w33 + 0);
      xnn_prefetch_to_l1((const int8_t*) w33 + 64);
      xnn_prefetch_to_l1((const int8_t*) w33 + 128);
      xnn_prefetch_to_l1((const int8_t*) w33 + 192);
      xnn_prefetch_to_l1((const int8_t*) w33 + 256);
      xnn_prefetch_to_l1((const int8_t*) w33 + 320);
      xnn_prefetch_to_l1((const int8_t*) w33 + 384);
      xnn_prefetch_to_l1((const int8_t*) w34 + 0);
      xnn_prefetch_to_l1((const int8_t*) w34 + 64);
      xnn_prefetch_to_l1((const int8_t*) w34 + 128);
      xnn_prefetch_to_l1((const int8_t*) w34 + 192);
      xnn_prefetch_to_l1((const int8_t*) w34 + 256);
      xnn_prefetch_to_l1((const int8_t*) w34 + 320);
      xnn_prefetch_to_l1((const int8_t*) w34 + 384);
      xnn_prefetch_to_l1((const int8_t*) w35 + 0);
      xnn_prefetch_to_l1((const int8_t*) w35 + 64);
      xnn_prefetch_to_l1((const int8_t*) w35 + 128);
      xnn_prefetch_to_l1((const int8_t*) w35 + 192);
      xnn_prefetch_to_l1((const int8_t*) w35 + 256);
      xnn_prefetch_to_l1((const int8_t*) w35 + 320);
      xnn_prefetch_to_l1((const int8_t*) w35 + 384);
      xnn_prefetch_to_l1((const int8_t*) w36 + 0);
      xnn_prefetch_to_l1((const int8_t*) w36 + 64);
      xnn_prefetch_to_l1((const int8_t*) w36 + 128);
      xnn_prefetch_to_l1((const int8_t*) w36 + 192);
      xnn_prefetch_to_l1((const int8_t*) w36 + 256);
      xnn_prefetch_to_l1((const int8_t*) w36 + 320);
      xnn_prefetch_to_l1((const int8_t*) w36 + 384);
      xnn_prefetch_to_l1((const int8_t*) w37 + 0);
      xnn_prefetch_to_l1((const int8_t*) w37 + 64);
      xnn_prefetch_to_l1((const int8_t*) w37 + 128);
      xnn_prefetch_to_l1((const int8_t*) w37 + 192);
      xnn_prefetch_to_l1((const int8_t*) w37 + 256);
      xnn_prefetch_to_l1((const int8_t*) w37 + 320);
      xnn_prefetch_to_l1((const int8_t*) w37 + 384);
      xnn_prefetch_to_l1((const int8_t*) w38 + 0);
      xnn_prefetch_to_l1((const int8_t*) w38 + 64);
      xnn_prefetch_to_l1((const int8_t*) w38 + 128);
      xnn_prefetch_to_l1((const int8_t*) w38 + 192);
      xnn_prefetch_to_l1((const int8_t*) w38 + 256);
      xnn_prefetch_to_l1((const int8_t*) w38 + 320);
      xnn_prefetch_to_l1((const int8_t*) w38 + 384);
      xnn_prefetch_to_l1((const int8_t*) w39 + 0);
      xnn_prefetch_to_l1((const int8_t*) w39 + 64);
      xnn_prefetch_to_l1((const int8_t*) w39 + 128);
      xnn_prefetch_to_l1((const int8_t*) w39 + 192);
      xnn_prefetch_to_l1((const int8_t*) w39 + 256);
      xnn_prefetch_to_l1((const int8_t*) w39 + 320);
      xnn_prefetch_to_l1((const int8_t*) w39 + 384);
      xnn_prefetch_to_l1((const int8_t*) w40 + 0);
      xnn_prefetch_to_l1((const int8_t*) w40 + 64);
      xnn_prefetch_to_l1((const int8_t*) w40 + 128);
      xnn_prefetch_to_l1((const int8_t*) w40 + 192);
      xnn_prefetch_to_l1((const int8_t*) w40 + 256);
      xnn_prefetch_to_l1((const int8_t*) w40 + 320);
      xnn_prefetch_to_l1((const int8_t*) w40 + 384);
      xnn_prefetch_to_l1((const int8_t*) w41 + 0);
      xnn_prefetch_to_l1((const int8_t*) w41 + 64);
      xnn_prefetch_to_l1((const int8_t*) w41 + 128);
      xnn_prefetch_to_l1((const int8_t*) w41 + 192);
      xnn_prefetch_to_l1((const int8_t*) w41 + 256);
      xnn_prefetch_to_l1((const int8_t*) w41 + 320);
      xnn_prefetch_to_l1((const int8_t*) w41 + 384);
      xnn_prefetch_to_l1((const int8_t*) w42 + 0);
      xnn_prefetch_to_l1((const int8_t*) w42 + 64);
      xnn_prefetch_to_l1((const int8_t*) w42 + 128);
      xnn_prefetch_to_l1((const int8_t*) w42 + 192);
      xnn_prefetch_to_l1((const int8_t*) w42 + 256);
      xnn_prefetch_to_l1((const int8_t*) w42 + 320);
      xnn_prefetch_to_l1((const int8_t*) w42 + 384);
      xnn_prefetch_to_l1((const int8_t*) w43 + 0);
      xnn_prefetch_to_l1((const int8_t*) w43 + 64);
      xnn_prefetch_to_l1((const int8_t*) w43 + 128);
      xnn_prefetch_to_l1((const int8_t*) w43 + 192);
      xnn_prefetch_to_l1((const int8_t*) w43 + 256);
      xnn_prefetch_to_l1((const int8_t*) w43 + 320);
      xnn_prefetch_to_l1((const int8_t*) w43 + 384);
      xnn_prefetch_to_l1((const int8_t*) w44 + 0);
      xnn_prefetch_to_l1((const int8_t*) w44 + 64);
      xnn_prefetch_to_l1((const int8_t*) w44 + 128);
      xnn_prefetch_to_l1((const int8_t*) w44 + 192);
      xnn_prefetch_to_l1((const int8_t*) w44 + 256);
      xnn_prefetch_to_l1((const int8_t*) w44 + 320);
      xnn_prefetch_to_l1((const int8_t*) w44 + 384);
      xnn_prefetch_to_l1((const int8_t*) w45 + 0);
      xnn_prefetch_to_l1((const int8_t*) w45 + 64);
      xnn_prefetch_to_l1((const int8_t*) w45 + 128);
      xnn_prefetch_to_l1((const int8_t*) w45 + 192);
      xnn_prefetch_to_l1((const int8_t*) w45 + 256);
      xnn_prefetch_to_l1((const int8_t*) w45 + 320);
      xnn_prefetch_to_l1((const int8_t*) w45 + 384);
      xnn_prefetch_to_l1((const int8_t*) w46 + 0);
      xnn_prefetch_to_l1((const int8_t*) w46 + 64);
      xnn_prefetch_to_l1((const int8_t*) w46 + 128);
      xnn_prefetch_to_l1((const int8_t*) w46 + 192);
      xnn_prefetch_to_l1((const int8_t*) w46 + 256);
      xnn_prefetch_to_l1((const int8_t*) w46 + 320);
      xnn_prefetch_to_l1((const int8_t*) w46 + 384);
      xnn_prefetch_to_l1((const int8_t*) w47 + 0);
      xnn_prefetch_to_l1((const int8_t*) w47 + 64);
      xnn_prefetch_to_l1((const int8_t*) w47 + 128);
      xnn_prefetch_to_l1((const int8_t*) w47 + 192);
      xnn_prefetch_to_l1((const int8_t*) w47 + 256);
      xnn_prefetch_to_l1((const int8_t*) w47 + 320);
      xnn_prefetch_to_l1((const int8_t*) w47 + 384);
      xnn_prefetch_to_l1((const int8_t*) w48 + 0);
      xnn_prefetch_to_l1((const int8_t*) w48 + 64);
      xnn_prefetch_to_l1((const int8_t*) w48 + 128);
      xnn_prefetch_to_l1((const int8_t*) w48 + 192);
      xnn_prefetch_to_l1((const int8_t*) w48 + 256);
      xnn_prefetch_to_l1((const int8_t*) w48 + 320);
      xnn_prefetch_to_l1((const int8_t*) w48 + 384);
      xnn_prefetch_to_l1((const int8_t*) w49 + 0);
      xnn_prefetch_to_l1((const int8_t*) w49 + 64);
      xnn_prefetch_to_l1((const int8_t*) w49 + 128);
      xnn_prefetch_to_l1((const int8_t*) w49 + 192);
      xnn_prefetch_to_l1((const int8_t*) w49 + 256);
      xnn_prefetch_to_l1((const int8_t*) w49 + 320);
      xnn_prefetch_to_l1((const int8_t*) w49 + 384);
      xnn_prefetch_to_l1((const int8_t*) w50 + 0);
      xnn_prefetch_to_l1((const int8_t*) w50 + 64);
      xnn_prefetch_to_l1((const int8_t*) w50 + 128);
      xnn_prefetch_to_l1((const int8_t*) w50 + 192);
      xnn_prefetch_to_l1((const int8_t*) w50 + 256);
      xnn_prefetch_to_l1((const int8_t*) w50 + 320);
      xnn_prefetch_to_l1((const int8_t*) w50 + 384);
      xnn_prefetch_to_l1((const int8_t*) w51 + 0);
      xnn_prefetch_to_l1((const int8_t*) w51 + 64);
      xnn_prefetch_to_l1((const int8_t*) w51 + 128);
      xnn_prefetch_to_l1((const int8_t*) w51 + 192);
      xnn_prefetch_to_l1((const int8_t*) w51 + 256);
      xnn_prefetch_to_l1((const int8_t*) w51 + 320);
      xnn_prefetch_to_l1((const int8_t*) w51 + 384);
      xnn_prefetch_to_l1((const int8_t*) w52 + 0);
      xnn_prefetch_to_l1((const int8_t*) w52 + 64);
      xnn_prefetch_to_l1((const int8_t*) w52 + 128);
      xnn_prefetch_to_l1((const int8_t*) w52 + 192);
      xnn_prefetch_to_l1((const int8_t*) w52 + 256);
      xnn_prefetch_to_l1((const int8_t*) w52 + 320);
      xnn_prefetch_to_l1((const int8_t*) w52 + 384);
      xnn_prefetch_to_l1((const int8_t*) w53 + 0);
      xnn_prefetch_to_l1((const int8_t*) w53 + 64);
      xnn_prefetch_to_l1((const int8_t*) w53 + 128);
      xnn_prefetch_to_l1((const int8_t*) w53 + 192);
      xnn_prefetch_to_l1((const int8_t*) w53 + 256);
      xnn_prefetch_to_l1((const int8_t*) w53 + 320);
      xnn_prefetch_to_l1((const int8_t*) w53 + 384);
      xnn_prefetch_to_l1((const int8_t*) w54 + 0);
      xnn_prefetch_to_l1((const int8_t*) w54 + 64);
      xnn_prefetch_to_l1((const int8_t*) w54 + 128);
      xnn_prefetch_to_l1((const int8_t*) w54 + 192);
      xnn_prefetch_to_l1((const int8_t*) w54 + 256);
      xnn_prefetch_to_l1((const int8_t*) w54 + 320);
      xnn_prefetch_to_l1((const int8_t*) w54 + 384);
      xnn_prefetch_to_l1((const int8_t*) w55 + 0);
      xnn_prefetch_to_l1((const int8_t*) w55 + 64);
      xnn_prefetch_to_l1((const int8_t*) w55 + 128);
      xnn_prefetch_to_l1((const int8_t*) w55 + 192);
      xnn_prefetch_to_l1((const int8_t*) w55 + 256);
      xnn_prefetch_to_l1((const int8_t*) w55 + 320);
      xnn_prefetch_to_l1((const int8_t*) w55 + 384);
      xnn_prefetch_to_l1((const int8_t*) w56 + 0);
      xnn_prefetch_to_l1((const int8_t*) w56 + 64);
      xnn_prefetch_to_l1((const int8_t*) w56 + 128);
      xnn_prefetch_to_l1((const int8_t*) w56 + 192);
      xnn_prefetch_to_l1((const int8_t*) w56 + 256);
      xnn_prefetch_to_l1((const int8_t*) w56 + 320);
      xnn_prefetch_to_l1((const int8_t*) w56 + 384);
      xnn_prefetch_to_l1((const int8_t*) w57 + 0);
      xnn_prefetch_to_l1((const int8_t*) w57 + 64);
      xnn_prefetch_to_l1((const int8_t*) w57 + 128);
      xnn_prefetch_to_l1((const int8_t*) w57 + 192);
      xnn_prefetch_to_l1((const int8_t*) w57 + 256);
      xnn_prefetch_to_l1((const int8_t*) w57 + 320);
      xnn_prefetch_to_l1((const int8_t*) w57 + 384);
      xnn_prefetch_to_l1((const int8_t*) w58 + 0);
      xnn_prefetch_to_l1((const int8_t*) w58 + 64);
      xnn_prefetch_to_l1((const int8_t*) w58 + 128);
      xnn_prefetch_to_l1((const int8_t*) w58 + 192);
      xnn_prefetch_to_l1((const int8_t*) w58 + 256);
      xnn_prefetch_to_l1((const int8_t*) w58 + 320);
      xnn_prefetch_to_l1((const int8_t*) w58 + 384);
      xnn_prefetch_to_l1((const int8_t*) w59 + 0);
      xnn_prefetch_to_l1((const int8_t*) w59 + 64);
      xnn_prefetch_to_l1((const int8_t*) w59 + 128);
      xnn_prefetch_to_l1((const int8_t*) w59 + 192);
      xnn_prefetch_to_l1((const int8_t*) w59 + 256);
      xnn_prefetch_to_l1((const int8_t*) w59 + 320);
      xnn_prefetch_to_l1((const int8_t*) w59 + 384);
      xnn_prefetch_to_l1((const int8_t*) w60 + 0);
      xnn_prefetch_to_l1((const int8_t*) w60 + 64);
      xnn_prefetch_to_l1((const int8_t*) w60 + 128);
      xnn_prefetch_to_l1((const int8_t*) w60 + 192);
      xnn_prefetch_to_l1((const int8_t*) w60 + 256);
      xnn_prefetch_to_l1((const int8_t*) w60 + 320);
      xnn_prefetch_to_l1((const int8_t*) w60 + 384);
      xnn_prefetch_to_l1((const int8_t*) w61 + 0);
      xnn_prefetch_to_l1((const int8_t*) w61 + 64);
      xnn_prefetch_to_l1((const int8_t*) w61 + 128);
      xnn_prefetch_to_l1((const int8_t*) w61 + 192);
      xnn_prefetch_to_l1((const int8_t*) w61 + 256);
      xnn_prefetch_to_l1((const int8_t*) w61 + 320);
      xnn_prefetch_to_l1((const int8_t*) w61 + 384);
      xnn_prefetch_to_l1((const int8_t*) w62 + 0);
      xnn_prefetch_to_l1((const int8_t*) w62 + 64);
      xnn_prefetch_to_l1((const int8_t*) w62 + 128);
      xnn_prefetch_to_l1((const int8_t*) w62 + 192);
      xnn_prefetch_to_l1((const int8_t*) w62 + 256);
      xnn_prefetch_to_l1((const int8_t*) w62 + 320);
      xnn_prefetch_to_l1((const int8_t*) w62 + 384);
      xnn_prefetch_to_l1((const int8_t*) w63 + 0);
      xnn_prefetch_to_l1((const int8_t*) w63 + 64);
      xnn_prefetch_to_l1((const int8_t*) w63 + 128);
      xnn_prefetch_to_l1((const int8_t*) w63 + 192);
      xnn_prefetch_to_l1((const int8_t*) w63 + 256);
      xnn_prefetch_to_l1((const int8_t*) w63 + 320);
      xnn_prefetch_to_l1((const int8_t*) w63 + 384);

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc8 = _mm256_setzero_si256();
      __m256i vacc16 = _mm256_setzero_si256();
      __m256i vacc24 = _mm256_setzero_si256();
      __m256i vacc32 = _mm256_setzero_si256();
      __m256i vacc40 = _mm256_setzero_si256();
      __m256i vacc48 = _mm256_setzero_si256();
      __m256i vacc56 = _mm256_setzero_si256();

      size_t k = kc;

      // KC main loop multiple of 64x4
      for (; k >= 4; k -= 4) {
        __m256i v0 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w1)), 0x02);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w2)), 0x04);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w3)), 0x08);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w4)), 0x10);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w5)), 0x20);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w6)), 0x40);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w7)), 0x80);
        __m256i v8 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w8));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w9)), 0x02);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w10)), 0x04);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w11)), 0x08);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w12)), 0x10);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w13)), 0x20);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w14)), 0x40);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w15)), 0x80);
        __m256i v16 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w16));
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w17)), 0x02);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w18)), 0x04);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w19)), 0x08);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w20)), 0x10);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w21)), 0x20);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w22)), 0x40);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w23)), 0x80);
        __m256i v24 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w24));
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w25)), 0x02);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w26)), 0x04);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w27)), 0x08);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w28)), 0x10);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w29)), 0x20);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w30)), 0x40);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w31)), 0x80);
        __m256i v32 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w32));
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w33)), 0x02);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w34)), 0x04);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w35)), 0x08);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w36)), 0x10);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w37)), 0x20);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w38)), 0x40);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w39)), 0x80);
        __m256i v40 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w40));
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w41)), 0x02);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w42)), 0x04);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w43)), 0x08);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w44)), 0x10);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w45)), 0x20);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w46)), 0x40);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w47)), 0x80);
        __m256i v48 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w48));
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w49)), 0x02);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w50)), 0x04);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w51)), 0x08);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w52)), 0x10);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w53)), 0x20);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w54)), 0x40);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w55)), 0x80);
        __m256i v56 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w56));
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w57)), 0x02);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w58)), 0x04);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w59)), 0x08);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w60)), 0x10);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w61)), 0x20);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w62)), 0x40);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w63)), 0x80);
        xnn_prefetch_to_l1((const int8_t*) w0 + 448);
        xnn_prefetch_to_l1((const int8_t*) w1 + 448);
        xnn_prefetch_to_l1((const int8_t*) w2 + 448);
        xnn_prefetch_to_l1((const int8_t*) w3 + 448);
        xnn_prefetch_to_l1((const int8_t*) w4 + 448);
        xnn_prefetch_to_l1((const int8_t*) w5 + 448);
        xnn_prefetch_to_l1((const int8_t*) w6 + 448);
        xnn_prefetch_to_l1((const int8_t*) w7 + 448);
        xnn_prefetch_to_l1((const int8_t*) w8 + 448);
        xnn_prefetch_to_l1((const int8_t*) w9 + 448);
        xnn_prefetch_to_l1((const int8_t*) w10 + 448);
        xnn_prefetch_to_l1((const int8_t*) w11 + 448);
        xnn_prefetch_to_l1((const int8_t*) w12 + 448);
        xnn_prefetch_to_l1((const int8_t*) w13 + 448);
        xnn_prefetch_to_l1((const int8_t*) w14 + 448);
        xnn_prefetch_to_l1((const int8_t*) w15 + 448);
        xnn_prefetch_to_l1((const int8_t*) w16 + 448);
        xnn_prefetch_to_l1((const int8_t*) w17 + 448);
        xnn_prefetch_to_l1((const int8_t*) w18 + 448);
        xnn_prefetch_to_l1((const int8_t*) w19 + 448);
        xnn_prefetch_to_l1((const int8_t*) w20 + 448);
        xnn_prefetch_to_l1((const int8_t*) w21 + 448);
        xnn_prefetch_to_l1((const int8_t*) w22 + 448);
        xnn_prefetch_to_l1((const int8_t*) w23 + 448);
        xnn_prefetch_to_l1((const int8_t*) w24 + 448);
        xnn_prefetch_to_l1((const int8_t*) w25 + 448);
        xnn_prefetch_to_l1((const int8_t*) w26 + 448);
        xnn_prefetch_to_l1((const int8_t*) w27 + 448);
        xnn_prefetch_to_l1((const int8_t*) w28 + 448);
        xnn_prefetch_to_l1((const int8_t*) w29 + 448);
        xnn_prefetch_to_l1((const int8_t*) w30 + 448);
        xnn_prefetch_to_l1((const int8_t*) w31 + 448);
        xnn_prefetch_to_l1((const int8_t*) w32 + 448);
        xnn_prefetch_to_l1((const int8_t*) w33 + 448);
        xnn_prefetch_to_l1((const int8_t*) w34 + 448);
        xnn_prefetch_to_l1((const int8_t*) w35 + 448);
        xnn_prefetch_to_l1((const int8_t*) w36 + 448);
        xnn_prefetch_to_l1((const int8_t*) w37 + 448);
        xnn_prefetch_to_l1((const int8_t*) w38 + 448);
        xnn_prefetch_to_l1((const int8_t*) w39 + 448);
        xnn_prefetch_to_l1((const int8_t*) w40 + 448);
        xnn_prefetch_to_l1((const int8_t*) w41 + 448);
        xnn_prefetch_to_l1((const int8_t*) w42 + 448);
        xnn_prefetch_to_l1((const int8_t*) w43 + 448);
        xnn_prefetch_to_l1((const int8_t*) w44 + 448);
        xnn_prefetch_to_l1((const int8_t*) w45 + 448);
        xnn_prefetch_to_l1((const int8_t*) w46 + 448);
        xnn_prefetch_to_l1((const int8_t*) w47 + 448);
        xnn_prefetch_to_l1((const int8_t*) w48 + 448);
        xnn_prefetch_to_l1((const int8_t*) w49 + 448);
        xnn_prefetch_to_l1((const int8_t*) w50 + 448);
        xnn_prefetch_to_l1((const int8_t*) w51 + 448);
        xnn_prefetch_to_l1((const int8_t*) w52 + 448);
        xnn_prefetch_to_l1((const int8_t*) w53 + 448);
        xnn_prefetch_to_l1((const int8_t*) w54 + 448);
        xnn_prefetch_to_l1((const int8_t*) w55 + 448);
        xnn_prefetch_to_l1((const int8_t*) w56 + 448);
        xnn_prefetch_to_l1((const int8_t*) w57 + 448);
        xnn_prefetch_to_l1((const int8_t*) w58 + 448);
        xnn_prefetch_to_l1((const int8_t*) w59 + 448);
        xnn_prefetch_to_l1((const int8_t*) w60 + 448);
        xnn_prefetch_to_l1((const int8_t*) w61 + 448);
        xnn_prefetch_to_l1((const int8_t*) w62 + 448);
        xnn_prefetch_to_l1((const int8_t*) w63 + 448);

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc8 = _mm256_dpbusd_epi32(vacc8, vone, v8);
        vacc16 = _mm256_dpbusd_epi32(vacc16, vone, v16);
        vacc24 = _mm256_dpbusd_epi32(vacc24, vone, v24);
        vacc32 = _mm256_dpbusd_epi32(vacc32, vone, v32);
        vacc40 = _mm256_dpbusd_epi32(vacc40, vone, v40);
        vacc48 = _mm256_dpbusd_epi32(vacc48, vone, v48);
        vacc56 = _mm256_dpbusd_epi32(vacc56, vone, v56);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v8);
        _mm256_storeu_si256((__m256i *)&out[64],  v16);
        _mm256_storeu_si256((__m256i *)&out[96],  v24);
        _mm256_storeu_si256((__m256i *)&out[128],  v32);
        _mm256_storeu_si256((__m256i *)&out[160],  v40);
        _mm256_storeu_si256((__m256i *)&out[192],  v48);
        _mm256_storeu_si256((__m256i *)&out[224],  v56);

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
        w31 += 4;
        w32 += 4;
        w33 += 4;
        w34 += 4;
        w35 += 4;
        w36 += 4;
        w37 += 4;
        w38 += 4;
        w39 += 4;
        w40 += 4;
        w41 += 4;
        w42 += 4;
        w43 += 4;
        w44 += 4;
        w45 += 4;
        w46 += 4;
        w47 += 4;
        w48 += 4;
        w49 += 4;
        w50 += 4;
        w51 += 4;
        w52 += 4;
        w53 += 4;
        w54 += 4;
        w55 += 4;
        w56 += 4;
        w57 += 4;
        w58 += 4;
        w59 += 4;
        w60 += 4;
        w61 += 4;
        w62 += 4;
        w63 += 4;
        out += 256;
      }

      // KC remainder of 1..3
      if (k != 0) {
        assert(k >= 1 && k <= 3);

        __m256i v0 = _mm256_set1_epi32((int32_t) safe_load_u32(w0, k));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w1, k)), 0x02);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w2, k)), 0x04);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w3, k)), 0x08);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w4, k)), 0x10);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w5, k)), 0x20);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w6, k)), 0x40);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w7, k)), 0x80);
        __m256i v8 = _mm256_set1_epi32((int32_t) safe_load_u32(w8, k));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w9, k)), 0x02);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w10, k)), 0x04);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w11, k)), 0x08);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w12, k)), 0x10);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w13, k)), 0x20);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w14, k)), 0x40);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w15, k)), 0x80);
        __m256i v16 = _mm256_set1_epi32((int32_t) safe_load_u32(w16, k));
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w17, k)), 0x02);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w18, k)), 0x04);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w19, k)), 0x08);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w20, k)), 0x10);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w21, k)), 0x20);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w22, k)), 0x40);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w23, k)), 0x80);
        __m256i v24 = _mm256_set1_epi32((int32_t) safe_load_u32(w24, k));
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w25, k)), 0x02);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w26, k)), 0x04);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w27, k)), 0x08);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w28, k)), 0x10);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w29, k)), 0x20);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w30, k)), 0x40);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w31, k)), 0x80);
        __m256i v32 = _mm256_set1_epi32((int32_t) safe_load_u32(w32, k));
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w33, k)), 0x02);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w34, k)), 0x04);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w35, k)), 0x08);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w36, k)), 0x10);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w37, k)), 0x20);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w38, k)), 0x40);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w39, k)), 0x80);
        __m256i v40 = _mm256_set1_epi32((int32_t) safe_load_u32(w40, k));
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w41, k)), 0x02);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w42, k)), 0x04);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w43, k)), 0x08);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w44, k)), 0x10);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w45, k)), 0x20);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w46, k)), 0x40);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w47, k)), 0x80);
        __m256i v48 = _mm256_set1_epi32((int32_t) safe_load_u32(w48, k));
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w49, k)), 0x02);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w50, k)), 0x04);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w51, k)), 0x08);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w52, k)), 0x10);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w53, k)), 0x20);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w54, k)), 0x40);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w55, k)), 0x80);
        __m256i v56 = _mm256_set1_epi32((int32_t) safe_load_u32(w56, k));
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w57, k)), 0x02);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w58, k)), 0x04);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w59, k)), 0x08);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w60, k)), 0x10);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w61, k)), 0x20);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w62, k)), 0x40);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w63, k)), 0x80);

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
        w16 += k;
        w17 += k;
        w18 += k;
        w19 += k;
        w20 += k;
        w21 += k;
        w22 += k;
        w23 += k;
        w24 += k;
        w25 += k;
        w26 += k;
        w27 += k;
        w28 += k;
        w29 += k;
        w30 += k;
        w31 += k;
        w32 += k;
        w33 += k;
        w34 += k;
        w35 += k;
        w36 += k;
        w37 += k;
        w38 += k;
        w39 += k;
        w40 += k;
        w41 += k;
        w42 += k;
        w43 += k;
        w44 += k;
        w45 += k;
        w46 += k;
        w47 += k;
        w48 += k;
        w49 += k;
        w50 += k;
        w51 += k;
        w52 += k;
        w53 += k;
        w54 += k;
        w55 += k;
        w56 += k;
        w57 += k;
        w58 += k;
        w59 += k;
        w60 += k;
        w61 += k;
        w62 += k;
        w63 += k;

        xnn_prefetch_to_l1((const int8_t*) w0 + 448);
        xnn_prefetch_to_l1((const int8_t*) w1 + 448);
        xnn_prefetch_to_l1((const int8_t*) w2 + 448);
        xnn_prefetch_to_l1((const int8_t*) w3 + 448);
        xnn_prefetch_to_l1((const int8_t*) w4 + 448);
        xnn_prefetch_to_l1((const int8_t*) w5 + 448);
        xnn_prefetch_to_l1((const int8_t*) w6 + 448);
        xnn_prefetch_to_l1((const int8_t*) w7 + 448);
        xnn_prefetch_to_l1((const int8_t*) w8 + 448);
        xnn_prefetch_to_l1((const int8_t*) w9 + 448);
        xnn_prefetch_to_l1((const int8_t*) w10 + 448);
        xnn_prefetch_to_l1((const int8_t*) w11 + 448);
        xnn_prefetch_to_l1((const int8_t*) w12 + 448);
        xnn_prefetch_to_l1((const int8_t*) w13 + 448);
        xnn_prefetch_to_l1((const int8_t*) w14 + 448);
        xnn_prefetch_to_l1((const int8_t*) w15 + 448);
        xnn_prefetch_to_l1((const int8_t*) w16 + 448);
        xnn_prefetch_to_l1((const int8_t*) w17 + 448);
        xnn_prefetch_to_l1((const int8_t*) w18 + 448);
        xnn_prefetch_to_l1((const int8_t*) w19 + 448);
        xnn_prefetch_to_l1((const int8_t*) w20 + 448);
        xnn_prefetch_to_l1((const int8_t*) w21 + 448);
        xnn_prefetch_to_l1((const int8_t*) w22 + 448);
        xnn_prefetch_to_l1((const int8_t*) w23 + 448);
        xnn_prefetch_to_l1((const int8_t*) w24 + 448);
        xnn_prefetch_to_l1((const int8_t*) w25 + 448);
        xnn_prefetch_to_l1((const int8_t*) w26 + 448);
        xnn_prefetch_to_l1((const int8_t*) w27 + 448);
        xnn_prefetch_to_l1((const int8_t*) w28 + 448);
        xnn_prefetch_to_l1((const int8_t*) w29 + 448);
        xnn_prefetch_to_l1((const int8_t*) w30 + 448);
        xnn_prefetch_to_l1((const int8_t*) w31 + 448);
        xnn_prefetch_to_l1((const int8_t*) w32 + 448);
        xnn_prefetch_to_l1((const int8_t*) w33 + 448);
        xnn_prefetch_to_l1((const int8_t*) w34 + 448);
        xnn_prefetch_to_l1((const int8_t*) w35 + 448);
        xnn_prefetch_to_l1((const int8_t*) w36 + 448);
        xnn_prefetch_to_l1((const int8_t*) w37 + 448);
        xnn_prefetch_to_l1((const int8_t*) w38 + 448);
        xnn_prefetch_to_l1((const int8_t*) w39 + 448);
        xnn_prefetch_to_l1((const int8_t*) w40 + 448);
        xnn_prefetch_to_l1((const int8_t*) w41 + 448);
        xnn_prefetch_to_l1((const int8_t*) w42 + 448);
        xnn_prefetch_to_l1((const int8_t*) w43 + 448);
        xnn_prefetch_to_l1((const int8_t*) w44 + 448);
        xnn_prefetch_to_l1((const int8_t*) w45 + 448);
        xnn_prefetch_to_l1((const int8_t*) w46 + 448);
        xnn_prefetch_to_l1((const int8_t*) w47 + 448);
        xnn_prefetch_to_l1((const int8_t*) w48 + 448);
        xnn_prefetch_to_l1((const int8_t*) w49 + 448);
        xnn_prefetch_to_l1((const int8_t*) w50 + 448);
        xnn_prefetch_to_l1((const int8_t*) w51 + 448);
        xnn_prefetch_to_l1((const int8_t*) w52 + 448);
        xnn_prefetch_to_l1((const int8_t*) w53 + 448);
        xnn_prefetch_to_l1((const int8_t*) w54 + 448);
        xnn_prefetch_to_l1((const int8_t*) w55 + 448);
        xnn_prefetch_to_l1((const int8_t*) w56 + 448);
        xnn_prefetch_to_l1((const int8_t*) w57 + 448);
        xnn_prefetch_to_l1((const int8_t*) w58 + 448);
        xnn_prefetch_to_l1((const int8_t*) w59 + 448);
        xnn_prefetch_to_l1((const int8_t*) w60 + 448);
        xnn_prefetch_to_l1((const int8_t*) w61 + 448);
        xnn_prefetch_to_l1((const int8_t*) w62 + 448);
        xnn_prefetch_to_l1((const int8_t*) w63 + 448);

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc8 = _mm256_dpbusd_epi32(vacc8, vone, v8);
        vacc16 = _mm256_dpbusd_epi32(vacc16, vone, v16);
        vacc24 = _mm256_dpbusd_epi32(vacc24, vone, v24);
        vacc32 = _mm256_dpbusd_epi32(vacc32, vone, v32);
        vacc40 = _mm256_dpbusd_epi32(vacc40, vone, v40);
        vacc48 = _mm256_dpbusd_epi32(vacc48, vone, v48);
        vacc56 = _mm256_dpbusd_epi32(vacc56, vone, v56);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v8);
        _mm256_storeu_si256((__m256i *)&out[64],  v16);
        _mm256_storeu_si256((__m256i *)&out[96],  v24);
        _mm256_storeu_si256((__m256i *)&out[128],  v32);
        _mm256_storeu_si256((__m256i *)&out[160],  v40);
        _mm256_storeu_si256((__m256i *)&out[192],  v48);
        _mm256_storeu_si256((__m256i *)&out[224],  v56);

        out += 256;
      }

      __m256i vksum0 = _mm256_mullo_epi32(vacc0, vzeropoint);
      __m256i vksum8 = _mm256_mullo_epi32(vacc8, vzeropoint);
      __m256i vksum16 = _mm256_mullo_epi32(vacc16, vzeropoint);
      __m256i vksum24 = _mm256_mullo_epi32(vacc24, vzeropoint);
      __m256i vksum32 = _mm256_mullo_epi32(vacc32, vzeropoint);
      __m256i vksum40 = _mm256_mullo_epi32(vacc40, vzeropoint);
      __m256i vksum48 = _mm256_mullo_epi32(vacc48, vzeropoint);
      __m256i vksum56 = _mm256_mullo_epi32(vacc56, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      __m256i vpack8 =  _mm256_loadu_si256((const __m256i*) (packed_b + 8));
      __m256i vpack16 =  _mm256_loadu_si256((const __m256i*) (packed_b + 16));
      __m256i vpack24 =  _mm256_loadu_si256((const __m256i*) (packed_b + 24));
      __m256i vpack32 =  _mm256_loadu_si256((const __m256i*) (packed_b + 32));
      __m256i vpack40 =  _mm256_loadu_si256((const __m256i*) (packed_b + 40));
      __m256i vpack48 =  _mm256_loadu_si256((const __m256i*) (packed_b + 48));
      __m256i vpack56 =  _mm256_loadu_si256((const __m256i*) (packed_b + 56));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      vpack8 = _mm256_sub_epi32(vpack8, vksum8);
      vpack16 = _mm256_sub_epi32(vpack16, vksum16);
      vpack24 = _mm256_sub_epi32(vpack24, vksum24);
      vpack32 = _mm256_sub_epi32(vpack32, vksum32);
      vpack40 = _mm256_sub_epi32(vpack40, vksum40);
      vpack48 = _mm256_sub_epi32(vpack48, vksum48);
      vpack56 = _mm256_sub_epi32(vpack56, vksum56);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      _mm256_storeu_si256((__m256i *) (packed_b + 8), vpack8);
      _mm256_storeu_si256((__m256i *) (packed_b + 16), vpack16);
      _mm256_storeu_si256((__m256i *) (packed_b + 24), vpack24);
      _mm256_storeu_si256((__m256i *) (packed_b + 32), vpack32);
      _mm256_storeu_si256((__m256i *) (packed_b + 40), vpack40);
      _mm256_storeu_si256((__m256i *) (packed_b + 48), vpack48);
      _mm256_storeu_si256((__m256i *) (packed_b + 56), vpack56);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w63;
    }

    // NC remainder (1..63)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 63);

      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        for (nb = 0; nb < n; ++nb) {
          ((int32_t*) out)[nb] = b[nb];
        }
        b += n;
      } else {
        _mm256_storeu_si256((__m256i*) (out + 0), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 32), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 64), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 96), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 128), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 160), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 192), _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i*) (out + 224), _mm256_setzero_si256());
      }
      out += 64 * sizeof(int32_t);

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
      const int8_t* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const int8_t* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const int8_t* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const int8_t* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const int8_t* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const int8_t* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const int8_t* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const int8_t* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const int8_t* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const int8_t* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const int8_t* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const int8_t* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const int8_t* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const int8_t* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const int8_t* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }
      const int8_t* w31 = w30 + kc;
      if XNN_UNPREDICTABLE(n < 32) {
        w31 = w30;
      }
      const int8_t* w32 = w31 + kc;
      if XNN_UNPREDICTABLE(n <= 32) {
        w32 = w31;
      }
      const int8_t* w33 = w32 + kc;
      if XNN_UNPREDICTABLE(n < 34) {
        w33 = w32;
      }
      const int8_t* w34 = w33 + kc;
      if XNN_UNPREDICTABLE(n <= 34) {
        w34 = w33;
      }
      const int8_t* w35 = w34 + kc;
      if XNN_UNPREDICTABLE(n < 36) {
        w35 = w34;
      }
      const int8_t* w36 = w35 + kc;
      if XNN_UNPREDICTABLE(n <= 36) {
        w36 = w35;
      }
      const int8_t* w37 = w36 + kc;
      if XNN_UNPREDICTABLE(n < 38) {
        w37 = w36;
      }
      const int8_t* w38 = w37 + kc;
      if XNN_UNPREDICTABLE(n <= 38) {
        w38 = w37;
      }
      const int8_t* w39 = w38 + kc;
      if XNN_UNPREDICTABLE(n < 40) {
        w39 = w38;
      }
      const int8_t* w40 = w39 + kc;
      if XNN_UNPREDICTABLE(n <= 40) {
        w40 = w39;
      }
      const int8_t* w41 = w40 + kc;
      if XNN_UNPREDICTABLE(n < 42) {
        w41 = w40;
      }
      const int8_t* w42 = w41 + kc;
      if XNN_UNPREDICTABLE(n <= 42) {
        w42 = w41;
      }
      const int8_t* w43 = w42 + kc;
      if XNN_UNPREDICTABLE(n < 44) {
        w43 = w42;
      }
      const int8_t* w44 = w43 + kc;
      if XNN_UNPREDICTABLE(n <= 44) {
        w44 = w43;
      }
      const int8_t* w45 = w44 + kc;
      if XNN_UNPREDICTABLE(n < 46) {
        w45 = w44;
      }
      const int8_t* w46 = w45 + kc;
      if XNN_UNPREDICTABLE(n <= 46) {
        w46 = w45;
      }
      const int8_t* w47 = w46 + kc;
      if XNN_UNPREDICTABLE(n < 48) {
        w47 = w46;
      }
      const int8_t* w48 = w47 + kc;
      if XNN_UNPREDICTABLE(n <= 48) {
        w48 = w47;
      }
      const int8_t* w49 = w48 + kc;
      if XNN_UNPREDICTABLE(n < 50) {
        w49 = w48;
      }
      const int8_t* w50 = w49 + kc;
      if XNN_UNPREDICTABLE(n <= 50) {
        w50 = w49;
      }
      const int8_t* w51 = w50 + kc;
      if XNN_UNPREDICTABLE(n < 52) {
        w51 = w50;
      }
      const int8_t* w52 = w51 + kc;
      if XNN_UNPREDICTABLE(n <= 52) {
        w52 = w51;
      }
      const int8_t* w53 = w52 + kc;
      if XNN_UNPREDICTABLE(n < 54) {
        w53 = w52;
      }
      const int8_t* w54 = w53 + kc;
      if XNN_UNPREDICTABLE(n <= 54) {
        w54 = w53;
      }
      const int8_t* w55 = w54 + kc;
      if XNN_UNPREDICTABLE(n < 56) {
        w55 = w54;
      }
      const int8_t* w56 = w55 + kc;
      if XNN_UNPREDICTABLE(n <= 56) {
        w56 = w55;
      }
      const int8_t* w57 = w56 + kc;
      if XNN_UNPREDICTABLE(n < 58) {
        w57 = w56;
      }
      const int8_t* w58 = w57 + kc;
      if XNN_UNPREDICTABLE(n <= 58) {
        w58 = w57;
      }
      const int8_t* w59 = w58 + kc;
      if XNN_UNPREDICTABLE(n < 60) {
        w59 = w58;
      }
      const int8_t* w60 = w59 + kc;
      if XNN_UNPREDICTABLE(n <= 60) {
        w60 = w59;
      }
      const int8_t* w61 = w60 + kc;
      if XNN_UNPREDICTABLE(n < 62) {
        w61 = w60;
      }
      const int8_t* w62 = w61 + kc;
      if XNN_UNPREDICTABLE(n <= 62) {
        w62 = w61;
      }
      const int8_t* w63 = w62 + kc;
      if XNN_UNPREDICTABLE(n < 64) {
        w63 = w62;
      }

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc8 = _mm256_setzero_si256();
      __m256i vacc16 = _mm256_setzero_si256();
      __m256i vacc24 = _mm256_setzero_si256();
      __m256i vacc32 = _mm256_setzero_si256();
      __m256i vacc40 = _mm256_setzero_si256();
      __m256i vacc48 = _mm256_setzero_si256();
      __m256i vacc56 = _mm256_setzero_si256();

      size_t k = kc;
      // KC main loop multiple of 64x4
      for (; k >= 4; k -= 4) {
        __m256i v0 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w1)), 0x02);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w2)), 0x04);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w3)), 0x08);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w4)), 0x10);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w5)), 0x20);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w6)), 0x40);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) unaligned_load_u32(w7)), 0x80);
        __m256i v8 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w8));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w9)), 0x02);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w10)), 0x04);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w11)), 0x08);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w12)), 0x10);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w13)), 0x20);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w14)), 0x40);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) unaligned_load_u32(w15)), 0x80);
        __m256i v16 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w16));
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w17)), 0x02);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w18)), 0x04);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w19)), 0x08);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w20)), 0x10);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w21)), 0x20);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w22)), 0x40);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) unaligned_load_u32(w23)), 0x80);
        __m256i v24 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w24));
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w25)), 0x02);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w26)), 0x04);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w27)), 0x08);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w28)), 0x10);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w29)), 0x20);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w30)), 0x40);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) unaligned_load_u32(w31)), 0x80);
        __m256i v32 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w32));
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w33)), 0x02);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w34)), 0x04);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w35)), 0x08);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w36)), 0x10);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w37)), 0x20);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w38)), 0x40);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) unaligned_load_u32(w39)), 0x80);
        __m256i v40 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w40));
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w41)), 0x02);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w42)), 0x04);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w43)), 0x08);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w44)), 0x10);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w45)), 0x20);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w46)), 0x40);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) unaligned_load_u32(w47)), 0x80);
        __m256i v48 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w48));
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w49)), 0x02);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w50)), 0x04);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w51)), 0x08);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w52)), 0x10);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w53)), 0x20);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w54)), 0x40);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) unaligned_load_u32(w55)), 0x80);
        __m256i v56 = _mm256_set1_epi32((int32_t) unaligned_load_u32(w56));
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w57)), 0x02);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w58)), 0x04);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w59)), 0x08);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w60)), 0x10);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w61)), 0x20);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w62)), 0x40);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) unaligned_load_u32(w63)), 0x80);
        xnn_prefetch_to_l1((const int8_t*) w0 + 448);
        xnn_prefetch_to_l1((const int8_t*) w1 + 448);
        xnn_prefetch_to_l1((const int8_t*) w2 + 448);
        xnn_prefetch_to_l1((const int8_t*) w3 + 448);
        xnn_prefetch_to_l1((const int8_t*) w4 + 448);
        xnn_prefetch_to_l1((const int8_t*) w5 + 448);
        xnn_prefetch_to_l1((const int8_t*) w6 + 448);
        xnn_prefetch_to_l1((const int8_t*) w7 + 448);
        xnn_prefetch_to_l1((const int8_t*) w8 + 448);
        xnn_prefetch_to_l1((const int8_t*) w9 + 448);
        xnn_prefetch_to_l1((const int8_t*) w10 + 448);
        xnn_prefetch_to_l1((const int8_t*) w11 + 448);
        xnn_prefetch_to_l1((const int8_t*) w12 + 448);
        xnn_prefetch_to_l1((const int8_t*) w13 + 448);
        xnn_prefetch_to_l1((const int8_t*) w14 + 448);
        xnn_prefetch_to_l1((const int8_t*) w15 + 448);
        xnn_prefetch_to_l1((const int8_t*) w16 + 448);
        xnn_prefetch_to_l1((const int8_t*) w17 + 448);
        xnn_prefetch_to_l1((const int8_t*) w18 + 448);
        xnn_prefetch_to_l1((const int8_t*) w19 + 448);
        xnn_prefetch_to_l1((const int8_t*) w20 + 448);
        xnn_prefetch_to_l1((const int8_t*) w21 + 448);
        xnn_prefetch_to_l1((const int8_t*) w22 + 448);
        xnn_prefetch_to_l1((const int8_t*) w23 + 448);
        xnn_prefetch_to_l1((const int8_t*) w24 + 448);
        xnn_prefetch_to_l1((const int8_t*) w25 + 448);
        xnn_prefetch_to_l1((const int8_t*) w26 + 448);
        xnn_prefetch_to_l1((const int8_t*) w27 + 448);
        xnn_prefetch_to_l1((const int8_t*) w28 + 448);
        xnn_prefetch_to_l1((const int8_t*) w29 + 448);
        xnn_prefetch_to_l1((const int8_t*) w30 + 448);
        xnn_prefetch_to_l1((const int8_t*) w31 + 448);
        xnn_prefetch_to_l1((const int8_t*) w32 + 448);
        xnn_prefetch_to_l1((const int8_t*) w33 + 448);
        xnn_prefetch_to_l1((const int8_t*) w34 + 448);
        xnn_prefetch_to_l1((const int8_t*) w35 + 448);
        xnn_prefetch_to_l1((const int8_t*) w36 + 448);
        xnn_prefetch_to_l1((const int8_t*) w37 + 448);
        xnn_prefetch_to_l1((const int8_t*) w38 + 448);
        xnn_prefetch_to_l1((const int8_t*) w39 + 448);
        xnn_prefetch_to_l1((const int8_t*) w40 + 448);
        xnn_prefetch_to_l1((const int8_t*) w41 + 448);
        xnn_prefetch_to_l1((const int8_t*) w42 + 448);
        xnn_prefetch_to_l1((const int8_t*) w43 + 448);
        xnn_prefetch_to_l1((const int8_t*) w44 + 448);
        xnn_prefetch_to_l1((const int8_t*) w45 + 448);
        xnn_prefetch_to_l1((const int8_t*) w46 + 448);
        xnn_prefetch_to_l1((const int8_t*) w47 + 448);
        xnn_prefetch_to_l1((const int8_t*) w48 + 448);
        xnn_prefetch_to_l1((const int8_t*) w49 + 448);
        xnn_prefetch_to_l1((const int8_t*) w50 + 448);
        xnn_prefetch_to_l1((const int8_t*) w51 + 448);
        xnn_prefetch_to_l1((const int8_t*) w52 + 448);
        xnn_prefetch_to_l1((const int8_t*) w53 + 448);
        xnn_prefetch_to_l1((const int8_t*) w54 + 448);
        xnn_prefetch_to_l1((const int8_t*) w55 + 448);
        xnn_prefetch_to_l1((const int8_t*) w56 + 448);
        xnn_prefetch_to_l1((const int8_t*) w57 + 448);
        xnn_prefetch_to_l1((const int8_t*) w58 + 448);
        xnn_prefetch_to_l1((const int8_t*) w59 + 448);
        xnn_prefetch_to_l1((const int8_t*) w60 + 448);
        xnn_prefetch_to_l1((const int8_t*) w61 + 448);
        xnn_prefetch_to_l1((const int8_t*) w62 + 448);
        xnn_prefetch_to_l1((const int8_t*) w63 + 448);

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc8 = _mm256_dpbusd_epi32(vacc8, vone, v8);
        vacc16 = _mm256_dpbusd_epi32(vacc16, vone, v16);
        vacc24 = _mm256_dpbusd_epi32(vacc24, vone, v24);
        vacc32 = _mm256_dpbusd_epi32(vacc32, vone, v32);
        vacc40 = _mm256_dpbusd_epi32(vacc40, vone, v40);
        vacc48 = _mm256_dpbusd_epi32(vacc48, vone, v48);
        vacc56 = _mm256_dpbusd_epi32(vacc56, vone, v56);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v8);
        _mm256_storeu_si256((__m256i *)&out[64],  v16);
        _mm256_storeu_si256((__m256i *)&out[96],  v24);
        _mm256_storeu_si256((__m256i *)&out[128],  v32);
        _mm256_storeu_si256((__m256i *)&out[160],  v40);
        _mm256_storeu_si256((__m256i *)&out[192],  v48);
        _mm256_storeu_si256((__m256i *)&out[224],  v56);

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
        w31 += 4;
        w32 += 4;
        w33 += 4;
        w34 += 4;
        w35 += 4;
        w36 += 4;
        w37 += 4;
        w38 += 4;
        w39 += 4;
        w40 += 4;
        w41 += 4;
        w42 += 4;
        w43 += 4;
        w44 += 4;
        w45 += 4;
        w46 += 4;
        w47 += 4;
        w48 += 4;
        w49 += 4;
        w50 += 4;
        w51 += 4;
        w52 += 4;
        w53 += 4;
        w54 += 4;
        w55 += 4;
        w56 += 4;
        w57 += 4;
        w58 += 4;
        w59 += 4;
        w60 += 4;
        w61 += 4;
        w62 += 4;
        w63 += 4;
        out += 256;
      }

      // KC remainder of 1..3
      if (k != 0) {
        assert(k >= 1 && k <= 3);

        __m256i v0 = _mm256_set1_epi32((int32_t) safe_load_u32(w0, k));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w1, k)), 0x02);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w2, k)), 0x04);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w3, k)), 0x08);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w4, k)), 0x10);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w5, k)), 0x20);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w6, k)), 0x40);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi32((int32_t) safe_load_u32(w7, k)), 0x80);
        __m256i v8 = _mm256_set1_epi32((int32_t) safe_load_u32(w8, k));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w9, k)), 0x02);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w10, k)), 0x04);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w11, k)), 0x08);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w12, k)), 0x10);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w13, k)), 0x20);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w14, k)), 0x40);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi32((int32_t) safe_load_u32(w15, k)), 0x80);
        __m256i v16 = _mm256_set1_epi32((int32_t) safe_load_u32(w16, k));
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w17, k)), 0x02);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w18, k)), 0x04);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w19, k)), 0x08);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w20, k)), 0x10);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w21, k)), 0x20);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w22, k)), 0x40);
        v16 = _mm256_blend_epi32(v16, _mm256_set1_epi32((int32_t) safe_load_u32(w23, k)), 0x80);
        __m256i v24 = _mm256_set1_epi32((int32_t) safe_load_u32(w24, k));
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w25, k)), 0x02);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w26, k)), 0x04);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w27, k)), 0x08);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w28, k)), 0x10);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w29, k)), 0x20);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w30, k)), 0x40);
        v24 = _mm256_blend_epi32(v24, _mm256_set1_epi32((int32_t) safe_load_u32(w31, k)), 0x80);
        __m256i v32 = _mm256_set1_epi32((int32_t) safe_load_u32(w32, k));
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w33, k)), 0x02);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w34, k)), 0x04);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w35, k)), 0x08);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w36, k)), 0x10);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w37, k)), 0x20);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w38, k)), 0x40);
        v32 = _mm256_blend_epi32(v32, _mm256_set1_epi32((int32_t) safe_load_u32(w39, k)), 0x80);
        __m256i v40 = _mm256_set1_epi32((int32_t) safe_load_u32(w40, k));
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w41, k)), 0x02);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w42, k)), 0x04);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w43, k)), 0x08);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w44, k)), 0x10);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w45, k)), 0x20);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w46, k)), 0x40);
        v40 = _mm256_blend_epi32(v40, _mm256_set1_epi32((int32_t) safe_load_u32(w47, k)), 0x80);
        __m256i v48 = _mm256_set1_epi32((int32_t) safe_load_u32(w48, k));
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w49, k)), 0x02);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w50, k)), 0x04);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w51, k)), 0x08);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w52, k)), 0x10);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w53, k)), 0x20);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w54, k)), 0x40);
        v48 = _mm256_blend_epi32(v48, _mm256_set1_epi32((int32_t) safe_load_u32(w55, k)), 0x80);
        __m256i v56 = _mm256_set1_epi32((int32_t) safe_load_u32(w56, k));
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w57, k)), 0x02);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w58, k)), 0x04);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w59, k)), 0x08);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w60, k)), 0x10);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w61, k)), 0x20);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w62, k)), 0x40);
        v56 = _mm256_blend_epi32(v56, _mm256_set1_epi32((int32_t) safe_load_u32(w63, k)), 0x80);

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
        w16 += k;
        w17 += k;
        w18 += k;
        w19 += k;
        w20 += k;
        w21 += k;
        w22 += k;
        w23 += k;
        w24 += k;
        w25 += k;
        w26 += k;
        w27 += k;
        w28 += k;
        w29 += k;
        w30 += k;
        w31 += k;
        w32 += k;
        w33 += k;
        w34 += k;
        w35 += k;
        w36 += k;
        w37 += k;
        w38 += k;
        w39 += k;
        w40 += k;
        w41 += k;
        w42 += k;
        w43 += k;
        w44 += k;
        w45 += k;
        w46 += k;
        w47 += k;
        w48 += k;
        w49 += k;
        w50 += k;
        w51 += k;
        w52 += k;
        w53 += k;
        w54 += k;
        w55 += k;
        w56 += k;
        w57 += k;
        w58 += k;
        w59 += k;
        w60 += k;
        w61 += k;
        w62 += k;
        w63 += k;

        xnn_prefetch_to_l1((const int8_t*) w0 + 448);
        xnn_prefetch_to_l1((const int8_t*) w1 + 448);
        xnn_prefetch_to_l1((const int8_t*) w2 + 448);
        xnn_prefetch_to_l1((const int8_t*) w3 + 448);
        xnn_prefetch_to_l1((const int8_t*) w4 + 448);
        xnn_prefetch_to_l1((const int8_t*) w5 + 448);
        xnn_prefetch_to_l1((const int8_t*) w6 + 448);
        xnn_prefetch_to_l1((const int8_t*) w7 + 448);
        xnn_prefetch_to_l1((const int8_t*) w8 + 448);
        xnn_prefetch_to_l1((const int8_t*) w9 + 448);
        xnn_prefetch_to_l1((const int8_t*) w10 + 448);
        xnn_prefetch_to_l1((const int8_t*) w11 + 448);
        xnn_prefetch_to_l1((const int8_t*) w12 + 448);
        xnn_prefetch_to_l1((const int8_t*) w13 + 448);
        xnn_prefetch_to_l1((const int8_t*) w14 + 448);
        xnn_prefetch_to_l1((const int8_t*) w15 + 448);
        xnn_prefetch_to_l1((const int8_t*) w16 + 448);
        xnn_prefetch_to_l1((const int8_t*) w17 + 448);
        xnn_prefetch_to_l1((const int8_t*) w18 + 448);
        xnn_prefetch_to_l1((const int8_t*) w19 + 448);
        xnn_prefetch_to_l1((const int8_t*) w20 + 448);
        xnn_prefetch_to_l1((const int8_t*) w21 + 448);
        xnn_prefetch_to_l1((const int8_t*) w22 + 448);
        xnn_prefetch_to_l1((const int8_t*) w23 + 448);
        xnn_prefetch_to_l1((const int8_t*) w24 + 448);
        xnn_prefetch_to_l1((const int8_t*) w25 + 448);
        xnn_prefetch_to_l1((const int8_t*) w26 + 448);
        xnn_prefetch_to_l1((const int8_t*) w27 + 448);
        xnn_prefetch_to_l1((const int8_t*) w28 + 448);
        xnn_prefetch_to_l1((const int8_t*) w29 + 448);
        xnn_prefetch_to_l1((const int8_t*) w30 + 448);
        xnn_prefetch_to_l1((const int8_t*) w31 + 448);
        xnn_prefetch_to_l1((const int8_t*) w32 + 448);
        xnn_prefetch_to_l1((const int8_t*) w33 + 448);
        xnn_prefetch_to_l1((const int8_t*) w34 + 448);
        xnn_prefetch_to_l1((const int8_t*) w35 + 448);
        xnn_prefetch_to_l1((const int8_t*) w36 + 448);
        xnn_prefetch_to_l1((const int8_t*) w37 + 448);
        xnn_prefetch_to_l1((const int8_t*) w38 + 448);
        xnn_prefetch_to_l1((const int8_t*) w39 + 448);
        xnn_prefetch_to_l1((const int8_t*) w40 + 448);
        xnn_prefetch_to_l1((const int8_t*) w41 + 448);
        xnn_prefetch_to_l1((const int8_t*) w42 + 448);
        xnn_prefetch_to_l1((const int8_t*) w43 + 448);
        xnn_prefetch_to_l1((const int8_t*) w44 + 448);
        xnn_prefetch_to_l1((const int8_t*) w45 + 448);
        xnn_prefetch_to_l1((const int8_t*) w46 + 448);
        xnn_prefetch_to_l1((const int8_t*) w47 + 448);
        xnn_prefetch_to_l1((const int8_t*) w48 + 448);
        xnn_prefetch_to_l1((const int8_t*) w49 + 448);
        xnn_prefetch_to_l1((const int8_t*) w50 + 448);
        xnn_prefetch_to_l1((const int8_t*) w51 + 448);
        xnn_prefetch_to_l1((const int8_t*) w52 + 448);
        xnn_prefetch_to_l1((const int8_t*) w53 + 448);
        xnn_prefetch_to_l1((const int8_t*) w54 + 448);
        xnn_prefetch_to_l1((const int8_t*) w55 + 448);
        xnn_prefetch_to_l1((const int8_t*) w56 + 448);
        xnn_prefetch_to_l1((const int8_t*) w57 + 448);
        xnn_prefetch_to_l1((const int8_t*) w58 + 448);
        xnn_prefetch_to_l1((const int8_t*) w59 + 448);
        xnn_prefetch_to_l1((const int8_t*) w60 + 448);
        xnn_prefetch_to_l1((const int8_t*) w61 + 448);
        xnn_prefetch_to_l1((const int8_t*) w62 + 448);
        xnn_prefetch_to_l1((const int8_t*) w63 + 448);

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc8 = _mm256_dpbusd_epi32(vacc8, vone, v8);
        vacc16 = _mm256_dpbusd_epi32(vacc16, vone, v16);
        vacc24 = _mm256_dpbusd_epi32(vacc24, vone, v24);
        vacc32 = _mm256_dpbusd_epi32(vacc32, vone, v32);
        vacc40 = _mm256_dpbusd_epi32(vacc40, vone, v40);
        vacc48 = _mm256_dpbusd_epi32(vacc48, vone, v48);
        vacc56 = _mm256_dpbusd_epi32(vacc56, vone, v56);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v8);
        _mm256_storeu_si256((__m256i *)&out[64],  v16);
        _mm256_storeu_si256((__m256i *)&out[96],  v24);
        _mm256_storeu_si256((__m256i *)&out[128],  v32);
        _mm256_storeu_si256((__m256i *)&out[160],  v40);
        _mm256_storeu_si256((__m256i *)&out[192],  v48);
        _mm256_storeu_si256((__m256i *)&out[224],  v56);

        out += 256;
      }

      __m256i vksum0 = _mm256_mullo_epi32(vacc0, vzeropoint);
      __m256i vksum8 = _mm256_mullo_epi32(vacc8, vzeropoint);
      __m256i vksum16 = _mm256_mullo_epi32(vacc16, vzeropoint);
      __m256i vksum24 = _mm256_mullo_epi32(vacc24, vzeropoint);
      __m256i vksum32 = _mm256_mullo_epi32(vacc32, vzeropoint);
      __m256i vksum40 = _mm256_mullo_epi32(vacc40, vzeropoint);
      __m256i vksum48 = _mm256_mullo_epi32(vacc48, vzeropoint);
      __m256i vksum56 = _mm256_mullo_epi32(vacc56, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      __m256i vpack8 =  _mm256_loadu_si256((const __m256i*) (packed_b + 8));
      __m256i vpack16 =  _mm256_loadu_si256((const __m256i*) (packed_b + 16));
      __m256i vpack24 =  _mm256_loadu_si256((const __m256i*) (packed_b + 24));
      __m256i vpack32 =  _mm256_loadu_si256((const __m256i*) (packed_b + 32));
      __m256i vpack40 =  _mm256_loadu_si256((const __m256i*) (packed_b + 40));
      __m256i vpack48 =  _mm256_loadu_si256((const __m256i*) (packed_b + 48));
      __m256i vpack56 =  _mm256_loadu_si256((const __m256i*) (packed_b + 56));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      vpack8 = _mm256_sub_epi32(vpack8, vksum8);
      vpack16 = _mm256_sub_epi32(vpack16, vksum16);
      vpack24 = _mm256_sub_epi32(vpack24, vksum24);
      vpack32 = _mm256_sub_epi32(vpack32, vksum32);
      vpack40 = _mm256_sub_epi32(vpack40, vksum40);
      vpack48 = _mm256_sub_epi32(vpack48, vksum48);
      vpack56 = _mm256_sub_epi32(vpack56, vksum56);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      _mm256_storeu_si256((__m256i *) (packed_b + 8), vpack8);
      _mm256_storeu_si256((__m256i *) (packed_b + 16), vpack16);
      _mm256_storeu_si256((__m256i *) (packed_b + 24), vpack24);
      _mm256_storeu_si256((__m256i *) (packed_b + 32), vpack32);
      _mm256_storeu_si256((__m256i *) (packed_b + 40), vpack40);
      _mm256_storeu_si256((__m256i *) (packed_b + 48), vpack48);
      _mm256_storeu_si256((__m256i *) (packed_b + 56), vpack56);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * kc;
  } while (--g != 0);
}
