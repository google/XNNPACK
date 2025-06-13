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

XNN_INLINE static uint32_t safe_load_u32(const void* src, size_t k) {
  uint32_t value = 0;
  const uint8_t* bytes = (const uint8_t*)src;
  for (size_t i = 0; i < k; ++i) {
    value |= (uint32_t) bytes[i] << (i * 8);
  }
  return value;
}


void xnn_qs8_packw_gemm_goi_ukernel_x64c4__avx256vnni(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
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
      const int8_t* w1 = w0 + n_stride;
      const int8_t* w2 = w1 + n_stride;
      const int8_t* w3 = w2 + n_stride;
      const int8_t* w4 = w3 + n_stride;
      const int8_t* w5 = w4 + n_stride;
      const int8_t* w6 = w5 + n_stride;
      const int8_t* w7 = w6 + n_stride;
      const int8_t* w8 = w7 + n_stride;
      const int8_t* w9 = w8 + n_stride;
      const int8_t* w10 = w9 + n_stride;
      const int8_t* w11 = w10 + n_stride;
      const int8_t* w12 = w11 + n_stride;
      const int8_t* w13 = w12 + n_stride;
      const int8_t* w14 = w13 + n_stride;
      const int8_t* w15 = w14 + n_stride;
      const int8_t* w16 = w15 + n_stride;
      const int8_t* w17 = w16 + n_stride;
      const int8_t* w18 = w17 + n_stride;
      const int8_t* w19 = w18 + n_stride;
      const int8_t* w20 = w19 + n_stride;
      const int8_t* w21 = w20 + n_stride;
      const int8_t* w22 = w21 + n_stride;
      const int8_t* w23 = w22 + n_stride;
      const int8_t* w24 = w23 + n_stride;
      const int8_t* w25 = w24 + n_stride;
      const int8_t* w26 = w25 + n_stride;
      const int8_t* w27 = w26 + n_stride;
      const int8_t* w28 = w27 + n_stride;
      const int8_t* w29 = w28 + n_stride;
      const int8_t* w30 = w29 + n_stride;
      const int8_t* w31 = w30 + n_stride;
      const int8_t* w32 = w31 + n_stride;
      const int8_t* w33 = w32 + n_stride;
      const int8_t* w34 = w33 + n_stride;
      const int8_t* w35 = w34 + n_stride;
      const int8_t* w36 = w35 + n_stride;
      const int8_t* w37 = w36 + n_stride;
      const int8_t* w38 = w37 + n_stride;
      const int8_t* w39 = w38 + n_stride;
      const int8_t* w40 = w39 + n_stride;
      const int8_t* w41 = w40 + n_stride;
      const int8_t* w42 = w41 + n_stride;
      const int8_t* w43 = w42 + n_stride;
      const int8_t* w44 = w43 + n_stride;
      const int8_t* w45 = w44 + n_stride;
      const int8_t* w46 = w45 + n_stride;
      const int8_t* w47 = w46 + n_stride;
      const int8_t* w48 = w47 + n_stride;
      const int8_t* w49 = w48 + n_stride;
      const int8_t* w50 = w49 + n_stride;
      const int8_t* w51 = w50 + n_stride;
      const int8_t* w52 = w51 + n_stride;
      const int8_t* w53 = w52 + n_stride;
      const int8_t* w54 = w53 + n_stride;
      const int8_t* w55 = w54 + n_stride;
      const int8_t* w56 = w55 + n_stride;
      const int8_t* w57 = w56 + n_stride;
      const int8_t* w58 = w57 + n_stride;
      const int8_t* w59 = w58 + n_stride;
      const int8_t* w60 = w59 + n_stride;
      const int8_t* w61 = w60 + n_stride;
      const int8_t* w62 = w61 + n_stride;
      const int8_t* w63 = w62 + n_stride;

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

      const int8_t* w1 = w0 + n_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + n_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + n_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + n_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + n_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + n_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + n_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const int8_t* w8 = w7 + n_stride;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int8_t* w9 = w8 + n_stride;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int8_t* w10 = w9 + n_stride;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int8_t* w11 = w10 + n_stride;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int8_t* w12 = w11 + n_stride;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int8_t* w13 = w12 + n_stride;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int8_t* w14 = w13 + n_stride;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const int8_t* w15 = w14 + n_stride;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const int8_t* w16 = w15 + n_stride;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const int8_t* w17 = w16 + n_stride;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const int8_t* w18 = w17 + n_stride;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const int8_t* w19 = w18 + n_stride;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const int8_t* w20 = w19 + n_stride;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const int8_t* w21 = w20 + n_stride;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const int8_t* w22 = w21 + n_stride;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const int8_t* w23 = w22 + n_stride;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const int8_t* w24 = w23 + n_stride;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const int8_t* w25 = w24 + n_stride;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const int8_t* w26 = w25 + n_stride;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const int8_t* w27 = w26 + n_stride;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const int8_t* w28 = w27 + n_stride;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const int8_t* w29 = w28 + n_stride;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const int8_t* w30 = w29 + n_stride;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }
      const int8_t* w31 = w30 + n_stride;
      if XNN_UNPREDICTABLE(n < 32) {
        w31 = w30;
      }
      const int8_t* w32 = w31 + n_stride;
      if XNN_UNPREDICTABLE(n <= 32) {
        w32 = w31;
      }
      const int8_t* w33 = w32 + n_stride;
      if XNN_UNPREDICTABLE(n < 34) {
        w33 = w32;
      }
      const int8_t* w34 = w33 + n_stride;
      if XNN_UNPREDICTABLE(n <= 34) {
        w34 = w33;
      }
      const int8_t* w35 = w34 + n_stride;
      if XNN_UNPREDICTABLE(n < 36) {
        w35 = w34;
      }
      const int8_t* w36 = w35 + n_stride;
      if XNN_UNPREDICTABLE(n <= 36) {
        w36 = w35;
      }
      const int8_t* w37 = w36 + n_stride;
      if XNN_UNPREDICTABLE(n < 38) {
        w37 = w36;
      }
      const int8_t* w38 = w37 + n_stride;
      if XNN_UNPREDICTABLE(n <= 38) {
        w38 = w37;
      }
      const int8_t* w39 = w38 + n_stride;
      if XNN_UNPREDICTABLE(n < 40) {
        w39 = w38;
      }
      const int8_t* w40 = w39 + n_stride;
      if XNN_UNPREDICTABLE(n <= 40) {
        w40 = w39;
      }
      const int8_t* w41 = w40 + n_stride;
      if XNN_UNPREDICTABLE(n < 42) {
        w41 = w40;
      }
      const int8_t* w42 = w41 + n_stride;
      if XNN_UNPREDICTABLE(n <= 42) {
        w42 = w41;
      }
      const int8_t* w43 = w42 + n_stride;
      if XNN_UNPREDICTABLE(n < 44) {
        w43 = w42;
      }
      const int8_t* w44 = w43 + n_stride;
      if XNN_UNPREDICTABLE(n <= 44) {
        w44 = w43;
      }
      const int8_t* w45 = w44 + n_stride;
      if XNN_UNPREDICTABLE(n < 46) {
        w45 = w44;
      }
      const int8_t* w46 = w45 + n_stride;
      if XNN_UNPREDICTABLE(n <= 46) {
        w46 = w45;
      }
      const int8_t* w47 = w46 + n_stride;
      if XNN_UNPREDICTABLE(n < 48) {
        w47 = w46;
      }
      const int8_t* w48 = w47 + n_stride;
      if XNN_UNPREDICTABLE(n <= 48) {
        w48 = w47;
      }
      const int8_t* w49 = w48 + n_stride;
      if XNN_UNPREDICTABLE(n < 50) {
        w49 = w48;
      }
      const int8_t* w50 = w49 + n_stride;
      if XNN_UNPREDICTABLE(n <= 50) {
        w50 = w49;
      }
      const int8_t* w51 = w50 + n_stride;
      if XNN_UNPREDICTABLE(n < 52) {
        w51 = w50;
      }
      const int8_t* w52 = w51 + n_stride;
      if XNN_UNPREDICTABLE(n <= 52) {
        w52 = w51;
      }
      const int8_t* w53 = w52 + n_stride;
      if XNN_UNPREDICTABLE(n < 54) {
        w53 = w52;
      }
      const int8_t* w54 = w53 + n_stride;
      if XNN_UNPREDICTABLE(n <= 54) {
        w54 = w53;
      }
      const int8_t* w55 = w54 + n_stride;
      if XNN_UNPREDICTABLE(n < 56) {
        w55 = w54;
      }
      const int8_t* w56 = w55 + n_stride;
      if XNN_UNPREDICTABLE(n <= 56) {
        w56 = w55;
      }
      const int8_t* w57 = w56 + n_stride;
      if XNN_UNPREDICTABLE(n < 58) {
        w57 = w56;
      }
      const int8_t* w58 = w57 + n_stride;
      if XNN_UNPREDICTABLE(n <= 58) {
        w58 = w57;
      }
      const int8_t* w59 = w58 + n_stride;
      if XNN_UNPREDICTABLE(n < 60) {
        w59 = w58;
      }
      const int8_t* w60 = w59 + n_stride;
      if XNN_UNPREDICTABLE(n <= 60) {
        w60 = w59;
      }
      const int8_t* w61 = w60 + n_stride;
      if XNN_UNPREDICTABLE(n < 62) {
        w61 = w60;
      }
      const int8_t* w62 = w61 + n_stride;
      if XNN_UNPREDICTABLE(n <= 62) {
        w62 = w61;
      }
      const int8_t* w63 = w62 + n_stride;
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
