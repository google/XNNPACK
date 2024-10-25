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

#include <immintrin.h>

#include "xnnpack/packw.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__avxvnni(
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
  const void* params) XNN_OOB_READS
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const __m256i vone = _mm256_set1_epi8(1);
  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 128): 128);
  __m256i vzeropoint = _mm256_set1_epi32((int32_t) izp);

  do {
    // NC main loop multiple of 16
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 16; n -= 16) {
      int32_t* packed_b = (int32_t*) out;
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

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc4 = _mm256_setzero_si256();
      __m256i vacc8 = _mm256_setzero_si256();
      __m256i vacc12 = _mm256_setzero_si256();

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

        vacc0 = _mm256_dpbusd_avx_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_avx_epi32(vacc4, vone, v4);
        vacc8 = _mm256_dpbusd_avx_epi32(vacc8, vone, v8);
        vacc12 = _mm256_dpbusd_avx_epi32(vacc12, vone, v12);

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

        const __m256i vmask = _mm256_srli_epi64(_mm256_set1_epi32(-1), (8 - k) * sizeof(int8_t) * 8);

        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        v0 = _mm256_and_si256(v0, vmask);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);
        v4 = _mm256_and_si256(v4, vmask);
        __m256i v8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w8));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w9)), 0x0C);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w10)), 0x30);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w11)), 0xC0);
        v8 = _mm256_and_si256(v8, vmask);
        __m256i v12 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w12));
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w13)), 0x0C);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w14)), 0x30);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w15)), 0xC0);
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

        vacc0 = _mm256_dpbusd_avx_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_avx_epi32(vacc4, vone, v4);
        vacc8 = _mm256_dpbusd_avx_epi32(vacc8, vone, v8);
        vacc12 = _mm256_dpbusd_avx_epi32(vacc12, vone, v12);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);
        _mm256_storeu_si256((__m256i *)&out[64],  v8);
        _mm256_storeu_si256((__m256i *)&out[96],  v12);

        out += 128;
      }

      __m256i vksum0 = _mm256_hadd_epi32(vacc0, vacc4);
      vksum0 = _mm256_permute4x64_epi64(vksum0, _MM_SHUFFLE(3, 1, 2, 0));
      __m256i vksum8 = _mm256_hadd_epi32(vacc8, vacc12);
      vksum8 = _mm256_permute4x64_epi64(vksum8, _MM_SHUFFLE(3, 1, 2, 0));
      vksum0 = _mm256_mullo_epi32(vksum0, vzeropoint);
      vksum8 = _mm256_mullo_epi32(vksum8, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      __m256i vpack8 =  _mm256_loadu_si256((const __m256i*) (packed_b + 8));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      vpack8 = _mm256_sub_epi32(vpack8, vksum8);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      _mm256_storeu_si256((__m256i *) (packed_b + 8), vpack8);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 15);

      int32_t* packed_b = (int32_t*) out;
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

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc4 = _mm256_setzero_si256();
      __m256i vacc8 = _mm256_setzero_si256();
      __m256i vacc12 = _mm256_setzero_si256();

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

        vacc0 = _mm256_dpbusd_avx_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_avx_epi32(vacc4, vone, v4);
        vacc8 = _mm256_dpbusd_avx_epi32(vacc8, vone, v8);
        vacc12 = _mm256_dpbusd_avx_epi32(vacc12, vone, v12);

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

        const __m256i vmask = _mm256_srli_epi64(_mm256_set1_epi32(-1), (8 - k) * sizeof(int8_t) * 8);

        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        v0 = _mm256_and_si256(v0, vmask);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);
        v4 = _mm256_and_si256(v4, vmask);
        __m256i v8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w8));
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w9)), 0x0C);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w10)), 0x30);
        v8 = _mm256_blend_epi32(v8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w11)), 0xC0);
        v8 = _mm256_and_si256(v8, vmask);
        __m256i v12 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w12));
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w13)), 0x0C);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w14)), 0x30);
        v12 = _mm256_blend_epi32(v12, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w15)), 0xC0);
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

        vacc0 = _mm256_dpbusd_avx_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_avx_epi32(vacc4, vone, v4);
        vacc8 = _mm256_dpbusd_avx_epi32(vacc8, vone, v8);
        vacc12 = _mm256_dpbusd_avx_epi32(vacc12, vone, v12);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);
        _mm256_storeu_si256((__m256i *)&out[64],  v8);
        _mm256_storeu_si256((__m256i *)&out[96],  v12);

        out += 128;
      }

      __m256i vksum0 = _mm256_hadd_epi32(vacc0, vacc4);
      vksum0 = _mm256_permute4x64_epi64(vksum0, _MM_SHUFFLE(3, 1, 2, 0));
      __m256i vksum8 = _mm256_hadd_epi32(vacc8, vacc12);
      vksum8 = _mm256_permute4x64_epi64(vksum8, _MM_SHUFFLE(3, 1, 2, 0));
      vksum0 = _mm256_mullo_epi32(vksum0, vzeropoint);
      vksum8 = _mm256_mullo_epi32(vksum8, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      __m256i vpack8 =  _mm256_loadu_si256((const __m256i*) (packed_b + 8));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      vpack8 = _mm256_sub_epi32(vpack8, vksum8);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      _mm256_storeu_si256((__m256i *) (packed_b + 8), vpack8);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * kc;
  } while (--g != 0);
}
